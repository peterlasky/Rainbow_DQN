import gymnasium as gym
import numpy as np
import torch
import random
import copy
from torch.optim import Adam
from types import SimpleNamespace
from typing import Dict, Tuple, List

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale_py not found. Some Atari environments may not be available.")

from modules.parameter_handler import ParameterHandler
from modules.policy.atari_policy_net import AtariPolicyNet
from modules.policy.policy_updater import PolicyUpdater
from modules.action_handlers import ActionHandler, VecActionHandler
from modules.evaluator import Evaluator
from modules.replay_buffers.get_buffers import get_replay_buffers
from modules.environment.get_environment import get_vectorized_envs, get_single_env
from modules.utils import PBar, Logger, Plotter, ipynb
from modules.filepath_manager import FilePathManager
from IPython.display import clear_output

class DQN:
    def __init__(self, params: SimpleNamespace):
        # Verifies & combines user and default parameters 
        parameter_handler = ParameterHandler(user_parameters=params)
        self.p = parameter_handler.get_parameters() # SimpleNamespace
    
        # Set seeds
        torch.manual_seed(self.p.seed); np.random.seed(self.p.seed); random.seed(self.p.seed)

        # Create file paths for logging
        self.filepaths = FilePathManager(self.p)

        # Get environments
        self.train_envs = get_vectorized_envs(train=True, p=self.p)
        self.eval_env = get_single_env(train=False, p=self.p)
        if self.p.record_interval is not None:
            self.record_env = get_single_env(train=False, record_video=True, video_dir = self.filepaths.video_dir, p=self.p)
        
        # Create the policy network and target network
        self.policy_net = AtariPolicyNet(self.p).to(self.p.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.p.device)
        self.target_net.eval()

        # Create the optimizer
        self.optimizer = Adam(self.policy_net.parameters(), lr=self.p.learning_rate)

        # Get replay buffers
        self.memory, self.n_step_memory = get_replay_buffers(self.p)

        # Create the policy updater
        self.policy_updater = PolicyUpdater(params=self.p, memory=self.memory, policy_net=self.policy_net, target_net=self.target_net, optimizer=self.optimizer)

        # Action handlers: generate actions and manages epsilon decay
        self.vec_action_handler = VecActionHandler(params=self.p, policy_net=self.policy_net, action_space=self.eval_env.action_space)
        self.action_handler = ActionHandler(params=self.p, policy_net=self.policy_net, action_space=self.eval_env.action_space)

        # Evaluator: evaluates the policy network
        self.evaluator = Evaluator(params=self.p, env=self.eval_env, action_handler=self.action_handler)

        # Record video: A separate mini-evaluator used to take advantage of the gymnasium video wrapper
        if self.p.record_interval is not None:
            self.video_recorder = Evaluator(params=self.p, env=self.record_env, action_handler=self.action_handler)

        # Create logger
        self.logger = Logger(filepaths=self.filepaths, note=self.p.note, params=vars(self.p))

        # Helper class for plotting in Jupyter
        self.plotter = Plotter(self.filepaths.plot_filepath)
        
        # Initialize progress bar
        self.pbar = PBar(max_steps=self.p.max_steps, increment=self.p.pbar_update_interval)
    
    def train(self) -> tuple[float, float]:
        ''' Train the policy network '''
        
        steps, episodes = 0, 0
        loss = 0
        dones = np.array([True] * self.p.n_envs)
        truncateds = np.array([False] * self.p.n_envs)
        frames, _ = self.train_envs.reset()
        if ipynb(): clear_output()
        
        # Start progress bar
        self.pbar.start()
        
        while True:
            steps += self.p.n_envs

            # Increment the episode count if done or truncated
            episodes += sum(dones | truncateds)

            # Reset environments if needed
            if any(dones | truncateds):
                frames, _ = self.train_envs.reset()

            # Determine if the agent should be in random mode
            rand_mode = (steps < self.p.random_starts)

            # Get the action from the action handler
            frames = frames[:,1:,:,:]  # remove the first frame from the state
            states = torch.from_numpy(frames).to(self.p.device)
            actions = self.vec_action_handler.get_actions(states, rand_mode=rand_mode, eval_mode=False)
            

            # Take a step in the environment
            frames, rewards, dones, truncateds, _ = self.train_envs.step(actions)

            # Store transitions in memory
            for i in range(self.p.n_envs):
                self.memory.add((frames[i], actions[i].item(), rewards[i], dones[i]))

            ### Periodic updates ####
            # Update the policy network
            if not rand_mode and steps % (self.p.policy_update_interval * self.p.n_envs) == 0:
                for _ in range(self.p.n_batch_updates):
                    loss = self.policy_updater.update()

            # Update the target network
            if steps % self.p.target_update_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Evaluate the policy network
            if steps % self.p.eval_interval == 0:
                self.evaluator.evaluate(steps=steps, recent_loss=loss)
                self.logger.save_to_log(self.evaluator.history_df)
                if ipynb():
                    if self.p.data_plotting:
                        clear_output()
                        self.plotter.plot_data(self.evaluator.history_df)
    

            # Record video
            if self.p.record_interval is not None and steps % self.p.record_interval == 0:
                self.video_recorder.evaluate(steps=steps, recent_loss=loss)

            # Save checkpoint
            if steps % self.p.checkpoint_interval == 0:
                self.logger.save_checkpoint(model=self.policy_net, optimizer=self.optimizer, steps=steps)

            # Update progress bar
            if steps % self.p.pbar_update_interval == 0:
                self.pbar.update(steps=steps, eps=episodes, avg=self.evaluator.avg, trailing_avg=self.evaluator.trailing_avg)

            # Check exit conditions
            if steps >= self.p.max_steps:
                break
            if self.evaluator.trailing_avg >= self.p.exit_trailing_average:
                break
            if self.pbar.elapsed_time >= self.p.exit_time_limit_seconds * 60:
                break

        # Clean up
        self._cleanup()
        time_elapsed = self.pbar.elapsed_time / 60
        trailing_avg = self.evaluator.trailing_avg
        return time_elapsed, trailing_avg

    def _cleanup(self):
        ''' Clean up resources to avoid memory leaks '''
        del self.memory._state_history
        self.train_envs.close()
        self.eval_env.close()
        if self.p.record_interval is not None:
            self.record_env.close()
