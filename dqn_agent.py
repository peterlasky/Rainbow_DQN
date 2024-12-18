from inspect import Parameter
import torch
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np

import gymnasium as gym
import ale_py; gym.register_envs(ale_py)


import datetime, copy, random
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# typing, dataclasses, and debugging
from typing import Dict

from IPython.display import clear_output

# Import custom classes
from modules.policy.atari_policy_net import AtariPolicyNet
from modules.action_handlers import ActionHandler, VecActionHandler
from modules.utils import FilePathManager, PBar, Logger, Plotter, ipynb
from modules.evaluator import Evaluator
from modules.replay_buffer.get_buffers import get_replay_buffers
from modules.environment.get_environment import get_vectorized_envs, get_single_env
from modules.parameters import ParameterHandler
from modules.policy.policy_updater import PolicyUpdater

class DQN:
    def __init__(self, p: Dict):

        ## Verify correct input parameters, add defaults, and unpack.  
        # Parameter handler will throw an error if the parameters are incorrect
        ph = ParameterHandler(p)
        verified_parameters = ph.get_parameters()
        for k, v in verified_parameters.items():
            setattr(self, k, v)
   
        # Set seeds. Seed will also be passed to the environment
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)

        # Set up the file structure for logging, create and/or erase as needed
        self.log_dir = os.path.join('logs',self.log_dir)
        self.filepaths = FilePathManager(log_dir            =self.log_dir, 
                                         name               =self.name, 
                                         overwrite_previous =self.overwrite_previous)

        # Custom-wrapped gymnasium environments (evaluations are non-vectorized for now)
        self.train_envs = get_vectorized_envs(
                        env_name        = self.env_name,
                        n_envs          = self.n_envs,
                        asynchronous    = self.asynchronous,
                        train           = True,
                        noop_min        = self.noop_min,
                        noop_max        = self.noop_max,
                        screen_size     = self.screen_size,
                        seed            = self.seed)
        self.eval_env = get_single_env(
                        env_name        = self.env_name,
                        noop_min        = self.noop_min,
                        noop_max        = self.noop_max,
                        screen_size     = self.screen_size,
                        seed            = self.seed)
        if self.record_interval is not None:
            self.record_env = get_single_env(
                        env_name        = self.env_name,
                        noop_min        = self.noop_min,
                        noop_max        = self.noop_max,
                        screen_size     = self.screen_size,
                        seed            = self.seed,
                        record_video    = True,
                        video_dir       = self.filepaths.video_dir)
        
        # Policy network and target network
        self.policy_net =    AtariPolicyNet(
                screen_size=        self.screen_size,
                n_actions=          self.eval_env.action_space.n,
                device=             self.device,
                categorical_DQN=    self.categorical_DQN,   # Categorical DQN requires a different output layer
                dueling=            self.dueling,           # Dueling DQN requires a different output layer
                noisy_linear=       self.noisy_linear,      # Noisy linear requires linear layers to be modified
                batch_norm=         self.batch_norm,        # Optional batch norm for convolutional layers
                layer_norm=         self.layer_norm,        # Optional layer norm for linear layers
                scale_around_zero=  self.scale_around_zero  # True lambda x: x/255.-0.5, False: lambda x: x/255.
                ).to(self.device)
        self.optimizer = Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        # Replay buffers (inc. optional per experience replay memory and configured for optional n-step learning)
        (self.memory, 
         self.n_step_memory) = get_replay_buffers(
                                prioritized_replay= self.prioritized_replay,
                                n_step_learning=    self.n_step_learning,
                                batch_size=         self.batch_size,
                                memory_size=        self.memory_size,
                                screen_size=        self.screen_size,
                                output_device=      self.device,
                                n_step_params=      self.n_step_params,
                                per_params=         self.per_params)
        
        ### Policy Updater -- vlass that manages and updates the policy network
        self.policy_updater = PolicyUpdater(
            batch_size=            self.batch_size,
            n_batch_updates=       self.n_batch_updates,
            memory=                self.memory,
            noisy_linear=          self.noisy_linear,
            doubleQ=               self.doubleQ,
            policy_net=            self.policy_net,
            target_net=            self.target_net,
            optimizer=             self.optimizer,
            gamma=                 self.gamma,
            gradient_clamping=     self.gradient_clamping,
            group_training_losses= self.group_training_losses,
            device=                self.device
        )
 

        # Action handlers: generates actions and manages epsilon decay
        self.vec_action_handler =  VecActionHandler(
                                        policy_net=     self.policy_net,
                                        n_envs=         self.n_envs,
                                        action_space=   self.eval_env.action_space,
                                        screen_size=    self.screen_size,
                                        device=         self.device,    
                                        epsilons=       self.epsilons)
        # Evaluation uses the non-vectorized environment
        self.action_handler =  ActionHandler(
                                        policy_net=     self.policy_net,
                                        action_space=   self.eval_env.action_space, 
                                        screen_size=    self.screen_size,
                                        device=         self.device,    
                                        epsilons=       self.epsilons)
        
        # Evaluator: evaluates the policy network, Evaluator.eval_df keeps the data history
        self.evaluator = Evaluator(env=     self.eval_env, 
                                   ah=      self.action_handler, 
                                   n=       self.n_games_per_eval, 
                                   trail=   self.trailing_avg_trail)
        
        # Record video: A separate mini-evaluator used to take advantage of the gymnasium video wrapper
        if self.record_interval is not None:
            self.video_dir = self.filepaths.video_dir
            self.video_recorder = Evaluator(env=self.record_env, ah=self.action_handler)
   
        # Helper classes for logging data to csv files and checkpoints
        self.logger = Logger(
            filepaths =         self.filepaths,  
            note=               self.note,
            params=             self.__dict__)
        
        # Helper class for plotting in Jupyter
        self.plotter = Plotter(self.filepaths.plot_filepath)
        
        # Initialize pbar using custom PBar class
        self.pbar = PBar(max_steps=self.max_steps, 
                         increment=self.pbar_update_interval)
    # ----  end .__init__() ----

    def train(self):
        ''' Train the policy network '''
        
        steps, episodes = 0, 0
        dones = np.array([True] * self.n_envs)
        truncateds = np.array([False] * self.n_envs)
        frames, _ = self.train_envs.reset()
        if ipynb(): clear_output()
        self.pbar.start()
        
        while True:
            steps += self.n_envs

            # Increment the episode count i if done or truncated
            episodes += sum(dones | truncateds)

            # Determine if the agent should be in random mode
            rand_mode = (steps < self.random_starts)

            # Get the action from the action handler
            frames = frames[:,1:,:,:]  # remove the first frame from the state
            # assert frames.shape == (self.n_envs, 4, self.screen_size, self.screen_size)

            actions = self.vec_action_handler.get_actions(states=frames, rand_mode=rand_mode, eval_mode=False)
            # assert actions.shape == (self.n_envs,), f"Actions shape is incorrect: expected {(self.n_envs,)}, got {actions.shape}"

            # Take a step in the environment
            frames, rewards, dones, truncateds, _ = self.train_envs.step(actions)

            # Concatenate the transitions and add them to the memory
            for i in range(self.n_envs):
                transition = (frames[i], actions[i], rewards[i], dones[i])

                # If n_step_learning is enabled, modify the transition
                if self.n_step_learning:
                    transitions = self.n_step_memory.add(transition)
                    if transitions is None:
                        continue  # Skip if transition is not ready

                # Store the transition in the replay buffer
                self.memory.add(transition)
            
            ''' Perform Periodic actions and check for early exit conditions '''
            # 1. Update pbar
            if steps % self.pbar_update_interval == 0: 
                self.pbar.update(steps=steps, eps=episodes, avg= self.evaluator.avg, trailing_avg=self.evaluator.trailing_avg)

            # 2. Update the policy network
            if steps % self.policy_update_interval_adjusted == 0:
                loss = self.policy_updater.update()

            # 3. Update the target network
            if steps % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # 4. Record video
            if self.record_interval is not None:
                if steps % self.record_interval == 0:
                    self.video_recorder.evaluate(steps, loss)
                
            # 5. Evaluate the policy network
            if steps % self.eval_interval == 0:
                self.evaluator.evaluate(steps, loss)
                if self.data_logging: self.logger.save_to_log(self.evaluator.history_df)
                if self.data_plotting: self.plotter.plot_data(self.evaluator.history_df)

            # 6. Save checkpoint
            if self.checkpoint_interval is not None:
                if steps % self.checkpoint_interval == 0:
                    self.logger.save_checkpoint(self.policy_net, self.optimizer, steps)

            # 7. Check exit conditions
            if (self.evaluator.avg > self.exit_trailing_average) or \
               (self.pbar.elapsed_time > self.exit_time_limit_seconds) or \
               (steps >= self.max_steps):
                break
        # -- end training loop --

    # Exit steps: After the training loop ends, perform the following actions:
        if self.record_interval is not None: self.video_recorder.evaluate(steps, 0)
        if ipynb(): clear_output()
        print('Exiting training loop....')
        if self.evaluator.avg > self.exit_trailing_average:
            print(f'Exit condition: trailing average reached {self.exit_trailing_average}')
        elif self.pbar.elapsed_time > self.exit_time_limit_seconds:
            print(f'Exit condition: time limit reached {self.exit_time_limit} mins')
        elif steps >= self.max_steps:
            print('Exit condition: max steps reached {self.max_steps}')
        
        hh_mm_ss = str(datetime.timedelta(seconds=round(self.pbar.elapsed_time,0)))

        if self.data_logging:
            self.logger.save_to_log(history_df=self.evaluator.history_df)
            if self.data_plotting:
                self.plotter.plot_data(history_df=self.evaluator.history_df, save_plot=True)
        if self.checkpoint_interval is not None:
            self.logger.save_checkpoint(self.policy_net, self.optimizer, steps)

        self.logger.append_elapsed_time(hh_mm_ss, steps)
        print(f'steps={steps}, episodes={episodes}')
        print(f'Best Score: {self.evaluator.best_score}')
        print(f'Trailing Avg (last {self.trailing_avg_trail}): {self.evaluator.trailing_avg}')
        print(f'Time elapsed: {hh_mm_ss}')
        
        time_elapsed = hh_mm_ss
        trailing_avg = self.evaluator.trailing_avg
        self._cleanup() # critical step. see definition of _cleanup()
        return time_elapsed, trailing_avg
    # ----  end .train() ----

    def _cleanup(self):
        '''
        1. Delete memory buffer to avoid a system crash.  Memory buffer will consume as much as 35GB of RAM.  Deletion avoids a memory crash in the 
         case where DQN objects are instantiated and remain open in a notebook.
        2. Close gymnasium environments.  Probably not critical.

         '''
        del self.memory._state_history
        del self.pbar
        self.train_envs.close
        self.eval_env.close
        if self.record_interval is not None: self.record_env.close
