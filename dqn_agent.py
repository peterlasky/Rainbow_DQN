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
        ''' 
        Verify correct input parameters, add defaults, and unpack
        '''
        ph = ParameterHandler(p)
        verified_paramters = ph.get_parameters()
        for k, v in verified_paramters.items():
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

    def _update_policy(self):
        if len(self.memory) < (self.batch_size * self.n_batch_updates):
            return 0.0

        def forward_pass():
            if self.noisy_linear:
                with torch.no_grad():
                    self.policy_net.reset_noise()
                    self.target_net.reset_noise()
            # Sample the batch
            (s_batch, a_batch, r_batch, ns_batch, d_batch) = self.memory.sample()

            # Calculate the Q-Values
            Q = self.policy_net(s_batch).gather(1, a_batch)

            # Calculate the target Q-Values using DQN or Double DQN
            if self.doubleQ:
                next_actions = self.policy_net(ns_batch).argmax(1, keepdim=True)
                with torch.no_grad():
                    next_Q = self.target_net(ns_batch).gather(1, next_actions)
            else:
                with torch.no_grad():
                    next_Q = self.target_net(ns_batch).max(1)[0].detach().unsqueeze(1)

            expected_next_Q = (next_Q * self.gamma) * (1 - d_batch.view(-1,1)) + r_batch.view(-1,1)
            loss = F.smooth_l1_loss(Q, expected_next_Q)
            return loss
        
        def backward_pass(loss):
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clamping:
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        # 
        self.policy_net.train()
        if self.group_training_losses == True:  # If single threaded, perform the forward pass and update the policy network n_batch_updates times
            l = 0.0
            for i in range(self.n_batch_updates):
                loss = forward_pass()
                backward_pass(loss)
                l += loss.item()
            loss = l / self.n_batch_updates
            return loss

        else:  # Accumulate gradients over all batch updates before applying them
            loss = torch.tensor(0.0, device=self.device)
            for _ in range(self.n_batch_updates):
                loss += forward_pass()
            # Average the losses before backward pass
            loss = loss / self.n_batch_updates
            backward_pass(loss)
            return loss.item()

    # ----  end ._update_policy() ----


    def _cleanup(self):
        '''
        1. Delete memory buffer: 84 x 84 x 1_000_000 x uint8 =~ 35gb.  Deletion avoids a memory crash in the 
         case where DQN objects are created and left open in a notebook.
        2. 
         '''
        del self.memory._state_history
        del self.pbar
        self.train_envs.close
        self.eval_env.close
        if self.record_interval is not None: self.record_env.close
