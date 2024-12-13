import torch
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np

import gymnasium as gym
import ale_py; gym.register_envs(ale_py)


import datetime, copy, random
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# typing, dataclasses, and debugging
from types import SimpleNamespace
from typing import Tuple, Dict
import inspect

from IPython.display import clear_output

# Import custom classes
from modules.policy.atari_policy_net import AtariPolicyNet
from modules.action_handlers import ActionHandler, VecActionHandler
from modules.utils import FilePathManager, PBar, Logger, Plotter
from modules.evaluator import Evaluator



from modules.replay_buffer.get_buffers import get_replay_buffers
from modules.environment.get_environment import get_vectorized_envs, get_single_env

class DQN:
    def __init__(self, p: Dict):
        clear_output()
        ## Input parameters                                              # Default  
        # Rainbow DQN options
        self.doubleQ =                  p.get('doubleQ',                    False                         )    
        self.dueling =                  p.get('dueling',                    False                         )
        self.categorical_DQN =          p.get('categorical_DQN',            False                         )
        self.noisy_linear =             p.get('noisy_linear',               False                         )
        self.prioritized_replay =       p.get('prioritized_replay',         False                         )
        self.n_step_learning =          p.get('n_step_learning',            False                         )
        
        # Environment parameters
        self.n_envs =                   p.get('n_envs',                     4                             ) # 1, 2, 4, 8, 16] only
        self.asynchronous =             p.get('asynchronous',               False                         ) # KEEP FALSE FOR NOW
        self.seed =                     p.get('seed',                       0                             )
        self.env_name =                 p.get('env_name',                   "Breakout") + "NoFrameskip-v4"
        self.screen_size =              p.get('screen_size',                84                            )
        self.noop_min =                 p.get('noop_min',                   10                            )
        self.noop_max =                 p.get('noop_max',                   10                            )
        self.fire_on_life_loss =        p.get('fire_on_life_loss',          False                         )
        
        # Model parameters
        self.memory_size =              p.get('memory_size',                1_000_000                     )
        self.batch_size =               p.get('batch_size',                 32                            )
        self.random_starts =            p.get('random_starts',              50_000                        )
        self.learning_rate =            p.get('learning_rate',              0.0000625                     )
        self.gradient_clamping =        p.get('gradient_clamping',          True                          )
        self.gamma =                    p.get('gamma',                      0.99                          )
        self.scale_around_zero =        p.get('scale_around_zero',          False                         )

        # Experimental parameters
        self.batch_norm =               p.get('batch_norm',                 False                         )
        self.layer_norm =               p.get('layer_norm',                 False                         )

        # Epsilon parameters
        self.epsilon_start =            p.get('epsilon_start',              1.0                           )
        self.epsilon_final =            p.get('epsilon_final',              0.1                           )
        self.epsilon_decay_steps =      p.get('epsilon_decay_steps',        1_000_000                     )
        self.eval_epsilon =             p.get('eval_epsilon',               0.05                          )

        # Interval parameters
        self.policy_update_interval =   p.get('policy_update_interval',     4                             ) 
        self.pbar_update_interval  =    p.get('pbar_update_interval',       100,                          )
        self.target_update_interval =   p.get('target_update_interval',     10_000                        )
        self.eval_interval =            p.get('eval_interval',              50_000                        )
        self.n_games_per_eval =         p.get('n_games_per_eval',           10                            )
        self.checkpoint_interval =      p.get('checkpoint_interval',        2_500_000                     )
        self.record_interval=           p.get('record_interval',            2_500_000                     ) 
    
        # Exit conditions   (time in minutes)
        self.max_steps =                p.get('max_steps',                  20_000_000                    )
        self.exit_trailing_average =    p.get('exit_trailing_average',      10000                         )
        self.exit_time_limit =          p.get('exit_time_limit',            1200                          ) # mins

        ## Rainbow parameters
        # Categorical DQN parameters
        self.categorical_params =       p.get('categorical_params',         dict(atom_size=     51, 
                                                                                 Vmin=         -10, 
                                                                                 Vmax=          10)       )
        # Priority Replay parameters
        self.per_params =               p.get('per_params',                 dict(alpha=         0.6, 
                                                                                 beta_start=    0.4, 
                                                                                 beta_frames=   100_000, 
                                                                                 pr_epsilon=    1e-5)     )
        # N-step learning parameters
        self.n_step_params =            p.get('n_step_params',              dict(n_steps=       3, 
                                                                                 memory_size=   500, 
                                                                                 gamma=         0.99)     )
        # Noisy linear parameters (set this parameter in the modules/noisy_linear.py file)
        #self.std_init =                 p.get('std_init',                   0.017                        )

        # Logging parameters
        self.trailing_avg_trail =       p.get('trailing_avg_trail',         20                            ) 
        self.name =                     p.get('name',                       '[no name]'                   )
        self.log_dir=                   p.get('log_dir',                    '[no name]'                   )
        self.overwrite_previous=        p.get('overwrite_previous',         False                         )
        self.data_logging =             p.get('data_logging',               True                          )
        self.note =                     p.get('note',                       '...'                         )
        self.data_plotting =            p.get('data_plotting',              True                          )
        self.dummy_dqn=                 p.get('dummy_dqn',                  False                         )

        # ------ end input parameters ----

        self._verify_parameters(p)      # Ensure no illegal parameters are passed    
        self.initial_params = self.__dict__     # Capture all the initial DQN parameters for logging

        # set seeds. Seed will also be passed to the environment
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)
        
        # Convert time limit minutes to seconds
        self.exit_time_limit_seconds = self.exit_time_limit * 60

        # Set policy_update interval and number of batches to update (goal is to keep the policy update interval at 4 steps)
        (self.policy_update_interval_adjusted, 
         self.n_batch_updates)                  = self._policy_updates(self.n_envs, 
                                                                       self.policy_update_interval)
            
        # For clarity when passing parameters
        self.categorical_params = SimpleNamespace(**self.categorical_params)
        self.per_params = SimpleNamespace(**self.per_params)
        self.n_step_params = SimpleNamespace(**self.n_step_params)

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
        self.record_env = get_single_env(
                        env_name        = self.env_name,
                        noop_min        = self.noop_min,
                        noop_max        = self.noop_max,
                        screen_size     = self.screen_size,
                        seed            = self.seed)
        #print(f'Early Return: Dummy DQN, exiting at {inspect.currentframe().f_lineno}'); return
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
    
        # Action handlers: generates actions and manages epsilon decay
        self.vec_action_handler =  VecActionHandler(
                                        policy_net=     self.policy_net,
                                        n_envs=         self.n_envs,
                                        action_space=   self.eval_env.action_space,
                                        screen_size=    self.screen_size,
                                        device=         self.device,    
                                        epsilons=       (self.epsilon_start, self.epsilon_final, self.epsilon_decay_steps, self.eval_epsilon))
        # Evaluation uses the non-vectorized environment
        self.action_handler =  ActionHandler(
                                        policy_net=     self.policy_net,
                                        action_space=   self.eval_env.action_space, 
                                        screen_size=    self.screen_size,
                                        device=         self.device,    
                                        epsilons=       (self.epsilon_start, self.epsilon_final, self.epsilon_decay_steps, self.eval_epsilon))
        
        # Evaluator: evaluates the policy network, Evaluator.eval_df keeps the data history
        self.evaluator = Evaluator(env=     self.eval_env, 
                                   ah=      self.action_handler, 
                                   n=       self.n_games_per_eval, 
                                   trail=   self.trailing_avg_trail)
        
        # Record video: A separate mini-evaluator used to take advantage of the gymnasium video wrapper
        if self.record_interval is not None:
            self.video_recorder = Evaluator(env=self.record_env, ah=self.action_handler)
   
        # Helper classes for logging data to csv files and checkpoints
        self.logger = Logger(
            filepaths =         self.filepaths,  
            note=               self.note,
            params=             self.initial_params)
        
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
        clear_output()
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

            # 2. Update the policy network: Here we perform multiple updates per step if warranted. For example, if the policy_update_interval = 4
            #    and n_envs = 16, then the policy network will be updated four times each step. If the n_envs is 1, then the policy network will be updated
            #    once every 4 steps.
            if steps % self.policy_update_interval_adjusted == 0:
                if len(self.memory) >= 512:
                    loss = 0
                    for i in range(self.n_batch_updates):
                        loss += self._update_policy()
                    loss /= self.n_batch_updates

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

    # Exit steps
        if self.record_interval is not None: self.video_recorder.evaluate(steps, 0)
        clear_output()
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

        self._cleanup()
    # ----  end .train() ----

    def _update_policy(self):
        '''
        Update the policy network:
        - Sample the batch from the replay buffer.
        - Calculate the loss.
        - Update the policy network.
        '''
        if self.noisy_linear:
            with torch.no_grad():
                self.policy_net.reset_noise()
                self.target_net.reset_noise()
        self.policy_net.train()

        # Sample the batch
        (s_batch, a_batch, r_batch, ns_batch, d_batch) = self.memory.sample()
        
        # Calculate the Q-values
        Q = self.policy_net(s_batch).gather(1, a_batch)
        
        # Calculate the target Q-values using DQN or Double DQN
        if self.doubleQ:
            next_actions = self.policy_net(ns_batch).argmax(1, keepdim=True)
            with torch.no_grad():
                next_Q = self.target_net(ns_batch).gather(1, next_actions)
        else:
            with torch.no_grad():
                next_Q = self.target_net(ns_batch).max(1)[0].detach().unsqueeze(1)

        expected_next_Q = (next_Q * self.gamma) * (1 - d_batch.view(-1,1)) + r_batch.view(-1,1)

        loss = F.smooth_l1_loss(Q, expected_next_Q)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clamping:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()
    # ----  end ._update_policy() ----

    def _verify_parameters(self, p: Dict):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        illegal_params = []
        for param in p.keys():
            if param not in members:
                illegal_params.append(param)
        if illegal_params != []: raise ValueError(f'Parameters {illegal_params} not found in {self.__class__.__name__} class')

        # make sure all periodic updates are divisible by the n_envs
        assert self.screen_size in [42, 84], "Screen size must be 42 or 84"
        assert self.pbar_update_interval % self.n_envs == 0, "pbar_update_interval must be divisible by n_envs"
        assert self.target_update_interval % self.n_envs == 0, "target_update_interval must be divisible by n_envs"
        assert self.eval_interval % self.n_envs == 0, "eval_interval must be divisible by n_envs"
        assert self.record_interval % self.n_envs == 0, "record_interval must be divisible by n_envs"
        assert self.checkpoint_interval % self.n_envs == 0, "checkpoint_interval must be divisible by n_envs"

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else: raise "Cuda not available: Use a machine with GPU"
    # ----  end ._verify_parameters() ----``

    def _cleanup(self):
        ''' Close environments
           Important: Delete memory buffer.  84 x 84 x 1_000_000 x uint8 =~ 35gb.  Deletion avoids a memory crash in the 
         case where DQN objects are created and left open in a notebook.
         '''
        del self.memory._state_history
        del self.pbar
        self.train_envs.close
        self.eval_env.close
        self.record_env.close

    @classmethod
    def _policy_updates(cls, 
                        n_envs:                 int, 
                        policy_update_interval: int) -> Tuple[int, int]:
        if n_envs <= policy_update_interval:
            policy_update_interval_adjusted = policy_update_interval // n_envs
            n_batch_updates = 1
        else:
            policy_update_interval_adjusted = 1
            n_batch_updates = n_envs // policy_update_interval
        return policy_update_interval_adjusted, n_batch_updates