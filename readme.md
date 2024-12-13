
## Vectorized Custom Rainbow DQN
A highly flexible implementation for applying "rainbow DQN methods" to Atari 2600 games using `gymnasium's` vectorized environments.  Currently `stable_baselines3` appears to have the most comprehensive package in the public domain, but I wanted to build some of them from the ground up to understand the implementations and fine tune various hyperparameters. I also wanted to explore speed tradeoffs with vectorization a 24-core processor.

#### Set-up
These steps build the environment as of October 2024, but the dependencies have been changing, so I haven't included an environment file.  
Please check the `Gymnasium` docs at the [Farama Foundation]('https://gymnasium.farama.org/') if this doesn't work.  The `RecordVideo` wrapper requires `moviepy`. 
```bash
conda create -n my_atari_env -c conda-forge python=3.10 pytorch numpy swig tqdm -y
pip install gymnasium[atari,accept-rom-license] moviepy
```

For `Gymnasium`-compatible wrappers from `stable-baselines3`:
```bash  
pip install stable-baselines3   
```

#### Basic use
Parameters are passed by dictionary. Each parameter has a default value (below). 
```python
  p = dict(
      name=               'DDQN',
      note=               '''DDQN with 16 parallel environments. ''',
      env_name =          'Breakout', # defaults to NoFrameskip-v4 version
      log_dir=            'breakout_tests',
      overwrite_logs_folder= True,
      asynchronous=       False,
      doubleQ=            True,
      dueling=            False,
      noisy_linear=       False,
      categorical_DQN=    False,
      prioritized_replay= False,
      n_step_learning=    False,
      screen_size=        42,
      memory_size=        1_000_000,
      eval_interval=      100_000,
      max_steps=          20_000_000, 
      record_interval=    10_000_000, 
      n_games_per_eval=   5,
      n_envs=             16,
      pbar_update_interval= 400,
    )
  dqn = DQN(p)
  dqn.train()
```

At `evalation_interval` steps, the evaluator simulates `n_games_per_eval` games (all lives) and updates the plots:
![](assets/plot_example.png)

#### Logging
Parameters, checkpoints, videos, and evaluation histories are all saved to or updated in the `[log_dir]/[name]` directory, based on parameter settings.    

#### Memory
The replay buffer takes the most memory.  The main constraint is the replay buffer.  Memory use is `memory_size` $* ($`screen_size`$^2) * 5$.  The default setting of $1,000,000 * 84 * 84 * 5 \sim 35$ GB.  We delete the memory buffer on exiting the training loop to avoid an out of memory crash if, e.g. , multiple instances of `DQN` are opened in *Jupyter*.

#### Vectorization
The training loop uses `gymnasium`'s vectorzed environment structure. The original *DeepMind* algortith performs a policy update every 4 steps, on a batch of $32$ transitions taken from the replay buffer.  In a vectorized environment, we need to adjust:  If `n_envs` $=1$, we perform a policy update every 4 steps.  If `n_envs` $= 4$, we perform a policy update each step. However, if `n_envs` $= 8$, we perform two updates of $32$ each step and, similarly, if `n_envs`=16 we perform four batch updates of $32$ each step.  The effect of training multiple batches consecutively (i.e., out of turn) becomes irrelevant as a large memory buffer is filled.

The evaluation loop is executed infrequently and uses a single, non-vectorized `gymnasium` environment.  Speed increase was significant, but not as much as I expected.  Using Intel-9 (24 cores) and NVIDIA RTX 4090:
- Basic DQN: 16 vectorized environments vs single environment: 20-22% faster.
- Rainbow DQN: 16 vectorized environments vs single environment: 25-29% faster.

#### Environment wrappers
I've created custom `gymnasium` wrappers that likely exist. I've also used a few `gymnasium`-compatible wrappers from the `stable_baselines3` library.

1. `five_stack`: stores each state / new state in a combined 5 frame stack observation, such that [:4] is the *state* and [1:] is the *next_state*.
2. `fire_on_life_loss`: the original **Deep Mind** algorithim used a 5% epsilon for evaluation mode to avoid games getting stuck.  For example, games like `breakout` that require a `FIRE` command to restart after each life lost will pause indefinitely if we use a pure `argmax` policy that returns an action other than `FIRE`.  This wrapper, if used, automatically triggers a fire when a life is lost, allowing us to lower the epsilon closer to zero to rely solely on the policy's best actions.  In many games the difference won't matter.
3. `noop_reset` allows for a range of noop_steps upon a reset.
4. `set_seed`: to seed single or vectorized environments.  In my implementation the same seed is applied as the random and numpy seed (although vectorized seeds are increments of the given seed)

#### Screen size
The standard approach resizes the default color screen (210,160,3) to b&w (84,84).  But for certain "boxy" games (e.g., **Breakout**), (42,42) works as well, allowing a 75% reduction in memory. I haven't run exact like-for-like comparisons nor have I run tests using 42x42 frame size on a wide range of Atari games.  The modification from the *DeepMind* convolutional layer format is constructed by altering the kernel and stride on the first convolutional layer:
```python
    # Adjust the kernal and stride for conv layer 1 based on screen size
    assert (screen_size in [42, 84]), "Screen size must be 42 or 84"
    kernel1, stride1 = (8, 4) if screen_size == 84 else (4, 2)

    # Define convolutional layers
    conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=kernel1, stride=stride1)
    conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
    conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
```

#### Comparing results
To graph results from all runs in the log folder:
```python
from modules.utils import plot_multiple_results
plot_multiple_results([log_dir], names, col)
```

#### Video
Videos are periodically recorded by setting the `record_interval` parameters.  Set to `None` if no video needed.

#### To-dos / Future updates 
- **Tensorboard**: Move the monitoring of progress to a tensorboard to avoid the need to run experiments in Jupyter. 
- **Checkpoint playback or training resumption**: Policy checkpoints are currently saved, but no the environment or other training data.  So there is currrently no way to run a simulation from the checkpoint, nor is there a way to resume training from a checkpoint.

#### Default options
```python   
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

  ```

#### Citations and acknolowledgements:
If you use ideas from this work, please cite these papers:
1. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2013). Playing Atari with Deep Reinforcement Learning. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
2. Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., ... & Silver, D. (2017). Rainbow: Combining Improvements in Deep Reinforcement Learning. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298). This paper integrates several key advancements in deep reinforcement learning, including:</small>
- **Double Q-Learning** ([Van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)),
- **Prioritized Experience Replay** ([Schaul et al., 2015](https://arxiv.org/abs/1511.05952))
- **Dueling Network Architectures** ([Wang et al., 2015](https://arxiv.org/abs/1511.06581))
- **Multi-step Learning** ([Sutton, 1988](https://webdocs.cs.ualberta.ca/~sutton/papers/sutton-88-with-erratum.pdf))
- **Distributional RL** ([Bellemare et al., 2017](https://arxiv.org/abs/1707.06887))
- **Noisy Nets** ([Fortunato et al., 2017](https://arxiv.org/abs/1706.10295))
</small>

For conding inspiration I used the following:
1. Wetlui's basic [DQN implementation](https://github.com/wetliu/dqn_pytorch) was a great starting point for this project. 
2. Curt Park's repository [rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need) was helpful in understanding the underlying concepts of each of the rainbow methods.
 