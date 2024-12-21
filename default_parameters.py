from dataclasses import dataclass

@dataclass
class DefaultParameters:
    # Rainbow DQN Flags
    doubleQ                   : bool = False
    dueling                   : bool = False
    categorical_DQN           : bool = False
    noisy_linear              : bool = False
    prioritized_replay        : bool = False
    n_step_learning           : bool = False

    # Vectorization Parameters
    n_envs                    : int = 8
    group_training_losses     : bool = False

    # Environment Parameters
    asynchronous              : bool = False
    seed                      : int = 0
    env_name                  : str = "BreakoutNoFrameskip-v4"
    screen_size               : int = 84
    noop_min                  : int = 10
    noop_max                  : int = 10
    fire_on_life_loss         : bool = False

    # Device
    device                    : str = 'cuda'

    # Model Parameters
    memory_size               : int = 1_000_000
    batch_size                : int = 32
    random_starts             : int = 50_000
    learning_rate             : float = 0.0000625
    gradient_clamping         : bool = True
    gamma                     : float = 0.99
    scale_around_zero         : bool = False
    fc1_out                   : int = 512

    # Experimental Parameters
    batch_norm                : bool = False
    layer_norm                : bool = False

    # Epsilon Parameters
    epsilon_start             : float = 1.0
    epsilon_final             : float = 0.1
    epsilon_decay_steps       : int = 1_000_000
    eval_epsilon              : float = 0.05

    # Interval Parameters
    policy_update_interval    : int = 4
    pbar_update_interval      : int = 100
    target_update_interval    : int = 10_000
    eval_interval             : int = 50_000
    n_games_per_eval          : int = 10
    checkpoint_interval       : int = 2_500_000
    record_interval           : None = None

    # Exit Conditions
    max_steps                 : int = 20_000_000
    exit_trailing_average     : int = 10_000
    exit_time_limit           : int = 1200  # Time in minutes

    # Rainbow Parameters
    # Categorical DQN Parameters
    atom_size                 : int = 51
    Vmin                      : int = -10
    Vmax                      : int = 10

    # Priority Replay Parameters
    alpha                     : float = 0.6
    beta_start                : float = 0.4
    beta_frames               : int = 100_000
    pr_epsilon                : float = 1e-5

    # N-step Learning Parameters
    n_steps                   : int = 3
    n_memory_size             : int = 500
    n_gamma                   : float = 0.99

    # Logging Parameters
    main_log_dir              : str = 'logs'
    group_dir                 : str = None
    video_dir                 : str = 'videos'
    name                      : str = None
    note                      : str = '[no note]'
    overwrite_previous        : bool = False
    data_logging              : bool = True
    data_plotting             : bool = False
    trailing_avg_trail        : int = 20
