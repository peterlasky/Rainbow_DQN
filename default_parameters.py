import torch
from types import SimpleNamespace
def get_default_parameters():
 return(
   SimpleNamespace(

    # Rainbow DQN Flags
    doubleQ=                   False,
    dueling=                   False,
    categorical_DQN=           False,
    noisy_linear=              False,
    prioritized_replay=        False,
    n_step_learning=           False,

    # Vectorization Parameters
    n_envs=                    8,
    group_training_losses=     False,

    # Environment Parameters
    asynchronous=             False,
    seed=                     0,
    env_name=                 "BreakoutNoFrameskip-v4",
    screen_size=              84,
    noop_min=                 10,
    noop_max=                 10,
    fire_on_life_loss=        False,

    # Device
    device=                   torch.device('cuda'),

    # Model Parameters
    memory_size=              1_000_000,
    batch_size=               32,
    random_starts=            50_000,
    learning_rate=            0.0000625,
    gradient_clamping=        True,
    gamma=                    0.99,
    scale_around_zero=        False,
    fc1_out=                  512,

    # Experimental Parameters
    batch_norm=               False,
    layer_norm=               False,

    # Epsilon Parameters
    epsilon_start=            1.0,
    epsilon_final=            0.1,
    epsilon_decay_steps=      1_000_000,
    eval_epsilon=             0.05,

    # Interval Parameters
    policy_update_interval=   4,
    pbar_update_interval=     100,
    target_update_interval=   10_000,
    eval_interval=            50_000,
    n_games_per_eval=        10,
    checkpoint_interval=      2_500_000,
    record_interval=         None,

    # Exit Conditions
    max_steps=               20_000_000,
    exit_trailing_average=   10_000,
    exit_time_limit=        1200,  # Time in minutes

    # Rainbow Parameters
    # Categorical DQN Parameters
    atom_size=               51,
    Vmin=                    -10,
    Vmax=                    10,

    # Priority Replay Parameters
    alpha=                   0.6,
    beta_start=              0.4,
    beta_frames=             100_000,
    pr_epsilon=              1e-5,

    # N-step Learning Parameters
    n_steps=                 3,
    n_memory_size=           500,
    n_gamma=                 0.99,

    # Logging Parameters
    main_log_dir=            'logs',
    group_dir=               '[no group name]',
    video_dir=               'videos',
    name=                    '[no name]',
    note=                    '[no note]',
    overwrite_previous=      False,
    data_logging=            True,
    data_plotting=           False,
    trailing_avg_trail=      20,
))