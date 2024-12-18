from dqn_agent import DQNAgent

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
names=[] # keep track of names for plotting at the end

# basic parameters
p = dict(
    env_name =              'BreakoutNoFrameskip-v4', # use NoFrameskip-v4 version
    group_dir=              'SI_tests',
    name =                  'DDQN',
    overwrite_previous=     False,
    asynchronous=           False,

    group_training_losses = True,
    data_plotting =         False,
    
    screen_size=        42,
    trailing_avg_trail= 40,
    eval_interval=      20_000,
    max_steps=          15_000_000, 
    record_interval=    5_000_000, 
    n_games_per_eval=   10,
    n_envs=             40,
    pbar_update_interval= 800,
    seed=               42
    )

p.update(
    name=               'DDQN',
    note=               f'''{p['n_envs']} vectorized environments. ''',
    n_envs=             20,
    
    )
dqn = DQNAgent(p)

dqn.train()
names.append(dqn.filepaths.sub_dir)