from .vectorized_wrappers import MyVecAtariWrapper
from .wrappers import MyAtariWrapper
from typing import Tuple, Callable
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gymnasium as gym

def get_single_env(
        env_name:           str  = None,
        train:              bool = False, 
        record_video:       bool = False,
        noop_min:           int  = 1,
        noop_max:           int  = 1,
        screen_size:        int  = 84,
        seed:               int  = 0,
        video_dir:          str  = None,
        fire_on_life_loss:  bool = False) -> gym.Env:
        ''' 
        Build gymnasium.Env objects:
        - train_env: Training environment with terminal on life loss and reward clipping
        - eval_env: Evaluation environment without terminal on life loss and reward clipping. Possible fire on life loss.
        - record_env: Evaluation environment with storage of raw frmaes.  Records every episode to an .mp4 file.
        
        Custom wrappers are applied to the environments:
        - Both environments' outer wrappers return the last 5 images (5, 84, 84) or (5, 42, 42)
        - Both fire on reset and apply a random number of no-ops (min=5, max=noop_max) at the start of each episode
        '''
        if train:
            terminal_on_life_loss = True
            clip_reward = True
            fire_on_life_loss = False
        else:
            terminal_on_life_loss = False
            clip_reward = False

        env = gym.make(env_name, render_mode='rgb_array')
        return MyAtariWrapper(env, noop_min, noop_max, screen_size, seed,
                             terminal_on_life_loss=terminal_on_life_loss,
                             clip_reward=clip_reward,
                             fire_on_life_loss=fire_on_life_loss,
                             record_video=record_video,
                             video_dir = video_dir)


def get_vectorized_envs(
            env_name:           str = None,
            n_envs:             int = 2,
            asynchronous:       bool= False,            # Not used in this implementation
            train:              bool= True, 
            record_video:       bool= False,
            noop_min:           int = 1,
            noop_max:           int = 1,
            screen_size:        int = 84,
            seed:               int = 0,
            video_dir:          str = None,
            fire_on_life_loss:  bool= False) -> gym.vector.AsyncVectorEnv:
    
    if train:
            terminal_on_life_loss = True
            clip_reward = True
            fire_on_life_loss = False
    else:
            terminal_on_life_loss = False
            clip_reward = False
            fire_on_life_loss = fire_on_life_loss
    ''' 
    Build gymnasium.Env or vectorized environments:
    - train_env: Training environment with terminal on life loss and reward clipping
    - eval_env: Evaluation environment without terminal on life loss and reward clipping. Possible fire on life loss.
    - record_env: Evaluation environment with storage of raw frames. Records every episode to an .mp4 file.

    Custom wrappers are applied to the environments:
    - Both environments' outer wrappers return the last 5 images (5, 84, 84) or (5, 42, 42)
    - Both fire on reset and apply a random number of no-ops (min=5, max=noop_max) at the start of each episode.
    '''

    def make_single_env(seed_offset=0) -> Callable[[], gym.Env]:
        """ Returns a factory function for creating a single environment instance. """
        def _init():
            env = gym.make(env_name, render_mode='rgb_array')
            clip_reward = True
            fire_on_life_loss = False
            return MyVecAtariWrapper(
                env=env, 
                noop_min=noop_min, 
                noop_max=noop_max, 
                screen_size=screen_size, 
                seed=seed+seed_offset,
                terminal_on_life_loss=terminal_on_life_loss,
                clip_reward=clip_reward,
                fire_on_life_loss=fire_on_life_loss,
                record_video=False,
                video_dir=None
            )
        return _init

    #if n_envs == 1:
    #    # Single environment
    #    return make_single_env()()
    #else:
    if asynchronous:
            # Asynchronous environments
        return AsyncVectorEnv([make_single_env(i) for i in range(n_envs)])
    else:
        return SyncVectorEnv([make_single_env(i) for i in range(n_envs)])
    