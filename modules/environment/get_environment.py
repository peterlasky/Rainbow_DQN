from .vectorized_wrappers import MyVecAtariWrapper
from .wrappers import MyAtariWrapper
from typing import Dict, Callable
from types import SimpleNamespace
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gymnasium as gym


def get_single_env(
        train:              bool = False, 
        record_video:       bool = False,
        video_dir:          str = None,
        p:                  SimpleNamespace = None) -> gym.Env:

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
            p.terminal_on_life_loss = True
            p.clip_reward = True
            p.fire_on_life_loss = False
        else:
            p.terminal_on_life_loss = False
            p.clip_reward = False

        env = gym.make(p.env_name, render_mode='rgb_array')
        
        p.n_actions = env.action_space.n # add to global parameters

        return MyAtariWrapper(env, p.noop_min, p.noop_max, p.screen_size, p.seed,
                             terminal_on_life_loss=p.terminal_on_life_loss,
                             clip_reward=p.clip_reward,
                             fire_on_life_loss=p.fire_on_life_loss,
                             record_video=record_video,
                             video_dir=video_dir)

def get_vectorized_envs(
                train:              bool = False,
                record_video:       bool = False,
                video_dir:          str  = None,
                p:                  SimpleNamespace = None
                ) -> gym.Env:
    
    '''
            Build gymnasium.Env or vectorized environments:
            - train_env: Training environment with terminal on life loss and reward clipping
            - eval_env: Evaluation environment without terminal on life loss and reward clipping. Possible fire on life loss.
            - record_env: Evaluation environment with storage of raw frames. Records every episode to an .mp4 file.

            Custom wrappers are applied to the environments:
            - Both environments' outer wrappers return the last 5 images (5, 84, 84) or (5, 42, 42)
            - Both fire on reset and apply a random number of no-ops (min=5, max=noop_max) at the start of each episode.
    '''
    
    if train:
        p.terminal_on_life_loss = True
        p.clip_reward = True
        p.fire_on_life_loss = False
    else:
        p.terminal_on_life_loss = False
        p.clip_reward = False

    def make_single_env(seed_offset=0) -> Callable[[], gym.Env]:
        """ Returns a factory function for creating a single environment instance. """
        def _init():
            env = gym.make(p.env_name, render_mode='rgb_array')
            return MyVecAtariWrapper(
                env=env, 
                noop_min=p.noop_min, 
                noop_max=p.noop_max, 
                screen_size=p.screen_size, 
                seed=p.seed+seed_offset,
                terminal_on_life_loss=p.terminal_on_life_loss,
                clip_reward=p.clip_reward,
                fire_on_life_loss=p.fire_on_life_loss,
                record_video=record_video,
                video_dir=video_dir,
            )
        return _init

    if p.asynchronous:
        return AsyncVectorEnv([make_single_env(i) for i in range(p.n_envs)])
    else:
        return SyncVectorEnv([make_single_env(i) for i in range(p.n_envs)])