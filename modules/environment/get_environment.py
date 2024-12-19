"""Environment factory functions for Atari environments.

This module provides functions to create single or vectorized Atari environments
with appropriate wrappers for training and evaluation. The environments are configured
with custom wrappers that handle:
- Frame stacking (last 5 frames)
- Screen resizing (84x84 or 42x42)
- Episode initialization (fire on reset, random no-ops)
- Reward clipping (optional)
- Life loss handling (optional terminal states)
- Video recording (optional)
"""

from typing import Callable
from types import SimpleNamespace
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from .vectorized_wrappers import MyVecAtariWrapper
from .wrappers import MyAtariWrapper


def get_single_env(
        p: SimpleNamespace,
        train: bool = False,
        record_video: bool = False,
        video_dir: str = None
) -> gym.Env:
    """Create a single Atari environment with appropriate wrappers.
    
    Args:
        p: Parameters containing environment configuration
        train: If True, configure for training (life loss = terminal, clip rewards)
        record_video: If True, save episode recordings
        video_dir: Directory to save video recordings if enabled
        
    Returns:
        Wrapped Atari environment
        
    The environment is configured differently for training vs evaluation:
    - Training: Terminal on life loss, reward clipping
    - Evaluation: No terminal on life loss, no reward clipping
    
    Both modes include:
    - Frame stacking (last 5 frames)
    - Random no-ops on reset (between noop_min and noop_max)
    - Fire action on reset (if applicable)
    """
    # Set environment mode
    p.terminal_on_life_loss = train
    p.clip_reward = train
    p.fire_on_life_loss = not train
    
    # Create base environment
    env = gym.make(p.env_name, render_mode='rgb_array')
    p.n_actions = env.action_space.n
    
    # Apply custom wrappers
    return MyAtariWrapper(
        env=env,
        noop_min=p.noop_min,
        noop_max=p.noop_max,
        screen_size=p.screen_size,
        seed=p.seed,
        terminal_on_life_loss=p.terminal_on_life_loss,
        clip_reward=p.clip_reward,
        fire_on_life_loss=p.fire_on_life_loss,
        record_video=record_video,
        video_dir=video_dir
    )
    
def get_vectorized_envs(
        p: SimpleNamespace,
        train: bool = False,
        record_video: bool = False,
        video_dir: str = None
) -> gym.Env:
    """Create multiple vectorized Atari environments.
    
    Args:
        p: Parameters containing environment configuration
        train: If True, configure for training (life loss = terminal, clip rewards)
        record_video: If True, save episode recordings
        video_dir: Directory to save video recordings if enabled
        
    Returns:
        Vectorized environment (either sync or async based on parameters)
        
    Creates n_envs copies of the environment, either synchronously or
    asynchronously based on the asynchronous parameter. Each environment
    is configured with the same wrappers as get_single_env().
    
    The environments share the same configuration but have different seeds
    offset from the base seed to ensure different trajectories.
    """
    # Set environment mode
    p.terminal_on_life_loss = train
    p.clip_reward = train
    p.fire_on_life_loss = not train
    
    def make_env(seed_offset: int = 0) -> Callable[[], gym.Env]:
        """Create a factory function for a single environment instance.
        
        Args:
            seed_offset: Offset to add to base seed for this environment
            
        Returns:
            Factory function that creates and wraps an environment
        """
        def _init() -> gym.Env:
            env = gym.make(p.env_name, render_mode='rgb_array')
            return MyVecAtariWrapper(
                env=env,
                noop_min=p.noop_min,
                noop_max=p.noop_max,
                screen_size=p.screen_size,
                seed=p.seed + seed_offset,
                terminal_on_life_loss=p.terminal_on_life_loss,
                clip_reward=p.clip_reward,
                fire_on_life_loss=p.fire_on_life_loss,
                record_video=record_video,
                video_dir=video_dir
            )
        return _init
    
    # Create environment factories with different seeds
    env_fns = [make_env(i) for i in range(p.n_envs)]
    
    # Return either async or sync vectorized environment
    return (AsyncVectorEnv(env_fns) if p.asynchronous 
            else SyncVectorEnv(env_fns))