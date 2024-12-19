"""Vectorized environment wrappers for Atari environments.

This module provides custom wrappers for Atari environments that are compatible
with vectorized environments. The wrappers handle:
- Frame stacking (5 frames)
- No-op resets
- Fire actions on life loss
- Seed management
- Video recording
"""

import os
from collections import deque
from typing import Tuple, Callable, Optional, List, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    StickyActionEnv,
    WarpFrame
)


class VecFiveStackWrapper(gym.Wrapper):
    """Stack 5 frames together as an observation.
    
    Allows for efficient storage of state and next_state together in memory
    by maintaining a rolling window of 5 frames. The observation shape is
    (5, H, W) where H and W are the height and width of each frame.
    """
    
    def __init__(self, env: gym.Env) -> None:
        """Initialize the frame stacking wrapper.
        
        Args:
            env: Environment to wrap
        """
        super().__init__(env)
        self.n_frames = 5
        self.frame_deque = deque([], maxlen=self.n_frames)
        
        # Update observation space for stacked frames
        obs_shape = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.n_frames, obs_shape[0], obs_shape[1]),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and initialize frame stack.
        
        Returns:
            Tuple of (stacked_frames, info)
        """
        observation, info = self.env.reset(**kwargs)
        observation = np.squeeze(observation, axis=-1)
        
        # Initialize frame stack with zeros
        for _ in range(self.n_frames):
            self.frame_deque.append(np.zeros_like(observation))
            
        self.frame_deque.append(observation)
        return self.frames, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and update frame stack.
        
        Args:
            action: Action to take in environment
            
        Returns:
            Tuple of (stacked_frames, reward, done, truncated, info)
        """
        observation, reward, done, truncated, info = self.env.step(action)
        observation = np.squeeze(observation, axis=-1)
        self.frame_deque.append(observation)
        return self.frames, reward, done, truncated, info

    @property
    def frames(self) -> np.ndarray:
        """Get the stacked frames as a numpy array."""
        return np.stack(self.frame_deque, axis=0)


class VecFireOnLifeLossWrapper(gym.Wrapper):
    """Force 'FIRE' action when a life is lost in Atari games.
    
    This wrapper helps maintain game state after life loss in games where
    the fire button needs to be pressed to continue playing.
    """
    
    def __init__(self, env: gym.Env) -> None:
        """Initialize the fire on life loss wrapper.
        
        Args:
            env: Environment to wrap
        """
        super().__init__(env)
        self.last_lives = None
        self.FIRE = self.env.unwrapped.get_action_meanings().index('FIRE')

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and track initial lives.
        
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        self.last_lives = (info["lives"] if isinstance(info, dict) 
                          else [i["lives"] for i in info])
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and fire if life was lost.
        
        Args:
            action: Action to take in environment
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Handle both vectorized and single environments
        if isinstance(info, list):
            for i, (lives, d) in enumerate(zip(info["lives"], done)):
                if lives < self.last_lives[i] and not d:
                    obs[i], reward[i], done[i], truncated[i], info[i] = self.env.step(self.FIRE)
        else:
            if info["lives"] < self.last_lives and not done:
                obs, reward, done, truncated, info = self.env.step(self.FIRE)
                
        self.last_lives = (info["lives"] if isinstance(info, dict) 
                          else [i["lives"] for i in info])
        return obs, reward, done, truncated, info


class VecNoopResetWrapper(gym.Wrapper):
    """Sample initial states by taking random no-ops on reset.
    
    This wrapper helps prevent the agent from learning start-state dependent
    strategies by randomizing the initial state through no-op actions.
    """
    
    def __init__(
            self,
            env: gym.Env,
            noop_min: int = 0,
            noop_max: int = 20
    ) -> None:
        """Initialize the no-op reset wrapper.
        
        Args:
            env: Environment to wrap
            noop_min: Minimum number of no-ops to perform
            noop_max: Maximum number of no-ops to perform
        """
        super().__init__(env)
        assert noop_max >= noop_min, "noop_max must be >= noop_min"
        self.noop_min = noop_min
        self.noop_max = noop_max
        self.NOOP = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and perform random number of no-ops.
        
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        
        if len(obs.shape) > 3:  # Vectorized environments
            batch_size = obs.shape[0]
            for i in range(batch_size):
                noops = np.random.randint(self.noop_min, self.noop_max + 1)
                for _ in range(noops):
                    obs[i], _, done, truncated, info[i] = self.env.step(self.NOOP)
                    if done or truncated:
                        obs[i], info[i] = self.env.reset()
        else:  # Single environment
            noops = np.random.randint(self.noop_min, self.noop_max + 1)
            for _ in range(noops):
                obs, _, done, truncated, info = self.env.step(self.NOOP)
                if done or truncated:
                    obs, info = self.env.reset()
                    
        return obs, info


class VecSetSeedWrapper(gym.Wrapper):
    """Handle environment seeding for reproducibility.
    
    This wrapper manages seed generation for environment resets, allowing
    both fixed seeds for deterministic behavior and random seeds from a
    sequence for controlled randomization.
    """
    
    def __init__(
            self,
            env: gym.Env,
            seed: Optional[int] = None,
            seed_sequence: Optional[np.random.SeedSequence] = None
    ) -> None:
        """Initialize the seed wrapper.
        
        Args:
            env: Environment to wrap
            seed: Fixed seed for deterministic behavior
            seed_sequence: Sequence for generating random seeds
        """
        super().__init__(env)
        self.fixed_seed = seed
        self.seed_sequence = seed_sequence or np.random.SeedSequence()
        self.rng = np.random.default_rng(self.seed_sequence)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with appropriate seed.
        
        Returns:
            Tuple of (observation, info)
        """
        kwargs["seed"] = (self.fixed_seed if self.fixed_seed is not None
                         else self.rng.integers(0, 1_000_000))
        return self.env.reset(**kwargs)

    def set_fixed_seed(self, seed: int) -> None:
        """Set a fixed seed for deterministic behavior.
        
        Args:
            seed: Seed value to use
        """
        self.fixed_seed = seed

    def unset_fixed_seed(self) -> None:
        """Remove fixed seed to allow random seeding."""
        self.fixed_seed = None


class MyVecAtariWrapper(gym.Wrapper):
    """Comprehensive wrapper combining multiple Atari preprocessing steps.
    
    This wrapper combines standard Atari preprocessing with custom additions:
    - Frame stacking (5 frames)
    - Frame skipping (4 frames)
    - Screen resizing (42x42 or 84x84)
    - Grayscale conversion
    - Reward clipping
    - Life loss handling
    - No-op resets
    - Sticky actions
    
    Warning:
        Use only with Atari v4 without frame skip: "*NoFrameskip-v4"
    """
    
    def __init__(
            self,
            env: gym.Env,
            noop_min: int = 0,
            noop_max: int = 1,
            screen_size: int = 84,
            seed: Optional[int] = None,
            terminal_on_life_loss: bool = False,
            fire_on_life_loss: bool = False,
            clip_reward: bool = False,
            record_video: bool = False,
            video_dir: Optional[str] = None,
            action_repeat_probability: float = 0.0
    ) -> None:
        """Initialize the Atari wrapper with all preprocessing steps.
        
        Args:
            env: Environment to wrap
            noop_min: Minimum number of no-ops on reset
            noop_max: Maximum number of no-ops on reset
            screen_size: Size to resize frames to (42 or 84)
            seed: Random seed for reproducibility
            terminal_on_life_loss: If True, end episode on life loss
            fire_on_life_loss: If True, press FIRE after life loss
            clip_reward: If True, clip rewards to {-1, 0, 1}
            record_video: If True, save episode recordings
            video_dir: Directory for video recordings
            action_repeat_probability: Chance of repeating previous action
        """
        assert noop_max >= noop_min, "noop_max must be >= noop_min"
        assert screen_size in [42, 84], "screen_size must be 42 or 84"
        
        # Optional video recording
        if record_video:
            assert video_dir is not None, "video_dir required if record_video=True"
            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda x: x % 1 == 0
            )
        
        # Apply wrappers in sequence
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
            
        env = VecNoopResetWrapper(env, noop_min=noop_min, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=4)
        
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
            
        if fire_on_life_loss:
            env = VecFireOnLifeLossWrapper(env)
            
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
            
        env = WarpFrame(env, width=screen_size, height=screen_size)
        
        if clip_reward:
            env = ClipRewardEnv(env)
            
        # Frame stacking must be last
        env = VecFiveStackWrapper(env)
        
        super().__init__(env)
