

import os
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import numpy as np
from typing import Tuple, Callable
from collections import deque

# gymnasium-compatible sb3 wrappers 
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, 
    EpisodicLifeEnv, 
    FireResetEnv, 
    MaxAndSkipEnv, 
    StickyActionEnv, 
    WarpFrame) 
from gymnasium.wrappers import RecordVideo


NUM_STACK = 5
from gymnasium.spaces import Box
import numpy as np

class VecFiveStackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = 5
        self.frame_deque = deque([], maxlen=self.n)
        obs_shape = env.observation_space.shape
        # Update observation space to reflect the stacked frames without the last channel dimension
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.n, obs_shape[0], obs_shape[1]),  # Remove the last dimension
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = np.squeeze(observation, axis=-1)  # Remove the channel dimension
        for _ in range(self.n):
            self.frame_deque.append(np.zeros_like(observation))  # Initialize deque
        self.frame_deque.append(observation)
        return self.frames, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        observation = np.squeeze(observation, axis=-1)  # Remove the channel dimension
        self.frame_deque.append(observation)
        return self.frames, reward, done, truncated, info

    @property
    def frames(self):
        stacked_frames = np.stack(self.frame_deque, axis=0)  # Stack along new axis for frames
        return stacked_frames
    
class VecFireOnLifeLossWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_lives = None
        self.FIRE = self.env.unwrapped.get_action_meanings().index('FIRE')

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lives = info["lives"] if isinstance(info, dict) else [i["lives"] for i in info]
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if isinstance(info, list):  # Handle vectorized environments
            for i, (lives, d) in enumerate(zip(info["lives"], done)):
                if lives < self.last_lives[i] and not d:
                    obs[i], reward[i], done[i], truncated[i], info[i] = self.env.step(self.FIRE)
        else:  # Single environment
            if info["lives"] < self.last_lives and not done:
                obs, reward, done, truncated, info = self.env.step(self.FIRE)
        self.last_lives = info["lives"] if isinstance(info, dict) else [i["lives"] for i in info]
        return obs, reward, done, truncated, info
    
class VecNoopResetWrapper(gym.Wrapper):
    """
    Sample initial states by taking a random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_min: Minimum number of no-ops to perform
    :param noop_max: Maximum number of no-ops to perform
    """
    def __init__(self, 
                 env: gym.Env,
                 noop_min: int = 0, 
                 noop_max: int = 20) -> None:
        super().__init__(env)
        self.noop_min = noop_min
        self.noop_max = noop_max
        self.NOOP = 0
        assert noop_max >= noop_min, "noop_max must be greater than or equal to noop_min"
        
    def reset(self, **kwargs):
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
    """
    Gymnasium Wrapper to handle environment seeding.
    Automatically sets the seed during environment reset.
    """

    def __init__(self, env, seed: int =None, seed_sequence=None):
        """
        Initialize the SetSeedWrapper.
        The seed can be set to a fixed value for each reset, or a unique seed can be generated for each reset based on the numpy seed (if set)

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            seed (int, optional): A fixed seed for deterministic behavior.
            seed_sequence (SeedSequence, optional): A SeedSequence instance for generating seeds.
        """
        super(VecSetSeedWrapper, self).__init__(env)
        self.fixed_seed = seed
        self.seed_sequence = seed_sequence or np.random.SeedSequence()
        self.rng = np.random.default_rng(self.seed_sequence)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment and set the seed.

        Args:
            **kwargs: Additional arguments to pass to env.reset().
        Returns:
            observation: The initial observation.
            info (dict): Additional information.
        """
        if self.fixed_seed is not None:
            seed = self.fixed_seed
        else:
            # Generate a unique seed for each reset
            seed = self.rng.integers(0, 1_000_000)
        kwargs["seed"] = int(seed)
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def set_fixed_seed(self, seed: int):
        """
        Set a fixed seed for deterministic behavior.

        Args:
            seed (int): The seed value to set.
        """
        self.fixed_seed = seed

    def unset_fixed_seed(self):
        """
        Remove the fixed seed, allowing for random seeding.
        """
        self.fixed_seed = None

class MyVecAtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * CUSTOM: Noop reset: obtain initial state by taking random number of no-ops on reset.(noop_min added)
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default
    * CUSTOM: Fire on life loss: force 'FIRE' action on life loss
    * CUSTOM: Five-stack: stack 5 frames together as an observation.  Allows for storage of state and next_state together.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This corresponds to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clipped to {-1, 0, 1} depending on its sign.
    :param frame_stacking: Number of frames to stack (default: None)
    :param record_video: If True, record video of the environment. (default: False)
    :param action_repeat_probability: Probability of repeating the last action
    """
    def __init__(
        self,
        env:                        gym.Env,
        noop_min:                   int = 0,
        noop_max:                   int = 1,
        screen_size:                int = 84,
        seed:                       int = None,       
        terminal_on_life_loss:      bool = False,
        fire_on_life_loss:          bool = False,
        clip_reward:                bool = False,
        record_video:               bool = False,
        video_dir:                  str = None,
        action_repeat_probability:  float = 0.0,
    ) -> None:
        
        assert noop_max >= noop_min, 'noop_max must be greater than or equal to noop_min'
        assert screen_size in [42, 84], 'screen_size must be either 42 or 84'

        # record video (used for playback only: not for training or testing)
        if record_video:
            assert video_dir is not None, 'video_dir must be provided if record_video is True'
            env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: x % 1 == 0)

        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        
        # Generate a random number of no-ops on reset.
        env = VecNoopResetWrapper(env, noop_min=noop_min, noop_max=noop_max)

        # Standard for Atari games
        env = MaxAndSkipEnv(env, skip=4)

        # DeepMind uses this methodolog for training but not testing
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)

        # Custom option added by me, may require some tinkering
        if fire_on_life_loss:
            env = VecFireOnLifeLossWrapper(env)

        # Fire on reset: standard option for Atari DeepMind games
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        # Resize to either 42x42 or 84x84 for fitting the convolutional layers of the model
        env = WarpFrame(env, width=screen_size, height=screen_size)
        
        # Used for training only.  
        if clip_reward:
            env = ClipRewardEnv(env)
  
        # Set the seed for reproducibility
        #if seed is not None:
        #    env = SetSeedWrapper(env, seed)

        # Custom wrapper to create an output obs of (5, 84, 84) or (5, 42, 42).
        ''' NOTE: 
            - This is not generally compatible with other wrappers and must be the last 
                 wrapper in the chain.
            - Saves memory on the replay buffer, allowing storage of state and next_state together.
            - Here, state = obs[:4], next_state = obs[1:]
            '''
        env = VecFiveStackWrapper(env)

        super().__init__(env)
