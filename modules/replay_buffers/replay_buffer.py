"""Experience Replay Buffer Implementation.

This module implements a standard experience replay buffer for deep reinforcement
learning, with support for n-step returns and prioritized experience replay (PER).
The buffer stores transitions as (state, action, reward, done) tuples and provides
functionality for random sampling of experiences.

Features:
    - Efficient storage using PyTorch tensors
    - Support for n-step returns
    - Optional prioritized experience replay
    - Configurable batch size and memory size
    - Device-agnostic tensor operations
"""

from collections import deque
from types import SimpleNamespace
from typing import Tuple, Optional, Union

import numpy as np
import torch


class ReplayBuffer:
    """Standard experience replay buffer for reinforcement learning.
    
    Stores transitions and provides functionality for random sampling of experiences.
    Supports both standard experience replay and n-step learning.
    
    Attributes:
        batch_size: Number of experiences to sample per batch
        memory_size: Maximum number of transitions to store
        screen_size: Size of observation frames
        output_device: Device to store tensors on
        n_step_learning: Whether n-step learning is enabled
        n_steps: Number of steps for n-step learning
        n_step_gamma: Discount factor for n-step learning
        _buffer_len: Current number of transitions stored
        index: Current position in the circular buffer
        _state_history: Tensor storing state observations
        _reward_history: Tensor storing rewards
        _action_history: Tensor storing actions
        _done_history: Tensor storing done flags
        n_step_buffer: Deque storing transitions for n-step learning
    """
    
    def __init__(
            self,
            batch_size: int,
            memory_size: int,
            screen_size: int,
            output_device: Optional[torch.device] = None,
            n_step_learning: bool = False,
            n_steps: Optional[int] = None,
            n_step_gamma: Optional[float] = None,
            per_params: Optional[SimpleNamespace] = None
    ) -> None:
        """Initialize the replay buffer.
        
        Args:
            batch_size: Number of experiences to sample per batch
            memory_size: Maximum number of transitions to store
            screen_size: Size of observation frames
            output_device: Device to store tensors on (defaults to CPU)
            n_step_learning: Whether to use n-step learning
            n_steps: Number of steps for n-step learning
            n_step_gamma: Discount factor for n-step learning
            per_params: Parameters for prioritized replay (if used)
            
        Raises:
            ValueError: If n-step parameters are missing when n_step_learning is True
        """
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.screen_size = screen_size
        self.output_device = output_device or torch.device('cpu')
        
        self._buffer_len = 0
        self.index = 0
        
        # Initialize history tensors
        self._state_history = torch.zeros(
            (self.memory_size, 5, self.screen_size, self.screen_size),
            dtype=torch.uint8
        )
        self._reward_history = torch.zeros(
            (self.memory_size, 1),
            dtype=torch.float32  # Changed to float32 for rewards
        )
        self._action_history = torch.zeros(
            (self.memory_size, 1),
            dtype=torch.int64  # Changed to int64 for actions
        )
        self._done_history = torch.zeros(
            (self.memory_size, 1),
            dtype=torch.bool  # Changed to bool for done flags
        )
        
        # Setup n-step learning
        self.n_step_learning = n_step_learning
        if n_step_learning:
            if n_steps is None:
                raise ValueError("n_steps required when n_step_learning is True")
            if n_step_gamma is None:
                raise ValueError("n_step_gamma required when n_step_learning is True")
                
            self.n_steps = n_steps
            self.n_step_gamma = n_step_gamma
            self.n_step_buffer = deque(maxlen=n_steps)

    def _store_transition(
            self,
            frames: np.ndarray,
            action: int,
            reward: float,
            done: bool
    ) -> None:
        """Store a transition in the buffer.
        
        Args:
            frames: Observation frames
            action: Action taken
            reward: Reward received
            done: Whether episode ended
        """
        self._state_history[self.index] = torch.from_numpy(frames).view(
            5, self.screen_size, self.screen_size
        )
        self._action_history[self.index, 0] = int(action)  # Convert to int
        self._reward_history[self.index, 0] = float(reward)  # Convert to float
        self._done_history[self.index, 0] = bool(done)  # Convert to bool
        
        self.index = (self.index + 1) % self.memory_size
        self._buffer_len = min(self._buffer_len + 1, self.memory_size)

    def add(
            self,
            transition: Tuple[np.ndarray, int, float, bool]
    ) -> Optional[Tuple[np.ndarray, int, float, bool]]:
        """Add a transition to the buffer.
        
        Handles both single-step and n-step transitions. For n-step learning,
        returns the processed n-step transition when enough steps are accumulated.
        
        Args:
            transition: Tuple of (state, action, reward, done)
            
        Returns:
            Processed transition for n-step learning, or None if not enough steps
        """
        frames, action, reward, done = transition
        
        if self.n_step_learning:
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_steps:
                return None
                
            # Create n-step transition
            _, _, reward, done = self._get_n_step_info()
            frames, action, _, _ = self.n_step_buffer[0]
            transition = (frames, action, reward, done)
            
        # Store transition in buffer
        self._store_transition(frames, action, reward, done)
        return transition

    def _get_n_step_info(self) -> Tuple[None, None, float, bool]:
        """Calculate n-step return information.
        
        Returns:
            Tuple of (None, None, cumulative_reward, terminal_done)
            
        Raises:
            AssertionError: If called when n_step_learning is False
        """
        if not self.n_step_learning:
            raise AssertionError("N-step learning is not enabled")
            
        _, _, reward, done = self.n_step_buffer[-1]
        transitions = list(self.n_step_buffer)[:-1]
        
        for _, _, r, d in reversed(transitions):
            reward = r + self.n_step_gamma * (reward * (1 - d))
            done = d if d else done
            
        return None, None, reward, done

    def sample(self) -> Union[
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Sample a batch of transitions from the buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            All tensors are moved to self.output_device
        """
        indices = torch.randint(0, self._buffer_len, (self.batch_size,))
        
        # Get batch tensors
        states = self._state_history[indices, :4]
        next_states = self._state_history[indices, 1:]
        actions = self._action_history[indices].long()
        rewards = self._reward_history[indices].float()
        dones = self._done_history[indices].float()
        
        # Move to device
        return (
            states.to(self.output_device),
            actions.to(self.output_device),
            rewards.to(self.output_device),
            next_states.to(self.output_device),
            dones.to(self.output_device)
        )

    def __len__(self) -> int:
        """Get the current number of transitions in the buffer."""
        return self._buffer_len
