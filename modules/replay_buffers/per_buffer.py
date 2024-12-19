"""Prioritized Experience Replay (PER) Buffer Implementation.

This module implements a prioritized experience replay buffer based on the paper
"Prioritized Experience Replay" by Schaul et al. (2015). It extends the basic
replay buffer with importance sampling and prioritized sampling of experiences
based on their TD errors.

The buffer uses segment trees for efficient storage and sampling of priorities,
and supports n-step learning for temporal difference updates.

Reference:
    Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015).
    Prioritized Experience Replay.
    arXiv preprint arXiv:1511.05952.
"""

from types import SimpleNamespace
from typing import List, Tuple, Optional, Union, Any

import numpy as np
import random
import torch

from .segment_tree import SumSegmentTree, MinSegmentTree
from .replay_buffer import ReplayBuffer


class PerBuffer(ReplayBuffer):
    """Prioritized Experience Replay Buffer implementation.
    
    Extends the basic replay buffer with prioritized sampling based on TD errors
    and importance sampling weights to correct for the bias introduced by
    non-uniform sampling.
    
    Attributes:
        alpha: float, determines how much prioritization is used (0 = uniform, 1 = full)
        beta: float, importance-sampling correction exponent
        beta_start: float, initial value of beta
        beta_frames: int, number of frames over which to anneal beta to 1.0
        pr_epsilon: float, small constant to add to priorities to ensure non-zero sampling
        max_priority: float, maximum priority seen so far
        step: int, current step count for beta annealing
        sum_tree: SumSegmentTree, for storing and sampling priorities
        min_tree: MinSegmentTree, for computing importance sampling weights
        last_weights: Optional[torch.Tensor], weights from last sampling
        last_indices: Optional[List[int]], indices from last sampling
    """
    
    def __init__(
            self,
            batch_size: int,
            memory_size: int,
            screen_size: int,
            output_device: torch.device = torch.device('cpu'),
            n_step_learning: bool = False,
            n_steps: Optional[int] = None,
            n_step_gamma: Optional[float] = None,
            per_params: Optional[SimpleNamespace] = None
    ) -> None:
        """Initialize the PER buffer.
        
        Args:
            batch_size: Number of experiences to sample per batch
            memory_size: Maximum number of experiences to store
            screen_size: Size of observation frames
            output_device: Device to store tensors on
            n_step_learning: Whether to use n-step learning
            n_steps: Number of steps for n-step learning
            n_step_gamma: Discount factor for n-step learning
            per_params: Parameters for PER (alpha, beta_start, beta_frames, pr_epsilon)
        """
        super().__init__(
            batch_size=batch_size,
            memory_size=memory_size,
            screen_size=screen_size,
            output_device=output_device,
            n_step_learning=n_step_learning,
            n_steps=n_steps,
            n_step_gamma=n_step_gamma
        )
        
        if per_params is None:
            raise ValueError("PER parameters must be provided")
            
        self.alpha = per_params.alpha
        self.beta_start = per_params.beta_start
        self.beta_frames = per_params.beta_frames
        self.pr_epsilon = per_params.pr_epsilon
        self.beta = self.beta_start
        self.max_priority = 1.0
        self.output_device = output_device
        self.step = 0
        
        # Initialize segment trees
        tree_capacity = 1
        while tree_capacity < self.memory_size:
            tree_capacity *= 2
            
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
        self.last_weights: Optional[torch.Tensor] = None
        self.last_indices: Optional[List[int]] = None

    def add(self, transition: Tuple[Any, ...]) -> None:
        """Add an experience to the buffer.
        
        Handles both single-step and n-step transitions, updating priorities
        appropriately.
        
        Args:
            transition: Tuple containing (state, action, reward, done)
        """
        if self.n_step_learning:
            transition = self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_steps:
                return
                
            _, _, reward, done = self._get_n_step_info()
            frames, action, _, _ = self.n_step_buffer[0]
            transition = (frames, action, reward, done)
            
        super().add(transition)
        
        idx = (self.index - 1) % self.memory_size
        priority = self.max_priority ** self.alpha
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

    def sample(self) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences using prioritized replay.
        
        Returns:
            Tuple containing (states, actions, rewards, next_states, dones)
            All returned tensors are moved to self.output_device
        """
        if len(self) < self.batch_size:
            raise RuntimeError(f"Buffer contains {len(self)} transitions, "
                             f"but batch_size is {self.batch_size}")
            
        self.update_beta()
        self.step += 1
        indices = self._sample_proportional()
        
        # Get experiences
        state_batch = self._state_history[indices, :4]
        next_state_batch = self._state_history[indices, 1:]
        reward_batch = self._reward_history[indices].float()
        action_batch = self._action_history[indices].long()
        done_batch = self._done_history[indices].float()
        
        # Calculate importance sampling weights
        weights = torch.tensor([
            self._calculate_weight(idx, self.beta)
            for idx in indices
        ], device=self.output_device).float()
        
        # Store for priority updates
        self.last_weights = weights
        self.last_indices = indices
        
        # Move to device
        return (
            state_batch.to(self.output_device),
            action_batch.to(self.output_device),
            reward_batch.to(self.output_device),
            next_state_batch.to(self.output_device),
            done_batch.to(self.output_device)
        )

    def update_beta(self) -> None:
        """Update beta parameter for importance sampling.
        
        Beta is annealed from beta_start to 1.0 over beta_frames steps.
        """
        progress = min(1.0, float(self.step) / self.beta_frames)
        self.beta = self.beta_start + (1.0 - self.beta_start) * progress

    def update_priorities(
            self,
            indices: List[int],
            priorities: Union[np.ndarray, List[float]]
    ) -> None:
        """Update priorities for sampled transitions.
        
        Args:
            indices: List of indices to update
            priorities: New priority values (typically TD errors)
            
        Raises:
            AssertionError: If indices and priorities have different lengths
                           or if any priority is <= 0
        """
        if len(indices) != len(priorities):
            raise ValueError("Length of indices and priorities must match")
            
        for idx, priority in zip(indices, priorities):
            if priority <= 0:
                raise ValueError(f"Priority must be positive, got {priority}")
            if not 0 <= idx < self.memory_size:
                raise ValueError(f"Invalid index {idx}")
                
            adjusted_priority = (priority + self.pr_epsilon) ** self.alpha
            self.sum_tree[idx] = adjusted_priority
            self.min_tree[idx] = adjusted_priority
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on their priorities.
        
        Returns:
            List of indices sampled proportionally to their priorities
        """
        indices = []
        length = len(self)
        p_total = self.sum_tree.sum(0, length - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            mass = random.uniform(segment * i, segment * (i + 1))
            idx = self.sum_tree.retrieve(mass)
            indices.append(idx)
            
        return indices

    def _calculate_weight(self, idx: int, beta: float) -> float:
        """Calculate importance-sampling weight for an experience.
        
        Args:
            idx: Index of the experience
            beta: Current beta value for importance sampling
            
        Returns:
            Normalized importance-sampling weight
        """
        length = len(self)
        tree_sum = max(self.sum_tree.sum(0, length - 1), self.pr_epsilon)
        tree_min = max(self.min_tree.min(0, length - 1), self.pr_epsilon)
        
        p_min = tree_min / tree_sum
        max_weight = (p_min * length) ** (-beta)
        
        p_sample = self.sum_tree[idx] / tree_sum
        weight = (p_sample * length) ** (-beta)
        
        return weight / max_weight
