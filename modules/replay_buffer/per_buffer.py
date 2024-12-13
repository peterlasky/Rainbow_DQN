from .segment_tree import SumSegmentTree, MinSegmentTree
from .replay_buffer import ReplayBuffer
from typing import List
from types import SimpleNamespace
import numpy as np
import torch
import random

class PerBuffer(ReplayBuffer):
    def __init__(self, 
                 batch_size:        int,
                 memory_size:       int,
                 screen_size:       int,
                 output_device:     torch.device = torch.device('cpu'),
                 n_step_learning:   int = False,     # New parameter to enable n-step learning
                 n_steps:           int = None,              # Number of steps for n-step learning
                 n_step_gamma:      float = None,         # Discount factor for n-step learning
                 per_params:        SimpleNamespace = None):
        
        super().__init__(batch_size=batch_size, 
                         memory_size=int(memory_size),
                         screen_size=screen_size,
                         output_device=output_device,
                         n_step_learning=n_step_learning,   # Pass n-step parameters to parent
                         n_steps=n_steps,         
                         n_step_gamma=n_step_gamma)
        
        self.alpha = per_params.alpha
        self.beta_start = per_params.beta_start
        self.beta_frames = per_params.beta_frames
        self.pr_epsilon = per_params.pr_epsilon
        self.beta = self.beta_start
        self.max_priority = 1.0
        self.output_device = output_device

        # Initialize the step counter for beta decay
        self.step = 0

        # Initialize the segment trees for prioritized sampling
        tree_capacity = 1
        while tree_capacity < self.memory_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        # Store last sampled weights and indices
        self.last_weights = None
        self.last_indices = None

    def add(self, transition):
        """
        Add an experience to the buffer, with support for n-step learning.
        """
        if self.n_step_learning:
            # Process the transition as an n-step transition
            transition = self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_steps:
                return  # Wait until we have enough transitions for n-step
            
            # Create an n-step transition
            _, _, reward, done = self._get_n_step_info()
            frames, action, _, _ = self.n_step_buffer[0]
            transition = (frames, action, reward, done)

        # Add the processed (single-step or n-step) transition to the primary buffer
        super().add(transition)
        
        # Update the priority for the new experience
        idx = (self.index - 1) % self.memory_size
        idx = int(idx)
        priority = self.max_priority ** self.alpha
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

    def sample(self):
        """
        Sample a batch of experiences with prioritized replay, incorporating n-step if enabled.
        """
        assert len(self) >= self.batch_size
        self.update_beta()
        self.step += 1
        indices = self._sample_proportional()

        # Retrieve the sampled experiences
        state_batch = self._state_history[indices, :4]
        next_state_batch = self._state_history[indices, 1:]
        reward_batch = self._reward_history[indices].float()
        action_batch = self._action_history[indices].long()
        done_batch = self._done_history[indices].float()

        # Calculate importance-sampling weights for the batch
        weights = []
        for idx in indices:
            weight = self._calculate_weight(idx, self.beta)
            weights.append(weight)
        weights = torch.tensor(weights, device=self.output_device).float()

        # Store weights and indices internally for later use (e.g., updating priorities)
        self.last_weights = weights
        self.last_indices = indices

        # Move tensors to the appropriate device
        state_batch = state_batch.to(self.output_device)
        next_state_batch = next_state_batch.to(self.output_device)
        reward_batch = reward_batch.to(self.output_device)
        action_batch = action_batch.to(self.output_device)
        done_batch = done_batch.to(self.output_device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def update_beta(self):
        """
        Update beta towards 1 over time to increase the importance of lower-probability samples.
        """
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * float(self.step) / self.beta_frames)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities of sampled transitions based on new TD errors.
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.memory_size

            # Adjust priority with epsilon to avoid zero
            adjusted_priority = (priority + self.pr_epsilon) ** self.alpha
            self.sum_tree[idx] = adjusted_priority
            self.min_tree[idx] = adjusted_priority

            # Track the max priority for adding new transitions
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """
        Sample indices based on priority proportions.
        """
        indices = []
        length = len(self)
        p_total = self.sum_tree.sum(0, length - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            sample_value = random.uniform(a, b)
            idx = self.sum_tree.retrieve(sample_value)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """
        Calculate the importance-sampling weight of an experience at the specified index.
        """
        length = len(self)
        tree_sum = self.sum_tree.sum(0, length - 1)
        tree_min = self.min_tree.min(0, length - 1)

        # Avoid division by zero
        tree_min = max(tree_min, self.pr_epsilon)
        tree_sum = max(tree_sum, self.pr_epsilon)

        p_min = tree_min / tree_sum
        max_weight = (p_min * length) ** (-beta)

        p_sample = self.sum_tree[idx] / tree_sum
        weight = (p_sample * length) ** (-beta)
        weight = weight / max_weight

        return weight





