import numpy as np
from .segment_tree import SumSegmentTree, MinSegmentTree

import torch
from typing import Tuple
from collections import deque
from types import SimpleNamespace

#import tensordict as td

class ReplayBuffer:
    def __init__(self,
                 batch_size:        int = None,
                 memory_size:       int = None,
                 screen_size:       int = None,
                 output_device:     torch.device = None,
                 n_step_learning:   bool = False,
                 n_steps:           int = None,   
                 n_step_gamma:      float = None,       # default usually is .99
                 per_params:        SimpleNamespace = None        
                    ):

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.screen_size = screen_size
        self.output_device = output_device

        self._buffer_len = 0
        self.index = 0

        self._state_history = torch.zeros((self.memory_size, 5, self.screen_size, self.screen_size), dtype=torch.uint8)
        self._reward_history = torch.zeros(size=(self.memory_size,1), dtype=torch.uint8)
        self._action_history = torch.zeros(size=(self.memory_size,1), dtype=torch.uint8)
        self._done_history = torch.zeros(size=(self.memory_size,1), dtype=torch.uint8)
        
        # for n-step learning
        self.n_step_learning = n_step_learning
        if n_step_learning:
            if n_steps is None: raise ValueError("n_steps must be provided if n_step_learning is enabled")
            if n_step_gamma is None: raise ValueError("n_step_gamma must be provided if n_step_learning is enabled")
            self.n_steps = n_steps
            self.n_step_gamma = n_step_gamma
            self.n_step_buffer = deque(maxlen=n_steps)

        # for PER
        self.per_params = per_params
        if self.per_params is not None:
            # Get next power of 2 for tree capacity
            tree_capacity = 1
            while tree_capacity < self.memory_size:
                tree_capacity *= 2
            self.tree = SumSegmentTree(tree_capacity)
            self.min_tree = MinSegmentTree(tree_capacity)
            self.max_priority = 1.0  # Start with max priority
            self.tree_idx = 0

    def add(self, transition: Tuple[np.ndarray, int, float, bool]) -> Tuple[np.ndarray, int, float, bool]:
        frames, action, reward, done = transition
        if self.n_step_learning:
            self.n_step_buffer.append(transition)

            # Single step transition is not ready
            if len(self.n_step_buffer) < self.n_steps:
                return
            
            # Make an n-step transition
            _, _, reward, done = self._get_n_step_info()
            frames, action, _, _ = self.n_step_buffer[0]
            transition = (frames, action, reward, done)

        # Store the tuple
        self._state_history[self.index] = torch.from_numpy(frames).view(5, self.screen_size, self.screen_size).to(torch.uint8)
        self._action_history[self.index,0] = torch.tensor(action).to(torch.uint8)
        self._reward_history[self.index,0] = torch.tensor(reward).to(torch.uint8)
        self._done_history[self.index,0] = torch.tensor(done).to(torch.uint8)

        # Increment the index pointer and wrap around if it exceeds memory_size
        self.index = (self.index + 1) % self.memory_size
        self._buffer_len = min(self._buffer_len + 1, self.memory_size)

        if self.per_params is not None:
            self.tree[self.tree_idx] = self.max_priority
            self.min_tree[self.tree_idx] = self.max_priority
            self.tree_idx = (self.tree_idx + 1) % self.memory_size

        return transition
    
    def _get_n_step_info(self):
        assert self.n_step_learning, "You should not be calling this method if n_step_learning is disabled"
        _, _, reward, done = self.n_step_buffer[-1]
        transitions = reversed(list(self.n_step_buffer)[:-1])       # remove the last transition and reverse
        for transition in transitions:
            _, _, r, d = transition
            reward = r + self.n_step_gamma * (reward * (1 - d))
            done = d if d else done
        
        return (None, None, reward, done)

    # Get a batch from the buffer, transferring it to the return_device
    def sample(self):

        # Use PyTorch's random number generator to sample indices (avoiding index 0)
        if self.per_params is not None:
            batch_idx = self._sample_proportional()
        else:
            batch_idx = torch.randint(0, self._buffer_len, (self.batch_size,))

        # Gather the batches
        state_batch = self._state_history[batch_idx,:4]           # First 4 frames of the 5 frame index
        next_state_batch = self._state_history[batch_idx,1:]      # Last 4 frames of the 5 frame index
        action_batch = self._action_history[batch_idx].long()     # Actions as long (for indexing)
        reward_batch = self._reward_history[batch_idx].float()    # Rewards as float for compatibility
        done_batch = self._done_history[batch_idx].float()        # Done flags as float for compatibility

        state_batch = state_batch.to(self.output_device)
        next_state_batch = next_state_batch.to(self.output_device)
        action_batch = action_batch.to(self.output_device)
        reward_batch = reward_batch.to(self.output_device)
        done_batch = done_batch.to(self.output_device)

        if self.per_params is not None:
            weights = self._calculate_weights(batch_idx)
            return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights
        else:
            return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def _sample_proportional(self):
        mass = self.tree.sum() / self.batch_size
        batch_idx = torch.zeros(self.batch_size, dtype=torch.long)
        for i in range(self.batch_size):
            mass_sample = mass * torch.rand(1)
            idx = self.tree.find_prefixsum_idx(mass_sample)
            batch_idx[i] = idx
        return batch_idx

    def _calculate_weights(self, batch_idx):
        weights = []
        p_min = self.min_tree.min() / self.tree.sum()
        max_weight = (p_min * self.batch_size) ** (-self.per_params.beta)
        for idx in batch_idx:
            p_sample = self.tree[idx] / self.tree.sum()
            weight = (p_sample * self.batch_size) ** (-self.per_params.beta)
            weights.append(weight / max_weight)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return self._buffer_len
