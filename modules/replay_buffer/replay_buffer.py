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
            if n_steps is None: raise ValueError("n_step must be provided if n_step_learning is enabled")
            if n_step_gamma is None: raise ValueError("n_step_gamma must be provided if n_step_learning is enabled")
            self.n_steps = n_steps
            self.n_step_gamma = n_step_gamma
            self.n_step_buffer = deque(maxlen=n_steps)

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
        idx = torch.randint(0, self._buffer_len, (self.batch_size,))

        # Gather the batches
        state_batch = self._state_history[idx,:4]           # First 4 frames of the 5 frame index
        next_state_batch = self._state_history[idx,1:]      # Last 4 frames of the 5 frame index
        action_batch = self._action_history[idx].long()     # Actions as long (for indexing)
        reward_batch = self._reward_history[idx].float()    # Rewards as float for compatibility
        done_batch = self._done_history[idx].float()        # Done flags as float for compatibility

        state_batch = state_batch.to(self.output_device)
        next_state_batch = next_state_batch.to(self.output_device)
        action_batch = action_batch.to(self.output_device)
        reward_batch = reward_batch.to(self.output_device)
        done_batch = done_batch.to(self.output_device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return self._buffer_len
    
    

