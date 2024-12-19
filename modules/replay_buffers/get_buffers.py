from .per_buffer import PerBuffer
from .replay_buffer import ReplayBuffer
import torch
from types import SimpleNamespace
from typing import Tuple, Dict

'''
A function that builds the memory buffer for the agent.  
Configured to allow priority replay buffering as well as n-step learning.

'''
def get_replay_buffers(params: SimpleNamespace = None,
                       output_device: torch.device = None
                       ) -> Tuple[ReplayBuffer, ReplayBuffer]: 
    
    p = params  # Already a SimpleNamespace
    BufferClass = PerBuffer if p.prioritized_replay else ReplayBuffer

    # Use the device from parameters if output_device is not specified
    device = output_device if output_device is not None else p.device

    memory = BufferClass(   batch_size=         p.batch_size,
                            memory_size=        p.memory_size,
                            screen_size=        p.screen_size,
                            output_device=      device,
                            n_step_learning=    p.n_step_learning,
                            n_steps=            p.n_step_params.n_steps if p.n_step_learning else None,
                            n_step_gamma=       p.n_step_params.gamma if p.n_step_learning else None,
                            per_params=         p.per_params)

    if p.n_step_learning:
        n_step_memory = BufferClass(
                            batch_size=         p.batch_size,
                            memory_size=        p.n_step_params.memory_size,
                            screen_size=        p.screen_size,
                            output_device=      device,
                            n_step_learning=    False,
                            n_steps=            None,
                            n_step_gamma=       None,
                            per_params=         p.per_params)
    else:
        n_step_memory = None

    return memory, n_step_memory