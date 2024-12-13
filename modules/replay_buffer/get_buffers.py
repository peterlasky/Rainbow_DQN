from .per_buffer import PerBuffer
from .replay_buffer import ReplayBuffer
import torch
from types import SimpleNamespace
from typing import Tuple 

'''
A function that builds the memory buffer for the agent.  
Configured to allow priority replay buffering as well as n-step learning.

'''
def get_replay_buffers( prioritized_replay:   bool,
                        n_step_learning:      bool,
                        batch_size:           int,
                        memory_size:          int,
                        screen_size:          int,
                        output_device:        torch.device,
                        n_step_params:        SimpleNamespace,
                        per_params:           SimpleNamespace) -> Tuple[ReplayBuffer, ReplayBuffer]:

    BufferClass = PerBuffer if prioritized_replay else ReplayBuffer

    memory = BufferClass(   batch_size=         batch_size,
                            memory_size=        memory_size,
                            screen_size=        screen_size,
                            output_device=      output_device,
                            n_step_learning=    n_step_learning,
                            n_steps=            1,
                            n_step_gamma=       n_step_params.gamma,
                            per_params=         per_params)
    if n_step_learning:
        n_step_memory = ReplayBuffer(
                    batch_size=         batch_size,
                    memory_size=        n_step_params.memory_size,
                    screen_size=        screen_size,
                    output_device=      output_device,
                    n_step_learning=    True,
                    n_steps=            n_step_params.n_steps,           # default usually is 3, passed as parameter to the class
                    n_step_gamma=       n_step_params.gamma)        # default usually is .99, passed as parameter to the class
    else:
        n_step_memory = None

    return (memory, n_step_memory)