"""Factory functions for creating replay buffers.

This module provides functions to create replay buffers for reinforcement learning
agents. It supports both standard and prioritized experience replay (PER) buffers,
as well as n-step learning configurations.

The buffers are configured based on parameters provided in a SimpleNamespace object,
which should contain all necessary settings for buffer initialization.
"""

from types import SimpleNamespace
from typing import Tuple, Dict, Optional, Union

import torch

from .per_buffer import PerBuffer
from .replay_buffer import ReplayBuffer


def get_replay_buffers(
        params: SimpleNamespace,
        output_device: Optional[torch.device] = None
) -> Tuple[Union[ReplayBuffer, PerBuffer], Optional[Union[ReplayBuffer, PerBuffer]]]:
    """Create replay buffer instances based on configuration parameters.
    
    Creates either standard or prioritized experience replay buffers based on the
    configuration. If n-step learning is enabled, creates an additional buffer
    for storing n-step transitions.
    
    Args:
        params: Configuration parameters containing:
            - batch_size: Number of samples per batch
            - memory_size: Maximum number of transitions to store
            - screen_size: Size of observation frames
            - device: Default device for tensor operations
            - n_step_learning: Whether to use n-step learning
            - n_step_params: Parameters for n-step learning (if enabled)
            - per_params: Parameters for prioritized replay (if enabled)
            - prioritized_replay: Whether to use prioritized replay
        output_device: Optional device override for tensor operations
            
    Returns:
        Tuple containing:
        - Main replay buffer (standard or PER)
        - N-step replay buffer if n-step learning enabled, else None
        
    Note:
        The n-step buffer uses the same buffer type (standard/PER) as the
        main buffer but with different memory size and no n-step configuration.
    """
    # Select buffer class based on configuration
    BufferClass = PerBuffer if params.prioritized_replay else ReplayBuffer
    
    # Use specified output device or fall back to params device
    device = output_device or params.device
    
    # Create main replay buffer
    main_buffer = BufferClass(
        batch_size=params.batch_size,
        memory_size=params.memory_size,
        screen_size=params.screen_size,
        output_device=device,
        n_step_learning=params.n_step_learning,
        n_steps=(params.n_step_params.n_steps 
                if params.n_step_learning else None),
        n_step_gamma=(params.n_step_params.gamma 
                     if params.n_step_learning else None),
        per_params=params.per_params
    )
    
    # Create n-step buffer if enabled
    if params.n_step_learning:
        n_step_buffer = BufferClass(
            batch_size=params.batch_size,
            memory_size=params.n_step_params.memory_size,
            screen_size=params.screen_size,
            output_device=device,
            n_step_learning=False,
            n_steps=None,
            n_step_gamma=None,
            per_params=params.per_params
        )
    else:
        n_step_buffer = None
        
    return main_buffer, n_step_buffer