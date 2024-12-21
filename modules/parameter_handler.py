"""Parameter management for reinforcement learning experiments.

This module handles the initialization, validation, and organization of
hyperparameters for deep reinforcement learning experiments. It provides
functionality to:
    - Merge user-provided and default parameters
    - Validate parameter combinations
    - Create organized parameter namespaces for different components
    - Handle device selection (CPU/CUDA)
"""

from types import SimpleNamespace
from typing import Union, Dict, Any
from pathlib import Path
import copy

from default_parameters import DefaultParameters
from modules.utils import validate_device


class ParameterHandler:
    """Manages and validates parameters for reinforcement learning experiments.
    
    This class handles parameter initialization, validation, and organization.
    It ensures that all parameters are valid and properly structured for use
    in the training process.
    
    Attributes:
        device: Selected computation device (CPU/CUDA)
        _p: Finalized parameter namespace
        _default_parameters: Default parameter values
    """
    
    def __init__(self, user_parameters: Union[SimpleNamespace, Dict[str, Any]]) -> None:
        """Initialize parameter handler with user-provided parameters.
        
        Args:
            user_parameters: User-specified parameters as namespace or dict
            
        Raises:
            ValueError: If invalid parameters are provided
            AssertionError: If parameter combinations are invalid
        """
        # Convert dict to SimpleNamespace if needed
        if isinstance(user_parameters, dict):
            user_parameters = SimpleNamespace(**user_parameters)
        if not isinstance(user_parameters, SimpleNamespace):
            raise TypeError('User parameters must be a SimpleNamespace or dictionary')

        self._user_parameters = user_parameters
        self._default_parameters = DefaultParameters()

        self._validate_parameter_names()
        self._merge_parameters()
        self._configure_update_frequency()
        self._add_helper_parameters()
        self._validate_parameter_values()
        

    def _validate_parameter_names(self) -> None:
        """Ensure all user parameters are valid."""
        default_keys = set(vars(self._default_parameters).keys())
        user_keys = set(vars(self._user_parameters).keys())
        illegal_params = user_keys - default_keys
        
        if illegal_params:
            raise ValueError(f"Invalid parameters: {', '.join(illegal_params)}")

    def _merge_parameters(self) -> None:
        """Merge user parameters with defaults."""
        vars(self._default_parameters).update(vars(copy.deepcopy(self._user_parameters)))
        self._p = self._default_parameters

    def _add_helper_parameters(self) -> None:
        """Add derived and helper parameters."""
        p = self._p
        
        # Timing and intervals
        p.target_update_interval = p.target_update_interval // p.n_envs * p.n_envs
        p.exit_time_limit_seconds = p.exit_time_limit * 60
        
        # Algorithm-specific parameters
        p.categorical_params = SimpleNamespace(
            atom_size=p.atom_size,
            Vmin=p.Vmin,
            Vmax=p.Vmax
        )
        p.per_params = SimpleNamespace(
            alpha=p.alpha,
            beta_start=p.beta_start,
            beta_frames=p.beta_frames,
            pr_epsilon=p.pr_epsilon
        )
        p.n_step_params = SimpleNamespace(
            n_steps=p.n_steps,
            memory_size=p.memory_size,
            gamma=p.gamma
        )
        p.epsilons = SimpleNamespace(
            epsilon_start=p.epsilon_start,
            epsilon_final=p.epsilon_final,
            eval_epsilon=p.eval_epsilon,
            epsilon_decrement=(p.epsilon_start - p.epsilon_final) / p.epsilon_decay_steps
        )

    def _validate_parameter_values(self) -> None:
        """Validate parameter values and combinations."""
        p = self._p
        
        # Filename validation
        if p.group_dir is None:
            raise ValueError("Group directory must be specified.")
        if p.name is None:
            raise ValueError("Experiment name must be specified.")
            
        # Screen size validation
        if p.screen_size not in [42, 84]:
            raise ValueError(f"Invalid screen size: {p.screen_size}. Must be 42 or 84.")
        
        # n_envs limitations
        assert p.n_envs <= p.policy_update_interval or p.n_envs % p.policy_update_interval == 0, \
            f"n_envs={p.n_envs} must be divisible by policy_update_interval={p.policy_update_interval}."
        # Interval validations
        intervals = {
            'policy_update_interval_adjusted': p.policy_update_interval_adjusted,
            'pbar_update_interval_adjusted': p.pbar_update_interval,
            'eval_interval_adjusted': p.eval_interval,
            'target_update_interval_adjusted': p.target_update_interval,
            'checkpoint_interval_adjusted': p.checkpoint_interval}

        if p.record_interval:
            intervals['record_interval_adjusted'] = p.record_interval
            
        for k, v in intervals.items():
            if k == "policy_update_interval_adjusted":
                continue
            if v % p.n_envs != 0:
                ## instead of raising error, let's adjust to the next multiple
                intervals[k] = v // p.n_envs * p.n_envs
                short_k = k.rsplit('_',1)[0]
                print(f"Warning: {short_k} must be divisible by n_envs={p.n_envs}. ", end='')
                print(f"Adjusting {short_k} from {v} to {intervals[k]}.")
        p.intervals = SimpleNamespace(**intervals) # store intervals

        # Device validation: uses outside function
        p.device = validate_device(p.device)
    
    def _configure_update_frequency(self) -> None:
        """Configure policy update frequency based on environment count."""
        p = self._p
        inteval_change, batch_update_change = False, False
        if p.n_envs <= p.policy_update_interval:
            # Fewer envs than update interval - do fewer updates
            p.policy_update_interval_adjusted = p.policy_update_interval // p.n_envs
            p.n_batch_updates_adjusted = 1
        else:
            # More envs than update interval - do more batch updates
            p.policy_update_interval_adjusted = p.policy_update_interval
            p.n_batch_updates_adjusted = p.n_envs // p.policy_update_interval

        # Store both original and adjusted values
        p.n_batch_updates = p.n_batch_updates_adjusted  # Store original for comparison
        
        if p.policy_update_interval != p.policy_update_interval_adjusted:
            print(f'Adjusting policy update interval from {p.policy_update_interval} to {p.policy_update_interval_adjusted} to account for n_envs={p.n_envs}.')
        if p.n_batch_updates != p.n_batch_updates_adjusted:
            print(f'Adjusting n_batch_updates from {p.n_batch_updates} to {p.n_batch_updates_adjusted} batch updates per policy update.')
    
    @property
    def all_parameters(self) -> SimpleNamespace:
        """Get organized parameter namespace.
        
        Returns:
            SimpleNamespace containing all parameters organized by component
        """
        # Create fresh parameter copy
        p = SimpleNamespace(**vars(self._default_parameters))
        for key, value in vars(self._p).items():
            setattr(p, key, value)

        # Organize algorithm-specific parameters
        if p.n_step_learning:
            p.n_step_params = SimpleNamespace(
                n_steps=p.n_steps,
                memory_size=p.n_memory_size,
                gamma=p.n_gamma
            )
            del p.n_steps, p.n_memory_size, p.n_gamma

        if p.categorical_DQN:
            p.categorical_params = SimpleNamespace(
                atom_size=p.atom_size,
                Vmin=p.Vmin,
                Vmax=p.Vmax
            )
            del p.atom_size, p.Vmin, p.Vmax

        if p.prioritized_replay:
            p.per_params = SimpleNamespace(
                alpha=p.alpha,
                beta_start=p.beta_start,
                beta_frames=p.beta_frames,
                pr_epsilon=p.pr_epsilon
            )
            del p.alpha, p.beta_start, p.beta_frames, p.pr_epsilon
        else:
            p.per_params = None

        return p

    # Public
    def save_user_parameters(self, filepath: Path):
        # save user parameters as json
        self._p.user_parameters = vars(self._p)
        del self._p.user_parameters
