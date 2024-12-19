import torch
from types import SimpleNamespace
from typing import Union
from default_parameters import get_default_parameters

class ParameterHandler:
    def __init__(self, user_parameters: Union[SimpleNamespace, dict]):
        # Ensure user_parameters is a SimpleNamespace
        if isinstance(user_parameters, dict):
            user_parameters = SimpleNamespace(**user_parameters)
        assert isinstance(user_parameters, SimpleNamespace), 'User parameters must be a SimpleNamespace or dictionary.'

        # Store user parameters
        self._p = user_parameters

        # Get default parameters
        self._default_parameters = get_default_parameters()

        # Check for invalid parameters
        default_keys = set(vars(self._default_parameters).keys())
        user_keys = set(vars(user_parameters).keys())
        illegal_params = user_keys - default_keys
        if illegal_params:
            raise ValueError(f"Invalid parameters: {', '.join(illegal_params)}")

        # Update defaults with user parameters
        vars(self._default_parameters).update(vars(user_parameters))
        p = self._default_parameters  # Alias for convenience

        # Add helper parameters
        p.target_update_interval = p.target_update_interval // p.n_envs * p.n_envs
        p.exit_time_limit_seconds = p.exit_time_limit * 60
        p.categorical_params = SimpleNamespace(atom_size=p.atom_size, Vmin=p.Vmin, Vmax=p.Vmax)
        p.per_params = SimpleNamespace(alpha=p.alpha, beta_start=p.beta_start, beta_frames=p.beta_frames, pr_epsilon=p.pr_epsilon)
        p.n_step_params = SimpleNamespace(n_steps=p.n_steps, memory_size=p.memory_size, gamma=p.gamma)
        p.epsilons = SimpleNamespace(
            epsilon_start=p.epsilon_start,
            epsilon_final=p.epsilon_final,
            eval_epsilon=p.eval_epsilon,
            epsilon_decrement=(p.epsilon_start - p.epsilon_final) / p.epsilon_decay_steps,
        )

        # Validate parameters
        assert p.screen_size in [42, 84], f"Invalid screen size: {p.screen_size}. Must be 42 or 84."
        assert p.pbar_update_interval % p.n_envs == 0, f"pbar_update_interval ({p.pbar_update_interval}) must be divisible by n_envs ({p.n_envs})."
        assert p.eval_interval % p.n_envs == 0, f"eval_interval ({p.eval_interval}) must be divisible by n_envs ({p.n_envs})."
        if p.record_interval:
            assert p.record_interval % p.n_envs == 0, f"record_interval ({p.record_interval}) must be divisible by n_envs ({p.n_envs})."
        assert p.checkpoint_interval % p.n_envs == 0, f"checkpoint_interval ({p.checkpoint_interval}) must be divisible by n_envs ({p.n_envs})."

        # Determine policy update frequency
        if p.n_envs <= p.policy_update_interval:
            p.policy_update_interval_adjusted = p.policy_update_interval // p.n_envs
            p.n_batch_updates = 1
        else:
            p.policy_update_interval_adjusted = 1
            p.n_batch_updates = p.n_envs // p.policy_update_interval

        self._p = p  # Store finalized parameters

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            print("Warning: CUDA not available, using CPU.")

    def get_parameters(self) -> SimpleNamespace:
        ''' Get the parameters '''
        # Create a SimpleNamespace with default parameters
        p = SimpleNamespace(**vars(self._default_parameters))

        # Update with user parameters
        for key, value in vars(self._p).items():
            setattr(p, key, value)

        # Organize N-step parameters
        if p.n_step_learning:
            p.n_step_params = SimpleNamespace(
                n_steps=p.n_steps,
                memory_size=p.n_memory_size,
                gamma=p.n_gamma
            )

        # Organize categorical parameters
        if p.categorical_DQN:
            p.categorical_params = SimpleNamespace(
                atom_size=p.atom_size,
                Vmin=p.Vmin,
                Vmax=p.Vmax
            )

        # Organize PER parameters
        if p.prioritized_replay:
            p.per_params = SimpleNamespace(
                alpha=p.alpha,
                beta_start=p.beta_start,
                beta_frames=p.beta_frames,
                pr_epsilon=p.pr_epsilon
            )
        else:
            p.per_params = None

        # Delete redundant parameters
        if p.n_step_learning:
            del p.n_steps, p.n_memory_size, p.n_gamma
        if p.categorical_DQN:
            del p.atom_size, p.Vmin, p.Vmax
        if p.prioritized_replay:
            del p.alpha, p.beta_start, p.beta_frames, p.pr_epsilon

        return p
