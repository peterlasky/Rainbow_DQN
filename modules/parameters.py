import torch
from typing import Dict
from types import SimpleNamespace

class ParameterHandler:
    ''' Simple class to manager and verify the parameters, reduce clutter '''
    def __init__(self, 
                 user_parameters: Dict):
        
        self._default_parameters = dict(
            # Rainbow DQN Flags
            doubleQ=                    False,
            dueling=                    False,
            categorical_DQN=            False,
            noisy_linear=               False,
            prioritized_replay=         False,
            n_step_learning=            False,

            # vectorization parameters
            n_envs=                     8,
            group_training_losses =     False,
            
            # Environment parameters
            asynchronous=               False,
            seed=                       0,
            env_name=                   "BreakoutNoFrameskip-v4",
            screen_size=                84,
            noop_min=                   10,
            noop_max=                   10,
            fire_on_life_loss=          False,

            device =                    torch.device('cuda'),

            # Model parameters
            memory_size=                1_000_000,
            batch_size=                 32,
            random_starts=              50_000,
            learning_rate=              0.0000625,
            gradient_clamping=          True,
            gamma=                      0.99,
            scale_around_zero=          False,

            # Experimental parameters
            batch_norm=                 False,
            layer_norm=                 False,

            # Epsilon parameters
            epsilon_start=              1.0,
            epsilon_final=              0.1,
            epsilon_decay_steps=        1_000_000,
            eval_epsilon=               0.05,

            # Interval parameters
            policy_update_interval=     4,
            pbar_update_interval=       100,
            target_update_interval=     10_000,
            eval_interval=              50_000,
            n_games_per_eval=           10,
            checkpoint_interval=        2_500_000,
            record_interval=            None,

            # Exit conditions   (time in minutes),
            max_steps=                  20_000_000,
            exit_trailing_average=      10000,
            exit_time_limit=            1200,

            ## Rainbow parameters
            # Categorical DQN-specific parameters
            atom_size=                  51,
            Vmin =                      -10,
            Vmax =                      10,
            # Priority Replay-specific parameters
            alpha=                      0.6,
            beta_start=                 0.4,
            beta_frames=                100_000,
            pr_epsilon=                 1e-5,
            # N-step learning-specific parameters
            n_steps=                    3,
            n_memory_size=                500,
            n_gamma=                      0.99,

            # Logging parameters
            trailing_avg_trail=         20,
            name=                       '[no name]',
            log_dir=                    '[no name]',
            overwrite_previous=         False,
            data_logging=               True,
            note=                       '[no note]',    
            data_plotting=              False,
        )
        #####################################################################
         
        ## Check for unnamed parameters
        has_illegal_params = False
        for param in user_parameters.keys():
            if param not in self._default_parameters:
                has_illegal_params = True
                print(f'Parameter {param} not valid.')
        if has_illegal_params:
            raise ValueError(f'Illegal parameters.')
     
        ## Overlay user parameters
        self._default_parameters.update(user_parameters)
        parameters = self._default_parameters

        ## Add helper parameters
        parameters.update(dict(
            target_update_interval = parameters['target_update_interval'] // parameters['n_envs'] * parameters['n_envs'],
            exit_time_limit_seconds = parameters['exit_time_limit'] * 60,
            
            # Group related parameters
            categorical_params = SimpleNamespace(
                atom_size = parameters['atom_size'],
                Vmin = parameters['Vmin'],
                Vmax = parameters['Vmax']
            ),
            per_params = SimpleNamespace(
                alpha = parameters['alpha'],
                beta_start = parameters['beta_start'],
                beta_frames = parameters['beta_frames'],
                pr_epsilon = parameters['pr_epsilon']
            ),
            n_step_params = SimpleNamespace(
                n_steps = parameters['n_steps'],
                memory_size = parameters['memory_size'],
                gamma = parameters['gamma']
            ),

            epsilons = SimpleNamespace(
                epsilon_start = parameters['epsilon_start'],
                epsilon_final = parameters['epsilon_final'],
                eval_epsilon = parameters['eval_epsilon'],
                epsilon_decrement = (parameters['epsilon_start'] - parameters['epsilon_final']) / parameters['epsilon_decay_steps'],
            )
        ))

        ## check for illegal values
        assert parameters['screen_size'] in [42, 84], "Screen size must be 42 or 84"
        assert parameters['pbar_update_interval'] % parameters['n_envs'] == 0, "pbar_update_interval must be divisible by n_envs"
        parameters['target_update_interval'] = parameters['target_update_interval'] // parameters['n_envs'] * parameters['n_envs']
        assert parameters['eval_interval'] % parameters['n_envs'] == 0, "eval_interval must be divisible by n_envs"
        if parameters['record_interval'] is not None:
            assert parameters['record_interval'] % parameters['n_envs'] == 0, "record_interval must be divisible by n_envs"
        assert parameters['checkpoint_interval'] % parameters['n_envs'] == 0, "checkpoint_interval must be divisible by n_envs"

        ## Deterimine policy update frequency and number of policy updates
        if parameters['n_envs'] <= parameters['policy_update_interval']:
            parameters['policy_update_interval_adjusted'] = parameters['policy_update_interval'] // parameters['n_envs']
            parameters['n_batch_updates'] = 1
        else:
            parameters['policy_update_interval_adjusted'] = 1
            parameters['n_batch_updates'] = parameters['n_envs'] // parameters['policy_update_interval']

        # set parameters
        self._parameters = parameters
        
        # Check for cuda
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise RuntimeError("CUDA not available: Use a machine with GPU.")



    def get_parameters(self) -> Dict:
        return self._parameters
   