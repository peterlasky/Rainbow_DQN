import numpy as np
import torch
from typing import Tuple
from types import SimpleNamespace
import gymnasium as gym

__all__ = ['ActionHandler', 'VecActionHandler']

class ActionHandler:
    def __init__(self,
                 policy_net:    object,
                 action_space:  object,
                 screen_size:   int,
                 device:        torch.device,
                 epsilons:      SimpleNamespace):
        self.policy_net = policy_net
        self.screen_size = screen_size
        self.action_space = action_space
        self.device = device
        self.epsilon_start = epsilons.epsilon_start
        self.epsilon_final = epsilons.epsilon_final
        self.decrement = epsilons.epsilon_decrement
        self.eval_epsilon = epsilons.eval_epsilon
    
    def get_action(self, 
                   state: torch.Tensor, 
                   rand_mode: bool = False, 
                   eval_mode: bool = False) -> int:
        assert not (rand_mode and eval_mode), "Cannot be in both random and eval mode"
        assert state.shape == (4, self.screen_size, self.screen_size), "State shape is incorrect"

        # if rand_mode then just pick a random action
        if rand_mode:
            return self.action_space.sample()
        
        # otherwise set the epsilon value
        elif eval_mode:
            epsilon = self.eval_epsilon
        else:
            epsilon = self.epsilon
            # decrement epsilon
            self.epsilon = max(self.epsilon_final, self.epsilon - self.decrement)
        
        # Choose a random action with probability epsilon
        if np.random.rand() < epsilon:
            return self.action_space.sample()

        else: 
            # Otherwise pick the action with the highest Q-value

            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0) # (1, 4, 84, 84) or (1, 4, 42, 42)
            self.policy_net.eval()
            with torch.no_grad():
                action = self.policy_net(state).argmax().item()
            return action
# --- end of ActionHandler class

class VecActionHandler:
    """
    Handles action selection for vectorized environments, supporting epsilon-greedy exploration.

    Args:
        policy_net (torch.nn.Module): The neural network used for policy evaluation.
        n_envs (int): Number of vectorized environments.
        action_space (Discrete): Action space of the environments.
        screen_size (int): The size of the input frames (must be 42 or 84).
        device (torch.device): Device on which computations are performed.
        epsilons (Tuple[float, float, int, float]): Parameters for epsilon-greedy exploration.
    """

    def __init__(self,
                 policy_net:    torch.nn.Module,
                 n_envs:        int,
                 action_space:  gym.spaces.Discrete,
                 epsilons:      SimpleNamespace,
                 screen_size:   int = None,
                 device:        torch.device = torch.device('cuda')):
        
        assert ((n_envs in [1, 2]) or (n_envs % 4 == 0) and (n_envs <= 64)), "Number of environments must be in [1,2] or divisible by 4 and <= 64" 

        self.policy_net = policy_net
        self.n_envs = n_envs
        self.screen_size = screen_size
        self.action_space = action_space
        self.device = device

        self.epsilon = epsilons.epsilon_start
        self.epsilon_final = epsilons.epsilon_final
        self.decrement = epsilons.epsilon_decrement
        self.eval_epsilon = epsilons.eval_epsilon
        

    def get_actions(self,
                    states:    np.typing.NDArray[np.float32],
                    rand_mode: bool = False,
                    eval_mode: bool = False) -> np.typing.NDArray[np.int_]:
        """
        Select actions for vectorized environments using epsilon-greedy or policy evaluation.

        Args:
            states (NDArray[np.float32]): Input states of shape (n_envs, 4, screen_size, screen_size).
            rand_mode (bool): If True, selects random actions.
            eval_mode (bool): If True, uses fixed epsilon for evaluation.

        Returns:
            NDArray[np.int_]: Array of actions for each environment.
        """
        # Assertions for input validation
        assert isinstance(states, np.ndarray), f"States must be a numpy.ndarray, got {type(states)}"
        assert states.shape == (self.n_envs, 4, self.screen_size, self.screen_size), \
            f"States shape is incorrect: expected {(self.n_envs, 4, self.screen_size, self.screen_size)}, got {states.shape}"
        assert not (rand_mode and eval_mode), "Cannot be in both random and eval mode"
        
        # Random action mode
        if rand_mode:
            return np.random.randint(low=0, high=4, size=self.n_envs)

        # Set epsilon value
        if eval_mode:
            epsilon = self.eval_epsilon
        else:
            epsilon = self.epsilon
            # Each call to get_actions() represents n_envs timesteps of experience
            if not eval_mode:  # Only reduce epsilon in training mode
                self.epsilon = max(self.epsilon_final, self.epsilon - self.decrement * self.n_envs)

        # Epsilon-greedy: choose random action with probability epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(low=0, high=4, size=self.n_envs)

        # Policy-based action selection
        states = torch.from_numpy(states).float().to(self.device)  # Convert states to tensor on the correct device
        self.policy_net.eval()
        with torch.no_grad():
            actions = self.policy_net(states).argmax(dim=1).cpu().numpy()  # Get best actions for all envs
        return actions
