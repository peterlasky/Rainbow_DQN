"""Action selection handlers for reinforcement learning agents.

This module provides classes for handling action selection in both single and
vectorized environments. It implements epsilon-greedy exploration strategies
and supports both training and evaluation modes.

Classes:
    ActionHandler: Handles action selection for single environment
    VecActionHandler: Handles action selection for multiple parallel environments
"""

from types import SimpleNamespace
from typing import Union, Optional

import gymnasium as gym
import random
import torch

__all__ = ['ActionHandler', 'VecActionHandler']


class ActionHandler:
    """Handles action selection for a single environment.
    
    Implements epsilon-greedy exploration strategy with decaying epsilon value
    for training and fixed epsilon for evaluation.
    
    Attributes:
        policy_net: Neural network for action selection
        action_space: Gymnasium discrete action space
        step: Current step count
        epsilon: Current exploration rate
        epsilon_decrement: Rate of epsilon decay
        epsilon_final: Minimum epsilon value
    """
    
    def __init__(
            self,
            params: SimpleNamespace,
            policy_net: torch.nn.Module,
            action_space: gym.spaces.Discrete
    ) -> None:
        """Initialize the action handler.
        
        Args:
            params: Configuration parameters including epsilon settings
            policy_net: Neural network for action selection
            action_space: Gymnasium discrete action space
        """
        self.policy_net = policy_net
        self.action_space = action_space
        self.step = 0
        
        # Epsilon-greedy parameters
        self.epsilon = params.epsilons.epsilon_start
        self.epsilon_decrement = params.epsilons.epsilon_decrement
        self.epsilon_final = params.epsilons.epsilon_final
        self.eval_epsilon = params.epsilons.eval_epsilon
        self.device = params.device

    def get_action(self, state: torch.Tensor, eval: bool = False) -> int:
        """Select an action using epsilon-greedy strategy.
        
        Args:
            state: Current environment state
            eval: Whether to use evaluation mode with fixed epsilon
            
        Returns:
            Selected action index
            
        Raises:
            AssertionError: If state is not a torch.Tensor
        """
        if not isinstance(state, torch.Tensor):
            raise AssertionError("State must be a torch.Tensor")
        
        # Use evaluation epsilon if in eval mode
        epsilon = self.eval_epsilon if eval else self.epsilon
        
        # Random action with probability epsilon
        if random.random() < epsilon:
            return random.randrange(self.action_space.n)
        
        # Greedy action selection
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def update_epsilon(self) -> float:
        """Update epsilon value for exploration.
        
        Decrements epsilon by epsilon_decrement until reaching epsilon_final.
        
        Returns:
            Updated epsilon value
        """
        if self.epsilon > self.epsilon_final:
            self.epsilon = max(
                self.epsilon_final,
                self.epsilon - self.epsilon_decrement
            )
        self.step += 1
        return self.epsilon


class VecActionHandler:
    """Handles action selection for multiple parallel environments.
    
    Implements vectorized epsilon-greedy exploration strategy with support
    for random and evaluation modes.
    
    Attributes:
        policy_net: Neural network for action selection
        action_space: Gymnasium discrete action space
        step: Current step count
        epsilon: Current exploration rate
        epsilon_decrement: Rate of epsilon decay
        epsilon_final: Minimum epsilon value
        eval_epsilon: Fixed epsilon for evaluation
    """
    
    def __init__(
            self,
            params: SimpleNamespace,
            policy_net: torch.nn.Module,
            action_space: gym.spaces.Discrete
    ) -> None:
        """Initialize the vectorized action handler.
        
        Args:
            params: Configuration parameters including epsilon settings
                   and number of environments
            policy_net: Neural network for action selection
            action_space: Gymnasium discrete action space
            
        Raises:
            AssertionError: If number of environments is invalid
        """
        self._validate_n_envs(params.n_envs)
        
        self.policy_net = policy_net
        self.action_space = action_space
        self.step = 0
        self.n_envs = params.n_envs
        self.device = params.device
        
        # Epsilon-greedy parameters
        self.epsilon = params.epsilons.epsilon_start
        self.epsilon_decrement = params.epsilons.epsilon_decrement
        self.epsilon_final = params.epsilons.epsilon_final
        self.eval_epsilon = params.epsilons.eval_epsilon

    @staticmethod
    def _validate_n_envs(n_envs: int) -> None:
        """Validate number of environments.
        
        Args:
            n_envs: Number of parallel environments
            
        Raises:
            AssertionError: If n_envs is invalid
        """
        if n_envs not in [1, 2] and n_envs % 4 != 0:
            raise ValueError("Number of environments must be 1, 2, or divisible by 4")
        if n_envs > 64:
            raise ValueError("Number of environments must be <= 64")

    def get_actions(
            self,
            states: torch.Tensor,
            rand_mode: bool = False,
            eval_mode: bool = False
    ) -> torch.Tensor:
        """Select actions for all environments using vectorized epsilon-greedy.
        
        Args:
            states: Current states from all environments
            rand_mode: Whether to use purely random actions
            eval_mode: Whether to use evaluation mode with fixed epsilon
            
        Returns:
            Tensor of selected actions for each environment
            
        Raises:
            AssertionError: If states is not a torch.Tensor
        """
        if not isinstance(states, torch.Tensor):
            raise AssertionError("States must be a torch.Tensor")
        
        # Return random actions if in random mode
        if rand_mode:
            return torch.randint(0, self.action_space.n, (self.n_envs,))
        
        # Set epsilon based on mode
        epsilon = self.eval_epsilon if eval_mode else self.epsilon
        
        # Get greedy actions from policy network
        with torch.no_grad():
            states = states.to(self.device)
            q_values = self.policy_net(states)
            actions = q_values.max(1)[1]
        
        # Apply epsilon-greedy exploration
        random_actions = torch.randint(
            0, self.action_space.n, (self.n_envs,)
        ).to(self.device)
        random_mask = torch.rand(self.n_envs).to(self.device) < epsilon
        actions[random_mask] = random_actions[random_mask]
        
        # Update epsilon if not in evaluation mode
        if not eval_mode:
            self.update_epsilon()
            
        return actions

    def update_epsilon(self) -> float:
        """Update epsilon value for exploration.
        
        Decrements epsilon by epsilon_decrement * n_envs until reaching
        epsilon_final.
        
        Returns:
            Updated epsilon value
        """
        if self.epsilon > self.epsilon_final:
            self.epsilon = max(
                self.epsilon_final,
                self.epsilon - self.epsilon_decrement * self.n_envs
            )
        self.step += 1
        return self.epsilon
