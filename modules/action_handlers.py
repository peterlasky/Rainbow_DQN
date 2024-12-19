import numpy as np
import torch
from typing import Dict
from types import SimpleNamespace
import gymnasium as gym
import random

__all__ = ['ActionHandler', 'VecActionHandler']

class ActionHandler:
    def __init__(self,
                 params:        SimpleNamespace,
                 policy_net:    torch.nn.Module,
                 action_space:  gym.spaces.Discrete):
        
        self.p = params  # Already a SimpleNamespace
        self.policy_net = policy_net
        self.action_space = action_space
        self.step = 0
        self.epsilon = self.p.epsilons.epsilon_start
        self.epsilon_decrement = self.p.epsilons.epsilon_decrement
        self.epsilon_final = self.p.epsilons.epsilon_final

    def get_action(self, state: torch.Tensor, eval: bool = False) -> int:
        ''' Get an action from the policy network with epsilon-greedy exploration '''
        assert isinstance(state, torch.Tensor)
        
        # Epsilon for evaluation
        epsilon = self.p.epsilons.eval_epsilon if eval else self.epsilon

        # Random action with probability epsilon
        if random.random() < epsilon:
            return random.randrange(self.action_space.n)
        
        # Otherwise, get the action from the policy network
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.p.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def update_epsilon(self):
        ''' Update epsilon value for epsilon-greedy exploration '''
        if self.epsilon > self.epsilon_final:
            self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decrement)
        self.step += 1
        return self.epsilon


class VecActionHandler:
    """
    Vectorized version of ActionHandler.
    Handles action selection for multiple environments simultaneously.
    """

    def __init__(self,
                 params:        SimpleNamespace,
                 policy_net:    torch.nn.Module,
                 action_space:  gym.spaces.Discrete):

        self.p = params  # Already a SimpleNamespace
        
        assert self.p.n_envs in [1,2] or self.p.n_envs % 4 == 0, "Number of environments must be in [1,2] or divisible by 4"
        assert self.p.n_envs <= 64, "Number of environments must be <= 64"

        self.policy_net = policy_net
        self.action_space = action_space
        self.step = 0
        self.epsilon = self.p.epsilons.epsilon_start
        self.epsilon_decrement = self.p.epsilons.epsilon_decrement
        self.epsilon_final = self.p.epsilons.epsilon_final
        self.eval_epsilon = self.p.epsilons.eval_epsilon

    def get_actions(self, states: torch.Tensor, rand_mode: bool = False, eval_mode: bool = False) -> torch.Tensor:
        ''' Get actions from the policy network with epsilon-greedy exploration '''
        assert isinstance(states, torch.Tensor)
        # If in random mode, return random actions
        if rand_mode:
            return torch.randint(0, self.action_space.n, (self.p.n_envs,))
        
        # Set epsilon based on mode
        epsilon = self.eval_epsilon if eval_mode else self.epsilon
        
        # Get actions from the policy network
        with torch.no_grad():
            states = states.to(self.p.device)
            q_values = self.policy_net(states)
            actions = q_values.max(1)[1]

        # Random actions with probability epsilon
        random_actions = torch.randint(0, self.action_space.n, (self.p.n_envs,)).to(self.p.device)
        random_mask = torch.rand(self.p.n_envs).to(self.p.device) < epsilon
        actions[random_mask] = random_actions[random_mask]

        # Update epsilon if not in eval mode
        if not eval_mode:
            self.update_epsilon()

        return actions

    def update_epsilon(self):
        ''' Update epsilon value for epsilon-greedy exploration '''
        if self.epsilon > self.epsilon_final:
            self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decrement * self.p.n_envs)
        self.step += 1
        return self.epsilon
