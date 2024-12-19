"""Helper classes for Rainbow DQN implementation.

This module provides the building blocks for implementing Rainbow DQN features:
- Categorical DQN (C51) for distributional RL
- NoisyNet for exploration
- Dueling architecture for better value estimation
"""

import math
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DuelingLayer', 'NoisyLinear', 'CategoricalDQNHelper']


class CategoricalDQNHelper:
    """Helper class for Categorical DQN (C51) architecture.
    
    Handles the distribution calculations and Q-value computations for C51.
    Does not inherit from nn.Module as it has no trainable parameters.
    """
    
    def __init__(
            self,
            n_actions: int,
            categorical_params: SimpleNamespace,
            device: torch.device,
            layer_norm: bool = False
    ) -> None:
        """Initialize the Categorical DQN helper.
        
        Args:
            n_actions: Number of possible actions
            categorical_params: Parameters for categorical distribution
            device: Device to place tensors on
            layer_norm: Whether to use layer normalization
        """
        self.n_actions = n_actions
        self.atom_size = categorical_params.atom_size
        self.support = torch.linspace(
            categorical_params.Vmin,
            categorical_params.Vmax,
            self.atom_size
        ).to(device)
        self.layer_norm = layer_norm

    def get_q_values(
            self,
            features: torch.Tensor,
            final_layer: nn.Module,
            layer_norm_layer: nn.Module = None
    ) -> torch.Tensor:
        """Compute Q-values from categorical distribution.
        
        Args:
            features: Input features from previous network layers
            final_layer: Final network layer for computing logits
            layer_norm_layer: Optional layer normalization
            
        Returns:
            Computed Q-values for each action
        """
        # Apply final layer and optional normalization
        logits = final_layer(features)
        if self.layer_norm and layer_norm_layer is not None:
            logits = layer_norm_layer(logits)
        
        # Compute categorical distribution
        q_atoms = logits.view(-1, self.n_actions, self.atom_size)
        distribution = F.softmax(q_atoms, dim=-1)
        distribution = distribution.clamp(min=1e-3)  # Prevent log(0)
        
        # Calculate expected values
        return torch.sum(distribution * self.support, dim=2)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration in DQN.
    
    Implementation of NoisyNet (https://arxiv.org/abs/1706.10295) using
    factorized Gaussian noise for efficient exploration.
    """
    
    def __init__(
            self,
            in_features: int,
            out_features: int,
            sigma_init: float = 0.017
    ) -> None:
        """Initialize NoisyLinear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            sigma_init: Initial value for noise standard deviation
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise parameters
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Reset learnable parameters to initial values."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def reset_noise(self) -> None:
        """Reset the factorized Gaussian noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise if in training mode.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        weight = self.weight_mu
        bias = self.bias_mu
        
        if self.training:
            weight = weight + self.weight_sigma * self.weight_epsilon
            bias = bias + self.bias_sigma * self.bias_epsilon
            
        return F.linear(x, weight, bias)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise.
        
        Args:
            size: Size of noise tensor to generate
            
        Returns:
            Scaled noise tensor
        """
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()


class DuelingLayer(nn.Module):
    """Dueling network architecture for better value estimation.
    
    Implements the dueling architecture from https://arxiv.org/abs/1511.06581
    to separate state value and advantage streams.
    """
    
    def __init__(
            self,
            LinearLayer: nn.Module = nn.Linear,
            in_features: int = None,
            out_features: int = None,
            layer_norm: bool = False
    ) -> None:
        """Initialize dueling network layers.
        
        Args:
            LinearLayer: Linear layer class to use (can be NoisyLinear)
            in_features: Size of input features
            out_features: Size of output features
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        if any(x is None for x in [LinearLayer, in_features, out_features]):
            raise ValueError("LinearLayer, in_features, and out_features must be provided")
        
        # Value and advantage streams
        self.value_fc = LinearLayer(in_features=in_features, out_features=1)
        self.advantage_fc = LinearLayer(in_features=in_features, out_features=out_features)
        
        # Optional layer normalization
        self.layer_norm = layer_norm
        if layer_norm:
            self.value_ln = nn.LayerNorm(1)
            self.advantage_ln = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine value and advantage streams.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values tensor
        """
        # Compute value and advantage
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Apply layer normalization if enabled
        if self.layer_norm:
            value = self.value_ln(value)
            advantage = self.advantage_ln(advantage)
        
        # Combine streams with advantage normalization
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
