import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
import math

__all__ = ['DuelingLayer', 'NoisyLinear', 'CategoricalDQNHelper']

class CategoricalDQNHelper:
    '''
    Does not inherent from nn.Module and has no forward pass.

    Helper class for the Categorical DQN architecture.
    '''
    def __init__(self, 
                 n_actions: int,
                 params: Dict,
                 device: torch.device,
                 layer_norm: bool =False):
        self.n_actions = n_actions
        self.atom_size = params.get('atom_size', 51)
        Vmin = params.get('Vmin', -10)
        Vmax = params.get('Vmax', 10)
        
        self.support = torch.linspace(Vmin, Vmax, self.atom_size).to(device)
        self.layer_norm = layer_norm

    def get_q_values(self, x: torch.Tensor,
                     final_layer: nn.Module,
                     layer_norm_layer: nn.Module = None) -> torch.Tensor:
        # Apply the final fully connected layer
        x = final_layer(x)
        
        # Apply layer normalization if enabled
        if self.layer_norm and layer_norm_layer is not None:
            x = layer_norm_layer(x)
        
        # Reshape and compute distribution
        q_atoms = x.view(-1, self.n_actions, self.atom_size)
        distribution = F.softmax(q_atoms, dim=-1)
        distribution = distribution.clamp(min=1e-3)  # Avoiding log(0)
        
        # Calculate Q-values by weighting atoms by their support values
        q_values = torch.sum(distribution * self.support, dim=2)
        return q_values
# --- end of CategoricalDQNHelper class

''' This class contains the NoisyLinear layer for the NoisyNet architecture.
    Make sure reset_noise is called during the training loop.
'''
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
# --- end of NoisyLinear class

class DuelingLayer(nn.Module):
    '''
    Helper class adding the dueling architecture. Includes flexibility to use a noisy linear layer.
    '''
    def __init__(self, 
                 LinearLayer: nn.Module = nn.Linear,         # Default to nn.Linear but can be overridden with other fully-connected layers
                 in_features: int = None, 
                 out_features: int = None, 
                 layer_norm: bool= False):
        super(DuelingLayer, self).__init__()
        if any([LinearLayer, in_features, out_features]) is None:
           raise ValueError("LinearLayer, in_features, and out_features must be provided")
        
        self.value_fc = LinearLayer(in_features=in_features, out_features=1)
        self.advantage_fc = LinearLayer(in_features=in_features, out_features=out_features)

        self.layer_norm = layer_norm
        if layer_norm:
            self.value_ln = nn.LayerNorm(1)
            self.advantage_ln = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)

        if self.layer_norm:
            value = self.value_ln(value)
            advantage = self.advantage_ln(advantage)

        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
# --- end of DuelingLayer class

