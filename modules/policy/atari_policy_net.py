import torch
import torch.nn as nn
import torch.nn.functional as F
from .rainbow_helpers import NoisyLinear, DuelingLayer, CategoricalDQNHelper
from typing import Dict

class AtariPolicyNet(nn.Module):
    ''' Flexible architecture that adds to the original DeepMind paper optional:
        - dueling layers
        - categorical_DQN
        - noisy linear layers
        - batch normalization for convolutional layers
        - layer normalization for fully connected layers
    '''
    def __init__(self, 
                 screen_size:           int = 84,
                 fc1_out:               int = 512, 
                 n_actions:             int = 4,
                 categorical_DQN:       bool = False,
                 categorical_params:    Dict = dict(atom_size=51, Vmin=-10, Vmax=10),
                 noisy_linear:          bool = False,
                 dueling:               bool = False,
                 device:                torch.device = torch.device('cuda'),
                 layer_norm:            bool = False,
                 batch_norm:            bool = False,
                 scale_around_zero:     bool = False,
                 ) -> nn.Module:
        super(AtariPolicyNet, self).__init__()
        
        self.screen_size = screen_size
        self.n_actions = n_actions
        self.device = device
        self.layer_norm = layer_norm
        self.scaler_around_zero = scale_around_zero

        # User rainbow options
        self.dueling = dueling
        self.noisy_linear = noisy_linear
        self.categorical_DQN = categorical_DQN

        # If Categorical DQN is enabled, initialize the helper class
        if categorical_DQN:
            self.categ_helper = CategoricalDQNHelper(
                n_actions=n_actions,
                params = categorical_params,
                device=device,
                layer_norm=layer_norm
                )
            self.atom_size = categorical_params['atom_size']
        else:
            self.atom_size = 1  

        # Adjust the kernal and stride for conv layer 1 based on screen size
        assert (screen_size in [42, 84]), "Screen size must be 42 or 84"
        kernel1, stride1 = (8, 4) if screen_size == 84 else (4, 2)
  
        # Define convolutional layers
        conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=kernel1, stride=stride1)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Create batch normalization layers if enabled
        if batch_norm:
            self.convolutions = nn.Sequential(
                conv1, nn.BatchNorm2d(32), nn.ReLU(),
                conv2, nn.BatchNorm2d(64), nn.ReLU(),
                conv3, nn.BatchNorm2d(64), nn.ReLU()
            )
        else:
            self.convolutions = nn.Sequential(
                conv1, nn.ReLU(),
                conv2, nn.ReLU(),
                conv3, nn.ReLU()
            )

        # Compute convolutional output size
        conv_output_size = self._get_conv_output_size()

        # First fully connected layer
        self.fc1 = nn.Linear(in_features=conv_output_size, out_features=fc1_out) 
        if self.layer_norm:
            self.fc1_ln = nn.LayerNorm(fc1_out)

        # For the second fully connected layer, et the LinearLayer to NoisyLinear if noisy linear is enabled
        LinearLayerClass = NoisyLinear if self.noisy_linear else nn.Linear
        # Final layer with dueling or regular
        if self.dueling: 
            self.final = DuelingLayer(
                                      LinearLayer=      LinearLayerClass, 
                                      in_features=      fc1_out, 
                                      out_features=     self.atom_size * self.n_actions,
                                      layer_norm=       self.layer_norm)
        else:
            self.final = LinearLayerClass(in_features=fc1_out, 
                                          out_features= self.atom_size * self.n_actions)
    
    # called only if using NoisyLinear layers
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.float() /255.0
        if self.scaler_around_zero:
            x = x - 0.5
        
        x = self.convolutions(x)
        x = x.view(batch_size, -1)  # Flatten

        # Fully connected layer with optional layer normalization
        if self.layer_norm:
            x = self.fc1_ln(F.relu(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        
        if self.categorical_DQN:
            q_values = self.categ_helper.get_q_values(x, self.final, self.fc1_ln if self.layer_norm else None)
        else:
            q_values = self.final(x)
        
        return q_values

    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, self.screen_size, self.screen_size)
            x = self.convolutions(dummy_input)
            return x.view(1, -1).size(1)
        
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
