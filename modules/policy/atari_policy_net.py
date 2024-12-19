import torch
import torch.nn as nn
import torch.nn.functional as F
from .rainbow_helpers import NoisyLinear, DuelingLayer, CategoricalDQNHelper
from types import SimpleNamespace


class AtariPolicyNet(nn.Module):
    """Flexible architecture that adds to the original DeepMind paper optional:
        - dueling layers
        - categorical_DQN
        - noisy linear layers
        - batch normalization for convolutional layers
        - layer normalization for fully connected layers
    """

    def __init__(self, p: SimpleNamespace):
        """Initialize the Atari policy network.
        
        Args:
            p: SimpleNamespace containing network parameters
        """
        super(AtariPolicyNet, self).__init__()
        # Copy all attributes from p to self
        for k, v in vars(p).items():
            setattr(self, k, v)
        
        # If Categorical DQN is enabled, initialize the helper class
        if self.categorical_DQN:
            self.categ_helper = CategoricalDQNHelper(
                n_actions=p.n_actions,
                categorical_params=p.categorical_params,
                device=p.device,
                layer_norm=p.layer_norm
            )
            self.atom_size = p.categorical_params.atom_size
        else:
            self.atom_size = 1  

        # Adjust the kernel and stride for conv layer 1 based on screen size
        kernel1, stride1 = (8, 4) if self.screen_size == 84 else (4, 2)
  
        # Define convolutional layers
        conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=kernel1, stride=stride1)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Create convolutional sequence with optional batch normalization
        if self.batch_norm:
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

        # First fully connected layer with optional layer normalization
        self.fc1 = nn.Linear(in_features=conv_output_size, out_features=self.fc1_out) 
        if self.layer_norm:
            self.fc1_ln = nn.LayerNorm(self.fc1_out)

        # Set the LinearLayer to NoisyLinear if noisy linear is enabled
        LinearLayerClass = NoisyLinear if self.noisy_linear else nn.Linear
        
        # Final layer with optional dueling architecture
        if self.dueling: 
            self.final = DuelingLayer(
                LinearLayer=LinearLayerClass, 
                in_features=self.fc1_out, 
                out_features=self.atom_size * self.n_actions,
                layer_norm=self.layer_norm
            )
        else:
            self.final = LinearLayerClass(
                in_features=self.fc1_out, 
                out_features=self.atom_size * self.n_actions
            )
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4, screen_size, screen_size)
            
        Returns:
            Q-values tensor of shape (batch_size, n_actions) or 
            (batch_size, n_actions * atom_size) if using categorical DQN
        """
        batch_size = x.size(0)
        x = x.float() / 255.0
        if self.scale_around_zero:
            x = x - 0.5
        
        # Convolutional layers
        conv_out = self.convolutions(x)
        flat_features = conv_out.view(batch_size, -1)  # Flatten

        # Fully connected layer with optional layer normalization
        if self.layer_norm:
            fc_out = self.fc1_ln(F.relu(self.fc1(flat_features)))
        else:
            fc_out = F.relu(self.fc1(flat_features))
        
        # Final layer with optional categorical DQN
        if self.categorical_DQN:
            q_values = self.categ_helper.get_q_values(
                fc_out, 
                self.final, 
                self.fc1_ln if self.layer_norm else None
            )
        else:
            q_values = self.final(fc_out)
        
        return q_values

    def _get_conv_output_size(self) -> int:
        """Calculate the output size of the convolutional layers.
        
        Returns:
            Number of features output by the convolutional layers
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, self.screen_size, self.screen_size)
            conv_out = self.convolutions(dummy_input)
            return conv_out.view(1, -1).size(1)
        
    @property
    def num_parameters(self) -> int:
        """Get the number of trainable parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
