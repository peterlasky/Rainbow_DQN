import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from types import SimpleNamespace
from modules.replay_buffers.replay_buffer import ReplayBuffer


class PolicyUpdater:
    """Handles policy network updates including forward and backward passes.
    
    This class manages the policy network training process, including sampling from
    replay buffer, computing Q-values, and updating network weights through
    backpropagation.
    """
    
    def __init__(
            self,
            memory: ReplayBuffer,
            policy_net: nn.Module,
            target_net: nn.Module,
            optimizer: torch.optim.Optimizer,
            params: SimpleNamespace
    ) -> None:
        """Initialize the PolicyUpdater.
        
        Args:
            memory: Buffer containing experience replay samples
            policy_net: The main policy network being trained
            target_net: Target network for stable Q-value estimation
            optimizer: Optimizer for updating policy network weights
            params: Configuration parameters for training
        """
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.p = params
        self.device = next(policy_net.parameters()).device
        self.update_count = 0
        
    def _do_forward_pass(self) -> torch.Tensor:
        """Perform forward pass through policy and target networks.
        
        Samples a batch from replay memory, computes current Q-values and target
        Q-values (using Double DQN if enabled), and returns the loss.
        
        Returns:
            Computed loss between current and target Q-values
        """
        self.update_count+=1
        # Reset noise for NoisyLinear layers if enabled
        if self.p.noisy_linear:
            with torch.no_grad():
                self.policy_net.reset_noise()
                self.target_net.reset_noise()
                
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Calculate current Q-values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Calculate target Q-values using Double DQN or regular DQN
        with torch.no_grad():
            if self.p.doubleQ:
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Compute expected Q-values
        expected_q = rewards.view(-1, 1) + (self.p.gamma * next_q * (1 - dones.view(-1, 1)))
        
        # Compute Huber loss
        return F.smooth_l1_loss(current_q, expected_q)
        
    def _do_backward_pass(self, loss: torch.Tensor) -> None:
        """Perform backward pass to update policy network weights.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.p.gradient_clamping:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()

    def update(self) -> float:
        """Update policy network weights.
        
        Performs either single or grouped updates based on configuration.
        
        Returns:
            Average loss value over all updates
        """
        self.policy_net.train()
        
        # Group training losses: accumulate losses before update
        if self.p.group_training_losses:
            accumulated_loss = torch.tensor(0.0, device=self.device)
            for _ in range(self.p.n_batch_updates_adjusted):
                accumulated_loss += self._do_forward_pass()
            
            # Average losses and perform single backward pass
            average_loss = accumulated_loss / self.p.n_batch_updates_adjusted
            self._do_backward_pass(average_loss)
            return average_loss.item()
        
        # Single-threaded mode: perform updates sequentially
        total_loss = 0.0
        for _ in range(self.p.n_batch_updates_adjusted):
            loss = self._do_forward_pass()
            self._do_backward_pass(loss)
            total_loss += loss.item()
        return total_loss / self.p.n_batch_updates_adjusted
