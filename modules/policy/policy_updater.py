import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict
from types import SimpleNamespace
from modules.replay_buffers.replay_buffer import ReplayBuffer

class PolicyUpdater:
    def __init__(self,
                 memory:                ReplayBuffer = None,
                 policy_net:            nn.Module    = None,
                 target_net:            nn.Module    = None,
                 optimizer:             torch.optim  = None,
                 params:                SimpleNamespace = None):
        self.p = params  
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        
    def _do_forward_pass(self):
        if self.p.noisy_linear:
            with torch.no_grad():
                self.policy_net.reset_noise()
                self.target_net.reset_noise()
        # Sample the batch
        (s_batch, a_batch, r_batch, ns_batch, d_batch) = self.memory.sample()

        # Calculate the Q-Values
        Q = self.policy_net(s_batch).gather(1, a_batch)

        # Calculate the target Q-Values using DQN or Double DQN
        if self.p.doubleQ:
            next_actions = self.policy_net(ns_batch).argmax(1, keepdim=True)
            with torch.no_grad():
                next_Q = self.target_net(ns_batch).gather(1, next_actions)
        else:
            with torch.no_grad():
                next_Q = self.target_net(ns_batch).max(1)[0].detach().unsqueeze(1)

        expected_next_Q = (next_Q * self.p.gamma) * (1 - d_batch.view(-1,1)) + r_batch.view(-1,1)
        loss = F.smooth_l1_loss(Q, expected_next_Q)
        return loss
        
    def _do_backward_pass(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.p.gradient_clamping:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update(self):
        self.policy_net.train()
        # If single threaded, perform the forward pass and update the policy network n_batch_updates times
        if self.p.group_training_losses == True:  
            l = 0.0
            for i in range(self.p.n_batch_updates):
                loss = self._do_forward_pass()
                self._do_backward_pass(loss)
                l += loss.item()
            loss = l / self.p.n_batch_updates
            return loss

        # Else: accumulates loss over n_batch_updates before applying the average loss to the backward pass
      
        loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.p.n_batch_updates):
            loss += self._do_forward_pass()
        # Average the losses before backward pass
        loss = loss / self.p.n_batch_updates
        self._do_backward_pass(loss)
        return loss.item()
