"""
Batch-Constrained Q-Learning (BCQ) for offline RL.

Prevents out-of-distribution actions by learning behavioral policy
and constraining Q-learning to actions likely under that policy.

Reference:
    Fujimoto et al. (2019): Off-Policy Deep RL without Exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class Actor(nn.Module):
    """Behavioral policy network."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.sigmoid(self.l3(a))


class Critic(nn.Module):
    """Twin Q-networks."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


class VAE(nn.Module):
    """Variational Autoencoder for action modeling."""
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, max_action: float):
        super().__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 256)
        self.e2 = nn.Linear(256, 256)
        
        self.mean = nn.Linear(256, latent_dim)
        self.log_std = nn.Linear(256, latent_dim)
        
        self.d1 = nn.Linear(state_dim + latent_dim, 256)
        self.d2 = nn.Linear(256, 256)
        self.d3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        self.latent_dim = latent_dim
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)
        
        return u, mean, std
    
    def decode(self, state: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(state.device).clamp(-0.5, 0.5)
        
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.sigmoid(self.d3(a))


class BCQ:
    """Batch-Constrained Q-Learning agent."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 device: str = 'cpu',
                 discount: float = 0.99,
                 tau: float = 0.005,
                 lam: float = 0.75,
                 phi: float = 0.05):
        """
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_action: Maximum action value
            device: torch device
            discount: Discount factor γ
            tau: Target network update rate
            lam: Weighting for Q-value selection
            phi: VAE perturbation threshold
        """
        self.device = torch.device(device)
        self.discount = discount
        self.tau = tau
        self.lam = lam
        self.phi = phi
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.vae = VAE(state_dim, action_dim, 2 * action_dim, max_action).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using behavioral policy + perturbation."""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)
            return action.cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size: int = 256) -> dict:
        """Train BCQ on batch from replay buffer."""
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Train VAE
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        # Train Critic
        with torch.no_grad():
            # Sample actions from VAE
            next_action = self.vae.decode(next_state)
            
            # Compute target Q
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = self.lam * torch.min(target_q1, target_q2) + (1 - self.lam) * torch.max(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q
        
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train Actor
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state)
        
        # Q-value of perturbed actions
        actor_loss = -self.critic.q1(state, perturbed_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'vae_loss': vae_loss.item()
        }
    
    def save(self, filename: str):
        """Save model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'vae': self.vae.state_dict()
        }, filename)
    
    def load(self, filename: str):
        """Load model."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.vae.load_state_dict(checkpoint['vae'])
