"""Replay buffer for offline RL."""

import numpy as np
import torch


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6), device: str = 'cpu'):
        """
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_size: Maximum buffer size
            device: torch device
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        self.device = torch.device(device)
    
    def add(self, state, action, next_state, reward, done):
        """Add transition to buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int):
        """Sample batch of transitions."""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def save(self, filename: str):
        """Save buffer to disk."""
        np.savez(
            filename,
            state=self.state[:self.size],
            action=self.action[:self.size],
            next_state=self.next_state[:self.size],
            reward=self.reward[:self.size],
            not_done=self.not_done[:self.size]
        )
    
    def load(self, filename: str):
        """Load buffer from disk."""
        data = np.load(filename)
        self.size = data['state'].shape[0]
        self.ptr = self.size
        
        self.state[:self.size] = data['state']
        self.action[:self.size] = data['action']
        self.next_state[:self.size] = data['next_state']
        self.reward[:self.size] = data['reward']
        self.not_done[:self.size] = data['not_done']
