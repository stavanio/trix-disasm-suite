"""
SAC (Soft Actor-Critic) implementation stub.

For production: use stable-baselines3.SAC
For custom implementation: expand this stub with full SAC algorithm.

This stub shows the interface needed for the training script.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
    """Simple replay buffer."""
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done, info):
        self.buffer.append((obs, action, reward, next_obs, done, info))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, infos = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
            [info for info in infos]  # Keep as list
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Gaussian policy network."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std
    
    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        if deterministic:
            return mean, None
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Log probability
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Q-network."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc3 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2


class SACPolicy:
    """
    Simplified SAC policy interface for TRiX.
    """
    def __init__(self, observation_dim, action_dim, device="cpu"):
        self.device = device
        self.actor = Actor(observation_dim, action_dim).to(device)
        
    def select_action(self, obs, deterministic=False):
        """Select action (interface for TRiX)."""
        with torch.no_grad():
            action, _ = self.actor.sample(obs, deterministic=deterministic)
        return action.cpu().numpy()[0]


class SACAgent:
    """
    Complete SAC agent with training.
    
    For production: replace with stable_baselines3.SAC
    """
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device="cpu"):
        self.device = device
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Networks
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Stats
        self.actor_loss = 0
        self.critic_loss = 0
        
    def select_action(self, obs, deterministic=False):
        """Select action."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_tensor, deterministic=deterministic)
        return action.cpu().numpy()[0]
    
    def store_transition(self, obs, action, reward, next_obs, done, info):
        """Store transition in replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, done, info)
    
    def update(self, batch_size=256):
        """Update SAC networks."""
        if len(self.replay_buffer) < batch_size:
            return {"actor_loss": 0, "critic_loss": 0}
        
        # Sample batch
        obs, actions, rewards, next_obs, dones, _ = self.replay_buffer.sample(batch_size)
        
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()
        
        return {
            "actor_loss": self.actor_loss,
            "critic_loss": self.critic_loss
        }
    
    def state_dict(self):
        """Get state dict for saving."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict()
        }
