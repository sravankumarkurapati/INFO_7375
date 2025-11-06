#!/usr/bin/env python3
"""
DQN Agent with Boltzmann (Softmax) Exploration
Alternative to epsilon-greedy policy
"""
import sys
sys.path.append('../phase2_implementation')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn_network import DQN
from replay_buffer import ReplayBuffer


class BoltzmannDQNAgent:
    """
    DQN Agent using Boltzmann (Softmax) exploration instead of epsilon-greedy
    
    Softmax action selection:
    P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
    
    where τ (tau/temperature) controls exploration:
    - High τ: more random (more exploration)
    - Low τ: more greedy (more exploitation)
    """
    
    def __init__(
        self,
        state_shape=(4, 84, 84),
        num_actions=18,
        learning_rate=0.00025,
        gamma=0.99,
        temperature_start=1.0,
        temperature_end=0.1,
        temperature_decay=0.995,
        buffer_capacity=100000,
        batch_size=32,
        target_update_freq=1000,
        device=None
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.temperature = temperature_start
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Exploration: Boltzmann (Temperature: {temperature_start} → {temperature_end})")
        
        # Networks
        self.policy_net = DQN(state_shape[0], num_actions).to(self.device)
        self.target_net = DQN(state_shape[0], num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        self.training_step = 0
    
    def select_action(self, state):
        """
        Select action using Boltzmann (softmax) exploration
        
        Args:
            state (np.array): Current state
            
        Returns:
            int: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Apply softmax with temperature
        # P(a) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        
        # Sample action according to probabilities
        action = np.random.choice(self.num_actions, p=probabilities)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent (same as epsilon-greedy DQN)"""
        if not self.memory.is_ready(self.batch_size):
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.training_step += 1
        
        return loss.item()
    
    def decay_temperature(self):
        """Decay temperature after each episode"""
        self.temperature = max(self.temperature_end, 
                              self.temperature * self.temperature_decay)
    
    def save(self, filepath):
        """Save agent"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'temperature': self.temperature,
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath):
        """Load agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.temperature = checkpoint['temperature']
        self.training_step = checkpoint['training_step']