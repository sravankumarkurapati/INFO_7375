#!/usr/bin/env python3
"""
DQN Agent
Combines network, replay buffer, and training logic
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn_network import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Learning Agent
    Implements DQN algorithm with experience replay and target network
    """
    
    def __init__(
        self,
        state_shape=(4, 84, 84),
        num_actions=18,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=32,
        target_update_freq=1000,
        device=None
    ):
        """
        Initialize DQN Agent
        
        Args:
            state_shape (tuple): Shape of input state
            num_actions (int): Number of possible actions
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Minimum exploration rate
            epsilon_decay (float): Epsilon decay rate per episode
            buffer_capacity (int): Replay buffer capacity
            batch_size (int): Training batch size
            target_update_freq (int): Steps between target network updates
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = DQN(state_shape[0], num_actions).to(self.device)
        self.target_net = DQN(state_shape[0], num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Training stats
        self.training_step = 0
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state (np.array): Current state
            
        Returns:
            int: Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # Convert to tensor and move to device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        Perform one training step
        
        Returns:
            float: Loss value (or None if not enough samples)
        """
        # Check if enough samples in buffer
        if not self.memory.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.training_step += 1
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save agent state
        
        Args:
            filepath (str): Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load agent state
        
        Args:
            filepath (str): Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {filepath}")


def test_dqn_agent():
    """Test the DQN agent"""
    print("=" * 60)
    print("TESTING DQN AGENT")
    print("=" * 60)
    
    # Create agent
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        num_actions=18,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=32
    )
    
    print(f"\n1. Agent created successfully")
    print(f"   Device: {agent.device}")
    print(f"   Initial epsilon: {agent.epsilon}")
    print(f"   Batch size: {agent.batch_size}")
    
    # Test action selection
    print(f"\n2. Testing action selection...")
    dummy_state = np.random.rand(4, 84, 84).astype(np.float32)
    action = agent.select_action(dummy_state)
    print(f"   Selected action: {action}")
    
    # Add some transitions
    print(f"\n3. Adding transitions to replay buffer...")
    for i in range(100):
        state = np.random.rand(4, 84, 84).astype(np.float32)
        action = np.random.randint(18)
        reward = np.random.rand()
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        done = False
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(agent.memory)}")
    
    # Test training
    print(f"\n4. Testing training step...")
    loss = agent.train()
    if loss is not None:
        print(f"   Training loss: {loss:.6f}")
    else:
        print(f"   Not enough samples to train")
    
    # Test epsilon decay
    print(f"\n5. Testing epsilon decay...")
    print(f"   Epsilon before decay: {agent.epsilon:.4f}")
    agent.decay_epsilon()
    print(f"   Epsilon after decay: {agent.epsilon:.4f}")
    
    print("\n✓ DQN agent test successful!")
    print("=" * 60)


if __name__ == "__main__":
    test_dqn_agent()