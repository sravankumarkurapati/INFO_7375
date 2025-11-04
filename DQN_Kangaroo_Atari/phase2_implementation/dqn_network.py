#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Architecture
CNN-based network for processing Atari frames
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """
    Deep Q-Network with Convolutional layers
    Architecture based on DeepMind's Nature paper (2015)
    """
    
    def __init__(self, input_channels=4, num_actions=18):
        """
        Args:
            input_channels (int): Number of stacked frames
            num_actions (int): Number of possible actions
        """
        super(DQN, self).__init__()
        
        self.num_actions = num_actions
        
        # Convolutional layers
        # Input: (batch, 4, 84, 84)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        # Output: (batch, 32, 20, 20)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Output: (batch, 64, 9, 9)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Output: (batch, 64, 7, 7)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        """
        Forward pass through network
        
        Args:
            x (torch.Tensor): Input state (batch, 4, 84, 84)
            
        Returns:
            torch.Tensor: Q-values for each action (batch, num_actions)
        """
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def act(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state (np.array): Current state
            epsilon (float): Exploration rate
            
        Returns:
            int: Selected action
        """
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        
        # Return action with highest Q-value
        return q_values.argmax().item()


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_dqn_network():
    """Test the DQN network"""
    print("=" * 60)
    print("TESTING DQN NETWORK")
    print("=" * 60)
    
    # Create network
    dqn = DQN(input_channels=4, num_actions=18)
    
    print(f"\n1. Network Architecture:")
    print(dqn)
    
    print(f"\n2. Total trainable parameters: {count_parameters(dqn):,}")
    
    # Test forward pass
    print(f"\n3. Testing forward pass...")
    batch_size = 32
    dummy_input = torch.randn(batch_size, 4, 84, 84)
    
    output = dqn(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output (Q-values for first sample): {output[0].detach().numpy()[:5]}...")
    
    # Test action selection
    print(f"\n4. Testing action selection...")
    single_state = np.random.rand(4, 84, 84).astype(np.float32)
    
    # Greedy action (epsilon=0)
    action_greedy = dqn.act(single_state, epsilon=0.0)
    print(f"   Greedy action (ε=0): {action_greedy}")
    
    # Exploration action (epsilon=1)
    action_explore = dqn.act(single_state, epsilon=1.0)
    print(f"   Random action (ε=1): {action_explore}")
    
    # Epsilon-greedy (epsilon=0.1)
    actions = [dqn.act(single_state, epsilon=0.1) for _ in range(10)]
    print(f"   Actions with ε=0.1: {actions}")
    
    print("\n✓ DQN network test successful!")
    print("=" * 60)


if __name__ == "__main__":
    test_dqn_network()