#!/usr/bin/env python3
"""
Experience Replay Buffer for DQN
Stores transitions and samples random minibatches for training
"""
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN
    Stores (state, action, reward, next_state, done) transitions
    """
    
    def __init__(self, capacity=100000):
        """
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode terminated
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        # Random sample
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= batch_size


def test_replay_buffer():
    """Test the replay buffer"""
    print("=" * 60)
    print("TESTING REPLAY BUFFER")
    print("=" * 60)
    
    # Create buffer
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some dummy transitions
    print("\n1. Adding transitions...")
    for i in range(100):
        state = np.random.rand(84, 84, 4).astype(np.float32)
        action = np.random.randint(0, 18)
        reward = np.random.rand()
        next_state = np.random.rand(84, 84, 4).astype(np.float32)
        done = np.random.choice([True, False])
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Buffer capacity: {buffer.capacity}")
    
    # Test sampling
    print("\n2. Sampling batch...")
    batch_size = 32
    
    if buffer.is_ready(batch_size):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        print(f"   Batch size: {batch_size}")
        print(f"   States shape: {states.shape}")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Next states shape: {next_states.shape}")
        print(f"   Dones shape: {dones.shape}")
        
        # Calculate memory usage
        total_memory = (states.nbytes + next_states.nbytes) / (1024 ** 2)
        print(f"\n3. Memory usage for batch: {total_memory:.2f} MB")
        
        # Estimate full buffer memory
        full_buffer_memory = (total_memory / batch_size) * buffer.capacity
        print(f"   Estimated full buffer (1000 transitions): {full_buffer_memory:.2f} MB")
    
    print("\n✓ Replay buffer test successful!")
    print("=" * 60)


if __name__ == "__main__":
    test_replay_buffer()