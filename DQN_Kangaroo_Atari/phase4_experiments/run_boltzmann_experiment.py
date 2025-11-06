#!/usr/bin/env python3
"""
Experiment 9: Boltzmann (Softmax) Exploration for Kangaroo
Alternative exploration policy to epsilon-greedy

Author: Sravan Kumar Kurapati
Course: INFO 7375 - Fine-Tuning Large Language Models
Date: November 2025
"""

import sys
import os

import gymnasium as gym
import ale_py  # Register ALE environments
gym.register_envs(ale_py)  # Explicitly register ALE

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from collections import deque
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# DQN Network Architecture
# ============================================================================

class DQN(nn.Module):
    """Deep Q-Network for Atari games"""
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def is_ready(self, batch_size):
        return len(self.buffer) >= batch_size


# ============================================================================
# Boltzmann DQN Agent
# ============================================================================

class BoltzmannDQNAgent:
    """
    DQN Agent using Boltzmann (Softmax) exploration
    
    Softmax action selection:
    P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
    """
    
    def __init__(
        self,
        state_shape=(4, 84, 84),
        num_actions=18,
        learning_rate=0.00025,
        gamma=0.8,
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
        """Select action using Boltzmann exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        # Apply softmax with temperature
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        
        # Sample action according to probabilities
        action = np.random.choice(self.num_actions, p=probabilities)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent"""
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


# ============================================================================
# Frame Processing
# ============================================================================


class FrameProcessor:
    """Preprocess frames for DQN"""
    def __init__(self, height=84, width=84):
        self.height = height
        self.width = width
    
    def process(self, frame):
        """Convert frame to grayscale and resize"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return resized.astype(np.float32) / 255.0


def stack_frames(frame_buffer, frame, is_new):
    """Stack frames for temporal information"""
    if is_new:
        frame_buffer.clear()
        for _ in range(4):
            frame_buffer.append(frame)
    else:
        frame_buffer.append(frame)
    
    return np.array(frame_buffer)


def run_boltzmann_experiment():
    """Run Experiment 9: Boltzmann Exploration"""
    
    # Experiment configuration
    config = {
        'name': 'exp9_boltzmann',
        'description': 'Boltzmann (softmax) exploration with temperature decay instead of epsilon-greedy',
        'episodes': 1000,
        'max_steps': 500,
        'learning_rate': 0.00025,
        'gamma': 0.8,
        'temperature_start': 1.0,
        'temperature_end': 0.1,
        'temperature_decay': 0.995,
        'buffer_capacity': 100000,
        'batch_size': 32,
        'target_update_freq': 1000,
        'train_freq': 4,
        'save_freq': 500,
        'log_freq': 10
    }
    
    # Create results directory
    results_dir = f'experiment_results/{config["name"]}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Print experiment info
    print("\n" + "="*70)
    print(f"EXPERIMENT: {config['name']}")
    print("="*70)
    print(f"Description: {config['description']}")
    print(f"Episodes: {config['episodes']}")
    print(f"Max Steps: {config['max_steps']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Gamma: {config['gamma']}")
    print(f"Temperature: {config['temperature_start']} → {config['temperature_end']}")
    print(f"Temperature Decay: {config['temperature_decay']}")
    print("="*70)
    print()
    
    # Initialize environment
    env = gym.make('ALE/Kangaroo-v5')
    frame_processor = FrameProcessor()
    
    # Initialize agent
    agent = BoltzmannDQNAgent(
        state_shape=(4, 84, 84),
        num_actions=env.action_space.n,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        temperature_start=config['temperature_start'],
        temperature_end=config['temperature_end'],
        temperature_decay=config['temperature_decay'],
        buffer_capacity=config['buffer_capacity'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq']
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    temperature_history = []
    
    # Training loop
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Starting experiment: {config['name']}")
    print(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    global_step = 0
    max_reward = 0
    
    for episode in range(1, config['episodes'] + 1):
        # Reset environment
        state, _ = env.reset()
        state = frame_processor.process(state)
        frame_buffer = deque(maxlen=4)
        stacked_state = stack_frames(frame_buffer, state, is_new=True)
        
        episode_reward = 0
        episode_loss = []
        
        for step in range(config['max_steps']):
            # Select action using Boltzmann exploration
            action = agent.select_action(stacked_state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = frame_processor.process(next_state)
            next_stacked_state = stack_frames(frame_buffer, next_state, is_new=False)
            
            # Store transition
            agent.store_transition(
                stacked_state,
                action,
                reward,
                next_stacked_state,
                float(done)
            )
            
            # Train agent
            if global_step % config['train_freq'] == 0:
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)
            
            episode_reward += reward
            stacked_state = next_stacked_state
            global_step += 1
            
            if done:
                break
        
        # Decay temperature after episode
        agent.decay_temperature()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        temperature_history.append(agent.temperature)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        else:
            losses.append(0)
        
        # Update max reward
        if episode_reward > max_reward:
            max_reward = episode_reward
        
        # Logging
        if episode % config['log_freq'] == 0:
            elapsed_time = (time.time() - start_time) / 60  # minutes
            episodes_remaining = config['episodes'] - episode
            time_per_episode = elapsed_time / episode
            eta = episodes_remaining * time_per_episode
            
            print(f"Ep {episode}/{config['episodes']} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Max: {max_reward:.1f} | "
                  f"Steps: {step + 1} | "
                  f"τ: {agent.temperature:.3f} | "
                  f"Time: {elapsed_time:.0f}m | "
                  f"ETA: {eta:.0f}m")
        
        # Save model checkpoint
        if episode % config['save_freq'] == 0:
            model_path = f"{results_dir}/model_ep{episode}.pth"
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = f"{results_dir}/model_ep{config['episodes']}.pth"
    agent.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    final_model_path2 = f"{results_dir}/model_epfinal.pth"
    agent.save(final_model_path2)
    print(f"Model saved to {final_model_path2}")
    
    # Calculate statistics
    total_time = (time.time() - start_time) / 60
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # Save experiment statistics
    stats = {
        'experiment_name': config['name'],
        'description': config['description'],
        'config': {
            'total_episodes': config['episodes'],
            'max_steps': config['max_steps'],
            'learning_rate': config['learning_rate'],
            'gamma': config['gamma'],
            'temperature_start': config['temperature_start'],
            'temperature_end': config['temperature_end'],
            'temperature_decay': config['temperature_decay']
        },
        'results': {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'temperature_history': temperature_history
        },
        'summary': {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'max_reward': float(max_reward),
            'min_reward': float(np.min(episode_rewards)),
            'mean_length': float(mean_length),
            'final_temperature': float(agent.temperature),
            'total_time_minutes': float(total_time)
        }
    }
    
    stats_path = f"{results_dir}/experiment_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, indent=2, fp=f)
    print(f"✓ Stats saved to {stats_path}")
    
    # Create training plots
    create_training_plots(
        episode_rewards,
        episode_lengths,
        losses,
        temperature_history,
        config,
        results_dir
    )
    
    # Print summary
    print()
    print("="*70)
    print(f"EXPERIMENT COMPLETE: {config['name']}")
    print(f"Total Time: {total_time:.1f} minutes")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Mean Steps: {mean_length:.1f}")
    print("="*70)
    
    env.close()
    
    return stats


def create_training_plots(rewards, lengths, losses, temperatures, config, save_dir):
    """Create and save training visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{config['name']}\n{config['description']}", 
                 fontsize=14, fontweight='bold')
    
    episodes = range(1, len(rewards) + 1)
    
    # Plot 1: Training Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.6, linewidth=0.8, label='Episode Reward')
    if len(rewards) >= 50:
        ma50 = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax1.plot(range(50, len(rewards)+1), ma50, color='orange', 
                linewidth=2, label='MA(50)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax2 = axes[0, 1]
    ax2.plot(episodes, lengths, alpha=0.6, linewidth=0.8, label='Episode Length')
    if len(lengths) >= 50:
        ma50 = np.convolve(lengths, np.ones(50)/50, mode='valid')
        ax2.plot(range(50, len(lengths)+1), ma50, color='orange', 
                linewidth=2, label='MA(50)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    ax3 = axes[1, 0]
    if losses and any(losses):
        ax3.plot(episodes, losses, alpha=0.6, linewidth=0.8, label='Loss')
        if len(losses) >= 50:
            ma50 = np.convolve(losses, np.ones(50)/50, mode='valid')
            ax3.plot(range(50, len(losses)+1), ma50, color='orange', 
                    linewidth=2, label='MA(50)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Decay
    ax4 = axes[1, 1]
    ax4.plot(episodes, temperatures, color='purple', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Temperature (τ)')
    ax4.set_title('Temperature Decay (Exploration Rate)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=config['temperature_end'], color='r', linestyle='--', 
               alpha=0.5, label=f"Min τ={config['temperature_end']}")
    ax4.legend()
    
    plt.tight_layout()
    
    plot_path = f"{save_dir}/training_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BOLTZMANN EXPLORATION EXPERIMENT")
    print("Alternative Policy to Epsilon-Greedy")
    print("="*70)
    
    try:
        stats = run_boltzmann_experiment()
        print("\n✓ Experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()