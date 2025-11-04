#!/usr/bin/env python3
"""
Frame Preprocessing for Atari Kangaroo
Converts RGB frames to grayscale, resizes, and handles frame stacking
"""
import numpy as np
import cv2
from collections import deque

class FramePreprocessor:
    """
    Preprocesses Atari frames for DQN input
    - Converts to grayscale
    - Resizes to 84x84
    - Normalizes pixel values
    """
    
    def __init__(self, height=84, width=84):
        """
        Args:
            height (int): Target frame height
            width (int): Target frame width
        """
        self.height = height
        self.width = width
    
    def preprocess(self, frame):
        """
        Preprocess a single frame
        
        Args:
            frame (np.array): Raw frame from environment (210, 160, 3)
            
        Returns:
            np.array: Preprocessed frame (84, 84, 1)
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (self.width, self.height), 
                            interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized / 255.0
        
        # Add channel dimension
        processed = np.expand_dims(normalized, axis=-1)
        
        return processed.astype(np.float32)


class FrameStack:
    """
    Stacks consecutive frames to capture motion information
    DQN uses last 4 frames as input
    """
    
    def __init__(self, num_frames=4):
        """
        Args:
            num_frames (int): Number of frames to stack
        """
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        """
        Reset the frame stack with initial frame
        
        Args:
            frame (np.array): Initial preprocessed frame
        """
        # Fill stack with same frame initially
        for _ in range(self.num_frames):
            self.frames.append(frame)
    
    def append(self, frame):
        """
        Add new frame to stack
        
        Args:
            frame (np.array): New preprocessed frame
        """
        self.frames.append(frame)
    
    def get_state(self):
        """
        Get current stacked state
        
        Returns:
            np.array: Stacked frames (84, 84, 4)
        """
        # Stack along channel dimension
        stacked = np.concatenate(list(self.frames), axis=-1)
        return stacked


def test_preprocessing():
    """Test the preprocessing pipeline"""
    import gymnasium as gym
    import ale_py
    
    gym.register_envs(ale_py)
    
    print("=" * 60)
    print("TESTING FRAME PREPROCESSING")
    print("=" * 60)
    
    # Create environment
    env = gym.make('ALE/Kangaroo-v5', render_mode='rgb_array')
    observation, _ = env.reset()
    
    print(f"\n1. Original frame shape: {observation.shape}")
    print(f"   Original frame dtype: {observation.dtype}")
    print(f"   Original value range: [{observation.min()}, {observation.max()}]")
    
    # Preprocess single frame
    preprocessor = FramePreprocessor()
    processed_frame = preprocessor.preprocess(observation)
    
    print(f"\n2. Preprocessed frame shape: {processed_frame.shape}")
    print(f"   Preprocessed dtype: {processed_frame.dtype}")
    print(f"   Preprocessed value range: [{processed_frame.min():.3f}, {processed_frame.max():.3f}]")
    
    # Test frame stacking
    frame_stack = FrameStack(num_frames=4)
    frame_stack.reset(processed_frame)
    
    # Add a few more frames
    for _ in range(3):
        observation, _, _, _, _ = env.step(env.action_space.sample())
        processed = preprocessor.preprocess(observation)
        frame_stack.append(processed)
    
    stacked_state = frame_stack.get_state()
    
    print(f"\n3. Stacked state shape: {stacked_state.shape}")
    print(f"   Stacked state dtype: {stacked_state.dtype}")
    print(f"   Memory per state: {stacked_state.nbytes / 1024:.2f} KB")
    
    print("\n✓ Preprocessing test successful!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    test_preprocessing()