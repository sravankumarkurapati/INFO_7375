"""
Environment wrappers for compatibility
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FlattenMultiDiscreteWrapper(gym.Wrapper):
    """
    Wrapper to flatten MultiDiscrete action space to Discrete
    
    Converts MultiDiscrete([5, 5, 5]) to Discrete(125)
    where 125 = 5 × 5 × 5
    """
    
    def __init__(self, env):
        """
        Initialize wrapper
        
        Args:
            env: Environment with MultiDiscrete action space
        """
        super().__init__(env)
        
        # Store original action space
        self.original_action_space = env.action_space
        
        # Check that action space is MultiDiscrete
        assert isinstance(self.original_action_space, spaces.MultiDiscrete), \
            "This wrapper only works with MultiDiscrete action spaces"
        
        # Get dimensions
        self.nvec = self.original_action_space.nvec
        self.n_actions = len(self.nvec)
        
        # Calculate total number of action combinations
        self.n_total = int(np.prod(self.nvec))
        
        # Create new flattened action space
        self.action_space = spaces.Discrete(self.n_total)
        
        print(f"  ✓ Action space flattened: MultiDiscrete({list(self.nvec)}) → Discrete({self.n_total})")
    
    def _flatten_action(self, action: int) -> np.ndarray:
        """
        Convert flat action to multi-discrete action
        
        Args:
            action: Flat action (0 to n_total-1)
            
        Returns:
            Multi-discrete action array
        """
        # Convert flat index to multi-dimensional indices
        multi_action = np.zeros(self.n_actions, dtype=np.int32)
        
        remaining = action
        for i in range(self.n_actions - 1, -1, -1):
            multi_action[i] = remaining % self.nvec[i]
            remaining = remaining // self.nvec[i]
        
        return multi_action
    
    def step(self, action: int):
        """
        Take step with flat action
        
        Args:
            action: Flat action integer
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert flat action to multi-discrete
        multi_action = self._flatten_action(action)
        
        # Execute action in base environment
        return self.env.step(multi_action)
    
    def reset(self, **kwargs):
        """Reset environment"""
        return self.env.reset(**kwargs)


def create_training_env(wrapped: bool = True, **env_kwargs):
    """
    Create training environment with optional wrapper
    
    Args:
        wrapped: Whether to wrap with flattening wrapper
        **env_kwargs: Arguments for SupplyChainEnv
        
    Returns:
        Environment (wrapped or unwrapped)
    """
    from .supply_chain_env import SupplyChainEnv
    
    env = SupplyChainEnv(**env_kwargs)
    
    if wrapped:
        env = FlattenMultiDiscreteWrapper(env)
    
    return env