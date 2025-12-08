"""
Coordinated multi-agent system with shared learning
INNOVATION: Communication + reward sharing
"""
import numpy as np
from stable_baselines3 import DQN
from src.environment.multi_warehouse_env import MultiWarehouseEnv
from src.environment.env_wrappers import FlattenMultiDiscreteWrapper


class CoordinatedMultiAgent:
    """
    Coordinated agents with communication
    
    Innovation:
    - Single policy trained on full system state
    - Sees all warehouses (communication)
    - Optimizes global cost (shared reward)
    - Learns coordination strategies
    """
    
    def __init__(
        self,
        env: MultiWarehouseEnv,
        learning_rate: float = 0.0003,
        buffer_size: int = 100000,
        verbose: int = 0
    ):
        """
        Initialize coordinated agent
        
        Args:
            env: Multi-warehouse environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            verbose: Verbosity
        """
        self.env = env
        
        print(f"\n🤖 Creating Coordinated Multi-Agent System...")
        print(f"  Innovation: Single policy sees ALL warehouses")
        print(f"  State dimension: {env.observation_space.shape[0]}")
        print(f"  Includes: All warehouses' inventory (communication)")
        
        # Wrap environment to flatten action space
        wrapped_env = FlattenMultiDiscreteWrapper(env)
        
        # Create coordinated DQN agent
        self.agent = DQN(
            policy="MlpPolicy",
            env=wrapped_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=128,
            gamma=0.99,
            exploration_fraction=0.5,
            exploration_final_eps=0.1,
            policy_kwargs=dict(
                net_arch=[512, 512, 256]  # Larger network for coordination
            ),
            tensorboard_log="./logs/multi_agent/",
            verbose=verbose
        )
        
        print(f"  ✓ Coordinated agent created")
        print(f"  ✓ Network: [512, 512, 256] (handles full system state)")
    
    def train(self, total_timesteps: int = 150000, callback=None):
        """
        Train coordinated agent
        
        Args:
            total_timesteps: Training steps
            callback: Training callback
        """
        print(f"\n🚀 Training Coordinated Multi-Agent System...")
        print(f"  Timesteps: {total_timesteps:,}")
        print(f"  Learning: Global coordination strategy")
        
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        print(f"\n✅ Coordinated agent trained")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Predict coordinated actions"""
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save coordinated agent"""
        self.agent.save(path)
        print(f"💾 Coordinated agent saved to: {path}")
    
    def load(self, path: str):
        """Load coordinated agent"""
        wrapped_env = FlattenMultiDiscreteWrapper(self.env)
        self.agent = DQN.load(path, env=wrapped_env)
        print(f"📂 Coordinated agent loaded from: {path}")