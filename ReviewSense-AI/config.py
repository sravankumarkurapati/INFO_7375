#!/usr/bin/env python3
"""
Complete project configuration for ReviewSense AI
Optimized for Mac (MPS/CPU)
"""

from dataclasses import dataclass, asdict
import json
import torch
from pathlib import Path

@dataclass
class Config:
    """Master configuration for the entire project"""
    
    # ========================================================================
    # PROJECT INFO
    # ========================================================================
    project_name: str = "ReviewSense-AI"
    assignment: str = "LLM Fine-Tuning Assignment"
    student: str = "Sravan Kumar Kurapati"
    university: str = "Northeastern University"
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    # Using smaller model for Mac (7B is too large for most Macs)
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # Alternative for limited memory: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    model_max_length: int = 2048
    
    # ========================================================================
    # QUANTIZATION (For memory efficiency on Mac)
    # ========================================================================
    use_4bit: bool = False  # Mac doesn't support bitsandbytes well
    use_8bit: bool = False  # Mac doesn't support bitsandbytes well
    
    # ========================================================================
    # LoRA CONFIGURATION (Parameter-Efficient Fine-Tuning)
    # ========================================================================
    lora_r: int = 16               # LoRA rank
    lora_alpha: int = 32           # LoRA alpha
    lora_dropout: float = 0.05     # Dropout
    lora_bias: str = "none"        # Bias training
    lora_task_type: str = "CAUSAL_LM"
    
    # Target modules (Mistral-specific)
    lora_target_modules: list = None
    
    # ========================================================================
    # TRAINING HYPERPARAMETERS (Mac-optimized)
    # ========================================================================
    num_epochs: int = 3
    batch_size: int = 1            # Small for Mac memory
    gradient_accumulation_steps: int = 8  # Effective batch = 8
    
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    optimizer: str = "adamw_torch"  # Use torch's AdamW (works on Mac)
    lr_scheduler: str = "cosine"
    
    # ========================================================================
    # PRECISION (Mac-optimized)
    # ========================================================================
    fp16: bool = False  # Mac MPS doesn't support FP16 well
    bf16: bool = False  # Mac doesn't support BF16
    # Will use FP32 (slower but stable on Mac)
    
    # ========================================================================
    # LOGGING & CHECKPOINTING
    # ========================================================================
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 2  # Save only 2 checkpoints (save space)
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Use smaller dataset for Mac (faster training)
    max_train_samples: int = 10000
    max_val_samples: int = 1000
    max_test_samples: int = 1000
    
    # ========================================================================
    # PATHS
    # ========================================================================
    data_dir: str = "./data"
    output_dir: str = "./models/checkpoints"
    final_model_dir: str = "./models/final"
    log_dir: str = "./outputs/logs"
    results_dir: str = "./outputs/results"
    figures_dir: str = "./outputs/figures"
    
    # ========================================================================
    # MISC
    # ========================================================================
    seed: int = 42
    report_to: str = "none"  # Change to "wandb" if you want experiment tracking
    
    def __post_init__(self):
        """Initialize dynamic fields"""
        if self.lora_target_modules is None:
            # Mistral-specific target modules
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        
        # Create directories
        for path in [self.data_dir, self.output_dir, self.final_model_dir, 
                     self.log_dir, self.results_dir, self.figures_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_device(self):
        """Get best available device for Mac"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def save(self, path: str = "config.json"):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"✅ Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: str = "config.json"):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def print_config(self):
        """Print configuration in readable format"""
        print("="*80)
        print(f"⚙️  {self.project_name} CONFIGURATION")
        print("="*80)
        print(f"\n📚 Project Info:")
        print(f"   Assignment: {self.assignment}")
        print(f"   Student: {self.student}")
        print(f"   University: {self.university}")
        
        print(f"\n🤖 Model:")
        print(f"   Base Model: {self.base_model}")
        print(f"   Max Length: {self.model_max_length}")
        print(f"   Device: {self.get_device().upper()}")
        
        print(f"\n🔧 LoRA:")
        print(f"   Rank (r): {self.lora_r}")
        print(f"   Alpha: {self.lora_alpha}")
        print(f"   Dropout: {self.lora_dropout}")
        
        print(f"\n📊 Training:")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"   Effective Batch Size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Optimizer: {self.optimizer}")
        
        print(f"\n💾 Data:")
        print(f"   Train: {self.train_split*100:.0f}% ({self.max_train_samples:,} samples)")
        print(f"   Validation: {self.val_split*100:.0f}% ({self.max_val_samples:,} samples)")
        print(f"   Test: {self.test_split*100:.0f}% ({self.max_test_samples:,} samples)")
        
        print(f"\n📁 Paths:")
        print(f"   Data: {self.data_dir}")
        print(f"   Checkpoints: {self.output_dir}")
        print(f"   Final Model: {self.final_model_dir}")
        print(f"   Logs: {self.log_dir}")
        
        print("\n" + "="*80)


# Create and export default config
if __name__ == "__main__":
    config = Config()
    config.print_config()
    config.save()
    print("\n✅ Configuration file created!")
