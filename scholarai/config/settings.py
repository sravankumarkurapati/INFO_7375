"""
Configuration settings for ScholarAI
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Settings:
    """Application settings"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    # Models
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Application
    APP_NAME = os.getenv("APP_NAME", "ScholarAI")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "180"))
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    
    # Memory
    ENABLE_MEMORY = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
    MEMORY_TYPE = os.getenv("MEMORY_TYPE", "short_term")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/scholarai.log")
    
    # Directories
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
    VISUALIZATION_DIR = Path(os.getenv("VISUALIZATION_DIR", "outputs/visualizations"))
    REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "outputs/reports"))
    MEMORY_DIR = Path("memory")
    
    @classmethod
    def validate(cls):
        """Validate required settings"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")
        if not cls.SERPER_API_KEY:
            raise ValueError("SERPER_API_KEY not found in environment")
        
        # Create directories if they don't exist
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.VISUALIZATION_DIR.mkdir(exist_ok=True)
        cls.REPORTS_DIR.mkdir(exist_ok=True)
        cls.MEMORY_DIR.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        return True

# Create singleton instance
settings = Settings()