# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    """Comprehensive configuration management"""
    
    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic_data"
    BENCHMARKS_DIR = DATA_DIR / "benchmarks"
    CHROMA_DIR = DATA_DIR / "chroma_db"
    
    # ==================== API KEYS ====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # ==================== MODEL SETTINGS ====================
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # ==================== RAG SETTINGS ====================
    
    # Vector Store
    COLLECTION_NAME = "contextweaver_docs"
    EMBEDDING_DIMENSION = 1536
    
    # Chunking Strategies
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "hybrid")
    
    # Retrieval Settings
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K", "10"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
    
    # Ranking & Filtering
    RANKING_WEIGHTS = {
        'similarity': 0.35,
        'credibility': 0.20,
        'recency': 0.20,
        'quality': 0.15,
        'alignment': 0.10
    }
    
    CREDIBILITY_SCORES = {
        'peer_reviewed': 1.0,
        'official_documentation': 0.9,
        'preprint': 0.7,
        'blog_post': 0.5,
        'general_source': 0.6
    }
    
    # ==================== PROMPT ENGINEERING ====================
    
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    RESERVED_TOKENS_FOR_RESPONSE = 2000
    
    USE_FEW_SHOT = os.getenv("USE_FEW_SHOT", "true").lower() == "true"
    NUM_FEW_SHOT_EXAMPLES = int(os.getenv("NUM_FEW_SHOT_EXAMPLES", "2"))
    USE_CHAIN_OF_THOUGHT = os.getenv("USE_CHAIN_OF_THOUGHT", "true").lower() == "true"
    
    # ==================== SYNTHETIC DATA ====================
    
    SYNTHETIC_SAMPLES_PER_TYPE = int(os.getenv("SYNTHETIC_SAMPLES", "50"))
    AUGMENTATION_FACTOR = int(os.getenv("AUGMENTATION_FACTOR", "3"))
    
    MIN_SYNTHETIC_QUALITY_SCORE = float(os.getenv("MIN_SYNTHETIC_QUALITY", "0.7"))
    DIVERSITY_THRESHOLD = float(os.getenv("DIVERSITY_THRESHOLD", "0.6"))
    
    # ==================== EVALUATION ====================
    
    ENABLE_EVALUATION = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
    BENCHMARK_SUITE = ["multi_hop", "contradiction", "temporal", "citation_accuracy"]
    
    TARGET_RETRIEVAL_PRECISION = 0.80
    TARGET_MULTI_HOP_ACCURACY = 0.75
    TARGET_CONTRADICTION_F1 = 0.85
    TARGET_RESPONSE_TIME = 5.0
    
    # ==================== SYSTEM ====================
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            errors.append("Either OPENAI_API_KEY or ANTHROPIC_API_KEY must be set")
        
        for dir_path in [cls.DATA_DIR, cls.SAMPLE_DOCS_DIR, cls.CHROMA_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if not 0 <= cls.TEMPERATURE <= 1:
            errors.append(f"TEMPERATURE must be between 0 and 1, got {cls.TEMPERATURE}")
        
        if not 0 <= cls.HYBRID_ALPHA <= 1:
            errors.append(f"HYBRID_ALPHA must be between 0 and 1, got {cls.HYBRID_ALPHA}")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "Model": cls.MODEL_NAME,
            "Embedding": cls.EMBEDDING_MODEL,
            "Chunking Strategy": cls.CHUNKING_STRATEGY,
            "Chunk Size": cls.CHUNK_SIZE,
            "Top K": cls.TOP_K_DOCUMENTS,
            "Hybrid Search": cls.USE_HYBRID_SEARCH,
            "Few-Shot Learning": cls.USE_FEW_SHOT,
            "Chain of Thought": cls.USE_CHAIN_OF_THOUGHT,
            "Debug Mode": cls.DEBUG_MODE
        }
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("=" * 60)
        print("⚙️  CONTEXTWEAVER CONFIGURATION")
        print("=" * 60)
        for key, value in cls.get_summary().items():
            print(f"  {key:.<40} {value}")
        print("=" * 60)


if __name__ == "__main__":
    Config.validate()
    Config.print_summary()