"""
Memory management system for ScholarAI
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from config.settings import settings
from utils.logger import logger

class MemoryManager:
    """Manages short-term and long-term memory for agents"""
    
    def __init__(self):
        self.memory_dir = settings.MEMORY_DIR
        self.memory_dir.mkdir(exist_ok=True)
        
        # Short-term memory (current session)
        self.short_term: Dict[str, Any] = {
            "query_history": [],
            "agent_outputs": {},
            "errors": [],
            "metrics": {}
        }
        
        # Long-term memory file
        self.long_term_file = self.memory_dir / "long_term_memory.pkl"
        self.long_term: Dict[str, Any] = self._load_long_term()
    
    def _load_long_term(self) -> Dict[str, Any]:
        """Load long-term memory from disk"""
        if self.long_term_file.exists():
            try:
                with open(self.long_term_file, 'rb') as f:
                    memory = pickle.load(f)
                logger.info("Long-term memory loaded")
                return memory
            except Exception as e:
                logger.error(f"Failed to load long-term memory: {e}")
                return self._initialize_long_term()
        return self._initialize_long_term()
    
    def _initialize_long_term(self) -> Dict[str, Any]:
        """Initialize empty long-term memory"""
        return {
            "successful_searches": [],
            "domain_knowledge": {},
            "quality_scores": [],
            "user_preferences": {},
            "total_queries": 0,
            "created_at": datetime.now().isoformat()
        }
    
    def save_long_term(self):
        """Save long-term memory to disk"""
        try:
            with open(self.long_term_file, 'wb') as f:
                pickle.dump(self.long_term, f)
            logger.info("Long-term memory saved")
        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")
    
    def add_query(self, query: str):
        """Add query to history"""
        self.short_term["query_history"].append({
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        self.long_term["total_queries"] += 1
        logger.info(f"Query added to memory: {query}")
    
    def add_agent_output(self, agent_name: str, output: Any):
        """Store agent output"""
        self.short_term["agent_outputs"][agent_name] = {
            "output": output,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Agent output stored: {agent_name}")
    
    def get_agent_output(self, agent_name: str) -> Optional[Any]:
        """Retrieve agent output"""
        return self.short_term["agent_outputs"].get(agent_name, {}).get("output")
    
    def add_error(self, error: str, context: str = ""):
        """Log error to memory"""
        self.short_term["errors"].append({
            "error": error,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        logger.error(f"Error logged: {error}")
    
    def update_metric(self, metric_name: str, value: float):
        """Update performance metric"""
        if metric_name not in self.short_term["metrics"]:
            self.short_term["metrics"][metric_name] = []
        self.short_term["metrics"][metric_name].append(value)
        logger.debug(f"Metric updated: {metric_name}={value}")
    
    def add_successful_search(self, query: str, results_count: int):
        """Add successful search to long-term memory"""
        self.long_term["successful_searches"].append({
            "query": query,
            "results_count": results_count,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 100 searches
        if len(self.long_term["successful_searches"]) > 100:
            self.long_term["successful_searches"] = \
                self.long_term["successful_searches"][-100:]
        logger.info(f"Successful search logged: {query}")
    
    def update_domain_knowledge(self, term: str, frequency: int = 1):
        """Update domain-specific knowledge"""
        if term not in self.long_term["domain_knowledge"]:
            self.long_term["domain_knowledge"][term] = 0
        self.long_term["domain_knowledge"][term] += frequency
        logger.debug(f"Domain knowledge updated: {term}")
    
    def add_quality_score(self, score: float):
        """Add quality assessment score"""
        self.long_term["quality_scores"].append({
            "score": score,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Quality score added: {score}")
    
    def get_average_quality(self) -> float:
        """Calculate average quality score"""
        if not self.long_term["quality_scores"]:
            return 0.0
        scores = [s["score"] for s in self.long_term["quality_scores"]]
        return sum(scores) / len(scores)
    
    def clear_short_term(self):
        """Clear short-term memory"""
        self.short_term = {
            "query_history": [],
            "agent_outputs": {},
            "errors": [],
            "metrics": {}
        }
        logger.info("Short-term memory cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory summary"""
        return {
            "current_session": {
                "queries": len(self.short_term["query_history"]),
                "agent_outputs": len(self.short_term["agent_outputs"]),
                "errors": len(self.short_term["errors"]),
                "metrics": self.short_term["metrics"]
            },
            "long_term": {
                "total_queries": self.long_term["total_queries"],
                "successful_searches": len(self.long_term["successful_searches"]),
                "domain_terms": len(self.long_term["domain_knowledge"]),
                "average_quality": self.get_average_quality()
            }
        }
    
    def export_session(self, filename: Optional[str] = None) -> Path:
        """Export current session to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
        
        export_path = self.memory_dir / filename
        
        with open(export_path, 'w') as f:
            json.dump(self.short_term, f, indent=2, default=str)
        
        logger.info(f"Session exported to: {export_path}")
        return export_path

# Create singleton instance
memory_manager = MemoryManager()