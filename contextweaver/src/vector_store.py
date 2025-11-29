# src/vector_store.py
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List, Optional, Tuple
import os
import logging

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB vector store operations
    
    Requirement: "Implement vector storage and retrieval"
    
    Features:
    - Vector storage with ChromaDB
    - Similarity search
    - Metadata filtering
    - Persistence
    """
    
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        logger.info(f"🗄️ Initializing vector store at: {persist_directory}")
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create or update vector store with documents"""
        logger.info(f"📊 Creating embeddings for {len(documents)} document chunks...")
        
        # Filter complex metadata (lists, dicts) that ChromaDB doesn't support
        filtered_docs = filter_complex_metadata(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=filtered_docs,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        logger.info(f"✅ Vector store created successfully!")
        return self.vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Test if vectorstore has documents
            count = self.vectorstore._collection.count()
            logger.info(f"✅ Loaded existing vector store with {count} chunks")
            return self.vectorstore
            
        except Exception as e:
            logger.warning(f"❌ No existing vector store found: {e}")
            return None
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load first.")
        
        logger.info(f"📊 Adding {len(documents)} new documents...")
        # Filter complex metadata before adding
        filtered_docs = filter_complex_metadata(documents)
        self.vectorstore.add_documents(filtered_docs)
        logger.info(f"✅ Documents added successfully!")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Perform similarity search"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def delete_collection(self):
        """Delete the entire collection"""
        if self.vectorstore is not None:
            self.vectorstore._client.delete_collection(self.collection_name)
            logger.info(f"🗑️ Deleted collection: {self.collection_name}")
    
    def get_statistics(self) -> dict:
        """Get vector store statistics"""
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        count = self.vectorstore._collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }


class AdvancedRetriever:
    """
    Advanced retrieval with multiple strategies
    
    Requirement: "Create effective ranking and filtering mechanisms"
    
    Features:
    - Hybrid search (semantic + keyword)
    - Multi-factor ranking
    - Diversity filtering
    """
    
    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        logger.info("🔍 Advanced retriever initialized")
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        weights: Optional[dict] = None
    ) -> List[Tuple[Document, float, dict]]:
        """
        Multi-factor reranking of documents
        
        Factors:
        1. Semantic similarity (from vector search)
        2. Source credibility
        3. Temporal relevance (recency bias)
        4. Content quality
        """
        
        if weights is None:
            weights = Config.RANKING_WEIGHTS
        
        ranked_docs = []
        
        for doc in documents:
            scores = {
                'similarity': doc.metadata.get('similarity_score', 0.5),
                'credibility': self._get_credibility_score(doc),
                'recency': self._get_recency_score(doc),
                'quality': doc.metadata.get('quality_score', 0.5),
            }
            
            # Calculate weighted final score
            final_score = sum(
                scores[factor] * weights.get(factor, 0)
                for factor in scores
            )
            
            ranked_docs.append((doc, final_score, scores))
        
        # Sort by final score
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_docs
    
    def _get_credibility_score(self, doc: Document) -> float:
        """Score based on source type credibility"""
        source_type = doc.metadata.get('source_type', 'general_source')
        return Config.CREDIBILITY_SCORES.get(source_type, 0.5)
    
    def _get_recency_score(self, doc: Document, decay_rate: float = 0.1) -> float:
        """Score based on document recency with exponential decay"""
        from datetime import datetime
        import math
        
        year = doc.metadata.get('year')
        if not year:
            return 0.5  # Unknown date gets neutral score
        
        current_year = datetime.now().year
        years_old = current_year - year
        
        # Exponential decay: score = e^(-decay_rate * years_old)
        recency_score = math.exp(-decay_rate * years_old)
        
        return recency_score
    
    def filter_documents(
        self,
        documents: List[Document],
        filters: dict
    ) -> List[Document]:
        """Filter documents based on metadata criteria"""
        filtered = documents
        
        # Year range filter
        if 'year_min' in filters:
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('year', 0) >= filters['year_min']
            ]
        
        if 'year_max' in filters:
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('year', float('inf')) <= filters['year_max']
            ]
        
        # Source type filter
        if 'source_types' in filters:
            allowed_types = set(filters['source_types'])
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('source_type') in allowed_types
            ]
        
        # Domain filter
        if 'domains' in filters:
            allowed_domains = set(filters['domains'])
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('domain') in allowed_domains
            ]
        
        # Quality filter
        if 'min_quality' in filters:
            min_quality = filters['min_quality']
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('quality_score', 0) >= min_quality
            ]
        
        return filtered
    
    def diverse_filtering(
        self,
        documents: List[Document],
        max_per_source: int = 2,
        max_per_year: int = 3
    ) -> List[Document]:
        """Ensure diversity in retrieved documents"""
        from collections import defaultdict
        
        source_counts = defaultdict(int)
        year_counts = defaultdict(int)
        diverse_docs = []
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            year = doc.metadata.get('year', 'unknown')
            
            # Check diversity constraints
            if (source_counts[source] < max_per_source and 
                year_counts[year] < max_per_year):
                diverse_docs.append(doc)
                source_counts[source] += 1
                year_counts[year] += 1
        
        return diverse_docs


# Test the vector store
if __name__ == "__main__":
    from config import Config
    
    # Initialize
    vector_manager = VectorStoreManager(
        persist_directory=str(Config.CHROMA_DIR),
        collection_name=Config.COLLECTION_NAME
    )
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Coffee consumption has been linked to increased cardiovascular risk in some studies from 2018.",
            metadata={"source": "study_a.pdf", "year": 2018, "topic": "coffee", "source_type": "peer_reviewed", "quality_score": 0.8}
        ),
        Document(
            page_content="Recent research in 2023 shows that moderate coffee intake may have protective effects on heart health.",
            metadata={"source": "study_b.pdf", "year": 2023, "topic": "coffee", "source_type": "peer_reviewed", "quality_score": 0.9}
        ),
        Document(
            page_content="A 2022 meta-analysis found no significant correlation between coffee consumption and heart disease.",
            metadata={"source": "meta_analysis.pdf", "year": 2022, "topic": "coffee", "source_type": "peer_reviewed", "quality_score": 0.95}
        )
    ]
    
    # Create vector store
    print("\n" + "="*60)
    print("Testing Vector Store Creation")
    print("="*60)
    vectorstore = vector_manager.create_vectorstore(sample_docs)
    
    # Test search
    print("\n" + "="*60)
    print("Testing Similarity Search")
    print("="*60)
    query = "Is coffee bad for your heart?"
    print(f"\nQuery: '{query}'")
    results = vector_manager.similarity_search_with_score(query, k=3)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n📄 Result #{i} (Score: {score:.3f}):")
        print(f"   Content: {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Year: {doc.metadata.get('year', 'N/A')}")
    
    # Statistics
    print("\n" + "="*60)
    print("Vector Store Statistics")
    print("="*60)
    stats = vector_manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Vector store test complete!")