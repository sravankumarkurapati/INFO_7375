# src/contextweaver_pipeline.py
from typing import List, Dict, Optional
import logging
from pathlib import Path
from langchain_core.documents import Document

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager, AdvancedRetriever
from prompt_engineering import PromptEngineeringSystem, EdgeCaseHandler
from reasoning_engine import MultiHopReasoningEngine, ContradictionDetector, CitationTracker, TemporalAnalyzer
from document_graph import DocumentGraph
from uncertainty_quantification import UncertaintyQuantifier, EvidenceSufficiencyAnalyzer
from fact_checker import AutomatedFactChecker, MisinformationDetector, VerificationScorecard
from web_search_fallback import WebSearchFallback, HybridRetriever, RetrievalSourceIndicator

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ContextWeaverPipeline:
    """
    Main ContextWeaver Pipeline with Hybrid Retrieval
    """
    
    def __init__(self, use_existing_db: bool = False):
        """Initialize all components"""
        
        logger.info("="*60)
        logger.info("🚀 Initializing ContextWeaver Pipeline")
        logger.info("="*60)
        
        # Core components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager(
            persist_directory=str(Config.CHROMA_DIR),
            collection_name=Config.COLLECTION_NAME
        )
        self.retriever = AdvancedRetriever(self.vector_store)
        
        # Prompt engineering
        self.prompt_system = PromptEngineeringSystem()
        self.edge_handler = EdgeCaseHandler(self.prompt_system)
        
        # Reasoning
        self.reasoning_engine = MultiHopReasoningEngine()
        self.contradiction_detector = ContradictionDetector()
        self.citation_tracker = CitationTracker()
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Innovations
        self.document_graph = DocumentGraph()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.evidence_analyzer = EvidenceSufficiencyAnalyzer()
        self.fact_checker = AutomatedFactChecker()
        self.misinfo_detector = MisinformationDetector()
        self.scorecard_generator = VerificationScorecard()
        
        # Hybrid Retrieval
        self.web_search = WebSearchFallback()
        self.hybrid_retriever = HybridRetriever(self.vector_store, self.web_search)
        
        if use_existing_db:
            try:
                self.vector_store.load_vectorstore()
            except Exception as e:
                logger.warning(f"⚠️ Could not load DB: {e}")
        
        logger.info("✅ All components initialized!")
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, any]:
        """Ingest documents"""
        
        logger.info(f"📥 Ingesting {len(file_paths)} documents...")
        
        documents = self.document_processor.load_documents(file_paths)
        self.vector_store.create_vectorstore(documents)
        
        logger.info("🔨 Building document graph...")
        unique_docs = self._get_unique_documents(documents)
        contradictions = []
        
        if len(unique_docs) >= 2:
            try:
                contradiction_result = self.contradiction_detector.detect_contradictions(unique_docs)
                contradictions = contradiction_result.get('contradictions', [])
            except:
                pass
        
        self.document_graph.build_graph(unique_docs, contradictions=contradictions, temporal_order=True)
        
        report = {
            'files_processed': len(file_paths),
            'chunks_created': len(documents),
            'knowledge_base_stats': self.document_processor.knowledge_base.statistics,
            'vector_store_stats': self.vector_store.get_statistics(),
            'graph_stats': self.document_graph.get_graph_statistics(),
            'contradictions_found': len(contradictions)
        }
        
        logger.info(f"✅ Ingestion complete!")
        return report
    
    def _get_unique_documents(self, chunks: List[Document]) -> List[Document]:
        """Get unique documents"""
        seen_sources = set()
        unique = []
        for doc in chunks:
            source = doc.metadata.get('source', '')
            if source and source not in seen_sources:
                seen_sources.add(source)
                unique.append(doc)
        return unique
    
    def query(
        self,
        query: str,
        enable_multi_hop: bool = True,
        enable_contradiction_detection: bool = True,
        enable_uncertainty: bool = True,
        enable_fact_checking: bool = True,
        top_k: int = Config.TOP_K_DOCUMENTS
    ) -> Dict[str, any]:
        """Main query pipeline with HYBRID RETRIEVAL"""
        
        logger.info(f"🔍 Processing: '{query}'")
        
        if not query or len(query.strip()) < 3:
            return {'query': query, 'status': 'ERROR', 'answer': 'Invalid query'}
        
        # HYBRID RETRIEVAL
        logger.info("   🔄 Hybrid retrieval...")
        
        try:
            retrieval_result = self.hybrid_retriever.retrieve(query, k=top_k)
            
            documents = retrieval_result['documents']
            retrieval_source = retrieval_result['source']
            retrieval_confidence = retrieval_result['confidence']
            
            if not documents:
                return {'query': query, 'status': 'NO_DOCUMENTS', 'answer': 'No information found'}
            
            logger.info(f"   ✅ Source: {retrieval_source}, Docs: {len(documents)}")
        
        except Exception as e:
            logger.error(f"   ❌ Retrieval failed: {e}")
            return {'query': query, 'status': 'ERROR', 'answer': 'Retrieval error'}
        
        # For WEB or LLM sources, skip complex processing
        if retrieval_source in ['web', 'llm_direct']:
            logger.info(f"   🌐 Using {retrieval_source} source - simplified processing")
            
            # Generate simple answer from web/LLM docs
            if retrieval_source == 'llm_direct':
                answer = documents[0].page_content
            else:
                answer = self._generate_simple_answer(query, documents[:5])
            
            # Basic uncertainty
            uncertainty_result = None
            if enable_uncertainty:
                try:
                    uncertainty_result = {
                        'confidence_score': retrieval_confidence * 0.85,
                        'confidence_level': 'MEDIUM (Web/LLM Source)',
                        'uncertainty_sources': [f'Using {retrieval_source} sources - not from local knowledge base'],
                        'evidence_gaps': ['Local knowledge base does not cover this topic'],
                        'component_scores': {},
                        'sensitivity_analysis': {}
                    }
                except:
                    pass
            
            formatted_answer = RetrievalSourceIndicator.format_answer_with_source_indicator(
                answer, retrieval_source, retrieval_confidence
            )
            
            return {
                'query': query,
                'answer': formatted_answer,
                'raw_answer': answer,
                'status': 'SUCCESS',
                'retrieval': {
                    'num_documents': len(documents),
                    'top_sources': [d.metadata.get('source', 'unknown') for d in documents[:5]],
                    'retrieval_source': retrieval_source,
                    'retrieval_confidence': retrieval_confidence,
                    'fallback_used': True,
                    'source_summary': RetrievalSourceIndicator.generate_source_summary(retrieval_result),
                    'web_results_used': len(documents) if retrieval_source == 'web' else 0,
                    'local_results_used': 0
                },
                'reasoning': None,
                'contradictions': None,
                'uncertainty': uncertainty_result,
                'fact_check': None,
                'red_flags': [],
                'citations': [],
                'components_used': {
                    'multi_hop_reasoning': False,
                    'contradiction_detection': False,
                    'uncertainty_quantification': enable_uncertainty,
                    'fact_checking': False,
                    'hybrid_retrieval': True
                }
            }
        
        # For LOCAL sources, use full pipeline
        logger.info("   📊 Local source - using full pipeline...")
        
        # Re-rank
        ranked_docs = self.retriever.rerank_documents(query, documents)
        all_docs = [doc for doc, score, _ in ranked_docs]
        
        # Graph expansion
        if self.document_graph.graph.number_of_nodes() > 0:
            try:
                seed_ids = [f"doc_{i}" for i in range(min(3, len(all_docs)))]
                graph_docs = self.document_graph.graph_based_retrieval(seed_ids, max_hops=1)
                all_docs = self._deduplicate_documents(all_docs + graph_docs)
            except:
                pass
        
        # Contradictions
        contradiction_result = None
        if enable_contradiction_detection and len(all_docs) >= 2:
            try:
                contradiction_result = self.contradiction_detector.detect_contradictions(all_docs[:5])
            except:
                pass
        
        # Multi-hop reasoning
        reasoning_result = None
        answer = ""
        
        if enable_multi_hop:
            try:
                reasoning_result = self.reasoning_engine.reason_across_documents(query, all_docs[:5], max_hops=3)
                answer = reasoning_result['answer']
            except:
                answer = self._generate_simple_answer(query, all_docs[:3])
        else:
            answer = self._generate_simple_answer(query, all_docs[:3])
        
        # Uncertainty
        uncertainty_result = None
        if enable_uncertainty:
            try:
                uncertainty_result = self.uncertainty_quantifier.quantify_uncertainty(
                    query, answer, all_docs[:5], []
                )
            except:
                pass
        
        # Fact-check
        fact_check_result = None
        red_flags = []
        if enable_fact_checking:
            try:
                fact_check_result = self.fact_checker.fact_check_answer(answer, all_docs[:5])
                red_flags = self.misinfo_detector.detect_red_flags(answer, fact_check_result, all_docs)
            except:
                pass
        
        formatted_answer = RetrievalSourceIndicator.format_answer_with_source_indicator(
            answer, retrieval_source, retrieval_confidence
        )
        
        return {
            'query': query,
            'answer': formatted_answer,
            'raw_answer': answer,
            'status': 'SUCCESS',
            'retrieval': {
                'num_documents': len(all_docs),
                'top_sources': [d.metadata.get('filename', d.metadata.get('source', 'unknown')) for d in all_docs[:5]],
                'retrieval_source': retrieval_source,
                'retrieval_confidence': retrieval_confidence,
                'fallback_used': False,
                'source_summary': RetrievalSourceIndicator.generate_source_summary(retrieval_result),
                'web_results_used': 0,
                'local_results_used': len(all_docs)
            },
            'reasoning': reasoning_result,
            'contradictions': contradiction_result,
            'uncertainty': uncertainty_result,
            'fact_check': fact_check_result,
            'red_flags': red_flags,
            'citations': reasoning_result.get('citations', []) if reasoning_result else [],
            'components_used': {
                'multi_hop_reasoning': enable_multi_hop,
                'contradiction_detection': enable_contradiction_detection,
                'uncertainty_quantification': enable_uncertainty,
                'fact_checking': enable_fact_checking,
                'hybrid_retrieval': True
            }
        }
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicates"""
        seen = set()
        unique = []
        for doc in documents:
            source = doc.metadata.get('source', id(doc))
            if source not in seen:
                seen.add(source)
                unique.append(doc)
        return unique
    
    def _generate_simple_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer"""
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.2)
        
        doc_text = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content[:400]}"
            for doc in documents
        ])
        
        prompt = f"""Answer based on these documents.

Question: {query}

Documents:
{doc_text}

Answer:"""
        
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return "Error generating answer"
    
    def generate_comprehensive_report(self, result: Dict) -> str:
        """Generate report"""
        
        report = f"""
{'='*60}
CONTEXTWEAVER ANALYSIS
{'='*60}

QUERY: {result['query']}

RETRIEVAL: {result['retrieval']['retrieval_source'].upper()}
- Web Results: {result['retrieval'].get('web_results_used', 0)}
- Local Results: {result['retrieval'].get('local_results_used', 0)}

{'='*60}
ANSWER
{'='*60}

{result.get('raw_answer', result['answer'])}

{'='*60}
"""
        return report
    
    def get_system_statistics(self) -> Dict:
        """Get stats"""
        return {
            'vector_store': self.vector_store.get_statistics(),
            'knowledge_base': self.document_processor.knowledge_base.statistics or {},
            'document_graph': self.document_graph.get_graph_statistics()
        }


# Test
if __name__ == "__main__":
    print("="*60)
    print("🧪 HYBRID PIPELINE TEST")
    print("="*60)
    
    pipeline = ContextWeaverPipeline(use_existing_db=False)
    
    # Ingest coffee docs
    sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
    if sample_files:
        pipeline.ingest_documents([str(f) for f in sample_files])
    
    # Test queries
    queries = [
        "Is coffee safe?",
        "Is chicken healthy?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        result = pipeline.query(query, enable_fact_checking=False, enable_contradiction_detection=False)
        
        print(f"\n📍 Source: {result['retrieval']['retrieval_source'].upper()}")
        print(f"🌐 Web: {result['retrieval']['web_results_used']}")
        print(f"📚 Local: {result['retrieval']['local_results_used']}")
        print(f"\n📝 Answer: {result.get('raw_answer', '')[:200]}...")
    
    print("\n✅ Test complete!")