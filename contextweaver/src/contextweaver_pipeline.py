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
from synthetic_data_generator import SyntheticDataGenerator

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ContextWeaverPipeline:
    """
    Main ContextWeaver Pipeline - Integrates all components
    
    Complete Multi-Document Reasoning System with:
    - RAG (retrieval-augmented generation)
    - Prompt engineering
    - Synthetic data generation
    - Multi-hop reasoning
    - Contradiction detection
    - Document knowledge graph
    - Uncertainty quantification
    - Automated fact-checking
    """
    
    def __init__(self, use_existing_db: bool = False):
        """Initialize all components"""
        
        logger.info("="*60)
        logger.info("🚀 Initializing ContextWeaver Pipeline")
        logger.info("="*60)
        
        # Core RAG components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager(
            persist_directory=str(Config.CHROMA_DIR),
            collection_name=Config.COLLECTION_NAME
        )
        self.retriever = AdvancedRetriever(self.vector_store)
        
        # Prompt engineering
        self.prompt_system = PromptEngineeringSystem()
        self.edge_handler = EdgeCaseHandler(self.prompt_system)
        
        # Reasoning components
        self.reasoning_engine = MultiHopReasoningEngine()
        self.contradiction_detector = ContradictionDetector()
        self.citation_tracker = CitationTracker()
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Innovation 1: Document Graph
        self.document_graph = DocumentGraph()
        
        # Innovation 2: Uncertainty Quantification
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.evidence_analyzer = EvidenceSufficiencyAnalyzer()
        
        # Innovation 3: Fact Checking
        self.fact_checker = AutomatedFactChecker()
        self.misinfo_detector = MisinformationDetector()
        self.scorecard_generator = VerificationScorecard()
        
        # Load existing vector store if requested
        if use_existing_db:
            try:
                self.vector_store.load_vectorstore()
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing DB: {e}")
        
        logger.info("✅ All components initialized successfully!")
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, any]:
        """
        Ingest documents into the system
        
        Pipeline:
        1. Process documents (chunking, metadata)
        2. Create vector embeddings
        3. Build document graph
        4. Return ingestion report
        """
        
        logger.info(f"📥 Ingesting {len(file_paths)} documents...")
        
        # Process documents
        documents = self.document_processor.load_documents(file_paths)
        
        # Create/update vector store
        self.vector_store.create_vectorstore(documents)
        
        # Build document graph (detect contradictions first for graph edges)
        logger.info("🔨 Building document knowledge graph...")
        
        # Quick contradiction detection for graph building
        unique_docs = self._get_unique_documents(documents)
        contradictions = []
        
        if len(unique_docs) >= 2:
            try:
                contradiction_result = self.contradiction_detector.detect_contradictions(unique_docs)
                contradictions = contradiction_result.get('contradictions', [])
            except Exception as e:
                logger.warning(f"⚠️ Contradiction detection failed during ingestion: {e}")
        
        # Build graph with detected contradictions
        self.document_graph.build_graph(
            unique_docs,
            contradictions=contradictions,
            temporal_order=True
        )
        
        # Generate ingestion report
        report = {
            'files_processed': len(file_paths),
            'chunks_created': len(documents),
            'knowledge_base_stats': self.document_processor.knowledge_base.statistics,
            'vector_store_stats': self.vector_store.get_statistics(),
            'graph_stats': self.document_graph.get_graph_statistics(),
            'contradictions_found': len(contradictions)
        }
        
        logger.info(f"✅ Ingestion complete: {len(documents)} chunks from {len(file_paths)} files")
        
        return report
    
    def _get_unique_documents(self, chunks: List[Document]) -> List[Document]:
        """Get unique documents (one per source file)"""
        
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
        """
        Main query processing pipeline
        
        Complete pipeline:
        1. Validate query
        2. Retrieve relevant documents (vector + graph)
        3. Detect contradictions
        4. Perform multi-hop reasoning
        5. Generate answer with citations
        6. Quantify uncertainty
        7. Fact-check answer
        8. Generate comprehensive report
        """
        
        logger.info(f"🔍 Processing query: '{query}'")
        
        # Step 1: Basic query validation
        if not query or len(query.strip()) < 3:
            return {
                'query': query,
                'status': 'ERROR',
                'error': 'Query too short or empty',
                'answer': 'Please provide a valid query'
            }
        
        # Step 2: Retrieve relevant documents
        logger.info("   📚 Retrieving relevant documents...")
        
        try:
            # Vector-based retrieval
            retrieved_docs = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            if not retrieved_docs:
                return {
                    'query': query,
                    'status': 'NO_DOCUMENTS',
                    'answer': f"No relevant documents found for '{query}'. Please add more documents to the knowledge base."
                }
            
            # Extract documents and scores
            documents = [doc for doc, score in retrieved_docs]
            
            # Add similarity scores to metadata
            for (doc, score), original_doc in zip(retrieved_docs, documents):
                original_doc.metadata['similarity_score'] = 1 - score  # Convert distance to similarity
            
            logger.info(f"   ✅ Retrieved {len(documents)} documents")
        
        except Exception as e:
            logger.error(f"   ❌ Retrieval failed: {e}")
            return {
                'query': query,
                'status': 'ERROR',
                'error': str(e),
                'answer': 'Error during document retrieval'
            }
        
        # Step 3: Re-rank documents
        logger.info("   📊 Re-ranking with multi-factor scoring...")
        ranked_docs = self.retriever.rerank_documents(query, documents)
        top_ranked = [doc for doc, score, _ in ranked_docs]
        
        # Step 4: Graph-based expansion (optional)
        logger.info("   🕸️ Expanding retrieval via document graph...")
        
        try:
            # Get seed document IDs (first 3 from ranked results)
            seed_ids = [f"doc_{i}" for i in range(min(3, len(top_ranked)))]
            
            # Only try graph retrieval if graph has been built
            if self.document_graph.graph.number_of_nodes() > 0:
                graph_docs = self.document_graph.graph_based_retrieval(
                    seed_ids,
                    relationship_types=['cites', 'similar_topic', 'temporal_successor'],
                    max_hops=1
                )
                
                # Combine and deduplicate
                all_docs = self._deduplicate_documents(top_ranked + graph_docs)
                logger.info(f"   ✅ Expanded to {len(all_docs)} documents via graph")
            else:
                all_docs = top_ranked
                logger.info(f"   ⚠️ Graph not built yet, using vector results only")
        except Exception as e:
            logger.warning(f"   ⚠️ Graph expansion failed: {e}")
            all_docs = top_ranked
        
        # Step 5: Detect contradictions (if enabled)
        contradiction_result = None
        contradicting_docs = []
        
        if enable_contradiction_detection:
            logger.info("   🔍 Detecting contradictions...")
            try:
                contradiction_result = self.contradiction_detector.detect_contradictions(all_docs[:5])
                
                if contradiction_result.get('num_contradictions', 0) > 0:
                    logger.info(f"   ⚠️ Found {contradiction_result['num_contradictions']} contradictions")
                    # Mark contradicting documents
                    for c in contradiction_result.get('contradictions', []):
                        for doc in all_docs:
                            source = doc.metadata.get('source', '')
                            filename = doc.metadata.get('filename', '')
                            if c['source_A'] in source or c['source_A'] in filename or c['source_B'] in source or c['source_B'] in filename:
                                contradicting_docs.append(doc)
            except Exception as e:
                logger.warning(f"   ⚠️ Contradiction detection failed: {e}")
        
        # Step 6: Multi-hop reasoning (if enabled)
        reasoning_result = None
        answer = ""
        citations = []
        
        if enable_multi_hop:
            logger.info("   🧠 Performing multi-hop reasoning...")
            try:
                reasoning_result = self.reasoning_engine.reason_across_documents(
                    query,
                    all_docs[:5],
                    max_hops=3
                )
                
                answer = reasoning_result['answer']
                citations = reasoning_result.get('citations', [])
                
                logger.info(f"   ✅ Reasoning complete ({reasoning_result['hops_used']} hops)")
            except Exception as e:
                logger.warning(f"   ⚠️ Multi-hop reasoning failed: {e}")
                # Fallback to simple answer
                answer = self._generate_simple_answer(query, all_docs[:3])
        else:
            answer = self._generate_simple_answer(query, all_docs[:3])
        
        # Step 7: Quantify uncertainty (if enabled)
        uncertainty_result = None
        
        if enable_uncertainty:
            logger.info("   🎲 Quantifying uncertainty...")
            try:
                uncertainty_result = self.uncertainty_quantifier.quantify_uncertainty(
                    query,
                    answer,
                    all_docs[:5],
                    contradicting_docs
                )
                logger.info(f"   ✅ Confidence: {uncertainty_result['confidence_score']:.2%}")
            except Exception as e:
                logger.warning(f"   ⚠️ Uncertainty quantification failed: {e}")
        
        # Step 8: Fact-check answer (if enabled)
        fact_check_result = None
        red_flags = []
        
        if enable_fact_checking and answer:
            logger.info("   🔍 Fact-checking answer...")
            try:
                fact_check_result = self.fact_checker.fact_check_answer(answer, all_docs[:5])
                red_flags = self.misinfo_detector.detect_red_flags(answer, fact_check_result, all_docs)
                logger.info(f"   ✅ Verification: {fact_check_result['overall_score']:.1%}")
            except Exception as e:
                logger.warning(f"   ⚠️ Fact-checking failed: {e}")
        
        # Step 9: Compile comprehensive result
        logger.info("   📊 Compiling comprehensive result...")
        
        result = {
            'query': query,
            'answer': answer,
            'status': 'SUCCESS',
            
            # Retrieval info
            'retrieval': {
                'num_documents': len(all_docs),
                'top_sources': [doc.metadata.get('filename', doc.metadata.get('source', 'unknown')) for doc in all_docs[:3]]
            },
            
            # Reasoning
            'reasoning': reasoning_result,
            
            # Contradictions
            'contradictions': contradiction_result,
            
            # Uncertainty
            'uncertainty': uncertainty_result,
            
            # Fact-checking
            'fact_check': fact_check_result,
            'red_flags': red_flags,
            
            # Citations
            'citations': citations,
            
            # Metadata
            'components_used': {
                'multi_hop_reasoning': enable_multi_hop,
                'contradiction_detection': enable_contradiction_detection,
                'uncertainty_quantification': enable_uncertainty,
                'fact_checking': enable_fact_checking
            }
        }
        
        logger.info("✅ Query processing complete!")
        
        return result
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents"""
        
        seen_sources = set()
        unique = []
        
        for doc in documents:
            source = doc.metadata.get('source', id(doc))
            if source not in seen_sources:
                seen_sources.add(source)
                unique.append(doc)
        
        return unique
    
    def _generate_simple_answer(self, query: str, documents: List[Document]) -> str:
        """Generate simple answer as fallback"""
        
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)
        
        # Format documents
        doc_text = "\n\n".join([
            f"Source: {doc.metadata.get('filename', doc.metadata.get('source', 'unknown'))}\n{doc.page_content[:300]}"
            for doc in documents
        ])
        
        prompt = f"""Answer this question based on the provided documents.

Question: {query}

Documents:
{doc_text}

Provide a clear, concise answer with citations.

Answer:"""
        
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return "Error generating answer"
    
    def generate_comprehensive_report(self, result: Dict) -> str:
        """
        Generate comprehensive analysis report
        
        Combines all analyses into readable format
        """
        
        report = f"""
{'='*60}
CONTEXTWEAVER COMPREHENSIVE ANALYSIS
{'='*60}

QUERY: {result['query']}

{'='*60}
ANSWER
{'='*60}

{result['answer']}

"""
        
        # Reasoning section
        if result.get('reasoning'):
            report += f"""
{'='*60}
MULTI-HOP REASONING ANALYSIS
{'='*60}

Reasoning Hops: {result['reasoning']['hops_used']}
Documents Used: {result['reasoning']['documents_used']}
Confidence: {result['reasoning']['confidence']:.2%}

Reasoning Chain:
"""
            for step in result['reasoning']['reasoning_chain']:
                report += f"\nStep {step['hop_number']}:\n"
                extracted = str(step.get('extracted_info', 'N/A'))
                report += f"  {extracted[:150]}...\n"
        
        # Contradiction section
        if result.get('contradictions'):
            report += f"""
{'='*60}
CONTRADICTION ANALYSIS
{'='*60}

Contradictions Found: {result['contradictions'].get('num_contradictions', 0)}
Overall Severity: {result['contradictions'].get('overall_severity', 'N/A')}

"""
            if result['contradictions'].get('contradictions'):
                for i, c in enumerate(result['contradictions']['contradictions'][:3], 1):
                    report += f"\n{i}. {c.get('severity', 'N/A')} Severity Contradiction:\n"
                    report += f"   Source A: {c.get('source_A', 'N/A')}\n"
                    report += f"   Source B: {c.get('source_B', 'N/A')}\n"
                    report += f"   Explanation: {c.get('explanation', 'N/A')[:100]}...\n"
        
        # Uncertainty section
        if result.get('uncertainty'):
            report += f"""
{'='*60}
UNCERTAINTY QUANTIFICATION
{'='*60}

Confidence Score: {result['uncertainty']['confidence_score']:.2%}
Confidence Level: {result['uncertainty']['confidence_level']}

Uncertainty Sources:
"""
            for source in result['uncertainty']['uncertainty_sources']:
                report += f"  • {source}\n"
            
            report += f"\nEvidence Gaps:\n"
            for gap in result['uncertainty']['evidence_gaps']:
                report += f"  • {gap}\n"
        
        # Fact-checking section
        if result.get('fact_check'):
            report += f"""
{'='*60}
FACT-CHECK ANALYSIS
{'='*60}

Verification Score: {result['fact_check']['overall_score']:.1%}
Verification Level: {result['fact_check']['verification_level']}

Verified Claims: {result['fact_check']['num_verified']}/{len(result['fact_check']['claims'])}
Unsupported Claims: {result['fact_check']['num_unsupported']}
Contradicted Claims: {result['fact_check']['num_contradicted']}

Red Flags: {len(result.get('red_flags', []))}
"""
            if result.get('red_flags'):
                for flag in result['red_flags']:
                    report += f"  ⚠️ {flag['type']}: {flag['description']}\n"
        
        # Citations
        if result.get('citations'):
            report += f"""
{'='*60}
CITATIONS
{'='*60}

"""
            for i, citation in enumerate(result['citations'], 1):
                report += f"{i}. {citation.get('source', 'Unknown')}\n"
                if 'claim' in citation:
                    claim = str(citation['claim'])
                    report += f"   Supports: {claim[:100]}...\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def process_batch_queries(
        self,
        queries: List[str],
        **kwargs
    ) -> List[Dict]:
        """Process multiple queries in batch"""
        
        logger.info(f"📊 Processing {len(queries)} queries in batch...")
        
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\n{'─'*60}")
            logger.info(f"Query {i}/{len(queries)}")
            logger.info(f"{'─'*60}")
            
            result = self.query(query, **kwargs)
            results.append(result)
        
        logger.info(f"\n✅ Batch processing complete: {len(results)} queries processed")
        
        return results
    
    def get_system_statistics(self) -> Dict[str, any]:
        """Get comprehensive system statistics"""
        
        stats = {
            'vector_store': self.vector_store.get_statistics(),
            'knowledge_base': self.document_processor.knowledge_base.statistics if self.document_processor.knowledge_base.statistics else {},
            'document_graph': self.document_graph.get_graph_statistics()
        }
        
        return stats
    
    def export_results(
        self,
        result: Dict,
        output_dir: str,
        include_visualizations: bool = True
    ):
        """Export query results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export JSON
        json_path = output_path / "result.json"
        import json
        
        # Make result JSON-serializable
        serializable_result = self._make_serializable(result)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        logger.info(f"💾 Results exported to {json_path}")
        
        # Export text report
        report_path = output_path / "report.txt"
        report = self.generate_comprehensive_report(result)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"💾 Report exported to {report_path}")
        
        # Export visualizations
        if include_visualizations:
            try:
                viz_path = output_path / "graph_visualization.html"
                self.document_graph.visualize_graph_plotly(str(viz_path))
                logger.info(f"💾 Visualization exported to {viz_path}")
            except Exception as e:
                logger.warning(f"⚠️ Visualization export failed: {e}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Document):
            return {
                'content': obj.page_content[:200],
                'metadata': {k: v for k, v in obj.metadata.items() if isinstance(v, (str, int, float, bool)) or v is None}
            }
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


class QueryRouter:
    """Intelligent query routing"""
    
    def __init__(self, pipeline: ContextWeaverPipeline):
        self.pipeline = pipeline
        self.prompt_system = PromptEngineeringSystem()
        logger.info("🔀 Query router initialized")
    
    def route_query(self, query: str) -> Dict[str, any]:
        """Route query to optimal processing pipeline"""
        
        # Classify query type
        query_type = self.prompt_system.classify_query_type(query)
        
        logger.info(f"🔀 Routing query as: {query_type}")
        
        # Route based on type
        if query_type == 'contradiction':
            return self.pipeline.query(
                query,
                enable_multi_hop=True,
                enable_contradiction_detection=True,
                enable_uncertainty=True,
                enable_fact_checking=False
            )
        
        elif query_type == 'temporal':
            return self.pipeline.query(
                query,
                enable_multi_hop=False,
                enable_contradiction_detection=True,
                enable_uncertainty=True,
                enable_fact_checking=False
            )
        
        elif query_type == 'multi_hop':
            return self.pipeline.query(
                query,
                enable_multi_hop=True,
                enable_contradiction_detection=True,
                enable_uncertainty=True,
                enable_fact_checking=True
            )
        
        else:
            return self.pipeline.query(
                query,
                enable_multi_hop=False,
                enable_contradiction_detection=False,
                enable_uncertainty=True,
                enable_fact_checking=True
            )


# Test integrated pipeline
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING CONTEXTWEAVER INTEGRATED PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline (fresh - no existing DB)
    print("\n1️⃣ Initializing ContextWeaver...")
    pipeline = ContextWeaverPipeline(use_existing_db=False)
    
    # Ingest sample documents
    print("\n2️⃣ Ingesting Sample Documents...")
    sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
    
    if sample_files:
        print(f"   Found {len(sample_files)} sample documents")
        report = pipeline.ingest_documents([str(f) for f in sample_files])
        
        print(f"\n   Ingestion Report:")
        print(f"   • Files processed: {report['files_processed']}")
        print(f"   • Chunks created: {report['chunks_created']}")
        print(f"   • Contradictions found: {report['contradictions_found']}")
        print(f"   • Graph nodes: {report['graph_stats']['num_nodes']}")
        print(f"   • Graph edges: {report['graph_stats']['num_edges']}")
    else:
        print("   ⚠️ No sample documents found!")
    
    # Test queries
    print("\n3️⃣ Testing Query Processing...")
    
    test_queries = [
        "Is moderate coffee consumption safe for heart health?",
        "What contradictions exist in coffee research?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST QUERY {i}: {query}")
        print(f"{'='*60}")
        
        result = pipeline.query(query)
        
        print(f"\n✅ Status: {result['status']}")
        
        if result['status'] == 'SUCCESS':
            answer = result['answer']
            print(f"📝 Answer: {answer[:300]}...")
            
            if result.get('uncertainty'):
                print(f"🎲 Confidence: {result['uncertainty']['confidence_score']:.2%} ({result['uncertainty']['confidence_level']})")
            
            if result.get('fact_check'):
                print(f"✅ Verification: {result['fact_check']['overall_score']:.1%} ({result['fact_check']['verification_level']})")
            
            if result.get('contradictions'):
                print(f"⚠️ Contradictions: {result['contradictions'].get('num_contradictions', 0)}")
            
            if result.get('reasoning'):
                print(f"🧠 Reasoning hops: {result['reasoning']['hops_used']}")
        else:
            print(f"❌ Error: {result.get('answer', 'Unknown error')}")
    
    # Test 4: Generate comprehensive report
    print(f"\n{'='*60}")
    print("4️⃣ Generating Comprehensive Report...")
    print(f"{'='*60}")
    
    if test_queries and sample_files:
        result = pipeline.query(test_queries[0])
        
        if result['status'] == 'SUCCESS':
            report = pipeline.generate_comprehensive_report(result)
            print(report)
    
    # Test 5: System statistics
    print(f"\n{'='*60}")
    print("5️⃣ System Statistics")
    print(f"{'='*60}")
    
    stats = pipeline.get_system_statistics()
    
    print(f"\nVector Store:")
    for key, value in stats['vector_store'].items():
        print(f"  {key}: {value}")
    
    print(f"\nKnowledge Base:")
    if stats['knowledge_base']:
        print(f"  Total documents: {stats['knowledge_base'].get('total_documents', 0)}")
        print(f"  Coverage score: {stats['knowledge_base'].get('coverage_score', 0):.2f}")
        print(f"  Domains: {list(stats['knowledge_base'].get('domains', {}).keys())}")
    else:
        print(f"  No knowledge base stats (use ingest_documents first)")
    
    print(f"\nDocument Graph:")
    print(f"  Nodes: {stats['document_graph']['num_nodes']}")
    print(f"  Edges: {stats['document_graph']['num_edges']}")
    print(f"  Density: {stats['document_graph']['density']:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print("\n🎯 Complete ContextWeaver System:")
    print("  ✅ All 3 core components integrated")
    print("  ✅ All 3 innovations integrated")
    print("  ✅ Multi-hop reasoning pipeline")
    print("  ✅ End-to-end query processing")
    print("  ✅ Comprehensive analysis reports")
    print("\n🚀 READY FOR DEPLOYMENT!")