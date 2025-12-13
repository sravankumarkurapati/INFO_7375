# tests/test_all_components.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager, AdvancedRetriever
from prompt_engineering import PromptEngineeringSystem
from reasoning_engine import MultiHopReasoningEngine, ContradictionDetector
from uncertainty_quantification import UncertaintyQuantifier
from fact_checker import AutomatedFactChecker
from document_graph import DocumentGraph
from web_search_fallback import WebSearchFallback, HybridRetriever
from synthetic_data_generator import SyntheticDataGenerator, QualityChecker, DiversityChecker
from contextweaver_pipeline import ContextWeaverPipeline

import json
import time
from datetime import datetime
from langchain_core.documents import Document

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def save_results(results, filename="test_results.json"):
    """Save test results to file"""
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filepath}")
    return filepath

# ==================== TEST SUITE ====================

def test_1_configuration():
    """Test 1: Configuration Validation"""
    print_section("TEST 1: Configuration Validation")
    
    try:
        Config.validate()
        print("✅ Configuration valid")
        
        summary = Config.get_summary()
        print("\n📊 Configuration Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        return {"status": "PASS", "config": summary}
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_2_document_processing():
    """Test 2: Document Processing"""
    print_section("TEST 2: Document Processing")
    
    try:
        processor = DocumentProcessor()
        
        # Get sample files
        sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
        print(f"\n📚 Found {len(sample_files)} sample documents:")
        for f in sample_files:
            print(f"   - {f.name}")
        
        if not sample_files:
            print("❌ No sample documents found!")
            return {"status": "FAIL", "error": "No sample documents"}
        
        # Process documents
        print("\n⚙️  Processing documents...")
        start_time = time.time()
        documents = processor.load_documents([str(f) for f in sample_files])
        process_time = time.time() - start_time
        
        print(f"✅ Processed {len(documents)} chunks in {process_time:.2f}s")
        
        # Knowledge base stats
        print("\n📊 Knowledge Base Statistics:")
        processor.knowledge_base.print_statistics()
        
        return {
            "status": "PASS",
            "num_files": len(sample_files),
            "num_chunks": len(documents),
            "processing_time": process_time,
            "kb_stats": processor.knowledge_base.statistics
        }
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_3_vector_store():
    """Test 3: Vector Store & Embeddings"""
    print_section("TEST 3: Vector Store & Embeddings")
    
    try:
        # Load documents
        processor = DocumentProcessor()
        sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
        documents = processor.load_documents([str(f) for f in sample_files])
        
        # Create vector store
        print("\n🗄️  Creating vector store...")
        vector_store = VectorStoreManager(
            persist_directory=str(Config.CHROMA_DIR),
            collection_name=Config.COLLECTION_NAME
        )
        
        start_time = time.time()
        vectorstore = vector_store.create_vectorstore(documents)
        embedding_time = time.time() - start_time
        
        print(f"✅ Vector store created in {embedding_time:.2f}s")
        
        # Test similarity search
        print("\n🔍 Testing similarity search...")
        test_query = "Is coffee safe for heart health?"
        print(f"   Query: '{test_query}'")
        
        results = vector_store.similarity_search_with_score(test_query, k=3)
        
        print(f"\n📄 Top 3 Results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n   Result #{i} (Similarity: {1-score:.3f})")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
            print(f"   Content: {doc.page_content[:100]}...")
        
        stats = vector_store.get_statistics()
        
        return {
            "status": "PASS",
            "embedding_time": embedding_time,
            "num_chunks": stats['total_chunks'],
            "test_query": test_query,
            "top_3_similarities": [1-score for doc, score in results]
        }
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_4_multi_factor_ranking():
    """Test 4: Advanced Multi-Factor Ranking"""
    print_section("TEST 4: Multi-Factor Ranking")
    
    try:
        # Setup
        processor = DocumentProcessor()
        sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
        documents = processor.load_documents([str(f) for f in sample_files])
        
        vector_store = VectorStoreManager(
            persist_directory=str(Config.CHROMA_DIR),
            collection_name=Config.COLLECTION_NAME
        )
        vector_store.create_vectorstore(documents)
        
        # Test ranking
        retriever = AdvancedRetriever(vector_store)
        query = "Coffee and cardiovascular health"
        
        print(f"\n🔍 Query: '{query}'")
        print("\n📊 Applying multi-factor ranking...")
        
        # Get documents
        search_results = vector_store.similarity_search_with_score(query, k=5)
        docs_with_sim = []
        for doc, score in search_results:
            doc.metadata['similarity_score'] = 1 - score
            docs_with_sim.append(doc)
        
        # Re-rank
        ranked = retriever.rerank_documents(query, docs_with_sim)
        
        print("\n🏆 Ranked Results:")
        for i, (doc, final_score, component_scores) in enumerate(ranked[:3], 1):
            print(f"\n   #{i} - Final Score: {final_score:.3f}")
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
            print(f"   Component Scores:")
            for component, score in component_scores.items():
                print(f"      {component}: {score:.3f}")
        
        return {
            "status": "PASS",
            "query": query,
            "num_ranked": len(ranked),
            "top_3_scores": [score for doc, score, _ in ranked[:3]],
            "ranking_weights": Config.RANKING_WEIGHTS
        }
    except Exception as e:
        print(f"❌ Ranking error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_5_multi_hop_reasoning():
    """Test 5: Multi-Hop Reasoning"""
    print_section("TEST 5: Multi-Hop Reasoning")
    
    try:
        # Create sample documents
        docs = [
            Document(
                page_content="2018 study found high coffee consumption increases cardiovascular risk.",
                metadata={'source': 'study_2018.txt', 'year': 2018, 'credibility_score': 0.7}
            ),
            Document(
                page_content="2022 meta-analysis found that confounders explain contradictions in coffee research.",
                metadata={'source': 'meta_2022.txt', 'year': 2022, 'credibility_score': 0.95}
            ),
            Document(
                page_content="2023 research shows moderate coffee consumption beneficial for heart health.",
                metadata={'source': 'study_2023.txt', 'year': 2023, 'credibility_score': 0.9}
            )
        ]
        
        print("\n🧠 Testing multi-hop reasoning...")
        query = "Is moderate coffee consumption safe for heart health?"
        print(f"   Query: '{query}'")
        
        engine = MultiHopReasoningEngine()
        
        start_time = time.time()
        result = engine.reason_across_documents(query, docs, max_hops=3)
        reasoning_time = time.time() - start_time
        
        print(f"\n✅ Reasoning complete in {reasoning_time:.2f}s")
        print(f"   Hops used: {result['hops_used']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        print("\n🔗 Reasoning Chain:")
        for step in result['reasoning_chain']:
            hop = step['hop_number']
            
            # FIX: Convert to string before slicing
            extracted_info = str(step.get('extracted_info', 'N/A'))
            conclusion = str(step.get('intermediate_conclusion', 'N/A'))
            
            print(f"\n   Hop {hop}:")
            print(f"   ├─ Info: {extracted_info[:100]}...")
            print(f"   └─ Conclusion: {conclusion[:100]}...")
        
        return {
            "status": "PASS",
            "query": query,
            "hops_used": result['hops_used'],
            "confidence": result['confidence'],
            "reasoning_time": reasoning_time,
            "answer": result['answer'][:200] if result.get('answer') else "N/A"
        }
    except Exception as e:
        print(f"❌ Multi-hop reasoning error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "FAIL", "error": str(e)}

def test_6_contradiction_detection():
    """Test 6: Contradiction Detection"""
    print_section("TEST 6: Contradiction Detection")
    
    try:
        docs = [
            Document(
                page_content="2018 study: High coffee consumption increases cardiovascular risk by 23%.",
                metadata={'source': 'study_2018.txt', 'year': 2018}
            ),
            Document(
                page_content="2023 study: Moderate coffee consumption shows 15% reduction in cardiovascular risk.",
                metadata={'source': 'study_2023.txt', 'year': 2023}
            )
        ]
        
        print("\n🔍 Detecting contradictions...")
        detector = ContradictionDetector()
        
        start_time = time.time()
        result = detector.detect_contradictions(docs)
        detection_time = time.time() - start_time
        
        print(f"\n✅ Detection complete in {detection_time:.2f}s")
        print(f"   Contradictions found: {result.get('num_contradictions', 0)}")
        print(f"   Overall severity: {result.get('overall_severity', 'NONE')}")
        
        if result.get('contradictions'):
            for i, c in enumerate(result['contradictions'], 1):
                print(f"\n   Contradiction #{i}:")
                print(f"   ├─ Severity: {c.get('severity', 'N/A')}")
                
                # FIX: Safely handle claim text
                claim_a = str(c.get('claim_A', 'N/A'))
                claim_b = str(c.get('claim_B', 'N/A'))
                
                print(f"   ├─ Claim A: {claim_a[:80]}...")
                print(f"   ├─ Claim B: {claim_b[:80]}...")
                print(f"   └─ Confidence: {c.get('confidence', 0):.2%}")
        
        return {
            "status": "PASS",
            "num_contradictions": result.get('num_contradictions', 0),
            "severity": result.get('overall_severity', 'NONE'),
            "detection_time": detection_time
        }
    except Exception as e:
        print(f"❌ Contradiction detection error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_7_uncertainty_quantification():
    """Test 7: Uncertainty Quantification"""
    print_section("TEST 7: Uncertainty Quantification")
    
    try:
        supporting_docs = [
            Document(
                page_content="2023 study shows moderate coffee beneficial.",
                metadata={'credibility_score': 0.9, 'quality_score': 0.85, 'year': 2023}
            ),
            Document(
                page_content="2022 meta-analysis confirms moderate coffee is safe.",
                metadata={'credibility_score': 0.95, 'quality_score': 0.95, 'year': 2022}
            )
        ]
        
        contradicting_docs = [
            Document(
                page_content="2018 study found high coffee increases risk.",
                metadata={'credibility_score': 0.7, 'year': 2018}
            )
        ]
        
        print("\n🎲 Quantifying uncertainty...")
        quantifier = UncertaintyQuantifier()
        
        query = "Is moderate coffee safe?"
        answer = "Yes, moderate coffee is safe and potentially beneficial."
        
        start_time = time.time()
        result = quantifier.quantify_uncertainty(
            query, answer, supporting_docs, contradicting_docs
        )
        uncertainty_time = time.time() - start_time
        
        print(f"\n✅ Uncertainty quantified in {uncertainty_time:.2f}s")
        print(f"   Confidence: {result['confidence_score']:.2%}")
        print(f"   Level: {result['confidence_level']}")
        
        print("\n📊 Component Scores:")
        for component, score in result['component_scores'].items():
            print(f"   {component}: {score:.2%}")
        
        print("\n⚠️  Uncertainty Sources:")
        for source in result['uncertainty_sources']:
            print(f"   • {source}")
        
        return {
            "status": "PASS",
            "confidence_score": result['confidence_score'],
            "confidence_level": result['confidence_level'],
            "component_scores": result['component_scores'],
            "uncertainty_time": uncertainty_time
        }
    except Exception as e:
        print(f"❌ Uncertainty quantification error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_8_fact_checking():
    """Test 8: Automated Fact-Checking"""
    print_section("TEST 8: Automated Fact-Checking")
    
    try:
        source_docs = [
            Document(
                page_content="2023 study shows 15% reduction in cardiovascular risk with moderate coffee.",
                metadata={'source': 'study_2023.txt', 'year': 2023}
            ),
            Document(
                page_content="Meta-analysis confirms moderate coffee is safe for heart health.",
                metadata={'source': 'meta_2022.txt', 'year': 2022}
            )
        ]
        
        answer = "Moderate coffee consumption shows a 15% reduction in cardiovascular risk according to recent research."
        
        print(f"\n✅ Fact-checking answer...")
        print(f"   Answer: {answer[:80]}...")
        
        fact_checker = AutomatedFactChecker()
        
        start_time = time.time()
        result = fact_checker.fact_check_answer(answer, source_docs)
        fact_check_time = time.time() - start_time
        
        print(f"\n✅ Fact-check complete in {fact_check_time:.2f}s")
        print(f"   Overall Score: {result['overall_score']:.2%}")
        print(f"   Verification Level: {result['verification_level']}")
        print(f"   Verified: {result['num_verified']}/{len(result['claims'])}")
        
        return {
            "status": "PASS",
            "overall_score": result['overall_score'],
            "verification_level": result['verification_level'],
            "num_verified": result['num_verified'],
            "num_claims": len(result['claims']),
            "fact_check_time": fact_check_time
        }
    except Exception as e:
        print(f"❌ Fact-checking error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_9_knowledge_graph():
    """Test 9: Knowledge Graph Construction"""
    print_section("TEST 9: Knowledge Graph Construction")
    
    try:
        docs = [
            Document(
                page_content="2018 study on coffee and cardiovascular risk.",
                metadata={'source': 'study_2018.txt', 'year': 2018, 'domain': 'medical', 
                         'credibility_score': 0.7, 'quality_score': 0.8}
            ),
            Document(
                page_content="2022 meta-analysis reviewing coffee studies.",
                metadata={'source': 'meta_2022.txt', 'year': 2022, 'domain': 'research',
                         'credibility_score': 0.95, 'quality_score': 0.95}
            ),
            Document(
                page_content="2023 research on moderate coffee benefits.",
                metadata={'source': 'study_2023.txt', 'year': 2023, 'domain': 'medical',
                         'credibility_score': 0.9, 'quality_score': 0.9}
            )
        ]
        
        print("\n🕸️  Building knowledge graph...")
        graph = DocumentGraph()
        
        start_time = time.time()
        graph.build_graph(docs, contradictions=[], temporal_order=True)
        graph_time = time.time() - start_time
        
        stats = graph.get_graph_statistics()
        
        print(f"\n✅ Graph built in {graph_time:.2f}s")
        print(f"   Nodes: {stats['num_nodes']}")
        print(f"   Edges: {stats['num_edges']}")
        print(f"   Density: {stats['density']:.3f}")
        
        # PageRank
        print("\n📊 Calculating PageRank importance...")
        importance = graph.calculate_document_importance()
        
        print("\n🏆 Top Documents by Importance:")
        for i, (doc_id, score) in enumerate(list(importance.items())[:3], 1):
            source = graph.graph.nodes[doc_id].get('source', 'Unknown')
            print(f"   #{i}. {source}: {score:.3f}")
        
        return {
            "status": "PASS",
            "num_nodes": stats['num_nodes'],
            "num_edges": stats['num_edges'],
            "density": stats['density'],
            "graph_time": graph_time,
            "top_3_importance": list(importance.values())[:3]
        }
    except Exception as e:
        print(f"❌ Knowledge graph error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_10_hybrid_retrieval():
    """Test 10: Hybrid Retrieval (Local + Web + LLM)"""
    print_section("TEST 10: Hybrid Retrieval")
    
    try:
        # Setup vector store with sample docs
        processor = DocumentProcessor()
        sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
        documents = processor.load_documents([str(f) for f in sample_files])
        
        vector_store = VectorStoreManager(
            persist_directory=str(Config.CHROMA_DIR),
            collection_name=Config.COLLECTION_NAME
        )
        vector_store.create_vectorstore(documents)
        
        # Setup hybrid retriever
        web_search = WebSearchFallback()
        hybrid = HybridRetriever(vector_store, web_search)
        
        # Test 1: In-domain query (should use local)
        print("\n📚 Test 1: In-domain query (coffee)")
        query1 = "Is coffee safe for heart health?"
        print(f"   Query: '{query1}'")
        
        result1 = hybrid.retrieve(query1, k=5)
        print(f"   ✅ Source: {result1['source'].upper()}")
        print(f"   Documents: {len(result1['documents'])}")
        print(f"   Confidence: {result1['confidence']:.2%}")
        
        # Test 2: Out-of-domain query (should fallback to web)
        print("\n🌐 Test 2: Out-of-domain query (chicken)")
        query2 = "Is chicken healthy?"
        print(f"   Query: '{query2}'")
        
        result2 = hybrid.retrieve(query2, k=5)
        print(f"   ✅ Source: {result2['source'].upper()}")
        print(f"   Documents: {len(result2['documents'])}")
        print(f"   Confidence: {result2['confidence']:.2%}")
        
        return {
            "status": "PASS",
            "test1": {
                "query": query1,
                "source": result1['source'],
                "confidence": result1['confidence']
            },
            "test2": {
                "query": query2,
                "source": result2['source'],
                "confidence": result2['confidence']
            }
        }
    except Exception as e:
        print(f"❌ Hybrid retrieval error: {e}")
        return {"status": "FAIL", "error": str(e)}

def test_11_full_pipeline():
    """Test 11: Complete ContextWeaver Pipeline"""
    print_section("TEST 11: Full Pipeline Integration")
    
    try:
        print("\n🚀 Initializing complete pipeline...")
        pipeline = ContextWeaverPipeline(use_existing_db=False)
        
        # Ingest documents
        print("\n📚 Ingesting sample documents...")
        sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
        
        start_time = time.time()
        report = pipeline.ingest_documents([str(f) for f in sample_files])
        ingest_time = time.time() - start_time
        
        print(f"✅ Ingestion complete in {ingest_time:.2f}s")
        print(f"   Files processed: {report['files_processed']}")
        print(f"   Chunks created: {report['chunks_created']}")
        print(f"   Graph nodes: {report['graph_stats']['num_nodes']}")
        print(f"   Contradictions: {report['contradictions_found']}")
        
        # Test query with all features
        print("\n🔍 Testing full pipeline query...")
        query = "Is moderate coffee consumption safe for heart health?"
        print(f"   Query: '{query}'")
        
        start_time = time.time()
        result = pipeline.query(
            query,
            enable_multi_hop=True,
            enable_contradiction_detection=True,
            enable_uncertainty=True,
            enable_fact_checking=True
        )
        query_time = time.time() - start_time
        
        print(f"\n✅ Query complete in {query_time:.2f}s")
        print(f"   Retrieval source: {result['retrieval']['retrieval_source'].upper()}")
        print(f"   Documents used: {result['retrieval']['num_documents']}")
        
        if result.get('reasoning'):
            print(f"   Hops used: {result['reasoning']['hops_used']}")
            print(f"   Reasoning confidence: {result['reasoning']['confidence']:.2%}")
        
        if result.get('uncertainty'):
            print(f"   Overall confidence: {result['uncertainty']['confidence_score']:.2%}")
        
        if result.get('fact_check'):
            print(f"   Fact-check score: {result['fact_check']['overall_score']:.2%}")
        
        # FIX: Safely extract answer
        answer_text = result.get('raw_answer', result.get('answer', 'N/A'))
        print(f"\n📝 Answer (first 200 chars):")
        print(f"   {str(answer_text)[:200]}...")
        
        return {
            "status": "PASS",
            "ingestion": {
                "files": report['files_processed'],
                "chunks": report['chunks_created'],
                "time": ingest_time
            },
            "query": {
                "query": query,
                "retrieval_source": result['retrieval']['retrieval_source'],
                "num_documents": result['retrieval']['num_documents'],
                "query_time": query_time
            },
            "reasoning": {
                "hops": result.get('reasoning', {}).get('hops_used', 0),
                "confidence": result.get('reasoning', {}).get('confidence', 0)
            },
            "uncertainty": {
                "confidence": result.get('uncertainty', {}).get('confidence_score', 0),
                "level": result.get('uncertainty', {}).get('confidence_level', 'N/A')
            },
            "fact_check": {
                "score": result.get('fact_check', {}).get('overall_score', 0),
                "verified": result.get('fact_check', {}).get('num_verified', 0)
            }
        }
    except Exception as e:
        print(f"❌ Full pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "FAIL", "error": str(e)}

def test_12_synthetic_data():
    """Test 12: Synthetic Data Generation"""
    print_section("TEST 12: Synthetic Data Generation")
    
    try:
        # Load sample documents
        processor = DocumentProcessor()
        sample_files = list(Config.SAMPLE_DOCS_DIR.glob("*.txt"))
        documents = processor.load_documents([str(f) for f in sample_files])
        
        print("\n🧬 Testing synthetic data generation...")
        generator = SyntheticDataGenerator()
        
        # Generate Q&A pairs
        print("\n📝 Generating Q&A pairs...")
        start_time = time.time()
        qa_pairs = generator.generate_qa_pairs(
            documents[:3],  # Use first 3 docs
            num_pairs=3,
            difficulty_levels=['easy', 'medium', 'hard']
        )
        qa_time = time.time() - start_time
        
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs in {qa_time:.2f}s")
        
        if qa_pairs:
            print("\n📄 Sample Q&A Pair:")
            sample = qa_pairs[0]
            
            # FIX: Safely handle Q&A text
            question = str(sample.get('question', 'N/A'))
            answer = str(sample.get('answer', 'N/A'))
            
            print(f"   Q: {question[:100]}...")
            print(f"   A: {answer[:100]}...")
            print(f"   Difficulty: {sample.get('difficulty', 'N/A')}")
        
        # Quality check - FIX: Use correct imports
        quality_checker = QualityChecker()
        quality_metrics = quality_checker.calculate_dataset_quality(qa_pairs)
        
        print(f"\n📊 Quality Metrics:")
        print(f"   Overall Quality: {quality_metrics['overall_quality']:.2%}")
        print(f"   High Quality Ratio: {quality_metrics['high_quality_ratio']:.2%}")
        
        diversity_checker = DiversityChecker()
        diversity_metrics = diversity_checker.calculate_diversity(qa_pairs)
        
        print(f"\n🌈 Diversity Metrics:")
        print(f"   Overall Diversity: {diversity_metrics['overall_diversity']:.2%}")
        print(f"   Lexical Diversity: {diversity_metrics['lexical_diversity']:.2%}")
        
        return {
            "status": "PASS",
            "num_qa_pairs": len(qa_pairs),
            "generation_time": qa_time,
            "quality": quality_metrics['overall_quality'],
            "diversity": diversity_metrics['overall_diversity']
        }
    except Exception as e:
        print(f"❌ Synthetic data error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "FAIL", "error": str(e)}

# ==================== MAIN TEST RUNNER ====================

def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "="*70)
    print("  🧪 CONTEXTWEAVER COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    # Run all tests
    test_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "project": "ContextWeaver",
            "version": "1.0.0"
        },
        "tests": {}
    }
    
    tests = [
        ("Configuration", test_1_configuration),
        ("Document Processing", test_2_document_processing),
        ("Vector Store", test_3_vector_store),
        ("Multi-Factor Ranking", test_4_multi_factor_ranking),
        ("Multi-Hop Reasoning", test_5_multi_hop_reasoning),
        ("Contradiction Detection", test_6_contradiction_detection),
        ("Uncertainty Quantification", test_7_uncertainty_quantification),
        ("Fact-Checking", test_8_fact_checking),
        ("Knowledge Graph", test_9_knowledge_graph),
        ("Hybrid Retrieval", test_10_hybrid_retrieval),
        ("Full Pipeline", test_11_full_pipeline),
        ("Synthetic Data", test_12_synthetic_data)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results["tests"][test_name] = result
            
            if result["status"] == "PASS":
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            test_results["tests"][test_name] = {
                "status": "CRASH",
                "error": str(e)
            }
            failed += 1
    
    overall_time = time.time() - overall_start
    
    # Summary
    print_section("TEST SUMMARY")
    print(f"\n✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print(f"⏱️  Total Time: {overall_time:.2f}s")
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results["summary"] = {
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(tests),
        "total_time": overall_time
    }
    
    # Save results
    filepath = save_results(test_results, "comprehensive_test_results.json")
    
    print("\n" + "="*70)
    print("  ✅ TEST SUITE COMPLETE!")
    print("="*70)
    
    return test_results

if __name__ == "__main__":
    results = run_all_tests()