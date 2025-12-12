# scripts/test_day2.py
import sys
sys.path.append('./src')

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager, AdvancedRetriever
from pathlib import Path

def main():
    print("=" * 60)
    print("🧪 DAY 2 INTEGRATION TEST")
    print("=" * 60)
    
    # Step 1: Validate configuration
    print("\n1️⃣ Validating configuration...")
    try:
        Config.validate()
        print("   ✅ Configuration valid")
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return
    
    # Step 2: Check sample documents exist
    print("\n2️⃣ Checking sample documents...")
    sample_dir = Config.SAMPLE_DOCS_DIR
    sample_files = list(sample_dir.glob("*.txt"))
    
    if not sample_files:
        print(f"   ⚠️  No sample documents found in {sample_dir}")
        print("   Creating sample documents...")
        create_sample_documents()
        sample_files = list(sample_dir.glob("*.txt"))
    
    print(f"   ✅ Found {len(sample_files)} sample documents")
    for f in sample_files:
        print(f"      - {f.name}")
    
    # Step 3: Process documents
    print("\n3️⃣ Processing documents...")
    processor = DocumentProcessor()
    file_paths = [str(f) for f in sample_files]
    documents = processor.load_documents(file_paths)
    print(f"   ✅ Processed {len(documents)} chunks")
    
    # Step 4: Display knowledge base statistics
    print("\n4️⃣ Knowledge Base Statistics:")
    processor.knowledge_base.print_statistics()
    
    # Step 5: Create vector store
    print("\n5️⃣ Creating vector store...")
    vector_manager = VectorStoreManager(
        persist_directory=str(Config.CHROMA_DIR),
        collection_name=Config.COLLECTION_NAME
    )
    vectorstore = vector_manager.create_vectorstore(documents)
    
    # Step 6: Test retrieval with multiple queries
    print("\n6️⃣ Testing retrieval with sample queries...")
    test_queries = [
        "Is coffee bad for your heart?",
        "What did the 2022 meta-analysis find?",
        "Why do studies contradict each other about coffee?"
    ]
    
    for query in test_queries:
        print(f"\n   🔍 Query: '{query}'")
        results = vector_manager.similarity_search_with_score(query, k=2)
        for i, (doc, score) in enumerate(results, 1):
            print(f"      Result #{i} (Score: {score:.3f})")
            print(f"         Source: {doc.metadata.get('filename', doc.metadata.get('source', 'N/A'))}")
            print(f"         Year: {doc.metadata.get('year', 'N/A')}")
            print(f"         Domain: {doc.metadata.get('domain', 'N/A')}")
            print(f"         Snippet: {doc.page_content[:120]}...")
    
    # Step 7: Test advanced retrieval
    print("\n7️⃣ Testing advanced retrieval & ranking...")
    retriever = AdvancedRetriever(vector_manager)
    
    query = "coffee and heart health"
    print(f"   Query: '{query}'")
    results = vector_manager.similarity_search(query, k=5)
    ranked_results = retriever.rerank_documents(query, results)
    
    print(f"\n   Multi-factor ranked results:")
    for i, (doc, final_score, scores) in enumerate(ranked_results[:3], 1):
        print(f"\n      Result #{i} (Final Score: {final_score:.3f})")
        print(f"         Source: {doc.metadata.get('filename', Path(doc.metadata.get('source', 'N/A')).name)}")
        print(f"         Similarity: {scores['similarity']:.3f}")
        print(f"         Credibility: {scores['credibility']:.3f}")
        print(f"         Recency: {scores['recency']:.3f}")
        print(f"         Quality: {scores['quality']:.3f}")
    
    # Step 8: Test filtering
    print("\n8️⃣ Testing document filtering...")
    
    # Filter by year
    recent_docs = retriever.filter_documents(
        results,
        {'year_min': 2022}
    )
    print(f"   Documents from 2022+: {len(recent_docs)}")
    
    # Filter by credibility
    high_quality = retriever.filter_documents(
        results,
        {'min_quality': 0.8}
    )
    print(f"   High quality documents (>0.8): {len(high_quality)}")
    
    # Step 9: Vector store statistics
    print("\n9️⃣ Final Vector Store Statistics:")
    stats = vector_manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ DAY 2 COMPLETE - All systems operational!")
    print("=" * 60)
    print("\n🎯 What you've built today:")
    print("  ✅ Document processing with 4 chunking strategies")
    print("  ✅ Knowledge base with hierarchical organization")
    print("  ✅ Vector store with ChromaDB + OpenAI embeddings")
    print("  ✅ Advanced retrieval with multi-factor ranking")
    print("  ✅ Document filtering by metadata")
    print("\n📊 RAG Component Coverage:")
    print("  ✅ Build knowledge base for domain")
    print("  ✅ Implement vector storage and retrieval")
    print("  ✅ Design relevant document chunking strategies")
    print("  ✅ Create effective ranking and filtering mechanisms")
    print("\n🚀 Ready for Day 3: Prompt Engineering!")

def create_sample_documents():
    """Create sample documents if they don't exist"""
    sample_dir = Config.SAMPLE_DOCS_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    samples = {
        'coffee_study_2018.txt': """Title: Coffee and Cardiovascular Risk Study (2018)

Abstract: This study examines the relationship between coffee consumption
and cardiovascular disease risk. We analyzed data from 10,000 participants
over a 5-year period.

Results: Our findings indicate that high coffee consumption (>5 cups/day)
is associated with a 23% increased risk of cardiovascular events.

Limitations: This study did not control for confounding variables such as
smoking, sugar intake, or exercise habits.

Conclusion: High coffee consumption may increase cardiovascular risk.""",

        'coffee_study_2023.txt': """Title: Protective Effects of Moderate Coffee Intake (2023)

Abstract: Recent research suggests that moderate coffee consumption may
have protective cardiovascular effects. This study examines 15,000 participants
with rigorous controls for lifestyle factors.

Methodology: Unlike previous studies, we controlled for smoking, diet,
exercise, and sugar intake to isolate coffee's effects.

Results: Moderate coffee consumption (2-3 cups/day) shows a 15% reduction
in cardiovascular disease risk. Heavy consumption (>5 cups) shows neutral effects.

Conclusion: When confounding factors are controlled, moderate coffee intake
appears beneficial for heart health.""",

        'meta_analysis_2022.txt': """Title: Meta-Analysis of Coffee and Heart Health Studies (2022)

Abstract: This meta-analysis reviews 50 studies published between 2010-2022
examining coffee's effects on cardiovascular health.

Key Findings:
- Early studies (2010-2019) showed mixed results
- Studies failing to control for confounders showed negative effects
- Recent studies with better controls show neutral or positive effects
- The apparent contradiction stems from methodological differences

Explanation: Previous negative findings were likely due to confounding
variables. Coffee itself does not increase cardiovascular risk when
consumed moderately.

Recommendation: Current evidence supports moderate coffee consumption
as safe or potentially beneficial for heart health."""
    }
    
    for filename, content in samples.items():
        filepath = sample_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
    
    print(f"   ✅ Created {len(samples)} sample documents in {sample_dir}")

if __name__ == "__main__":
    main()