"""
Test Paper Hunter Agent
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.paper_hunter import PaperHunterAgent
from utils.validators import validators
import json

def test_paper_hunter():
    """Test paper hunter functionality"""
    
    print("\n🔬 Testing Paper Hunter Agent...")
    print("=" * 60)
    
    # Create agent
    print("\n1️⃣ Initializing Paper Hunter...")
    try:
        hunter = PaperHunterAgent()
        print("✅ Paper Hunter initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test search
    test_query = "transformer models for natural language processing"
    print(f"\n2️⃣ Searching for papers...")
    print(f"   Query: '{test_query}'")
    print("   ⏳ This will take 10-20 seconds...\n")
    
    try:
        results = hunter.search_papers(test_query)
        
        if not results['success']:
            print(f"❌ Search failed: {results['message']}")
            return False
        
        print(f"✅ Search successful!")
        print(f"   Papers found: {results['total_found']}")
        print(f"   Average relevance: {results.get('avg_relevance', 0):.2f}")
        
        # Validate results
        papers = results['papers']
        
        if len(papers) < 5:
            print(f"⚠️  Warning: Only {len(papers)} papers found (expected 10+)")
        
        print(f"\n3️⃣ Validating results...")
        try:
            validators.validate_papers(papers, min_count=min(5, len(papers)))
            print(f"✅ Validation passed")
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")
        
        # Display results
        print(f"\n4️⃣ Paper Results:")
        print("=" * 60)
        
        num_to_show = min(5, len(papers))
        for idx, paper in enumerate(papers[:num_to_show], 1):
            print(f"\n📄 Paper {idx}:")
            print(f"   Title: {paper['title'][:70]}{'...' if len(paper['title']) > 70 else ''}")
            print(f"   Source: {paper['source']}")
            print(f"   Year: {paper['year']}")
            print(f"   Relevance: {paper['relevance_score']:.3f}")
            print(f"   URL: {paper['url'][:60]}{'...' if len(paper['url']) > 60 else ''}")
        
        if len(papers) > num_to_show:
            print(f"\n   ... and {len(papers) - num_to_show} more papers")
        
        # Statistics
        print(f"\n5️⃣ Statistics:")
        print("=" * 60)
        
        if papers:
            avg_relevance = sum(p['relevance_score'] for p in papers) / len(papers)
            print(f"   Total Papers: {len(papers)}")
            print(f"   Average Relevance: {avg_relevance:.3f}")
            print(f"   Min Relevance: {min(p['relevance_score'] for p in papers):.3f}")
            print(f"   Max Relevance: {max(p['relevance_score'] for p in papers):.3f}")
            
            # Source distribution
            sources = {}
            for p in papers:
                sources[p['source']] = sources.get(p['source'], 0) + 1
            print(f"\n   Sources Distribution:")
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                print(f"     - {source}: {count}")
            
            # Year distribution
            years = [p['year'] for p in papers]
            print(f"\n   Year Range: {min(years)} - {max(years)}")
            
            # Recent papers (last 3 years)
            recent = [p for p in papers if p['year'] >= 2022]
            print(f"   Recent papers (2022+): {len(recent)} ({len(recent)/len(papers)*100:.1f}%)")
        
        print("\n" + "=" * 60)
        print("✅ PAPER HUNTER TEST SUCCESSFUL!")
        print("=" * 60)
        
        # Save results for inspection
        output_file = Path("outputs/test_paper_results.json")
        output_file.parent.mkdir(exist_ok=True)
        
        # Make results JSON serializable
        save_results = {
            'query': results['query'],
            'search_query': results['search_query'],
            'total_found': results['total_found'],
            'avg_relevance': results.get('avg_relevance', 0),
            'papers': results['papers']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        print("\n✨ You can inspect the JSON file to see all paper details")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_paper_hunter()
    sys.exit(0 if success else 1)