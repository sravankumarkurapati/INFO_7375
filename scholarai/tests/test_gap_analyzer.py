"""
Test Research Gap Analyzer (Custom Tool)
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.paper_hunter import PaperHunterAgent
from agents.content_analyzer import ContentAnalyzerAgent
from agents.research_synthesizer import ResearchSynthesizerAgent

def test_gap_analyzer():
    """Test the complete custom tool pipeline"""
    
    print("\n🔬 Testing Research Gap Analyzer (Custom Tool)...")
    print("=" * 60)
    
    # Step 1: Get papers
    print("\n1️⃣ Getting papers...")
    hunter = PaperHunterAgent()
    paper_results = hunter.search_papers("deep learning applications")
    
    if not paper_results['success']:
        print("❌ Failed to get papers")
        return False
    
    papers = paper_results['papers']
    print(f"✅ Got {len(papers)} papers")
    
    # Step 2: Analyze papers
    print("\n2️⃣ Analyzing papers...")
    analyzer = ContentAnalyzerAgent()
    analysis_results = analyzer.analyze_papers(papers)
    
    analyses = analysis_results['analyses']
    print(f"✅ Analyzed {len(analyses)} papers")
    
    # Step 3: Synthesize with custom tool
    print("\n3️⃣ Running Research Gap Analyzer (Custom Tool)...")
    print("   ⏳ This may take 30-60 seconds...\n")
    
    synthesizer = ResearchSynthesizerAgent()
    synthesis_results = synthesizer.synthesize_research(papers, analyses)
    
    if not synthesis_results['success']:
        print("❌ Synthesis failed")
        return False
    
    print(f"✅ Synthesis complete!")
    
    # Step 4: Display results
    print(f"\n4️⃣ Research Gap Analysis Results:")
    print("=" * 60)
    
    # Statistics
    stats = synthesis_results['statistics']
    print(f"\n📊 Statistics:")
    print(f"   Papers analyzed: {stats['total_papers']}")
    print(f"   Research clusters: {stats['num_clusters']}")
    print(f"   Gaps identified: {stats['num_gaps']}")
    print(f"   Contradictions found: {stats['num_contradictions']}")
    
    # Clusters
    print(f"\n🔍 Research Clusters:")
    for cluster in synthesis_results['clusters'][:5]:
        if cluster['cluster_id'] != -1:
            print(f"   • {cluster['theme']} ({cluster['size']} papers)")
    
    # Research Gaps
    print(f"\n💡 Research Gaps:")
    for gap in synthesis_results['research_gaps'][:5]:
        print(f"\n   Gap #{gap['gap_id']}: {gap['title']}")
        print(f"      Type: {gap['type']}")
        print(f"      Confidence: {gap['confidence']:.2f}")
        print(f"      Impact: {gap['impact']}")
        print(f"      Description: {gap['description'][:80]}...")
    
    # Trends
    print(f"\n📈 Research Trends:")
    trends = synthesis_results['trends']
    if trends.get('growing_trends'):
        print(f"   Growing Areas:")
        for trend in trends['growing_trends'][:3]:
            print(f"      • {trend['term']} (momentum: {trend['momentum']})")
    
    # Recommendations
    print(f"\n🎯 Research Recommendations:")
    for rec in synthesis_results['recommendations'][:3]:
        print(f"\n   {rec['recommendation_id']}. {rec['title']}")
        print(f"      Priority: {rec['priority']}")
        print(f"      {rec['description'][:80]}...")
    
    # Visualizations
    print(f"\n📊 Visualizations Created:")
    for viz_name, viz_path in synthesis_results['visualizations'].items():
        print(f"   ✅ {viz_name}: {viz_path}")
    
    print("\n" + "=" * 60)
    print("✅ RESEARCH GAP ANALYZER TEST SUCCESSFUL!")
    print("=" * 60)
    print("\n💎 This is your CUSTOM TOOL - the competitive advantage!")
    
    # Save results
    output_file = Path("outputs/gap_analysis_results.json")
    
    # Make serializable
    save_results = {
        'research_gaps': synthesis_results['research_gaps'],
        'contradictions': synthesis_results['contradictions'],
        'trends': {
            k: v for k, v in synthesis_results['trends'].items() 
            if k != 'publication_timeline'  # Skip for now
        },
        'clusters': synthesis_results['clusters'],
        'recommendations': synthesis_results['recommendations'],
        'statistics': synthesis_results['statistics']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_gap_analyzer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)