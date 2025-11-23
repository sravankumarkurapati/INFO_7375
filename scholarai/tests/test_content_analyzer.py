"""
Test Content Analyzer Agent
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.content_analyzer import ContentAnalyzerAgent
from agents.paper_hunter import PaperHunterAgent

def test_content_analyzer():
    """Test content analyzer functionality"""
    
    print("\n🔬 Testing Content Analyzer Agent...")
    print("=" * 60)
    
    # Step 1: Get some papers first
    print("\n1️⃣ Getting papers for analysis...")
    hunter = PaperHunterAgent()
    paper_results = hunter.search_papers("transformer models nlp")
    
    if not paper_results['success']:
        print("❌ Failed to get papers")
        return False
    
    papers = paper_results['papers'][:5]  # Analyze first 5
    print(f"✅ Got {len(papers)} papers")
    
    # Step 2: Initialize analyzer
    print("\n2️⃣ Initializing Content Analyzer...")
    analyzer = ContentAnalyzerAgent()
    print("✅ Analyzer initialized")
    
    # Step 3: Analyze papers
    print(f"\n3️⃣ Analyzing {len(papers)} papers...")
    print("   ⏳ This may take 30-60 seconds...\n")
    
    analysis_results = analyzer.analyze_papers(papers)
    
    if not analysis_results['success']:
        print("❌ Analysis failed")
        return False
    
    print(f"✅ Analysis complete!")
    print(f"   Papers analyzed: {analysis_results['statistics']['analyzed']}")
    print(f"   Failed: {analysis_results['statistics']['failed']}")
    
    # Step 4: Display results
    print(f"\n4️⃣ Analysis Results:")
    print("=" * 60)
    
    analyses = analysis_results['analyses']
    for idx, analysis in enumerate(analyses[:3], 1):  # Show first 3
        print(f"\n📄 Paper {idx}: {analysis['title'][:60]}...")
        print(f"   Methodology: {analysis['methodology']['type']} - {analysis['methodology']['approach']}")
        print(f"   Key Findings:")
        for finding in analysis['key_findings'][:2]:
            print(f"     • {finding[:80]}...")
        print(f"   Technical Terms: {', '.join(analysis['technical_terms'][:5])}")
        print(f"   Limitations: {analysis['limitations'][0]}")
    
    if len(analyses) > 3:
        print(f"\n   ... and {len(analyses) - 3} more papers")
    
    # Step 5: Aggregate insights
    print(f"\n5️⃣ Aggregate Insights:")
    print("=" * 60)
    
    insights = analysis_results['aggregate_insights']
    print(f"   Total Findings: {insights.get('total_findings', 0)}")
    print(f"   Dominant Methodology: {insights.get('dominant_methodology', 'Unknown')}")
    print(f"\n   Common Themes:")
    for theme in insights.get('common_themes', [])[:8]:
        print(f"     • {theme}")
    
    print(f"\n   Methodology Distribution:")
    for method, ratio in insights.get('methodological_distribution', {}).items():
        print(f"     • {method}: {ratio*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ CONTENT ANALYZER TEST SUCCESSFUL!")
    print("=" * 60)
    
    # Save results
    output_file = Path("outputs/test_analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_content_analyzer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)