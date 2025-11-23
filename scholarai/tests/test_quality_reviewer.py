"""
Test Quality Reviewer Agent
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.paper_hunter import PaperHunterAgent
from agents.content_analyzer import ContentAnalyzerAgent
from agents.research_synthesizer import ResearchSynthesizerAgent
from agents.quality_reviewer import QualityReviewerAgent

def test_quality_reviewer():
    """Test quality reviewer functionality"""
    
    print("\n🔬 Testing Quality Reviewer Agent...")
    print("=" * 60)
    
    # Step 1: Get research data
    print("\n1️⃣ Gathering research data...")
    
    hunter = PaperHunterAgent()
    paper_results = hunter.search_papers("neural networks deep learning")
    papers = paper_results['papers']
    print(f"✅ Got {len(papers)} papers")
    
    analyzer = ContentAnalyzerAgent()
    analysis_results = analyzer.analyze_papers(papers)
    analyses = analysis_results['analyses']
    print(f"✅ Analyzed {len(analyses)} papers")
    
    synthesizer = ResearchSynthesizerAgent()
    synthesis = synthesizer.synthesize_research(papers, analyses)
    print(f"✅ Synthesis complete")
    
    # Step 2: Quality Review
    print("\n2️⃣ Running Quality Review...")
    reviewer = QualityReviewerAgent()
    
    review = reviewer.review_research(papers, analyses, synthesis)
    
    if not review['success']:
        print("❌ Review failed")
        return False
    
    print(f"✅ Review complete!")
    
    # Step 3: Display Results
    print(f"\n3️⃣ Quality Review Results:")
    print("=" * 60)
    
    print(f"\n📊 Overall Score: {review['overall_score']}/10")
    
    print(f"\n📈 Dimension Scores:")
    for dimension, score in review['dimension_scores'].items():
        print(f"   • {dimension.replace('_', ' ').title()}: {score}/10")
    
    print(f"\n✨ Strengths:")
    for strength in review['strengths']:
        print(f"   • {strength}")
    
    if review['weaknesses']:
        print(f"\n⚠️  Weaknesses:")
        for weakness in review['weaknesses']:
            print(f"   • {weakness['issue']} (Severity: {weakness['severity']})")
            print(f"     {weakness['description']}")
    
    print(f"\n🎯 Recommendation:")
    print(f"   {review['recommendation']}")
    
    if review['needs_refinement']:
        print(f"\n🔄 Refinement Actions Suggested:")
        for action in review['refinement_actions']:
            print(f"   • {action['action']} → {action['target_agent']}")
            print(f"     {action['description']} (Priority: {action['priority']})")
    else:
        print(f"\n✅ No refinement needed - quality threshold met!")
    
    print("\n" + "=" * 60)
    print("✅ QUALITY REVIEWER TEST SUCCESSFUL!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_quality_reviewer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)