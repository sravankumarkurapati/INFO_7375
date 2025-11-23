"""
Comprehensive Evaluation Test Suite
Runs multiple test cases and collects metrics for evaluation report
"""
import sys
from pathlib import Path
import json
import time
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.paper_hunter import PaperHunterAgent
from agents.content_analyzer import ContentAnalyzerAgent
from agents.research_synthesizer import ResearchSynthesizerAgent
from agents.quality_reviewer import QualityReviewerAgent
from utils.memory import memory_manager

class ComprehensiveEvaluator:
    """Runs comprehensive evaluation tests"""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        
    def run_test_case(self, query, test_name):
        """Run single test case and collect metrics"""
        print(f"\n{'='*70}")
        print(f"🧪 TEST CASE: {test_name}")
        print(f"📝 Query: {query}")
        print(f"{'='*70}")
        
        start = time.time()
        
        try:
            # Phase 1: Paper Discovery
            print("\n📚 Phase 1: Paper Discovery...")
            hunter = PaperHunterAgent()
            paper_results = hunter.search_papers(query)
            
            if not paper_results['success']:
                print(f"❌ Paper discovery failed")
                return None
            
            papers = paper_results['papers']
            print(f"✅ Found {len(papers)} papers")
            print(f"   Average relevance: {paper_results['avg_relevance']:.2f}")
            
            # Phase 2: Content Analysis
            print("\n📖 Phase 2: Content Analysis...")
            analyzer = ContentAnalyzerAgent()
            analysis_results = analyzer.analyze_papers(papers)
            
            analyses = analysis_results['analyses']
            print(f"✅ Analyzed {len(analyses)}/{len(papers)} papers ({len(analyses)/len(papers)*100:.0f}%)")
            
            # Phase 3: Research Synthesis
            print("\n🔬 Phase 3: Research Synthesis...")
            synthesizer = ResearchSynthesizerAgent()
            synthesis = synthesizer.synthesize_research(papers, analyses)
            
            gaps = synthesis.get('research_gaps', [])
            print(f"✅ Identified {len(gaps)} research gaps")
            
            # Phase 4: Quality Review
            print("\n✅ Phase 4: Quality Review...")
            reviewer = QualityReviewerAgent()
            quality = reviewer.review_research(papers, analyses, synthesis)
            
            print(f"✅ Quality Score: {quality['overall_score']:.1f}/10")
            
            duration = time.time() - start
            
            # Collect metrics
            result = {
                'test_name': test_name,
                'query': query,
                'papers_found': len(papers),
                'papers_analyzed': len(analyses),
                'analysis_success_rate': len(analyses) / len(papers),
                'avg_relevance': paper_results['avg_relevance'],
                'gaps_identified': len(gaps),
                'avg_gap_confidence': sum(g['confidence'] for g in gaps) / len(gaps) if gaps else 0,
                'quality_score': quality['overall_score'],
                'dimension_scores': quality['dimension_scores'],
                'visualizations_created': len(synthesis.get('visualizations', {})),
                'recommendations': len(synthesis.get('recommendations', [])),
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'status': 'PASS' if quality['overall_score'] >= 7.0 else 'FAIL'
            }
            
            self.results.append(result)
            
            print(f"\n📊 Test Summary:")
            print(f"   Papers: {len(papers)}")
            print(f"   Analysis Rate: {len(analyses)/len(papers)*100:.0f}%")
            print(f"   Gaps: {len(gaps)}")
            print(f"   Quality: {quality['overall_score']:.1f}/10")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Status: {result['status']}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print("🎯 SCHOLARAI COMPREHENSIVE EVALUATION")
        print("="*70)
        
        self.start_time = datetime.now()
        
        test_cases = [
            ("transformer models for natural language processing", "Transformer NLP"),
            ("deep learning for computer vision", "Deep Learning CV"),
            ("machine learning in healthcare", "ML Healthcare"),
            ("neural architecture search", "Neural Arch Search"),
            ("deep reinforcement learning", "Deep RL"),
            ("explainable artificial intelligence", "Explainable AI"),
        ]
        
        for query, name in test_cases:
            result = self.run_test_case(query, name)
            time.sleep(2)  # Brief pause between tests
        
        self.generate_report()
    
    def generate_report(self):
        """Generate evaluation report"""
        print(f"\n\n{'='*70}")
        print("📊 EVALUATION REPORT SUMMARY")
        print(f"{'='*70}")
        
        if not self.results:
            print("❌ No results to report")
            return
        
        # Overall Statistics
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r['status'] == 'PASS'])
        
        print(f"\n🎯 OVERALL RESULTS")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ({passed_tests/total_tests*100:.0f}%)")
        print(f"   Failed: {total_tests - passed_tests}")
        
        # Aggregate Metrics
        avg_quality = sum(r['quality_score'] for r in self.results) / len(self.results)
        avg_time = sum(r['duration_seconds'] for r in self.results) / len(self.results)
        avg_papers = sum(r['papers_found'] for r in self.results) / len(self.results)
        avg_gaps = sum(r['gaps_identified'] for r in self.results) / len(self.results)
        avg_relevance = sum(r['avg_relevance'] for r in self.results) / len(self.results)
        total_analysis = sum(r['papers_analyzed'] for r in self.results)
        total_papers = sum(r['papers_found'] for r in self.results)
        
        print(f"\n📈 AGGREGATE METRICS")
        print(f"   Average Quality Score: {avg_quality:.2f}/10")
        print(f"   Average Processing Time: {avg_time:.1f}s")
        print(f"   Average Papers/Query: {avg_papers:.1f}")
        print(f"   Average Gaps/Query: {avg_gaps:.1f}")
        print(f"   Average Relevance: {avg_relevance:.2f}")
        print(f"   Overall Analysis Rate: {total_analysis}/{total_papers} ({total_analysis/total_papers*100:.0f}%)")
        
        # Quality Distribution
        print(f"\n⭐ QUALITY SCORE DISTRIBUTION")
        excellent = len([r for r in self.results if r['quality_score'] >= 9.0])
        very_good = len([r for r in self.results if 8.0 <= r['quality_score'] < 9.0])
        good = len([r for r in self.results if 7.0 <= r['quality_score'] < 8.0])
        acceptable = len([r for r in self.results if r['quality_score'] < 7.0])
        
        print(f"   Excellent (9.0-10.0): {excellent} ({excellent/total_tests*100:.0f}%)")
        print(f"   Very Good (8.0-8.9): {very_good} ({very_good/total_tests*100:.0f}%)")
        print(f"   Good (7.0-7.9): {good} ({good/total_tests*100:.0f}%)")
        print(f"   Below Target (<7.0): {acceptable} ({acceptable/total_tests*100:.0f}%)")
        
        # Detailed Results Table
        print(f"\n📋 DETAILED RESULTS TABLE")
        print(f"{'='*70}")
        print(f"{'Test Name':<20} {'Papers':>7} {'Gaps':>5} {'Quality':>7} {'Time':>7}")
        print(f"{'-'*70}")
        
        for r in self.results:
            print(f"{r['test_name']:<20} {r['papers_found']:>7} {r['gaps_identified']:>5} "
                  f"{r['quality_score']:>7.1f} {r['duration_seconds']:>6.1f}s")
        
        print(f"{'='*70}")
        
        # Save results
        output_file = Path("outputs/evaluation_results.json")
        output_file.parent.mkdir(exist_ok=True)
        
        evaluation_data = {
            'timestamp': self.start_time.isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'aggregate_metrics': {
                'avg_quality_score': avg_quality,
                'avg_processing_time': avg_time,
                'avg_papers_found': avg_papers,
                'avg_gaps_identified': avg_gaps,
                'avg_relevance': avg_relevance,
                'overall_analysis_rate': total_analysis / total_papers
            },
            'detailed_results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"\n✅ Evaluation Complete!")
        print(f"{'='*70}")

def main():
    evaluator = ComprehensiveEvaluator()
    evaluator.run_all_tests()

if __name__ == "__main__":
    main()