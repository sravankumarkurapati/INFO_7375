"""
ScholarAI - Main Entry Point
Academic Research Assistant System with Multi-Agent Orchestration
"""
import sys
from pathlib import Path
from typing import Optional
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.paper_hunter import PaperHunterAgent
from agents.content_analyzer import ContentAnalyzerAgent
from agents.research_synthesizer import ResearchSynthesizerAgent
from agents.quality_reviewer import QualityReviewerAgent
from agents.controller import ControllerAgent
from config.settings import settings
from utils.logger import logger
from utils.memory import memory_manager
from utils.validators import validators
from datetime import datetime

class ScholarAI:
    """
    Main ScholarAI system orchestrator
    """
    
    def __init__(self):
        """Initialize ScholarAI system"""
        logger.info("=" * 60)
        logger.info("🎓 Initializing ScholarAI v1.0")
        logger.info("=" * 60)
        
        # Validate settings
        try:
            settings.validate()
            logger.info("✅ Configuration validated")
        except Exception as e:
            logger.error(f"❌ Configuration error: {e}")
            raise
        
        # Initialize agents
        self.controller = None
        self.paper_hunter = None
        self.content_analyzer = None
        self.research_synthesizer = None
        self.quality_reviewer = None
        self._init_agents()
        
        logger.info("✅ ScholarAI initialized successfully")
    
    def _init_agents(self):
        """Initialize all agents"""
        try:
            logger.info("Initializing agents...")
            
            # Initialize controller first
            self.controller = ControllerAgent()
            
            # Initialize specialist agents
            self.paper_hunter = PaperHunterAgent()
            self.content_analyzer = ContentAnalyzerAgent()
            self.research_synthesizer = ResearchSynthesizerAgent()
            self.quality_reviewer = QualityReviewerAgent()
            
            logger.info("✅ All 5 agents initialized")
            logger.info("   - Controller Agent (Orchestrator)")
            logger.info("   - Paper Hunter Agent (Search)")
            logger.info("   - Content Analyzer Agent (Analysis)")
            logger.info("   - Research Synthesizer Agent (Gap Detection)")
            logger.info("   - Quality Reviewer Agent (Validation)")
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise
    
    def research(self, query: str, save_results: bool = True, max_iterations: int = 2) -> dict:
        """
        Conduct research on a given topic with feedback loop support
        
        Args:
            query: Research query
            save_results: Whether to save results to file
            max_iterations: Maximum refinement iterations
            
        Returns:
            Research results dictionary
        """
        logger.info("=" * 60)
        logger.info(f"🔍 Starting research: {query}")
        logger.info(f"   Max iterations: {max_iterations}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        iteration = 0
        
        while iteration < max_iterations:
            try:
                iteration += 1
                logger.info(f"\n🔄 Iteration {iteration}/{max_iterations}")
                
                # Phase 1: Paper Discovery
                logger.info("\n📚 PHASE 1: Paper Discovery")
                logger.info("-" * 60)
                
                paper_results = self.paper_hunter.search_papers(query)
                
                if not paper_results['success']:
                    logger.error(f"❌ Paper discovery failed: {paper_results['message']}")
                    return {
                        'success': False,
                        'error': paper_results['message'],
                        'query': query
                    }
                
                papers = paper_results['papers']
                logger.info(f"✅ Found {len(papers)} papers")
                
                # Validate papers
                try:
                    validators.validate_papers(papers, min_count=2)
                    logger.info("✅ Papers validated")
                except Exception as e:
                    logger.warning(f"⚠️  Validation warning: {e}")
                
                # Phase 2: Content Analysis
                logger.info("\n📖 PHASE 2: Content Analysis")
                logger.info("-" * 60)
                
                analysis_results = self.content_analyzer.analyze_papers(papers)
                
                if analysis_results['success']:
                    logger.info(f"✅ Analyzed {analysis_results['statistics']['analyzed']} papers")
                    analyses = analysis_results['analyses']
                    aggregate_insights = analysis_results['aggregate_insights']
                else:
                    logger.warning("⚠️  Content analysis failed, continuing with basic data")
                    analyses = []
                    aggregate_insights = {}
                
                # Phase 3: Research Synthesis (CUSTOM TOOL)
                logger.info("\n🔬 PHASE 3: Research Synthesis & Gap Analysis (Custom Tool)")
                logger.info("-" * 60)
                
                synthesis_results = None
                if analyses:
                    synthesis_results = self.research_synthesizer.synthesize_research(
                        papers, analyses
                    )
                    
                    if synthesis_results['success']:
                        logger.info(f"✅ Synthesis complete")
                        logger.info(f"   - {len(synthesis_results['research_gaps'])} gaps identified")
                        logger.info(f"   - {len(synthesis_results['recommendations'])} recommendations")
                        logger.info(f"   - {len(synthesis_results.get('visualizations', {}))} visualizations created")
                    else:
                        logger.warning("⚠️  Research synthesis failed")
                        synthesis_results = None
                else:
                    logger.warning("⚠️  Skipping synthesis (no analyses available)")
                    synthesis_results = None
                
                # Phase 4: Quality Review
                logger.info("\n✅ PHASE 4: Quality Review & Validation")
                logger.info("-" * 60)
                
                quality_review = None
                if analyses and synthesis_results:
                    quality_review = self.quality_reviewer.review_research(
                        papers, analyses, synthesis_results
                    )
                    
                    if quality_review['success']:
                        logger.info(f"✅ Quality review complete")
                        logger.info(f"   - Overall score: {quality_review['overall_score']:.1f}/10")
                        logger.info(f"   - Needs refinement: {quality_review['needs_refinement']}")
                        
                        # FEEDBACK LOOP: Check if refinement needed
                        if quality_review['needs_refinement'] and iteration < max_iterations:
                            logger.warning(f"\n🔄 FEEDBACK LOOP TRIGGERED")
                            logger.warning(f"   Quality score {quality_review['overall_score']:.1f}/10 below threshold (7.5)")
                            logger.warning(f"   Initiating refinement iteration {iteration + 1}...")
                            
                            # Show refinement actions
                            if quality_review.get('refinement_actions'):
                                logger.info(f"\n   Planned refinements:")
                                for action in quality_review['refinement_actions']:
                                    logger.info(f"   • {action['action']} → {action['target_agent']}")
                                    logger.info(f"     {action['description']}")
                            
                            # Continue to next iteration
                            continue
                        else:
                            if quality_review['needs_refinement']:
                                logger.info(f"✅ Max iterations reached, proceeding with current results")
                            else:
                                logger.info(f"✅ Quality threshold met (>{quality_review['overall_score']:.1f}/10)")
                    else:
                        logger.warning("⚠️  Quality review failed")
                        quality_review = None
                else:
                    logger.warning("⚠️  Skipping quality review (insufficient data)")
                    quality_review = None
                
                # If we reach here, quality is acceptable or max iterations reached
                break
                
            except Exception as e:
                logger.error(f"❌ Iteration {iteration} failed: {e}")
                import traceback
                traceback.print_exc()
                
                if iteration >= max_iterations:
                    return {
                        'success': False,
                        'error': str(e),
                        'query': query,
                        'timestamp': start_time.isoformat()
                    }
                continue
        
        # Prepare final results
        try:
            # Calculate statistics
            stats = self._calculate_statistics(
                papers, 
                analyses, 
                aggregate_insights,
                synthesis_results,
                quality_review
            )
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'success': True,
                'query': query,
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration,
                'iterations': iteration,
                'papers': papers,
                'analyses': analyses,
                'aggregate_insights': aggregate_insights,
                'synthesis': synthesis_results,
                'quality_review': quality_review,
                'statistics': stats,
                'phases_completed': ['paper_discovery', 'content_analysis', 'research_synthesis', 'quality_review'],
                'message': f'Research completed successfully in {duration:.1f}s after {iteration} iteration(s)'
            }
            
            # Save results if requested
            if save_results:
                self._save_results(results)
            
            # Update memory with quality score
            quality_score = self._calculate_quality_score(results)
            memory_manager.add_quality_score(quality_score)
            
            logger.info("=" * 60)
            logger.info(f"✅ RESEARCH COMPLETED SUCCESSFULLY!")
            logger.info(f"   Papers found: {len(papers)}")
            logger.info(f"   Papers analyzed: {len(analyses)}")
            if synthesis_results:
                logger.info(f"   Gaps identified: {len(synthesis_results.get('research_gaps', []))}")
                logger.info(f"   Visualizations: {len(synthesis_results.get('visualizations', {}))}")
            if quality_review:
                logger.info(f"   Quality score (reviewer): {quality_review['overall_score']:.1f}/10")
            logger.info(f"   Iterations used: {iteration}")
            logger.info(f"   Total duration: {duration:.1f}s")
            logger.info(f"   Final quality: {quality_score:.1f}/10")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Final result preparation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'timestamp': start_time.isoformat()
            }
    
    def _calculate_statistics(
        self, 
        papers: list, 
        analyses: list = None, 
        aggregate_insights: dict = None,
        synthesis: dict = None,
        quality_review: dict = None
    ) -> dict:
        """Calculate comprehensive statistics"""
        if not papers:
            return {}
        
        # Source distribution
        sources = {}
        for paper in papers:
            source = paper.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        # Year distribution
        years = [p.get('year', 2023) for p in papers]
        
        # Relevance statistics
        relevances = [p.get('relevance_score', 0) for p in papers]
        
        stats = {
            'total_papers': len(papers),
            'sources': sources,
            'year_range': {
                'min': min(years),
                'max': max(years)
            },
            'recent_papers': len([y for y in years if y >= 2022]),
            'relevance': {
                'average': sum(relevances) / len(relevances) if relevances else 0,
                'min': min(relevances) if relevances else 0,
                'max': max(relevances) if relevances else 0
            }
        }
        
        # Add analysis statistics
        if analyses:
            stats['papers_analyzed'] = len(analyses)
            
            methodologies = {}
            for analysis in analyses:
                method = analysis.get('methodology', {}).get('type', 'Unknown')
                methodologies[method] = methodologies.get(method, 0) + 1
            stats['methodologies'] = methodologies
        
        # Add aggregate insights
        if aggregate_insights:
            stats['aggregate_insights'] = aggregate_insights
        
        # Add synthesis statistics
        if synthesis and synthesis.get('success'):
            stats['synthesis'] = {
                'gaps_identified': len(synthesis.get('research_gaps', [])),
                'contradictions_found': len(synthesis.get('contradictions', [])),
                'clusters_identified': len(synthesis.get('clusters', [])),
                'recommendations_generated': len(synthesis.get('recommendations', []))
            }
        
        # Add quality review statistics
        if quality_review and quality_review.get('success'):
            stats['quality_review'] = {
                'overall_score': quality_review['overall_score'],
                'needs_refinement': quality_review['needs_refinement'],
                'dimension_scores': quality_review['dimension_scores']
            }
        
        return stats
    
    def _calculate_quality_score(self, results: dict) -> float:
        """Calculate quality score (0-10)"""
        if not results['success']:
            return 0.0
        
        # Use quality reviewer score if available
        if results.get('quality_review') and results['quality_review'].get('success'):
            return results['quality_review']['overall_score']
        
        # Otherwise calculate
        score = 0.0
        papers = results['papers']
        stats = results['statistics']
        
        # Paper count (max 2 points)
        paper_count = len(papers)
        if paper_count >= 10:
            score += 2.0
        elif paper_count >= 5:
            score += 1.5
        elif paper_count >= 2:
            score += 1.0
        
        # Average relevance (max 2 points)
        avg_relevance = stats['relevance']['average']
        score += min(2.0, avg_relevance * 2)
        
        # Source diversity (max 1.5 points)
        num_sources = len(stats['sources'])
        score += min(1.5, num_sources * 0.5)
        
        # Recency (max 1.5 points)
        recent_ratio = stats['recent_papers'] / stats['total_papers']
        score += recent_ratio * 1.5
        
        # Content analysis bonus (max 1 point)
        if 'papers_analyzed' in stats:
            analysis_ratio = stats['papers_analyzed'] / stats['total_papers']
            score += analysis_ratio * 1.0
        
        # Synthesis bonus (max 2 points)
        if 'synthesis' in stats:
            synthesis_stats = stats['synthesis']
            if synthesis_stats['gaps_identified'] >= 5:
                score += 2.0
            elif synthesis_stats['gaps_identified'] >= 3:
                score += 1.5
            elif synthesis_stats['gaps_identified'] >= 1:
                score += 1.0
        
        return min(10.0, score)
    
    def _save_results(self, results: dict):
        """Save results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_{timestamp}.json"
            filepath = settings.REPORTS_DIR / filename
            
            settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Make JSON serializable
            save_data = {
                'success': results['success'],
                'query': results['query'],
                'timestamp': results['timestamp'],
                'duration_seconds': results['duration_seconds'],
                'iterations': results.get('iterations', 1),
                'papers': results['papers'],
                'analyses': results['analyses'],
                'aggregate_insights': results['aggregate_insights'],
                'statistics': results['statistics'],
                'phases_completed': results['phases_completed'],
                'message': results['message']
            }
            
            # Add synthesis
            if results.get('synthesis') and results['synthesis'].get('success'):
                synthesis = results['synthesis']
                save_data['synthesis'] = {
                    'research_gaps': synthesis.get('research_gaps', []),
                    'contradictions': synthesis.get('contradictions', []),
                    'trends': {
                        k: v for k, v in synthesis.get('trends', {}).items()
                        if k != 'publication_timeline'
                    },
                    'clusters': synthesis.get('clusters', []),
                    'recommendations': synthesis.get('recommendations', []),
                    'statistics': synthesis.get('statistics', {}),
                    'visualizations': synthesis.get('visualizations', {})
                }
            
            # Add quality review
            if results.get('quality_review') and results['quality_review'].get('success'):
                save_data['quality_review'] = results['quality_review']
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def display_results(self, results: dict):
        """Display results in formatted output"""
        if not results['success']:
            print(f"\n❌ Research failed: {results.get('error', 'Unknown error')}")
            return
        
        print("\n" + "=" * 70)
        print("📊 SCHOLARAI RESEARCH RESULTS")
        print("=" * 70)
        
        print(f"\n🔍 Query: {results['query']}")
        print(f"⏱️  Duration: {results['duration_seconds']:.1f}s")
        print(f"🔄 Iterations: {results.get('iterations', 1)}")
        print(f"📄 Papers Found: {len(results['papers'])}")
        print(f"📖 Papers Analyzed: {len(results.get('analyses', []))}")
        
        # Quality Review Summary
        if results.get('quality_review') and results['quality_review'].get('success'):
            qr = results['quality_review']
            print(f"\n⭐ QUALITY ASSESSMENT: {qr['overall_score']:.1f}/10")
            print(f"   Status: {qr['recommendation']}")
        
        # Statistics
        stats = results['statistics']
        print(f"\n📈 RESEARCH STATISTICS")
        print("-" * 70)
        print(f"   Average Relevance: {stats['relevance']['average']:.2f}")
        print(f"   Recent Papers (2022+): {stats['recent_papers']}/{stats['total_papers']} ({stats['recent_papers']/stats['total_papers']*100:.0f}%)")
        print(f"   Year Range: {stats['year_range']['min']}-{stats['year_range']['max']}")
        
        print(f"\n   Sources Distribution:")
        for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True):
            print(f"     • {source}: {count}")
        
        # Methodologies
        if 'methodologies' in stats:
            print(f"\n   Methodology Distribution:")
            for method, count in sorted(stats['methodologies'].items(), key=lambda x: x[1], reverse=True):
                print(f"     • {method}: {count}")
        
        # Aggregate insights
        if results.get('aggregate_insights'):
            insights = results['aggregate_insights']
            print(f"\n🔍 AGGREGATE INSIGHTS")
            print("-" * 70)
            print(f"   Dominant Methodology: {insights.get('dominant_methodology', 'Unknown')}")
            if insights.get('common_themes'):
                themes = ', '.join(insights['common_themes'][:5])
                print(f"   Common Themes: {themes}")
        
        # Synthesis Results (CUSTOM TOOL)
        if results.get('synthesis') and results['synthesis'].get('success'):
            synthesis = results['synthesis']
            
            print(f"\n💎 RESEARCH GAP ANALYSIS (CUSTOM TOOL)")
            print("=" * 70)
            
            # Statistics
            if 'statistics' in synthesis:
                syn_stats = synthesis['statistics']
                print(f"\n   Analysis Statistics:")
                print(f"   • Papers Processed: {syn_stats.get('total_papers', 0)}")
                print(f"   • Research Clusters: {syn_stats.get('num_clusters', 0)}")
                print(f"   • Gaps Identified: {syn_stats.get('num_gaps', 0)}")
                print(f"   • Contradictions: {syn_stats.get('num_contradictions', 0)}")
            
            # Clusters
            if synthesis.get('clusters'):
                print(f"\n   🔍 Research Clusters Detected:")
                for cluster in synthesis['clusters'][:4]:
                    if cluster['cluster_id'] != -1:
                        print(f"   • {cluster['theme']} ({cluster['size']} papers)")
            
            # Research Gaps
            if synthesis.get('research_gaps'):
                print(f"\n   💡 Research Gaps Identified:")
                for idx, gap in enumerate(synthesis['research_gaps'][:5], 1):
                    print(f"\n   Gap #{idx}: {gap['title']}")
                    print(f"      Type: {gap['type']}")
                    print(f"      Confidence: {gap['confidence']:.0%} | Impact: {gap['impact']}")
                    print(f"      {gap['description'][:120]}...")
                    if gap.get('evidence'):
                        print(f"      Evidence: {gap['evidence'][0]}")
                
                if len(synthesis['research_gaps']) > 5:
                    print(f"\n   ... and {len(synthesis['research_gaps']) - 5} more gaps")
            
            # Recommendations
            if synthesis.get('recommendations'):
                print(f"\n   🎯 Research Recommendations:")
                for idx, rec in enumerate(synthesis['recommendations'][:3], 1):
                    print(f"   {idx}. {rec['title']}")
                    print(f"      Priority: {rec['priority']} | {rec['description'][:80]}...")
            
            # Visualizations
            if synthesis.get('visualizations'):
                print(f"\n   📊 Visualizations Generated:")
                for viz_name, viz_path in synthesis['visualizations'].items():
                    viz_file = Path(viz_path).name
                    print(f"      ✓ {viz_file}")
                print(f"\n   📁 Location: outputs/visualizations/")
        
        # Quality Review Details
        if results.get('quality_review') and results['quality_review'].get('success'):
            qr = results['quality_review']
            
            print(f"\n✅ QUALITY REVIEW DETAILS")
            print("=" * 70)
            
            print(f"\n   Dimension Scores:")
            for dim, score in qr['dimension_scores'].items():
                bar = '█' * int(score) + '░' * (10 - int(score))
                print(f"   • {dim.replace('_', ' ').title()}: {score}/10 [{bar}]")
            
            if qr.get('strengths'):
                print(f"\n   ✨ Strengths:")
                for strength in qr['strengths']:
                    print(f"   ✓ {strength}")
            
            if qr.get('weaknesses') and len(qr['weaknesses']) > 0:
                print(f"\n   ⚠️  Areas for Improvement:")
                for weakness in qr['weaknesses']:
                    print(f"   • {weakness['issue']} ({weakness['severity']} severity)")
                    print(f"     → {weakness['description']}")
        
        # Papers List
        print(f"\n📚 PAPERS ANALYZED")
        print("=" * 70)
        
        for idx, paper in enumerate(results['papers'][:10], 1):
            print(f"\n{idx}. {paper['title']}")
            print(f"   📍 Source: {paper['source']} | 📅 Year: {paper['year']} | ⭐ Relevance: {paper['relevance_score']:.2f}")
            print(f"   🔗 {paper['url']}")
            
            # Show analysis if available
            if results.get('analyses'):
                for analysis in results['analyses']:
                    if analysis.get('paper_id') == paper['id'] and analysis.get('success'):
                        if analysis.get('key_findings'):
                            finding = analysis['key_findings'][0]
                            if len(finding) > 80:
                                finding = finding[:77] + "..."
                            print(f"   💡 {finding}")
                        
                        if analysis.get('methodology'):
                            method = analysis['methodology']
                            print(f"   🔬 {method.get('type', 'Unknown')} | {method.get('approach', 'N/A')}")
                        break
        
        if len(results['papers']) > 10:
            print(f"\n   ... and {len(results['papers']) - 10} more papers")
        
        print("\n" + "=" * 70)

def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("🎓 SCHOLARAI - ACADEMIC RESEARCH ASSISTANT")
    print("   Multi-Agent System with Custom Gap Analysis")
    print("=" * 70)
    
    print("\n📋 System Components:")
    print("   • Controller Agent (Workflow Orchestrator)")
    print("   • Paper Hunter Agent (Academic Search)")
    print("   • Content Analyzer Agent (Paper Analysis)")
    print("   • Research Synthesizer Agent (Gap Detection)")
    print("   • Quality Reviewer Agent (Quality Validation)")
    
    print("\n🛠️  Built-in Tools (3 tools):")
    print("   • SerperDevTool - Google Scholar search")
    print("   • FileReadTool - File operations")
    print("   • ScrapeWebsiteTool - Web content extraction")
    
    print("\n💎 Custom Tool:")
    print("   • Research Gap Analyzer")
    print("     - Sentence embeddings (384-dimensional)")
    print("     - DBSCAN clustering")
    print("     - Gap detection (4 methods)")
    print("     - Trend analysis")
    print("     - Visualization generation")
    
    try:
        # Initialize system
        scholar = ScholarAI()
        
        # Get research query
        query = input("\n🔍 Enter your research query: ").strip()
        
        if not query:
            query = "transformer models for natural language processing"
            print(f"   Using default query: {query}")
        
        print(f"\n⏳ Conducting research... This may take 60-90 seconds...")
        print(f"\n   🔄 Max iterations: 2 (with feedback loop)")
        print(f"   📊 Phases: Discovery → Analysis → Synthesis → Quality Review")
        print()
        
        # Conduct research with feedback loop
        results = scholar.research(query, save_results=True, max_iterations=2)
        
        # Display results
        scholar.display_results(results)
        
        # Show memory summary
        summary = memory_manager.get_summary()
        print(f"\n📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"   Total queries processed: {summary['long_term']['total_queries']}")
        print(f"   Average quality score: {summary['long_term']['average_quality']:.1f}/10")
        print(f"   Session queries: {summary['current_session']['queries']}")
        
        print("\n" + "=" * 70)
        print("✅ Research session completed successfully!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n👋 Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save memory
        memory_manager.save_long_term()
        print("\n💾 Session data saved to memory")
        print("=" * 70)

if __name__ == "__main__":
    main()