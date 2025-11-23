"""
Agent 3: Research Synthesizer
Uses the custom Research Gap Analyzer tool to synthesize insights
"""
from crewai import Agent
from langchain_openai import ChatOpenAI
from typing import Dict, List
from config.settings import settings
from utils.logger import logger
from utils.memory import memory_manager
from tools.gap_analyzer import research_gap_analyzer

class ResearchSynthesizerAgent:
    """
    Agent specialized in synthesizing research and identifying gaps
    """
    
    def __init__(self):
        """Initialize Research Synthesizer Agent"""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.4
        )
        
        self.gap_analyzer = research_gap_analyzer
        
        self.agent = Agent(
            role='Senior Research Advisor',
            goal='Synthesize research findings, identify gaps, and suggest future research directions',
            backstory="""You are a senior research advisor who has guided 100+ PhD students. 
            You excel at:
            - Identifying novel research opportunities
            - Spotting contradictions and gaps in literature
            - Understanding broader research trends
            - Providing actionable research directions
            - Synthesizing complex information into clear insights
            
            You use advanced analytical tools including embeddings, clustering, and 
            network analysis to uncover patterns that others might miss.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        logger.info("Research Synthesizer Agent initialized")
    
    def synthesize_research(self, papers: List[Dict], analyses: List[Dict]) -> Dict:
        """
        Main synthesis method using custom gap analyzer
        
        Args:
            papers: List of paper dictionaries
            analyses: List of analysis dictionaries
            
        Returns:
            Synthesis results with gaps, trends, and recommendations
        """
        logger.info(f"Starting research synthesis on {len(papers)} papers...")
        
        try:
            # Use custom tool for gap analysis
            gap_results = self.gap_analyzer.analyze(papers, analyses)
            
            if not gap_results['success']:
                logger.error("Gap analysis failed")
                return {
                    'success': False,
                    'error': 'Gap analysis failed',
                    'research_gaps': [],
                    'recommendations': []
                }
            
            # Enhance with LLM reasoning
            enhanced_synthesis = self._enhance_with_llm(
                gap_results, 
                papers, 
                analyses
            )
            
            # Combine results
            final_results = {
                'success': True,
                'research_gaps': gap_results['research_gaps'],
                'contradictions': gap_results['contradictions'],
                'trends': gap_results['trends'],
                'clusters': gap_results['clusters'],
                'network': gap_results['network'],
                'visualizations': gap_results['visualizations'],
                'recommendations': gap_results['recommendations'],
                'synthesis_summary': enhanced_synthesis,
                'statistics': gap_results['statistics']
            }
            
            # Update memory
            memory_manager.add_agent_output('research_synthesizer', {
                'gaps_found': len(gap_results['research_gaps']),
                'contradictions_found': len(gap_results['contradictions']),
                'clusters_identified': len(gap_results['clusters']),
                'recommendations_generated': len(gap_results['recommendations'])
            })
            
            logger.info(f"✅ Research synthesis complete!")
            logger.info(f"   - {len(gap_results['research_gaps'])} gaps identified")
            logger.info(f"   - {len(gap_results['recommendations'])} recommendations")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Research synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'research_gaps': [],
                'recommendations': []
            }
    
    def _enhance_with_llm(
        self, 
        gap_results: Dict, 
        papers: List[Dict], 
        analyses: List[Dict]
    ) -> str:
        """Use LLM to create a narrative synthesis"""
        
        try:
            # Prepare summary data
            num_papers = len(papers)
            num_gaps = len(gap_results['research_gaps'])
            num_clusters = len(gap_results['clusters'])
            
            # Get top themes
            top_clusters = gap_results['clusters'][:3]
            cluster_themes = [c['theme'] for c in top_clusters]
            
            # Get top gaps
            top_gaps = gap_results['research_gaps'][:3]
            gap_titles = [g['title'] for g in top_gaps]
            
            # Simple synthesis without LLM call (to save tokens)
            synthesis = f"""
Research Synthesis Summary:

Analyzed {num_papers} papers across {num_clusters} research clusters.

Main Research Areas:
{chr(10).join([f"- {theme}" for theme in cluster_themes])}

Key Research Gaps Identified:
{chr(10).join([f"- {title}" for title in gap_titles])}

The research landscape shows {'high' if num_clusters > 3 else 'moderate'} diversity 
with {'significant' if num_gaps > 5 else 'some'} opportunities for novel contributions.
"""
            
            return synthesis.strip()
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return "Synthesis summary generation failed."
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent"""
        return self.agent

# Convenience function
def create_research_synthesizer() -> ResearchSynthesizerAgent:
    """Create and return Research Synthesizer agent"""
    return ResearchSynthesizerAgent()