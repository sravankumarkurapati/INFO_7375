"""
Agent 4: Quality Reviewer
Reviews research quality and triggers refinement when needed
"""
from crewai import Agent
from langchain_openai import ChatOpenAI
from typing import Dict, List
from config.settings import settings
from utils.logger import logger
from utils.memory import memory_manager

class QualityReviewerAgent:
    """
    Agent specialized in quality assessment and validation
    """
    
    def __init__(self):
        """Initialize Quality Reviewer Agent"""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2  # Low temperature for consistent evaluation
        )
        
        self.agent = Agent(
            role='Academic Peer Reviewer',
            goal='Evaluate research quality and provide actionable feedback',
            backstory="""You are an experienced peer reviewer for top-tier academic conferences.
            You excel at:
            - Identifying strengths and weaknesses in research
            - Evaluating completeness and coherence
            - Assessing evidence quality and logical consistency
            - Providing constructive criticism
            - Determining when work needs refinement
            
            You maintain high standards while being fair and constructive.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        logger.info("Quality Reviewer Agent initialized")
    
    def review_research(
        self, 
        papers: List[Dict], 
        analyses: List[Dict],
        synthesis: Dict
    ) -> Dict:
        """
        Review overall research quality
        
        Args:
            papers: List of papers
            analyses: List of analyses
            synthesis: Synthesis results
            
        Returns:
            Quality review with scores and recommendations
        """
        logger.info("Starting quality review...")
        
        try:
            # Evaluate different dimensions
            completeness_score = self._evaluate_completeness(papers, analyses, synthesis)
            evidence_score = self._evaluate_evidence(analyses)
            coherence_score = self._evaluate_coherence(papers, analyses, synthesis)
            gap_quality_score = self._evaluate_gap_quality(synthesis)
            
            # Calculate overall score
            overall_score = (
                completeness_score * 0.25 +
                evidence_score * 0.25 +
                coherence_score * 0.25 +
                gap_quality_score * 0.25
            )
            
            # Determine if refinement is needed
            needs_refinement = overall_score < 7.5
            
            # Identify weaknesses
            weaknesses = self._identify_weaknesses(
                completeness_score,
                evidence_score,
                coherence_score,
                gap_quality_score,
                papers,
                analyses,
                synthesis
            )
            
            # Generate refinement actions
            refinement_actions = []
            if needs_refinement:
                refinement_actions = self._generate_refinement_actions(weaknesses)
            
            # Compile strengths
            strengths = self._identify_strengths(
                completeness_score,
                evidence_score,
                coherence_score,
                gap_quality_score
            )
            
            review_results = {
                'success': True,
                'overall_score': round(overall_score, 2),
                'dimension_scores': {
                    'completeness': round(completeness_score, 2),
                    'evidence_quality': round(evidence_score, 2),
                    'logical_coherence': round(coherence_score, 2),
                    'gap_analysis_quality': round(gap_quality_score, 2)
                },
                'needs_refinement': needs_refinement,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'refinement_actions': refinement_actions,
                'recommendation': self._generate_recommendation(overall_score)
            }
            
            # Update memory
            memory_manager.add_agent_output('quality_reviewer', {
                'overall_score': overall_score,
                'needs_refinement': needs_refinement,
                'num_weaknesses': len(weaknesses)
            })
            
            logger.info(f"✅ Quality review complete!")
            logger.info(f"   Overall score: {overall_score:.1f}/10")
            logger.info(f"   Needs refinement: {needs_refinement}")
            
            return review_results
            
        except Exception as e:
            logger.error(f"Quality review failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'overall_score': 0,
                'needs_refinement': True
            }
    
    def _evaluate_completeness(
        self, 
        papers: List[Dict], 
        analyses: List[Dict],
        synthesis: Dict
    ) -> float:
        """Evaluate research completeness (0-10)"""
        score = 0.0
        
        # Paper count (max 3 points)
        paper_count = len(papers)
        if paper_count >= 10:
            score += 3.0
        elif paper_count >= 5:
            score += 2.0
        elif paper_count >= 3:
            score += 1.0
        
        # Analysis coverage (max 3 points)
        if analyses:
            analysis_ratio = len(analyses) / len(papers)
            score += analysis_ratio * 3.0
        
        # Synthesis completeness (max 4 points)
        if synthesis and synthesis.get('success'):
            # Has gaps identified
            if synthesis.get('research_gaps'):
                score += 1.5
            
            # Has clusters
            if synthesis.get('clusters'):
                score += 1.0
            
            # Has recommendations
            if synthesis.get('recommendations'):
                score += 1.0
            
            # Has visualizations
            if synthesis.get('visualizations'):
                score += 0.5
        
        return min(10.0, score)
    
    def _evaluate_evidence(self, analyses: List[Dict]) -> float:
        """Evaluate quality of evidence (0-10)"""
        if not analyses:
            return 0.0
        
        score = 0.0
        
        # Check for key findings
        analyses_with_findings = sum(
            1 for a in analyses 
            if a.get('success') and a.get('key_findings')
        )
        
        if analyses_with_findings > 0:
            findings_ratio = analyses_with_findings / len(analyses)
            score += findings_ratio * 5.0
        
        # Check for methodology classification
        analyses_with_method = sum(
            1 for a in analyses
            if a.get('success') and a.get('methodology', {}).get('type') != 'Unknown'
        )
        
        if analyses_with_method > 0:
            method_ratio = analyses_with_method / len(analyses)
            score += method_ratio * 3.0
        
        # Check for technical terms
        analyses_with_terms = sum(
            1 for a in analyses
            if a.get('success') and len(a.get('technical_terms', [])) > 0
        )
        
        if analyses_with_terms > 0:
            terms_ratio = analyses_with_terms / len(analyses)
            score += terms_ratio * 2.0
        
        return min(10.0, score)
    
    def _evaluate_coherence(
        self,
        papers: List[Dict],
        analyses: List[Dict],
        synthesis: Dict
    ) -> float:
        """Evaluate logical coherence (0-10)"""
        score = 8.0  # Start with baseline
        
        # Check for source diversity
        sources = set(p.get('source', 'Unknown') for p in papers)
        if len(sources) >= 3:
            score += 1.0
        elif len(sources) >= 2:
            score += 0.5
        
        # Check for temporal spread
        years = [p.get('year', 2023) for p in papers]
        year_range = max(years) - min(years)
        if year_range >= 3:
            score += 1.0
        elif year_range >= 1:
            score += 0.5
        
        return min(10.0, score)
    
    def _evaluate_gap_quality(self, synthesis: Dict) -> float:
        """Evaluate quality of gap analysis (0-10)"""
        if not synthesis or not synthesis.get('success'):
            return 0.0
        
        score = 0.0
        
        # Number of gaps (max 4 points)
        gaps = synthesis.get('research_gaps', [])
        if len(gaps) >= 5:
            score += 4.0
        elif len(gaps) >= 3:
            score += 3.0
        elif len(gaps) >= 1:
            score += 2.0
        
        # Gap confidence (max 3 points)
        if gaps:
            avg_confidence = sum(g.get('confidence', 0) for g in gaps) / len(gaps)
            score += avg_confidence * 3.0
        
        # Has recommendations (max 2 points)
        recommendations = synthesis.get('recommendations', [])
        if len(recommendations) >= 3:
            score += 2.0
        elif len(recommendations) >= 1:
            score += 1.0
        
        # Has clusters (max 1 point)
        clusters = synthesis.get('clusters', [])
        if len(clusters) >= 2:
            score += 1.0
        elif len(clusters) >= 1:
            score += 0.5
        
        return min(10.0, score)
    
    def _identify_weaknesses(
        self,
        completeness: float,
        evidence: float,
        coherence: float,
        gap_quality: float,
        papers: List[Dict],
        analyses: List[Dict],
        synthesis: Dict
    ) -> List[Dict]:
        """Identify specific weaknesses"""
        weaknesses = []
        
        # Completeness issues
        if completeness < 7.0:
            if len(papers) < 5:
                weaknesses.append({
                    'issue': 'Limited paper coverage',
                    'severity': 'High',
                    'description': f'Only {len(papers)} papers analyzed. Recommend 10+ for comprehensive coverage.',
                    'dimension': 'completeness'
                })
            
            if analyses:
                analysis_ratio = len(analyses) / len(papers)
                if analysis_ratio < 0.8:
                    weaknesses.append({
                        'issue': 'Incomplete analysis coverage',
                        'severity': 'Medium',
                        'description': f'Only {len(analyses)}/{len(papers)} papers fully analyzed.',
                        'dimension': 'completeness'
                    })
        
        # Evidence issues
        if evidence < 7.0:
            weaknesses.append({
                'issue': 'Weak evidence extraction',
                'severity': 'Medium',
                'description': 'Key findings not extracted from all papers.',
                'dimension': 'evidence'
            })
        
        # Gap quality issues
        if gap_quality < 7.0:
            if synthesis and len(synthesis.get('research_gaps', [])) < 3:
                weaknesses.append({
                    'issue': 'Insufficient gap identification',
                    'severity': 'High',
                    'description': 'Fewer than 3 research gaps identified.',
                    'dimension': 'gap_quality'
                })
        
        return weaknesses
    
    def _identify_strengths(
        self,
        completeness: float,
        evidence: float,
        coherence: float,
        gap_quality: float
    ) -> List[str]:
        """Identify strengths"""
        strengths = []
        
        if completeness >= 8.0:
            strengths.append("Comprehensive paper coverage")
        
        if evidence >= 8.0:
            strengths.append("High-quality evidence extraction")
        
        if coherence >= 8.0:
            strengths.append("Strong logical coherence")
        
        if gap_quality >= 8.0:
            strengths.append("Excellent gap analysis")
        
        if not strengths:
            strengths.append("Baseline research quality achieved")
        
        return strengths
    
    def _generate_refinement_actions(self, weaknesses: List[Dict]) -> List[Dict]:
        """Generate specific refinement actions"""
        actions = []
        
        for weakness in weaknesses:
            if weakness['dimension'] == 'completeness':
                if 'Limited paper coverage' in weakness['issue']:
                    actions.append({
                        'action': 'expand_search',
                        'target_agent': 'paper_hunter',
                        'description': 'Broaden search query or reduce filters',
                        'priority': 'High'
                    })
                
                elif 'Incomplete analysis' in weakness['issue']:
                    actions.append({
                        'action': 'retry_analysis',
                        'target_agent': 'content_analyzer',
                        'description': 'Re-analyze failed papers',
                        'priority': 'Medium'
                    })
            
            elif weakness['dimension'] == 'gap_quality':
                actions.append({
                    'action': 'enhance_synthesis',
                    'target_agent': 'research_synthesizer',
                    'description': 'Lower confidence threshold or expand gap detection',
                    'priority': 'High'
                })
        
        return actions
    
    def _generate_recommendation(self, overall_score: float) -> str:
        """Generate overall recommendation"""
        if overall_score >= 9.0:
            return "Excellent quality. Ready for publication/submission."
        elif overall_score >= 8.0:
            return "High quality. Minor improvements possible."
        elif overall_score >= 7.0:
            return "Good quality. Some refinement recommended."
        elif overall_score >= 6.0:
            return "Acceptable quality. Refinement needed in specific areas."
        else:
            return "Needs significant refinement. Consider expanding search or improving analysis."
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent"""
        return self.agent

# Convenience function
def create_quality_reviewer() -> QualityReviewerAgent:
    """Create and return Quality Reviewer agent"""
    return QualityReviewerAgent()