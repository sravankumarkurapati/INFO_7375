"""
Agent 2: Content Analyzer
Analyzes paper content and extracts key information
"""
from crewai import Agent
from crewai_tools import ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
from typing import Dict, List
import re
from config.settings import settings
from utils.logger import logger
from utils.memory import memory_manager
from utils.web_scraper import web_scraper

class ContentAnalyzerAgent:
    """
    Agent specialized in analyzing academic paper content
    """
    
    def __init__(self):
        """Initialize Content Analyzer Agent"""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2
        )
        
        # Built-in Tool
        self.scrape_tool = ScrapeWebsiteTool()
        
        self.agent = Agent(
            role='PhD Research Analyst',
            goal='Analyze academic papers and extract key findings, methodologies, and insights',
            backstory="""You are a PhD researcher with expertise in quickly reading 
            and understanding complex academic papers. You excel at:
            - Identifying core contributions and key findings
            - Classifying research methodologies
            - Extracting limitations and future work
            - Understanding citation relationships
            - Summarizing complex ideas concisely
            
            You can process papers from various sources (ArXiv, IEEE, ACM) and 
            adapt your analysis to different paper formats.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.scrape_tool]
        )
        
        logger.info("Content Analyzer Agent initialized")
    
    def analyze_papers(self, papers: List[Dict]) -> Dict:
        """
        Analyze a list of papers
        
        Args:
            papers: List of paper dictionaries from Paper Hunter
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting analysis of {len(papers)} papers")
        
        analyses = []
        successful = 0
        failed = 0
        
        for idx, paper in enumerate(papers, 1):
            logger.info(f"Analyzing paper {idx}/{len(papers)}: {paper['title'][:50]}...")
            
            analysis = self.analyze_single_paper(paper)
            
            if analysis['success']:
                analyses.append(analysis)
                successful += 1
            else:
                failed += 1
                logger.warning(f"Failed to analyze: {paper['title'][:50]}")
        
        # Generate aggregate insights
        aggregate = self._generate_aggregate_insights(analyses)
        
        logger.info(f"✅ Analysis complete: {successful} successful, {failed} failed")
        
        # Update memory
        memory_manager.add_agent_output('content_analyzer', {
            'papers_analyzed': len(analyses),
            'successful': successful,
            'failed': failed
        })
        
        return {
            'success': True,
            'analyses': analyses,
            'aggregate_insights': aggregate,
            'statistics': {
                'total_papers': len(papers),
                'analyzed': successful,
                'failed': failed
            }
        }
    
    def analyze_single_paper(self, paper: Dict) -> Dict:
        """
        Analyze a single paper
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Analysis dictionary
        """
        try:
            # Fetch content
            content = self._fetch_paper_content(paper)
            
            if not content:
                return {
                    'success': False,
                    'paper_id': paper['id'],
                    'title': paper['title'],
                    'error': 'Failed to fetch content'
                }
            
            # Extract key findings
            key_findings = self._extract_key_findings(paper, content)
            
            # Classify methodology
            methodology = self._classify_methodology(content)
            
            # Extract main contribution
            contribution = self._extract_contribution(paper, content)
            
            # Identify limitations
            limitations = self._extract_limitations(content)
            
            # Extract technical terms
            technical_terms = self._extract_technical_terms(content)
            
            analysis = {
                'success': True,
                'paper_id': paper['id'],
                'title': paper['title'],
                'url': paper['url'],
                'year': paper['year'],
                'source': paper['source'],
                'key_findings': key_findings,
                'methodology': methodology,
                'main_contribution': contribution,
                'limitations': limitations,
                'technical_terms': technical_terms[:10],
                'content_length': len(content)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing paper: {e}")
            return {
                'success': False,
                'paper_id': paper.get('id', 'unknown'),
                'title': paper.get('title', 'Unknown'),
                'error': str(e)
            }
    
    def _fetch_paper_content(self, paper: Dict) -> str:
        """Fetch paper content from URL"""
        try:
            url = paper['url']
            
            # Use web scraper utility
            content_dict = web_scraper.fetch_content(url)
            
            if content_dict and content_dict.get('success'):
                return content_dict['text']
            
            # Fallback to snippet if scraping fails
            return paper.get('snippet', '')
            
        except Exception as e:
            logger.warning(f"Content fetch failed: {e}")
            return paper.get('snippet', '')
    
    def _extract_key_findings(self, paper: Dict, content: str) -> List[str]:
        """Extract key findings using simple heuristics"""
        findings = []
        
        # Use snippet as primary source
        snippet = paper.get('snippet', '')
        
        if snippet:
            # Split into sentences
            sentences = re.split(r'[.!?]+', snippet)
            
            # Look for result/finding indicators
            result_keywords = ['achieve', 'improve', 'demonstrate', 'show', 'result', 
                             'performance', 'accuracy', 'outperform', 'reduce']
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in result_keywords):
                    if len(sentence) > 20:
                        findings.append(sentence)
        
        # If no findings from snippet, extract from content
        if not findings and content:
            content_lower = content.lower()
            if 'result' in content_lower:
                start = content_lower.find('result')
                section = content[start:start+500]
                sentences = re.split(r'[.!?]+', section)
                findings = [s.strip() for s in sentences[1:4] if len(s.strip()) > 20]
        
        # Default finding if none found
        if not findings:
            findings = [f"Research on {paper.get('title', 'the topic')}"]
        
        return findings[:3]
    
    def _classify_methodology(self, content: str) -> Dict:
        """Classify research methodology"""
        content_lower = content.lower()
        
        methodology = {
            'type': 'Unknown',
            'approach': 'Not specified',
            'datasets': [],
            'metrics': []
        }
        
        # Detect methodology type
        if any(word in content_lower for word in ['experiment', 'implementation', 'evaluate']):
            methodology['type'] = 'Experimental'
        elif any(word in content_lower for word in ['survey', 'review', 'literature']):
            methodology['type'] = 'Survey'
        elif any(word in content_lower for word in ['theorem', 'proof', 'proposition']):
            methodology['type'] = 'Theoretical'
        else:
            methodology['type'] = 'Empirical'
        
        # Detect approach keywords
        if 'deep learning' in content_lower or 'neural network' in content_lower:
            methodology['approach'] = 'Deep Learning'
        elif 'machine learning' in content_lower:
            methodology['approach'] = 'Machine Learning'
        elif 'transformer' in content_lower:
            methodology['approach'] = 'Transformer-based'
        elif 'reinforcement' in content_lower:
            methodology['approach'] = 'Reinforcement Learning'
        
        # Extract common datasets
        datasets = ['imagenet', 'coco', 'mnist', 'cifar', 'glue', 'squad', 'wmt', 'nas-bench']
        methodology['datasets'] = [d for d in datasets if d in content_lower]
        
        # Extract metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'bleu', 'rouge', 'perplexity']
        methodology['metrics'] = [m for m in metrics if m in content_lower]
        
        return methodology
    
    def _extract_contribution(self, paper: Dict, content: str) -> str:
        """Extract main contribution"""
        title = paper.get('title', '')
        snippet = paper.get('snippet', '')
        
        # Look for contribution keywords in snippet
        contrib_keywords = ['propose', 'introduce', 'present', 'novel', 'new approach']
        
        for sentence in re.split(r'[.!?]+', snippet):
            if any(kw in sentence.lower() for kw in contrib_keywords):
                return sentence.strip()
        
        # Fallback: use first sentence of snippet
        first_sentence = snippet.split('.')[0] if snippet else title
        return first_sentence.strip()
    
    def _extract_limitations(self, content: str) -> List[str]:
        """Extract limitations from content"""
        limitations = []
        content_lower = content.lower()
        
        # Look for limitation section
        if 'limitation' in content_lower:
            start = content_lower.find('limitation')
            section = content[start:start+500]
            sentences = re.split(r'[.!?]+', section)
            limitations = [s.strip() for s in sentences[1:3] if len(s.strip()) > 20]
        
        # Common limitations
        if 'computational cost' in content_lower or 'expensive' in content_lower:
            limitations.append('High computational cost')
        
        if 'data' in content_lower and 'limited' in content_lower:
            limitations.append('Limited training data')
        
        if not limitations:
            limitations = ['Not explicitly stated']
        
        return limitations[:3]
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content"""
        technical_terms = [
            'transformer', 'attention', 'bert', 'gpt', 'neural network',
            'deep learning', 'machine learning', 'embedding', 'encoder',
            'decoder', 'fine-tuning', 'pre-training', 'architecture',
            'optimization', 'backpropagation', 'gradient', 'lstm',
            'convolution', 'layer', 'model', 'training', 'inference',
            'reinforcement', 'reward', 'policy', 'q-learning', 'actor-critic',
            'nas', 'automl', 'neural architecture', 'search space'
        ]
        
        content_lower = content.lower()
        found_terms = []
        
        for term in technical_terms:
            if term in content_lower:
                count = content_lower.count(term)
                if count > 0:
                    found_terms.append((term, count))
        
        # Sort by frequency
        found_terms.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the terms (not counts)
        return [term for term, count in found_terms[:15]]
    
    def _generate_aggregate_insights(self, analyses: List[Dict]) -> Dict:
        """Generate insights across all papers"""
        if not analyses:
            return {}
        
        # Collect all findings
        all_findings = []
        for analysis in analyses:
            all_findings.extend(analysis.get('key_findings', []))
        
        # Collect methodologies
        methodologies = {}
        for analysis in analyses:
            method_type = analysis.get('methodology', {}).get('type', 'Unknown')
            methodologies[method_type] = methodologies.get(method_type, 0) + 1
        
        # Collect common themes (technical terms)
        term_frequency = {}
        for analysis in analyses:
            for term in analysis.get('technical_terms', []):
                term_frequency[term] = term_frequency.get(term, 0) + 1
        
        # Get top themes
        top_themes = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Methodological trends
        total_papers = len(analyses)
        method_distribution = {
            method: count/total_papers 
            for method, count in methodologies.items()
        }
        
        return {
            'total_findings': len(all_findings),
            'common_themes': [theme for theme, count in top_themes],
            'methodological_distribution': method_distribution,
            'dominant_methodology': max(methodologies.items(), key=lambda x: x[1])[0] if methodologies else 'Unknown'
        }
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent"""
        return self.agent

# Convenience function
def create_content_analyzer() -> ContentAnalyzerAgent:
    """Create and return Content Analyzer agent"""
    return ContentAnalyzerAgent()