"""
Agent 1: Paper Hunter
Searches for academic papers and returns structured results
"""
from crewai import Agent
from crewai_tools import SerperDevTool, FileReadTool
from langchain_openai import ChatOpenAI
from typing import Dict, List
import json
from config.settings import settings
from utils.logger import logger
from utils.memory import memory_manager

class PaperHunterAgent:
    """
    Agent specialized in finding academic papers
    """
    
    def __init__(self):
        """Initialize Paper Hunter Agent"""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3
        )
        
        # Built-in Tools
        self.search_tool = SerperDevTool()
        self.file_tool = FileReadTool()
        
        self.agent = Agent(
            role='Academic Research Librarian',
            goal='Find 10-15 highly relevant academic papers on the given research topic',
            backstory="""You are an expert research librarian with 20+ years of experience 
            in academic databases. You excel at:
            - Crafting precise search queries for academic content
            - Identifying authoritative sources (arxiv, scholar.google, IEEE, ACM)
            - Filtering out low-quality or irrelevant papers
            - Extracting key metadata (title, authors, year, citations)
            
            You prioritize recent papers (last 5 years) but also include foundational work.
            You ensure diversity in approaches and avoid duplicate or very similar papers.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.search_tool, self.file_tool]
        )
        
        logger.info("Paper Hunter Agent initialized")
    
    def create_search_query(self, user_query: str) -> str:
        """
        Enhance user query for academic search
        
        Args:
            user_query: User's research query
            
        Returns:
            Enhanced search query
        """
        enhanced_query = f"{user_query} research paper OR study OR arxiv"
        logger.info(f"Enhanced query: {enhanced_query}")
        return enhanced_query
    
    def execute_search(self, query: str) -> dict:
        """
        Execute search using SerperDevTool
        
        Args:
            query: Search query
            
        Returns:
            Search results as dictionary
        """
        try:
            results = self.search_tool._run(search_query=query)
            
            if isinstance(results, str):
                return json.loads(results)
            return results
            
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return {"organic": []}
    
    def parse_search_results(self, results: dict) -> List[Dict]:
        """
        Parse and structure search results
        
        Args:
            results: Results dictionary from search
            
        Returns:
            List of structured paper dictionaries
        """
        papers = []
        
        try:
            organic = results.get('organic', [])
            
            for idx, item in enumerate(organic[:15]):
                paper = {
                    'id': f"paper_{idx + 1:03d}",
                    'title': item.get('title', 'Unknown Title'),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': self._extract_source(item.get('link', '')),
                    'relevance_score': 0.0
                }
                
                paper['year'] = self._extract_year(
                    item.get('title', '') + ' ' + item.get('snippet', '')
                )
                
                papers.append(paper)
            
            logger.info(f"Parsed {len(papers)} papers from search results")
            
        except Exception as e:
            logger.error(f"Error parsing results: {e}")
        
        return papers
    
    def _extract_source(self, url: str) -> str:
        """Extract source name from URL"""
        url_lower = url.lower()
        if 'arxiv.org' in url_lower:
            return 'ArXiv'
        elif 'scholar.google' in url_lower:
            return 'Google Scholar'
        elif 'ieee.org' in url_lower:
            return 'IEEE'
        elif 'acm.org' in url_lower:
            return 'ACM'
        elif 'semanticscholar' in url_lower:
            return 'Semantic Scholar'
        elif 'researchgate' in url_lower:
            return 'ResearchGate'
        else:
            return 'Other'
    
    def _extract_year(self, text: str) -> int:
        """Extract publication year from text"""
        import re
        years = re.findall(r'\b(20[0-2][0-9])\b', text)
        if years:
            return int(years[-1])
        return 2023
    
    def calculate_relevance_scores(self, papers: List[Dict], query: str) -> List[Dict]:
        """Calculate relevance scores using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not papers:
            return papers
        
        try:
            query_text = query.lower()
            paper_texts = [
                f"{p['title']} {p['snippet']}".lower() 
                for p in papers
            ]
            
            vectorizer = TfidfVectorizer(stop_words='english')
            all_texts = [query_text] + paper_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            for idx, paper in enumerate(papers):
                paper['relevance_score'] = float(similarities[idx])
            
            papers.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            avg_score = sum(similarities)/len(similarities) if similarities.size > 0 else 0
            logger.info(f"Calculated relevance scores (avg: {avg_score:.2f})")
            
        except Exception as e:
            logger.warning(f"Failed to calculate relevance scores: {e}")
            for idx, paper in enumerate(papers):
                paper['relevance_score'] = 1.0 - (idx * 0.05)
        
        return papers
    
    def filter_papers(self, papers: List[Dict], min_relevance: float = 0.15) -> List[Dict]:
        """Filter papers by relevance"""
        filtered = [
            p for p in papers 
            if p['relevance_score'] >= min_relevance and p['url']
        ]
        
        logger.info(f"Filtered {len(papers)} papers to {len(filtered)} (min_relevance={min_relevance})")
        return filtered[:15]
    
    def search_papers(self, query: str) -> Dict:
        """Main method to search for papers"""
        logger.info(f"Starting paper search for: {query}")
        memory_manager.add_query(query)
        
        try:
            search_query = self.create_search_query(query)
            logger.info("Executing search...")
            results = self.execute_search(search_query)
            
            papers = self.parse_search_results(results)
            
            if not papers:
                logger.warning("No papers found in search results")
                return {
                    'success': False,
                    'papers': [],
                    'query': query,
                    'message': 'No papers found'
                }
            
            papers = self.calculate_relevance_scores(papers, query)
            papers = self.filter_papers(papers)
            
            if not papers:
                logger.warning("All papers filtered out, lowering threshold")
                papers = self.parse_search_results(results)
                papers = self.calculate_relevance_scores(papers, query)
                papers = self.filter_papers(papers, min_relevance=0.05)
            
            memory_manager.add_successful_search(query, len(papers))
            
            avg_relevance = sum(p['relevance_score'] for p in papers) / len(papers) if papers else 0
            
            memory_manager.add_agent_output('paper_hunter', {
                'query': query,
                'papers_found': len(papers),
                'avg_relevance': avg_relevance
            })
            
            logger.info(f"✅ Successfully found {len(papers)} papers (avg relevance: {avg_relevance:.2f})")
            
            return {
                'success': True,
                'papers': papers,
                'query': query,
                'search_query': search_query,
                'total_found': len(papers),
                'avg_relevance': avg_relevance,
                'message': f'Found {len(papers)} relevant papers'
            }
            
        except Exception as e:
            logger.error(f"Paper search failed: {e}")
            import traceback
            traceback.print_exc()
            
            memory_manager.add_error(str(e), "paper_hunter.search_papers")
            
            return {
                'success': False,
                'papers': [],
                'query': query,
                'message': f'Search failed: {str(e)}'
            }
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent"""
        return self.agent

def create_paper_hunter() -> PaperHunterAgent:
    """Create and return Paper Hunter agent"""
    return PaperHunterAgent()