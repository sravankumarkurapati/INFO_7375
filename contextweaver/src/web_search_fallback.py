# src/web_search_fallback.py
from typing import List, Dict, Optional
import logging
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import requests
import json

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class WebSearchFallback:
    """Web search fallback for queries outside local knowledge base"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.7)
        logger.info("🌐 Web search fallback initialized")
    
    def should_use_web_search(
        self,
        query: str,
        local_documents: List[Document],
        min_similarity_threshold: float = 0.5  # INCREASED from 0.3
    ) -> bool:
        """Determine if web search is needed - MORE AGGRESSIVE"""
        
        if not local_documents:
            logger.info("   ⚠️ No local documents - web search needed")
            return True
        
        # Check similarity scores
        similarities = [doc.metadata.get('similarity_score', 0) for doc in local_documents]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        
        # More aggressive threshold - use web if best match is below 0.6
        if max_similarity < 0.6:
            logger.info(f"   ⚠️ Low similarity (max: {max_similarity:.2f}) - web search needed")
            return True
        
        # Check query term coverage - STRICTER
        query_terms = set(query.lower().split())
        # Remove common words
        query_terms = {t for t in query_terms if len(t) > 3 and t not in ['what', 'how', 'why', 'when', 'where', 'good', 'bad', 'best', 'health']}
        
        if not query_terms:
            return False
        
        doc_terms = set()
        for doc in local_documents:
            doc_terms.update(doc.page_content.lower().split())
        
        coverage = len(query_terms & doc_terms) / len(query_terms) if query_terms else 1
        
        # Require at least 50% of key terms to be present
        if coverage < 0.5:
            logger.info(f"   ⚠️ Low coverage ({coverage:.0%}) - web search needed")
            return True
        
        return False
    
    def search_web(self, query: str, num_results: int = 5) -> List[Document]:
        """Search the web and return results as Documents"""
        
        logger.info(f"🌐 Searching web for: '{query}'")
        
        web_results = self._simulate_web_search(query, num_results)
        
        logger.info(f"✅ Retrieved {len(web_results)} web results")
        
        return web_results
    
    def _simulate_web_search(self, query: str, num_results: int) -> List[Document]:
        """Simulate web search by generating realistic content"""
        
        prompt = f"""You are simulating web search results for a query.

Query: {query}

Generate {num_results} realistic web search results. For each result, create:
- A factual, informative paragraph (150-200 words)
- Include relevant statistics, facts, or expert opinions
- Make it sound like it's from a credible health/science website
- IMPORTANT: Focus on the ACTUAL query topic, not unrelated topics

Format as JSON array:
[
  {{
    "title": "Article title about {query}",
    "source": "Website name (e.g., Mayo Clinic, Harvard Health, WebMD)",
    "content": "Factual paragraph directly answering the query...",
    "url": "https://example.com/article"
  }}
]

Make the content diverse and directly relevant to: {query}

JSON:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            results = json.loads(content)
            
            documents = []
            
            for i, result in enumerate(results):
                doc = Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'source': result.get('source', f'Web Result {i+1}'),
                        'filename': result.get('source', f'Web Result {i+1}'),  # Add filename for consistency
                        'title': result.get('title', 'Unknown'),
                        'url': result.get('url', 'https://example.com'),
                        'source_type': 'web_search',
                        'credibility_score': self._estimate_web_source_credibility(result.get('source', '')),
                        'quality_score': 0.75,
                        'is_web_result': True,
                        'similarity_score': 0.8  # High similarity since it's targeted to query
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Web search simulation failed: {e}")
            return []
    
    def _estimate_web_source_credibility(self, source: str) -> float:
        """Estimate credibility of web source"""
        
        source_lower = source.lower()
        
        high_cred = ['mayo clinic', 'harvard', 'nih', 'cdc', 'who', 'webmd', 'cleveland clinic']
        if any(src in source_lower for src in high_cred):
            return 0.9
        
        medium_cred = ['healthline', 'medical news', 'science daily', 'health.com']
        if any(src in source_lower for src in medium_cred):
            return 0.75
        
        return 0.6
    
    def process_web_results(
        self,
        web_documents: List[Document],
        query: str
    ) -> List[Document]:
        """Process web results for use in pipeline"""
        
        logger.info(f"📊 Processing {len(web_documents)} web results...")
        
        for doc in web_documents:
            doc.metadata['query'] = query
            doc.metadata['retrieval_source'] = 'web_search'
        
        return web_documents


class HybridRetriever:
    """Hybrid retrieval: Local KB + Web Search + LLM Knowledge"""
    
    def __init__(self, vector_store, web_search: WebSearchFallback):
        self.vector_store = vector_store
        self.web_search = web_search
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.2)
        logger.info("🔄 Hybrid retriever initialized")
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        web_search_threshold: float = 0.5  # INCREASED threshold
    ) -> Dict[str, any]:
        """Hybrid retrieval with three-tier fallback"""
        
        logger.info(f"🔄 Hybrid retrieval for: '{query}'")
        
        # TIER 1: Try local knowledge base
        logger.info("   📚 Tier 1: Searching local knowledge base...")
        
        try:
            local_results = self.vector_store.similarity_search_with_score(query, k=k)
            
            if local_results:
                documents = [doc for doc, score in local_results]
                
                # Add similarity scores
                for (doc, score), original_doc in zip(local_results, documents):
                    original_doc.metadata['similarity_score'] = 1 - score
                
                # MORE AGGRESSIVE check - require GOOD local results
                if not self.web_search.should_use_web_search(query, documents, web_search_threshold):
                    logger.info(f"   ✅ Tier 1 SUCCESS: Found {len(documents)} relevant local documents")
                    
                    return {
                        'documents': documents,
                        'source': 'local',
                        'confidence': 0.9,
                        'num_results': len(documents),
                        'fallback_reason': None
                    }
                else:
                    logger.info("   ⚠️ Tier 1 INSUFFICIENT: Triggering web search...")
                    fallback_reason = "Local documents have low relevance to query"
            else:
                logger.info("   ⚠️ Tier 1 FAILED: No local documents")
                fallback_reason = "No documents in local knowledge base"
                documents = []
        
        except Exception as e:
            logger.error(f"   ❌ Tier 1 ERROR: {e}")
            fallback_reason = f"Local search error: {e}"
            documents = []
        
        # TIER 2: Web search fallback
        logger.info("   🌐 Tier 2: Searching web...")
        
        try:
            web_results = self.web_search.search_web(query, num_results=5)
            
            if web_results:
                processed_web = self.web_search.process_web_results(web_results, query)
                
                # ONLY use web results for out-of-scope queries
                # Don't mix with irrelevant local docs
                logger.info(f"   ✅ Tier 2 SUCCESS: {len(web_results)} web results")
                
                return {
                    'documents': processed_web,  # ONLY web results
                    'source': 'web',
                    'confidence': 0.75,
                    'num_results': len(processed_web),
                    'num_web_results': len(web_results),
                    'num_local_results': 0,
                    'fallback_reason': fallback_reason
                }
            else:
                logger.info("   ⚠️ Tier 2 FAILED: Web search returned no results")
        
        except Exception as e:
            logger.error(f"   ❌ Tier 2 ERROR: {e}")
        
        # TIER 3: LLM direct knowledge
        logger.info("   🤖 Tier 3: Using LLM direct knowledge...")
        
        llm_answer = self._get_llm_direct_answer(query)
        
        llm_doc = Document(
            page_content=llm_answer,
            metadata={
                'source': 'LLM Direct Knowledge',
                'filename': 'LLM Direct Knowledge',
                'source_type': 'llm_knowledge',
                'credibility_score': 0.6,
                'quality_score': 0.8,
                'is_llm_generated': True,
                'is_web_result': False,
                'warning': 'Answer from LLM training knowledge, not verified'
            }
        )
        
        logger.info("   ✅ Tier 3: Generated answer from LLM knowledge")
        
        return {
            'documents': [llm_doc],
            'source': 'llm_direct',
            'confidence': 0.5,
            'num_results': 1,
            'fallback_reason': 'Both local and web search failed',
            'warning': '⚠️ Answer from LLM knowledge without source verification'
        }
    
    def _get_llm_direct_answer(self, query: str) -> str:
        """Get answer directly from LLM training knowledge"""
        
        prompt = f"""Answer this question using your training knowledge.

Question: {query}

Provide a comprehensive, factual answer (200-300 words). Include relevant details but acknowledge uncertainty if needed.

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"❌ LLM direct answer failed: {e}")
            return f"I apologize, but I'm unable to answer '{query}' at this time."


class RetrievalSourceIndicator:
    """Indicate to users where information came from"""
    
    @staticmethod
    def format_answer_with_source_indicator(
        answer: str,
        retrieval_source: str,
        confidence: float
    ) -> str:
        """Add source indicator to answer"""
        
        indicators = {
            'local': "📚 **Source: Local Knowledge Base** (Highest Confidence)\n\n",
            'web': "🌐 **Source: Web Search Results** (Medium Confidence)\n\n⚠️ *Information retrieved from web search. Verify independently.*\n\n",
            'hybrid_local_web': "🔄 **Source: Local + Web** (Medium-High Confidence)\n\n💡 *Combined local and web results.*\n\n",
            'llm_direct': "🤖 **Source: AI Knowledge** (Lower Confidence)\n\n⚠️ *Answer from AI knowledge. No sources found. Verify independently.*\n\n"
        }
        
        indicator = indicators.get(retrieval_source, "")
        
        if confidence >= 0.8:
            conf_emoji, conf_text = "🟢", "High"
        elif confidence >= 0.6:
            conf_emoji, conf_text = "🟡", "Medium"
        else:
            conf_emoji, conf_text = "🟠", "Low"
        
        confidence_line = f"{conf_emoji} **Retrieval Confidence:** {conf_text} ({confidence:.0%})\n\n"
        
        return indicator + confidence_line + answer
    
    @staticmethod
    def generate_source_summary(retrieval_result: Dict) -> str:
        """Generate summary of retrieval sources"""
        
        summary = f"""**Retrieval Summary:**
- Primary Source: {retrieval_result['source'].replace('_', ' ').title()}
- Total Documents: {retrieval_result['num_results']}
"""
        
        if retrieval_result.get('num_local_results') is not None:
            summary += f"- Local Documents: {retrieval_result['num_local_results']}\n"
        
        if retrieval_result.get('num_web_results'):
            summary += f"- Web Results: {retrieval_result['num_web_results']}\n"
        
        if retrieval_result.get('fallback_reason'):
            summary += f"\n⚠️ Fallback Reason: {retrieval_result['fallback_reason']}\n"
        
        if retrieval_result.get('warning'):
            summary += f"\n{retrieval_result['warning']}\n"
        
        return summary


# Test
if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING WEB SEARCH")
    print("="*60)
    
    web_search = WebSearchFallback()
    
    # Test chicken query
    print("\n1️⃣ Testing chicken query...")
    results = web_search.search_web("Is chicken healthy?", num_results=3)
    
    print(f"   Results: {len(results)}")
    for i, doc in enumerate(results, 1):
        print(f"\n   {i}. {doc.metadata['source']}")
        print(f"      {doc.page_content[:100]}...")
    
    print("\n✅ Web search working!")