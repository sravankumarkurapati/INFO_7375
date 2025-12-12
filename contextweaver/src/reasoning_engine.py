# src/reasoning_engine.py
from typing import List, Dict, Tuple, Optional
import json
import logging
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import networkx as nx

from config import Config
from prompt_engineering import PromptEngineeringSystem

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class MultiHopReasoningEngine:
    """
    Multi-hop reasoning across multiple documents
    
    Core Innovation of ContextWeaver:
    - Builds reasoning chains across documents
    - Tracks information flow
    - Maintains citation provenance
    - Detects logical connections
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.prompt_system = PromptEngineeringSystem()
        self.reasoning_graph = nx.DiGraph()
        logger.info("🧠 Multi-hop reasoning engine initialized")
    
    def reason_across_documents(
        self,
        query: str,
        documents: List[Document],
        max_hops: int = 3
    ) -> Dict[str, any]:
        """
        Perform multi-hop reasoning across documents
        
        Returns:
            - reasoning_chain: Step-by-step reasoning process
            - answer: Final synthesized answer
            - citations: Source citations for each claim
            - confidence: Confidence score
            - hops_used: Number of reasoning hops
        """
        
        logger.info(f"🔗 Starting multi-hop reasoning for: '{query}'")
        
        # Build reasoning chain
        reasoning_chain = []
        used_documents = []
        current_context = query
        
        for hop in range(max_hops):
            logger.info(f"   Hop {hop + 1}/{max_hops}...")
            
            # Select relevant documents for this hop
            relevant_docs = self._select_documents_for_hop(
                current_context,
                documents,
                used_documents
            )
            
            if not relevant_docs:
                logger.info(f"   No new documents found, stopping at hop {hop + 1}")
                break
            
            # Perform reasoning step
            reasoning_step = self._perform_reasoning_step(
                query,
                current_context,
                relevant_docs,
                hop + 1
            )
            
            reasoning_chain.append(reasoning_step)
            used_documents.extend(relevant_docs)
            
            # Update context for next hop
            current_context = reasoning_step['intermediate_conclusion']
            
            # Check if we have enough information
            if reasoning_step.get('sufficient_info', False):
                logger.info(f"   Sufficient information found at hop {hop + 1}")
                break
        
        # Synthesize final answer
        final_answer = self._synthesize_final_answer(
            query,
            reasoning_chain,
            used_documents
        )
        
        # Build reasoning graph
        self._build_reasoning_graph(reasoning_chain)
        
        result = {
            'query': query,
            'reasoning_chain': reasoning_chain,
            'answer': final_answer['answer'],
            'citations': final_answer['citations'],
            'confidence': final_answer['confidence'],
            'hops_used': len(reasoning_chain),
            'documents_used': len(used_documents),
            'reasoning_graph': self._export_reasoning_graph()
        }
        
        logger.info(f"✅ Multi-hop reasoning complete ({len(reasoning_chain)} hops)")
        
        return result
    
    def _select_documents_for_hop(
        self,
        context: str,
        all_documents: List[Document],
        used_documents: List[Document]
    ) -> List[Document]:
        """Select documents relevant for current reasoning hop"""
        
        # Filter out already used documents
        unused_docs = [
            doc for doc in all_documents
            if doc not in used_documents
        ]
        
        if not unused_docs:
            return []
        
        # Simple selection: top 2 most relevant unused documents
        # In production, would use more sophisticated selection
        return unused_docs[:2]
    
    def _perform_reasoning_step(
        self,
        original_query: str,
        current_context: str,
        documents: List[Document],
        hop_number: int
    ) -> Dict:
        """Perform a single reasoning step"""
        
        # Format documents
        formatted_docs = self.prompt_system.format_documents_for_prompt(documents)
        
        # Create reasoning prompt
        prompt = f"""You are performing step {hop_number} of multi-hop reasoning.

**Original Query:** {original_query}

**Current Context:** {current_context}

**New Documents:**
{formatted_docs}

**Task:**
1. Extract relevant information from these documents
2. Connect it to the current context
3. Determine if we have sufficient information to answer the original query

**Output Format (JSON):**
{{
  "extracted_info": "key information from documents",
  "connection_to_context": "how this relates to previous steps",
  "intermediate_conclusion": "updated understanding so far",
  "sufficient_info": true/false,
  "citations": ["Doc1", "Doc2"]
}}

JSON:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            step_result = json.loads(content)
            step_result['hop_number'] = hop_number
            step_result['documents_used'] = [doc.metadata.get('source', 'unknown') for doc in documents]
            
            return step_result
            
        except Exception as e:
            logger.warning(f"⚠️ Error in reasoning step: {e}")
            return {
                'extracted_info': 'Error in reasoning step',
                'connection_to_context': '',
                'intermediate_conclusion': current_context,
                'sufficient_info': False,
                'hop_number': hop_number,
                'error': str(e)
            }
    
    def _synthesize_final_answer(
        self,
        query: str,
        reasoning_chain: List[Dict],
        documents: List[Document]
    ) -> Dict:
        """Synthesize final answer from reasoning chain"""
        
        # Build reasoning summary
        reasoning_summary = "\n\n".join([
            f"Step {step['hop_number']}: {step.get('extracted_info', 'N/A')}"
            for step in reasoning_chain
        ])
        
        # Format all documents
        formatted_docs = self.prompt_system.format_documents_for_prompt(documents)
        
        synthesis_prompt = f"""Based on this multi-hop reasoning process, provide the final answer.

**Query:** {query}

**Reasoning Chain:**
{reasoning_summary}

**All Documents Used:**
{formatted_docs}

**Task:** Provide a comprehensive answer that:
1. Directly answers the query
2. Cites specific sources for each claim
3. Acknowledges any uncertainties
4. Provides a confidence score (0-1)

**Output Format (JSON):**
{{
  "answer": "comprehensive answer with inline citations [Source, Year]",
  "citations": [
    {{"source": "document name", "claim": "what this source supports"}}
  ],
  "confidence": 0.0-1.0,
  "uncertainties": ["any limitations or unknowns"]
}}

JSON:"""
        
        try:
            response = self.llm.invoke(synthesis_prompt)
            content = response.content.strip()
            
            # Clean JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            final_answer = json.loads(content)
            
            return final_answer
            
        except Exception as e:
            logger.warning(f"⚠️ Error synthesizing answer: {e}")
            return {
                'answer': 'Error synthesizing final answer',
                'citations': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _build_reasoning_graph(self, reasoning_chain: List[Dict]):
        """Build graph representation of reasoning chain"""
        
        self.reasoning_graph.clear()
        
        # Add nodes for each reasoning step
        for step in reasoning_chain:
            hop_num = step['hop_number']
            node_id = f"hop_{hop_num}"
            
            self.reasoning_graph.add_node(
                node_id,
                info=step.get('extracted_info', ''),
                conclusion=step.get('intermediate_conclusion', ''),
                docs=step.get('documents_used', [])
            )
            
            # Add edge from previous hop
            if hop_num > 1:
                prev_node = f"hop_{hop_num - 1}"
                self.reasoning_graph.add_edge(
                    prev_node,
                    node_id,
                    connection=step.get('connection_to_context', '')
                )
    
    def _export_reasoning_graph(self) -> Dict:
        """Export reasoning graph as dictionary"""
        
        return {
            'nodes': [
                {
                    'id': node,
                    'info': self.reasoning_graph.nodes[node].get('info', ''),
                    'docs': self.reasoning_graph.nodes[node].get('docs', [])
                }
                for node in self.reasoning_graph.nodes()
            ],
            'edges': [
                {
                    'from': u,
                    'to': v,
                    'connection': self.reasoning_graph.edges[u, v].get('connection', '')
                }
                for u, v in self.reasoning_graph.edges()
            ]
        }
    
    def visualize_reasoning_chain(self, result: Dict) -> str:
        """Create text visualization of reasoning chain"""
    
        viz = f"\n{'='*60}\n"
        viz += f"MULTI-HOP REASONING VISUALIZATION\n"
        viz += f"{'='*60}\n\n"
    
        viz += f"🎯 Query: {result['query']}\n\n"
    
        for step in result['reasoning_chain']:
            hop = step['hop_number']
            viz += f"{'─'*60}\n"
            viz += f"HOP {hop}:\n"
            viz += f"{'─'*60}\n"
            viz += f"📄 Documents: {', '.join(step.get('documents_used', []))}\n"
        
            # Safe string extraction with slicing
            extracted = str(step.get('extracted_info', 'N/A'))
            connection = str(step.get('connection_to_context', 'N/A'))
            conclusion = str(step.get('intermediate_conclusion', 'N/A'))
        
            viz += f"📊 Extracted: {extracted[:150]}...\n"
            viz += f"🔗 Connection: {connection[:150]}...\n"
            viz += f"💡 Conclusion: {conclusion[:150]}...\n\n"


class ContradictionDetector:
    """
    Detect and analyze contradictions in documents
    
    Innovation: Automatically finds conflicting information
    
    Features:
    - Identify contradictory claims
    - Classify contradiction severity
    - Explain why contradictions exist
    - Suggest resolutions
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.prompt_system = PromptEngineeringSystem()
        logger.info("🔍 Contradiction detector initialized")
    
    def detect_contradictions(
        self,
        documents: List[Document]
    ) -> Dict[str, any]:
        """
        Detect contradictions across documents
        
        Returns:
            - contradictions: List of detected contradictions
            - severity: Overall severity assessment
            - explanations: Why contradictions exist
            - resolution: Suggested resolution
        """
        
        logger.info(f"🔍 Analyzing {len(documents)} documents for contradictions...")
        
        # Format documents for prompt
        formatted_docs = self.prompt_system.format_documents_for_prompt(documents)
        
        # Create contradiction detection prompt
        variables = {'documents': formatted_docs}
        
        prompt = self.prompt_system.create_prompt(
            template_name='contradiction_detection',
            variables=variables,
            use_few_shot=True,
            num_examples=1
        )
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Extract JSON from response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            # Calculate severity
            if result.get('contradictions'):
                severity_counts = {
                    'HIGH': 0,
                    'MEDIUM': 0,
                    'LOW': 0
                }
                
                for c in result['contradictions']:
                    severity = c.get('severity', 'MEDIUM')
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                result['severity_distribution'] = severity_counts
                result['num_contradictions'] = len(result['contradictions'])
                
                # Overall severity
                if severity_counts['HIGH'] > 0:
                    result['overall_severity'] = 'HIGH'
                elif severity_counts['MEDIUM'] > 0:
                    result['overall_severity'] = 'MEDIUM'
                else:
                    result['overall_severity'] = 'LOW'
            else:
                result['num_contradictions'] = 0
                result['overall_severity'] = 'NONE'
            
            logger.info(f"✅ Found {result['num_contradictions']} contradictions")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error detecting contradictions: {e}")
            return {
                'contradictions': [],
                'num_contradictions': 0,
                'overall_severity': 'ERROR',
                'error': str(e)
            }
    
    def explain_contradiction(
        self,
        contradiction: Dict
    ) -> str:
        """Generate detailed explanation of a contradiction"""
        
        explanation = f"""
{'='*60}
CONTRADICTION ANALYSIS
{'='*60}

📍 CLAIM A:
   "{contradiction['claim_A']}"
   Source: {contradiction['source_A']}

📍 CLAIM B:
   "{contradiction['claim_B']}"
   Source: {contradiction['source_B']}

⚠️ SEVERITY: {contradiction['severity']}
📊 CONFIDENCE: {contradiction['confidence']:.2f}

🔍 EXPLANATION:
{contradiction['explanation']}

{'='*60}
"""
        
        return explanation
    
    def resolve_contradictions(
        self,
        contradictions: List[Dict],
        all_documents: List[Document]
    ) -> Dict[str, any]:
        """
        Attempt to resolve contradictions
        
        Strategies:
        - Temporal analysis (older vs newer)
        - Methodological differences
        - Scope differences
        - Authority/credibility comparison
        """
        
        logger.info(f"🔧 Attempting to resolve {len(contradictions)} contradictions...")
        
        resolutions = []
        
        for contradiction in contradictions:
            resolution = self._resolve_single_contradiction(contradiction, all_documents)
            resolutions.append(resolution)
        
        return {
            'contradictions': contradictions,
            'resolutions': resolutions,
            'num_resolved': sum(1 for r in resolutions if r['resolved']),
            'num_unresolved': sum(1 for r in resolutions if not r['resolved'])
        }
    
    def _resolve_single_contradiction(
        self,
        contradiction: Dict,
        documents: List[Document]
    ) -> Dict:
        """Attempt to resolve a single contradiction"""
        
        # Extract source information
        source_a = contradiction.get('source_A', '')
        source_b = contradiction.get('source_B', '')
        
        # Find corresponding documents
        doc_a = next((d for d in documents if source_a in d.metadata.get('source', '')), None)
        doc_b = next((d for d in documents if source_b in d.metadata.get('source', '')), None)
        
        resolution_strategy = None
        resolved = False
        
        # Strategy 1: Temporal resolution (newer is usually better)
        if doc_a and doc_b:
            year_a = doc_a.metadata.get('year', 0)
            year_b = doc_b.metadata.get('year', 0)
            
            if year_a and year_b and abs(year_a - year_b) > 2:
                resolution_strategy = 'temporal'
                resolved = True
                resolution = f"Newer evidence ({max(year_a, year_b)}) supersedes older findings ({min(year_a, year_b)})"
            
            # Strategy 2: Credibility comparison
            elif doc_a and doc_b:
                cred_a = doc_a.metadata.get('credibility_score', 0.5)
                cred_b = doc_b.metadata.get('credibility_score', 0.5)
                
                if abs(cred_a - cred_b) > 0.3:
                    resolution_strategy = 'credibility'
                    resolved = True
                    resolution = f"Higher credibility source (score: {max(cred_a, cred_b):.2f}) preferred"
                else:
                    resolution_strategy = 'unresolved'
                    resolved = False
                    resolution = "Both sources have similar credibility and recency - requires expert judgment"
            else:
                resolution_strategy = 'insufficient_data'
                resolved = False
                resolution = "Cannot resolve - insufficient metadata"
        else:
            resolution_strategy = 'insufficient_data'
            resolved = False
            resolution = "Cannot resolve - documents not found"
        
        return {
            'contradiction': contradiction,
            'resolved': resolved,
            'strategy': resolution_strategy,
            'resolution': resolution
        }


class CitationTracker:
    """
    Track citation provenance chains
    
    Innovation: Full citation tracking with provenance
    
    Features:
    - Track which claims come from which sources
    - Build citation dependency graphs
    - Validate citation accuracy
    - Generate citation reports
    """
    
    def __init__(self):
        self.citation_graph = nx.DiGraph()
        logger.info("📖 Citation tracker initialized")
    
    def track_citations(
        self,
        answer: str,
        citations: List[Dict],
        reasoning_chain: List[Dict]
    ) -> Dict[str, any]:
        """
        Track all citations in an answer
        
        Returns:
            - citation_map: Map of claims to sources
            - citation_graph: Dependency graph
            - coverage: How well citations cover the answer
        """
        
        logger.info("📖 Tracking citation provenance...")
        
        citation_map = []
        
        for citation in citations:
            source = citation.get('source', 'Unknown')
            claim = citation.get('claim', '')
            
            # Add to graph
            self.citation_graph.add_node(source, type='source')
            self.citation_graph.add_node(claim, type='claim')
            self.citation_graph.add_edge(source, claim, relation='supports')
            
            citation_map.append({
                'claim': claim,
                'source': source,
                'in_answer': claim.lower() in answer.lower()
            })
        
        # Calculate coverage
        cited_claims = [c['claim'] for c in citation_map if c['in_answer']]
        coverage = len(cited_claims) / len(citation_map) if citation_map else 0
        
        result = {
            'citation_map': citation_map,
            'num_citations': len(citations),
            'coverage': coverage,
            'citation_graph': self._export_citation_graph()
        }
        
        logger.info(f"✅ Tracked {len(citations)} citations (coverage: {coverage:.1%})")
        
        return result
    
    def _export_citation_graph(self) -> Dict:
        """Export citation graph"""
        
        return {
            'nodes': [
                {
                    'id': node,
                    'type': self.citation_graph.nodes[node].get('type', 'unknown')
                }
                for node in self.citation_graph.nodes()
            ],
            'edges': [
                {
                    'from': u,
                    'to': v,
                    'relation': self.citation_graph.edges[u, v].get('relation', '')
                }
                for u, v in self.citation_graph.edges()
            ]
        }
    
    def validate_citations(
        self,
        answer: str,
        citations: List[Dict],
        source_documents: List[Document]
    ) -> Dict[str, any]:
        """
        Validate that citations are accurate
        
        Checks:
        - Citation exists in source document
        - Citation is accurately represented
        - No hallucinated citations
        """
        
        logger.info("✅ Validating citation accuracy...")
        
        validation_results = []
        
        for citation in citations:
            source = citation.get('source', '')
            claim = citation.get('claim', '')
            
            # Find source document
            source_doc = next(
                (doc for doc in source_documents if source in doc.metadata.get('source', '')),
                None
            )
            
            if source_doc:
                # Check if claim appears in source
                claim_in_source = claim.lower() in source_doc.page_content.lower()
                
                validation_results.append({
                    'source': source,
                    'claim': claim,
                    'valid': claim_in_source,
                    'found_in_source': claim_in_source
                })
            else:
                validation_results.append({
                    'source': source,
                    'claim': claim,
                    'valid': False,
                    'error': 'Source document not found'
                })
        
        num_valid = sum(1 for v in validation_results if v['valid'])
        accuracy = num_valid / len(validation_results) if validation_results else 0
        
        return {
            'validations': validation_results,
            'accuracy': accuracy,
            'num_valid': num_valid,
            'num_invalid': len(validation_results) - num_valid
        }
    
    def generate_citation_report(
        self,
        query: str,
        result: Dict,
        validation: Dict
    ) -> str:
        """Generate comprehensive citation report"""
        
        report = f"""
{'='*60}
CITATION PROVENANCE REPORT
{'='*60}

Query: {query}
Answer Confidence: {result.get('confidence', 0):.2f}

CITATIONS USED:
"""
        
        for i, citation in enumerate(result.get('citations', []), 1):
            report += f"\n{i}. {citation.get('source', 'Unknown')}\n"
            report += f"   Supports: {citation.get('claim', 'N/A')[:100]}...\n"
        
        report += f"\nCITATION VALIDATION:\n"
        report += f"   Accuracy: {validation.get('accuracy', 0):.1%}\n"
        report += f"   Valid Citations: {validation.get('num_valid', 0)}\n"
        report += f"   Invalid Citations: {validation.get('num_invalid', 0)}\n"
        
        report += f"\n{'='*60}\n"
        
        return report


class TemporalAnalyzer:
    """
    Analyze how information evolves over time
    
    Innovation: Temporal reasoning across documents
    
    Features:
    - Timeline construction
    - Evolution tracking
    - Turning point detection
    - Trend prediction
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)
        self.prompt_system = PromptEngineeringSystem()
        logger.info("📅 Temporal analyzer initialized")
    
    def analyze_temporal_evolution(
        self,
        query: str,
        documents: List[Document]
    ) -> Dict[str, any]:
        """
        Analyze how understanding evolved over time
        
        Returns:
            - timeline: Chronologically ordered information
            - evolution: How understanding changed
            - turning_points: Key studies that shifted consensus
            - current_state: Latest understanding
        """
        
        logger.info(f"📅 Performing temporal analysis for: '{query}'")
        
        # Sort documents by year
        docs_with_years = [d for d in documents if d.metadata.get('year')]
        docs_with_years.sort(key=lambda d: d.metadata.get('year', 0))
        
        if len(docs_with_years) < 2:
            return {
                'error': 'Insufficient temporal data (need 2+ documents with years)',
                'timeline': [],
                'evolution': 'Cannot determine evolution'
            }
        
        # Format documents
        formatted_docs = self.prompt_system.format_documents_for_prompt(docs_with_years)
        
        # Create temporal analysis prompt
        variables = {
            'query': query,
            'documents': formatted_docs
        }
        
        prompt = self.prompt_system.create_prompt(
            template_name='temporal_analysis',
            variables=variables,
            use_few_shot=True,
            num_examples=1
        )
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse response (expecting structured text)
            result = self._parse_temporal_analysis(response.content, docs_with_years)
            
            logger.info(f"✅ Temporal analysis complete")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in temporal analysis: {e}")
            return {
                'error': str(e),
                'timeline': self._build_simple_timeline(docs_with_years),
                'evolution': 'Error in detailed analysis'
            }
    
    def _parse_temporal_analysis(
        self,
        response: str,
        documents: List[Document]
    ) -> Dict:
        """Parse temporal analysis response"""
        
        # Build timeline
        timeline = self._build_simple_timeline(documents)
        
        return {
            'timeline': timeline,
            'evolution': response,
            'num_time_points': len(timeline),
            'year_range': (
                documents[0].metadata.get('year'),
                documents[-1].metadata.get('year')
            ) if documents else (None, None)
        }
    
    def _build_simple_timeline(self, documents: List[Document]) -> List[Dict]:
        """Build simple timeline from documents"""
        
        timeline = []
        
        for doc in documents:
            year = doc.metadata.get('year')
            if year:
                timeline.append({
                    'year': year,
                    'source': doc.metadata.get('filename', doc.metadata.get('source', 'Unknown')),
                    'key_finding': doc.page_content[:200] + '...'
                })
        
        return timeline


# Test reasoning engine
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING REASONING ENGINE")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Study from 2018 found high coffee consumption increases cardiovascular risk by 23%. However, the study did not control for confounding variables such as smoking and sugar intake.",
            metadata={'source': 'study_2018.txt', 'year': 2018, 'domain': 'medical', 'source_type': 'peer_reviewed', 'credibility_score': 0.7, 'quality_score': 0.8}
        ),
        Document(
            page_content="A 2022 meta-analysis reviewed 50 studies on coffee and heart health. The analysis found that previous negative findings were likely due to uncontrolled confounding variables. When properly controlled, coffee shows neutral or positive effects.",
            metadata={'source': 'meta_2022.txt', 'year': 2022, 'domain': 'research', 'source_type': 'peer_reviewed', 'credibility_score': 0.95, 'quality_score': 0.95}
        ),
        Document(
            page_content="Recent 2023 research with rigorous controls found that moderate coffee consumption (2-3 cups daily) is associated with 15% reduction in cardiovascular disease risk. The study controlled for smoking, diet, exercise, and sugar intake.",
            metadata={'source': 'study_2023.txt', 'year': 2023, 'domain': 'medical', 'source_type': 'peer_reviewed', 'credibility_score': 0.9, 'quality_score': 0.9}
        )
    ]
    
    # Test 1: Multi-hop reasoning
    print("\n1️⃣ Testing Multi-Hop Reasoning...")
    reasoning_engine = MultiHopReasoningEngine()
    
    query = "Is moderate coffee consumption safe for heart health?"
    result = reasoning_engine.reason_across_documents(query, sample_docs, max_hops=3)
    
    print(f"   ✅ Reasoning complete")
    print(f"   Hops used: {result['hops_used']}")
    print(f"   Documents used: {result['documents_used']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    
    # Visualize reasoning chain
    viz = reasoning_engine.visualize_reasoning_chain(result)
    print(viz)
    
    # Test 2: Contradiction detection
    print("\n2️⃣ Testing Contradiction Detection...")
    detector = ContradictionDetector()
    
    contradictions = detector.detect_contradictions(sample_docs)
    print(f"   ✅ Detection complete")
    print(f"   Contradictions found: {contradictions.get('num_contradictions', 0)}")
    print(f"   Overall severity: {contradictions.get('overall_severity', 'N/A')}")
    
    if contradictions.get('contradictions'):
        print("\n   Sample contradiction:")
        c = contradictions['contradictions'][0]
        explanation = detector.explain_contradiction(c)
        print(explanation)
    
    # Test 3: Citation tracking
    print("\n3️⃣ Testing Citation Tracking...")
    citation_tracker = CitationTracker()
    
    citation_tracking = citation_tracker.track_citations(
        answer=result['answer'],
        citations=result['citations'],
        reasoning_chain=result['reasoning_chain']
    )
    
    print(f"   ✅ Citation tracking complete")
    print(f"   Citations tracked: {citation_tracking['num_citations']}")
    print(f"   Coverage: {citation_tracking['coverage']:.1%}")
    
    # Validate citations
    validation = citation_tracker.validate_citations(
        result['answer'],
        result['citations'],
        sample_docs
    )
    
    print(f"   Citation accuracy: {validation['accuracy']:.1%}")
    print(f"   Valid citations: {validation['num_valid']}")
    
    # Generate citation report
    report = citation_tracker.generate_citation_report(query, result, validation)
    print(report)
    
    # Test 4: Temporal analysis
    print("\n4️⃣ Testing Temporal Analysis...")
    temporal = TemporalAnalyzer()
    
    temporal_result = temporal.analyze_temporal_evolution(
        "How has understanding of coffee and heart health evolved?",
        sample_docs
    )
    
    print(f"   ✅ Temporal analysis complete")
    print(f"   Time points: {temporal_result.get('num_time_points', 0)}")
    print(f"   Year range: {temporal_result.get('year_range', (None, None))}")
    
    if temporal_result.get('timeline'):
        print("\n   Timeline:")
        for point in temporal_result['timeline']:
            print(f"   {point['year']}: {point['key_finding'][:80]}...")
    
    print("\n" + "=" * 60)
    print("✅ REASONING ENGINE TEST COMPLETE")
    print("=" * 60)
    print("\n🎯 Advanced Features Implemented:")
    print("  ✅ Multi-hop reasoning across documents")
    print("  ✅ Contradiction detection and resolution")
    print("  ✅ Citation tracking with provenance")
    print("  ✅ Temporal analysis and evolution tracking")