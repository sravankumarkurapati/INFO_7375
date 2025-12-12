# src/document_graph.py
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
import logging
from langchain_core.documents import Document
import json
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DocumentGraph:
    """
    Knowledge graph of document relationships
    
    INNOVATION: Graph-based document organization and reasoning
    
    Nodes: Documents
    Edges: Relationships (cites, contradicts, supports, temporal_successor)
    
    Features:
    - Automatic relationship detection
    - Graph traversal for retrieval
    - Importance scoring (PageRank)
    - Visual network analysis
    - Path finding between documents
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.document_index = {}  # Map doc_id to Document
        logger.info("📊 Document graph initialized")
    
    def build_graph(
        self,
        documents: List[Document],
        contradictions: Optional[List[Dict]] = None,
        temporal_order: bool = True
    ):
        """
        Build knowledge graph from documents
        
        Args:
            documents: List of documents to add to graph
            contradictions: Pre-detected contradictions
            temporal_order: Add temporal succession edges
        """
        
        logger.info(f"🔨 Building graph from {len(documents)} documents...")
        
        # Add documents as nodes
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            self.document_index[doc_id] = doc
            
            self.graph.add_node(
                doc_id,
                content=doc.page_content[:200],  # Store snippet
                year=doc.metadata.get('year'),
                source=doc.metadata.get('filename', doc.metadata.get('source', 'unknown')),
                domain=doc.metadata.get('domain', 'general'),
                credibility=doc.metadata.get('credibility_score', 0.5),
                quality=doc.metadata.get('quality_score', 0.5)
            )
        
        # Add temporal edges
        if temporal_order:
            self._add_temporal_edges(documents)
        
        # Add citation edges (detected from content)
        self._add_citation_edges(documents)
        
        # Add contradiction edges
        if contradictions:
            self._add_contradiction_edges(contradictions)
        
        # Add similarity edges (documents on similar topics)
        self._add_similarity_edges(documents)
        
        logger.info(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _add_temporal_edges(self, documents: List[Document]):
        """Add edges for temporal succession"""
        
        # Group by domain
        from collections import defaultdict
        by_domain = defaultdict(list)
        
        for i, doc in enumerate(documents):
            domain = doc.metadata.get('domain', 'general')
            year = doc.metadata.get('year')
            if year:
                by_domain[domain].append((f"doc_{i}", year, doc))
        
        # Within each domain, create temporal edges
        for domain, doc_list in by_domain.items():
            # Sort by year
            doc_list.sort(key=lambda x: x[1])
            
            for i in range(len(doc_list) - 1):
                curr_id, curr_year, _ = doc_list[i]
                next_id, next_year, _ = doc_list[i + 1]
                
                self.graph.add_edge(
                    curr_id,
                    next_id,
                    relationship='temporal_successor',
                    years_apart=next_year - curr_year
                )
    
    def _add_citation_edges(self, documents: List[Document]):
        """Detect and add citation relationships"""
        
        # Simple approach: if document A mentions document B's source, add citation edge
        for i, doc_a in enumerate(documents):
            content_lower = doc_a.page_content.lower()
            
            for j, doc_b in enumerate(documents):
                if i == j:
                    continue
                
                source_b = doc_b.metadata.get('source', '').lower()
                filename_b = doc_b.metadata.get('filename', '').lower()
                
                # Check if doc_a mentions doc_b
                if (source_b and source_b in content_lower) or (filename_b and filename_b in content_lower):
                    self.graph.add_edge(
                        f"doc_{i}",
                        f"doc_{j}",
                        relationship='cites'
                    )
    
    def _add_contradiction_edges(self, contradictions: List[Dict]):
        """Add edges for detected contradictions"""
        
        for contradiction in contradictions:
            source_a = contradiction.get('source_A', '')
            source_b = contradiction.get('source_B', '')
            
            # Find document IDs
            doc_id_a = self._find_doc_id_by_source(source_a)
            doc_id_b = self._find_doc_id_by_source(source_b)
            
            if doc_id_a and doc_id_b:
                self.graph.add_edge(
                    doc_id_a,
                    doc_id_b,
                    relationship='contradicts',
                    severity=contradiction.get('severity', 'MEDIUM'),
                    confidence=contradiction.get('confidence', 0.5)
                )
    
    def _add_similarity_edges(self, documents: List[Document], threshold: float = 0.8):
        """Add edges for similar documents (same domain/topic)"""
        
        for i, doc_a in enumerate(documents):
            domain_a = doc_a.metadata.get('domain', 'general')
            
            for j, doc_b in enumerate(documents):
                if i >= j:  # Avoid duplicates
                    continue
                
                domain_b = doc_b.metadata.get('domain', 'general')
                
                # If same domain, add similarity edge
                if domain_a == domain_b and domain_a != 'general':
                    self.graph.add_edge(
                        f"doc_{i}",
                        f"doc_{j}",
                        relationship='similar_topic',
                        domain=domain_a
                    )
    
    def _find_doc_id_by_source(self, source: str) -> Optional[str]:
        """Find document ID by source name"""
        for doc_id, doc in self.document_index.items():
            if source in doc.metadata.get('source', '') or source in doc.metadata.get('filename', ''):
                return doc_id
        return None
    
    def graph_based_retrieval(
        self,
        seed_documents: List[str],
        relationship_types: List[str] = ['cites', 'similar_topic', 'temporal_successor'],
        max_hops: int = 2
    ) -> List[Document]:
        """
        Retrieve documents using graph traversal
        
        Innovation: Graph-based retrieval vs traditional vector similarity
        """
        
        logger.info(f"🔍 Graph-based retrieval from {len(seed_documents)} seeds...")
        
        reachable = set(seed_documents)
        
        for hop in range(max_hops):
            new_docs = set()
            
            for doc_id in reachable:
                if doc_id not in self.graph:
                    continue
                
                # Get neighbors via specified relationships
                for neighbor in self.graph.neighbors(doc_id):
                    edge_data = self.graph.edges[doc_id, neighbor]
                    
                    if edge_data.get('relationship') in relationship_types:
                        new_docs.add(neighbor)
            
            reachable.update(new_docs)
        
        # Convert doc IDs to Documents
        retrieved_docs = [
            self.document_index[doc_id]
            for doc_id in reachable
            if doc_id in self.document_index
        ]
        
        logger.info(f"✅ Retrieved {len(retrieved_docs)} documents via graph traversal")
        
        return retrieved_docs
    
    def calculate_document_importance(self) -> Dict[str, float]:
        """
        Calculate document importance using PageRank
        
        Innovation: Importance scoring based on graph structure
        """
        
        logger.info("📊 Calculating document importance (PageRank)...")
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        try:
            # Calculate PageRank
            pagerank_scores = nx.pagerank(self.graph)
            
            # Sort by importance
            sorted_scores = sorted(
                pagerank_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            logger.info(f"✅ Calculated importance for {len(pagerank_scores)} documents")
            
            return dict(sorted_scores)
        
        except Exception as e:
            logger.warning(f"⚠️ PageRank calculation failed: {e}")
            # Return uniform scores
            return {node: 1.0/self.graph.number_of_nodes() for node in self.graph.nodes()}
    
    def find_citation_chain(
        self,
        source_doc: str,
        target_doc: str
    ) -> List[List[str]]:
        """
        Find citation chains from source to target
        
        Innovation: Shows how information flows through citations
        """
        
        source_id = self._find_doc_id_by_source(source_doc)
        target_id = self._find_doc_id_by_source(target_doc)
        
        if not source_id or not target_id:
            logger.warning(f"⚠️ Could not find source or target document")
            return []
        
        # Create subgraph with only citation edges
        citation_graph = nx.DiGraph()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('relationship') == 'cites':
                citation_graph.add_edge(u, v)
        
        # Check if nodes exist in citation graph
        if source_id not in citation_graph or target_id not in citation_graph:
            logger.info(f"⚠️ No citation edges between documents")
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                citation_graph,
                source_id,
                target_id,
                cutoff=5
            ))
            
            # Convert to readable format
            readable_paths = []
            for path in paths:
                readable_path = [
                    self.graph.nodes[node].get('source', node)
                    for node in path
                ]
                readable_paths.append(readable_path)
            
            return readable_paths
            
        except nx.NetworkXNoPath:
            logger.info(f"⚠️ No path found between {source_doc} and {target_doc}")
            return []
        except nx.NodeNotFound:
            logger.info(f"⚠️ Nodes not found in citation graph")
            return []
    
    def find_paths_between_documents(
        self,
        source_doc_id: str,
        target_doc_id: str
    ) -> List[List[str]]:
        """Find reasoning paths between two documents"""
        
        if source_doc_id not in self.graph or target_doc_id not in self.graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source_doc_id,
                target_doc_id,
                cutoff=4
            ))
            
            return paths
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_document_context(self, source: str, hops: int = 1) -> Dict[str, List[Document]]:
        """Get full context around a document"""
        
        doc_id = self._find_doc_id_by_source(source)
        if not doc_id:
            return {}
        
        context = {
            'cites': [],
            'cited_by': [],
            'contradicts': [],
            'similar': [],
            'temporal_predecessors': [],
            'temporal_successors': []
        }
        
        # Analyze all edges
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            
            if u == doc_id:
                if rel_type == 'cites':
                    context['cites'].append(self.document_index.get(v))
                elif rel_type == 'contradicts':
                    context['contradicts'].append(self.document_index.get(v))
                elif rel_type == 'temporal_successor':
                    context['temporal_successors'].append(self.document_index.get(v))
                elif rel_type == 'similar_topic':
                    context['similar'].append(self.document_index.get(v))
            
            if v == doc_id:
                if rel_type == 'cites':
                    context['cited_by'].append(self.document_index.get(u))
                elif rel_type == 'contradicts':
                    context['contradicts'].append(self.document_index.get(u))
                elif rel_type == 'temporal_successor':
                    context['temporal_predecessors'].append(self.document_index.get(u))
                elif rel_type == 'similar_topic':
                    context['similar'].append(self.document_index.get(u))
        
        # Filter out None values
        context = {k: [d for d in v if d is not None] for k, v in context.items()}
        
        return context
    
    def query_graph(
        self,
        query_type: str,
        **kwargs
    ) -> List[Document]:
        """Query the graph using different strategies"""
        
        if query_type == 'cites':
            return self._query_cites(kwargs.get('source'))
        
        elif query_type == 'cited_by':
            return self._query_cited_by(kwargs.get('source'))
        
        elif query_type == 'contradicts':
            return self._query_contradicts(kwargs.get('source'))
        
        elif query_type == 'temporal_chain':
            return self._query_temporal_chain(kwargs.get('domain'))
        
        elif query_type == 'most_important':
            return self._query_most_important(kwargs.get('top_k', 5))
        
        else:
            logger.warning(f"⚠️ Unknown query type: {query_type}")
            return []
    
    def _query_cites(self, source: str) -> List[Document]:
        """Find documents that cite the given source"""
        
        doc_id = self._find_doc_id_by_source(source)
        if not doc_id:
            return []
        
        citing_ids = [
            u for u, v, data in self.graph.edges(data=True)
            if v == doc_id and data.get('relationship') == 'cites'
        ]
        
        return [self.document_index[doc_id] for doc_id in citing_ids if doc_id in self.document_index]
    
    def _query_cited_by(self, source: str) -> List[Document]:
        """Find documents cited by the given source"""
        
        doc_id = self._find_doc_id_by_source(source)
        if not doc_id:
            return []
        
        cited_ids = [
            v for u, v, data in self.graph.edges(data=True)
            if u == doc_id and data.get('relationship') == 'cites'
        ]
        
        return [self.document_index[doc_id] for doc_id in cited_ids if doc_id in self.document_index]
    
    def _query_contradicts(self, source: str) -> List[Document]:
        """Find documents that contradict the given source"""
        
        doc_id = self._find_doc_id_by_source(source)
        if not doc_id:
            return []
        
        contradicting_ids = set()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('relationship') == 'contradicts':
                if u == doc_id:
                    contradicting_ids.add(v)
                elif v == doc_id:
                    contradicting_ids.add(u)
        
        return [self.document_index[doc_id] for doc_id in contradicting_ids if doc_id in self.document_index]
    
    def _query_temporal_chain(self, domain: str) -> List[Document]:
        """Get temporal sequence of documents in a domain"""
        
        domain_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('domain') == domain
        ]
        
        sorted_nodes = sorted(
            domain_nodes,
            key=lambda n: self.graph.nodes[n].get('year', 0)
        )
        
        return [self.document_index[doc_id] for doc_id in sorted_nodes if doc_id in self.document_index]
    
    def _query_most_important(self, top_k: int) -> List[Document]:
        """Get most important documents by PageRank"""
        
        importance_scores = self.calculate_document_importance()
        
        top_docs = list(importance_scores.keys())[:top_k]
        
        return [self.document_index[doc_id] for doc_id in top_docs if doc_id in self.document_index]
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """Get comprehensive graph statistics"""
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        
        # Relationship type distribution
        relationship_counts = {}
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        stats['relationship_distribution'] = relationship_counts
        
        # Calculate centrality metrics
        if self.graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            stats['avg_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality)
        
        return stats
    
    def visualize_graph_plotly(self, output_path: Optional[str] = None) -> go.Figure:
        """Create interactive Plotly visualization"""
        
        logger.info("🎨 Creating graph visualization...")
        
        if self.graph.number_of_nodes() == 0:
            logger.warning("⚠️ No nodes in graph to visualize")
            return None
        
        # Use spring layout for positioning
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        
        # Group edges by relationship type
        from collections import defaultdict
        edges_by_type = defaultdict(list)
        
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            edges_by_type[rel_type].append((u, v))
        
        # Color map for relationship types
        color_map = {
            'cites': 'blue',
            'contradicts': 'red',
            'supports': 'green',
            'temporal_successor': 'purple',
            'similar_topic': 'orange'
        }
        
        for rel_type, edges in edges_by_type.items():
            edge_x = []
            edge_y = []
            
            for u, v in edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=2, color=color_map.get(rel_type, 'gray')),
                name=rel_type,
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.graph.nodes[node]
            source = node_data.get('source', 'Unknown')
            year = node_data.get('year', 'N/A')
            domain = node_data.get('domain', 'N/A')
            
            node_text.append(f"{source}<br>Year: {year}<br>Domain: {domain}")
            
            credibility = node_data.get('credibility', 0.5)
            node_color.append(credibility)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Credibility"),
                line=dict(width=2, color='white')
            ),
            text=[self.graph.nodes[node].get('source', '')[:10] for node in self.graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            name='Documents'
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Document Knowledge Graph',
                showlegend=True,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"💾 Graph visualization saved to {output_path}")
        
        return fig
    
    def export_graph(self, output_path: str):
        """Export graph for external analysis"""
        
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **data
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            'statistics': self.get_graph_statistics()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        logger.info(f"💾 Graph exported to {output_path}")


# Test document graph
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING DOCUMENT GRAPH")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Study from 2018 found high coffee consumption increases cardiovascular risk. This study is cited by the 2022 meta-analysis.",
            metadata={'source': 'study_2018.txt', 'filename': 'study_2018.txt', 'year': 2018, 'domain': 'medical', 'credibility_score': 0.7, 'quality_score': 0.8}
        ),
        Document(
            page_content="A 2022 meta-analysis reviewed previous studies including the 2018 study. Found that confounders explain contradictions.",
            metadata={'source': 'meta_2022.txt', 'filename': 'meta_2022.txt', 'year': 2022, 'domain': 'research', 'credibility_score': 0.95, 'quality_score': 0.95}
        ),
        Document(
            page_content="Recent 2023 research shows moderate coffee consumption beneficial. This contradicts the 2018 findings but aligns with 2022 meta-analysis.",
            metadata={'source': 'study_2023.txt', 'filename': 'study_2023.txt', 'year': 2023, 'domain': 'medical', 'credibility_score': 0.9, 'quality_score': 0.9}
        )
    ]
    
    # Create contradictions
    contradictions = [
        {
            'source_A': 'study_2018.txt',
            'source_B': 'study_2023.txt',
            'claim_A': 'Coffee increases risk',
            'claim_B': 'Coffee is protective',
            'severity': 'HIGH',
            'confidence': 0.9
        }
    ]
    
    # Test 1: Build graph
    print("\n1️⃣ Building Document Graph...")
    doc_graph = DocumentGraph()
    doc_graph.build_graph(sample_docs, contradictions=contradictions, temporal_order=True)
    
    stats = doc_graph.get_graph_statistics()
    print(f"   ✅ Graph built")
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Edges: {stats['num_edges']}")
    print(f"   Density: {stats['density']:.3f}")
    print(f"   Relationships: {stats['relationship_distribution']}")
    
    # Test 2: PageRank importance
    print("\n2️⃣ Calculating Document Importance...")
    importance = doc_graph.calculate_document_importance()
    
    print("   Top documents by importance:")
    for i, (doc_id, score) in enumerate(list(importance.items())[:3], 1):
        source = doc_graph.graph.nodes[doc_id].get('source', 'Unknown')
        print(f"   {i}. {source}: {score:.3f}")
    
    # Test 3: Graph-based retrieval
    print("\n3️⃣ Testing Graph-Based Retrieval...")
    
    seed_docs = ['doc_0']
    retrieved = doc_graph.graph_based_retrieval(
        seed_docs,
        relationship_types=['cites', 'temporal_successor', 'similar_topic'],
        max_hops=2
    )
    
    print(f"   ✅ Retrieved {len(retrieved)} documents via graph traversal")
    for doc in retrieved:
        print(f"      - {doc.metadata.get('filename', 'unknown')}")
    
    # Test 4: Find citation chains
    print("\n4️⃣ Finding Citation Chains...")
    
    chains = doc_graph.find_citation_chain('study_2018.txt', 'meta_2022.txt')
    
    if chains:
        print(f"   ✅ Found {len(chains)} citation chain(s)")
        for i, chain in enumerate(chains, 1):
            print(f"   Chain {i}: {' → '.join(chain)}")
    else:
        print("   ⚠️ No citation chains found (this is normal for test data)")
    
    # Test 5: Document context
    print("\n5️⃣ Getting Document Context...")
    
    context = doc_graph.get_document_context('study_2018.txt')
    
    print(f"   Context for study_2018.txt:")
    for rel_type, docs in context.items():
        if docs:
            print(f"      {rel_type}: {len(docs)} documents")
    
    # Test 6: Export graph
    print("\n6️⃣ Exporting Graph...")
    
    output_dir = Config.DATA_DIR / "graph_exports"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    doc_graph.export_graph(str(output_dir / "document_graph.json"))
    print(f"   ✅ Graph exported to {output_dir / 'document_graph.json'}")
    
    # Test 7: Visualize (save HTML)
    print("\n7️⃣ Creating Graph Visualization...")
    
    try:
        fig = doc_graph.visualize_graph_plotly(str(output_dir / "graph_visualization.html"))
        
        if fig:
            print(f"   ✅ Visualization saved to {output_dir / 'graph_visualization.html'}")
            print(f"   📊 Open this file in a browser to see interactive graph!")
    except Exception as e:
        print(f"   ⚠️ Visualization error (non-critical): {e}")
    
    print("\n" + "=" * 60)
    print("✅ DOCUMENT GRAPH TEST COMPLETE")
    print("=" * 60)
    print("\n🎯 Graph Innovation Added:")
    print("  ✅ Knowledge graph construction")
    print("  ✅ Relationship detection (cites, contradicts, temporal)")
    print("  ✅ Graph-based retrieval")
    print("  ✅ PageRank importance scoring")
    print("  ✅ Citation chain finding")
    print("  ✅ Interactive visualization")