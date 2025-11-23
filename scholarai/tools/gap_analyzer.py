"""
Custom Tool: Research Gap Analyzer
Uses embeddings, clustering, and LLM reasoning to identify research gaps
"""
from typing import Dict, List, Optional
import numpy as np
import json
from pathlib import Path

# ML imports
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from config.settings import settings
from utils.logger import logger

class ResearchGapAnalyzer:
    """
    Advanced tool for identifying research gaps using ML and NLP
    """
    
    def __init__(self):
        """Initialize the gap analyzer"""
        logger.info("Initializing Research Gap Analyzer...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
        
        # Ensure output directories exist
        settings.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, papers: List[Dict], analyses: List[Dict]) -> Dict:
        """
        Main analysis pipeline
        
        Args:
            papers: List of paper dictionaries
            analyses: List of analysis dictionaries
            
        Returns:
            Complete gap analysis results
        """
        logger.info(f"Starting gap analysis on {len(papers)} papers...")
        
        try:
            # Step 1: Generate embeddings
            embeddings = self._generate_embeddings(papers, analyses)
            
            # Step 2: Cluster papers
            clusters = self._cluster_papers(embeddings, papers)
            
            # Step 3: Identify gaps
            gaps = self._identify_gaps(clusters, papers, analyses)
            
            # Step 4: Detect contradictions
            contradictions = self._detect_contradictions(clusters, analyses)
            
            # Step 5: Analyze trends
            trends = self._analyze_trends(papers, analyses)
            
            # Step 6: Build citation network
            network_data = self._build_citation_network(papers)
            
            # Step 7: Create visualizations
            visualizations = self._create_visualizations(
                clusters, network_data, trends
            )
            
            # Step 8: Generate recommendations
            recommendations = self._generate_recommendations(gaps, trends)
            
            results = {
                'success': True,
                'research_gaps': gaps,
                'contradictions': contradictions,
                'trends': trends,
                'clusters': clusters,
                'network': network_data,
                'visualizations': visualizations,
                'recommendations': recommendations,
                'statistics': {
                    'total_papers': len(papers),
                    'num_clusters': len(clusters),
                    'num_gaps': len(gaps),
                    'num_contradictions': len(contradictions)
                }
            }
            
            logger.info(f"✅ Gap analysis complete!")
            logger.info(f"   - {len(gaps)} gaps identified")
            logger.info(f"   - {len(contradictions)} contradictions found")
            logger.info(f"   - {len(clusters)} research clusters")
            
            return results
            
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'research_gaps': [],
                'contradictions': [],
                'trends': {},
                'clusters': []
            }
    
    def _generate_embeddings(self, papers: List[Dict], analyses: List[Dict]) -> np.ndarray:
        """Generate embeddings for all papers"""
        logger.info("Generating embeddings...")
        
        # Combine title, snippet, and key findings
        texts = []
        for paper in papers:
            # Get analysis for this paper
            analysis = next((a for a in analyses if a.get('paper_id') == paper['id']), None)
            
            text_parts = [
                paper.get('title', ''),
                paper.get('snippet', '')
            ]
            
            if analysis and analysis.get('key_findings'):
                text_parts.extend(analysis['key_findings'][:2])
            
            text = ' '.join(text_parts)
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        logger.info(f"✅ Generated embeddings: shape {embeddings.shape}")
        return embeddings
    
    def _cluster_papers(self, embeddings: np.ndarray, papers: List[Dict]) -> List[Dict]:
        """Cluster papers using DBSCAN"""
        logger.info("Clustering papers...")
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(
            eps=0.5,
            min_samples=2,
            metric='cosine'
        ).fit(embeddings)
        
        labels = clustering.labels_
        
        # Group papers by cluster
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(papers[idx])
        
        # Create cluster objects
        clusters = []
        for label, cluster_papers in clusters_dict.items():
            if label == -1:  # Noise points
                theme = "Miscellaneous/Outliers"
            else:
                # Generate theme from paper titles
                theme = self._generate_cluster_theme(cluster_papers)
            
            clusters.append({
                'cluster_id': int(label),
                'theme': theme,
                'papers': [p['id'] for p in cluster_papers],
                'size': len(cluster_papers),
                'representative_papers': [p['title'] for p in cluster_papers[:2]]
            })
        
        # Sort by size
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        logger.info(f"✅ Created {len(clusters)} clusters")
        return clusters
    
    def _generate_cluster_theme(self, papers: List[Dict]) -> str:
        """Generate a theme name for a cluster"""
        # Extract common words from titles
        from collections import Counter
        import re
        
        all_words = []
        for paper in papers:
            title = paper.get('title', '').lower()
            # Extract meaningful words (3+ chars)
            words = re.findall(r'\b[a-z]{3,}\b', title)
            all_words.extend(words)
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'with', 'based', 'using', 'from', 'that', 'this', 'are', 'model', 'models'}
        filtered_words = [w for w in all_words if w not in stopwords]
        
        # Get most common words
        if filtered_words:
            common = Counter(filtered_words).most_common(3)
            theme = ' & '.join([word.capitalize() for word, count in common])
        else:
            theme = "General Research"
        
        return theme
    
    def _identify_gaps(self, clusters: List[Dict], papers: List[Dict], analyses: List[Dict]) -> List[Dict]:
        """Identify research gaps using multiple methods"""
        logger.info("Identifying research gaps...")
        
        gaps = []
        
        # Method 1: Small/Underexplored clusters
        cluster_sizes = [c['size'] for c in clusters if c['cluster_id'] != -1]
        if cluster_sizes:
            avg_size = np.mean(cluster_sizes)
            
            for cluster in clusters:
                if cluster['cluster_id'] != -1 and cluster['size'] < avg_size * 0.6:
                    gaps.append({
                        'gap_id': len(gaps) + 1,
                        'type': 'underexplored_area',
                        'title': f"Limited research in {cluster['theme']}",
                        'description': f"Only {cluster['size']} papers found in this area, suggesting it may be underexplored.",
                        'evidence': [
                            f"Cluster size: {cluster['size']} papers",
                            f"Below average cluster size ({avg_size:.1f})"
                        ],
                        'confidence': 0.75,
                        'impact': 'Medium',
                        'related_papers': cluster['papers'][:3]
                    })
        
        # Method 2: Methodology gaps
        methodologies = {}
        for analysis in analyses:
            if analysis.get('success'):
                method_type = analysis.get('methodology', {}).get('type', 'Unknown')
                methodologies[method_type] = methodologies.get(method_type, 0) + 1
        
        total_analyzed = len([a for a in analyses if a.get('success')])
        
        # Check for missing methodologies
        expected_methods = ['Experimental', 'Theoretical', 'Survey', 'Empirical']
        for method in expected_methods:
            if method not in methodologies:
                gaps.append({
                    'gap_id': len(gaps) + 1,
                    'type': 'methodological_gap',
                    'title': f"Lack of {method} studies",
                    'description': f"No {method.lower()} research found in the analyzed papers.",
                    'evidence': [
                        f"0/{total_analyzed} papers use {method} methodology",
                        "This approach could provide valuable insights"
                    ],
                    'confidence': 0.80,
                    'impact': 'High',
                    'related_papers': []
                })
        
        # Method 3: Technical term analysis - underrepresented terms
        all_terms = []
        for analysis in analyses:
            if analysis.get('success'):
                all_terms.extend(analysis.get('technical_terms', []))
        
        from collections import Counter
        term_freq = Counter(all_terms)
        
        # Terms mentioned 1-2 times might indicate emerging or neglected areas
        rare_terms = [term for term, count in term_freq.items() if 1 <= count <= 2]
        
        if rare_terms:
            for term in rare_terms[:3]:  # Top 3 rare terms
                gaps.append({
                    'gap_id': len(gaps) + 1,
                    'type': 'emerging_area',
                    'title': f"Emerging research on {term}",
                    'description': f"'{term}' is mentioned but not extensively studied, indicating a potential emerging area.",
                    'evidence': [
                        f"Mentioned in {term_freq[term]} paper(s)",
                        "Could be an emerging or underexplored topic"
                    ],
                    'confidence': 0.65,
                    'impact': 'Medium',
                    'related_papers': []
                })
        
        # Method 4: Temporal gaps (if old papers dominate)
        years = [p.get('year', 2023) for p in papers]
        avg_year = np.mean(years)
        current_year = 2024
        
        if avg_year < current_year - 3:
            gaps.append({
                'gap_id': len(gaps) + 1,
                'type': 'temporal_gap',
                'title': "Limited recent research",
                'description': f"Average publication year is {avg_year:.0f}, suggesting need for updated studies.",
                'evidence': [
                    f"Average publication year: {avg_year:.0f}",
                    f"Gap of {current_year - avg_year:.0f} years from present"
                ],
                'confidence': 0.85,
                'impact': 'High',
                'related_papers': []
            })
        
        # Limit to top gaps
        gaps = gaps[:8]
        
        logger.info(f"✅ Identified {len(gaps)} research gaps")
        return gaps
    
    def _detect_contradictions(self, clusters: List[Dict], analyses: List[Dict]) -> List[Dict]:
        """Detect contradictions in findings"""
        logger.info("Detecting contradictions...")
        
        contradictions = []
        
        # Look for opposing findings within same cluster
        for cluster in clusters:
            if cluster['size'] < 2:
                continue
            
            cluster_analyses = [
                a for a in analyses 
                if a.get('paper_id') in cluster['papers'] and a.get('success')
            ]
            
            if len(cluster_analyses) < 2:
                continue
            
            # Check for contradictory findings (simple heuristic)
            findings_text = []
            for analysis in cluster_analyses:
                findings = ' '.join(analysis.get('key_findings', [])).lower()
                findings_text.append(findings)
            
            # Look for opposing terms
            opposing_pairs = [
                ('improve', 'decrease'),
                ('better', 'worse'),
                ('increase', 'reduce'),
                ('effective', 'ineffective'),
                ('success', 'fail')
            ]
            
            for i, text_a in enumerate(findings_text):
                for j, text_b in enumerate(findings_text[i+1:], i+1):
                    for term_a, term_b in opposing_pairs:
                        if term_a in text_a and term_b in text_b:
                            contradictions.append({
                                'contradiction_id': len(contradictions) + 1,
                                'paper_a': cluster_analyses[i]['paper_id'],
                                'paper_b': cluster_analyses[j]['paper_id'],
                                'description': f"Potential contradiction: one paper reports '{term_a}' while another reports '{term_b}'",
                                'severity': 'Medium',
                                'requires_investigation': True
                            })
        
        logger.info(f"✅ Found {len(contradictions)} potential contradictions")
        return contradictions[:5]  # Limit to 5
    
    def _analyze_trends(self, papers: List[Dict], analyses: List[Dict]) -> Dict:
        """Analyze research trends over time"""
        logger.info("Analyzing trends...")
        
        # Publication timeline
        years = [p.get('year', 2023) for p in papers]
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        
        # Technical term frequency
        all_terms = []
        for analysis in analyses:
            if analysis.get('success'):
                all_terms.extend(analysis.get('technical_terms', []))
        
        from collections import Counter
        term_freq = Counter(all_terms)
        
        # Identify growing trends (terms in recent papers)
        recent_papers = [p for p in papers if p.get('year', 2020) >= 2022]
        recent_analyses = [a for a in analyses if a.get('paper_id') in [p['id'] for p in recent_papers]]
        
        recent_terms = []
        for analysis in recent_analyses:
            if analysis.get('success'):
                recent_terms.extend(analysis.get('technical_terms', []))
        
        recent_term_freq = Counter(recent_terms)
        
        # Growing terms (more frequent in recent papers)
        growing_trends = []
        for term in term_freq.keys():
            total_freq = term_freq[term]
            recent_freq = recent_term_freq.get(term, 0)
            
            if total_freq >= 3 and recent_freq / total_freq > 0.6:
                growing_trends.append({
                    'term': term,
                    'momentum': 'High',
                    'frequency': total_freq,
                    'recent_ratio': recent_freq / total_freq
                })
        
        growing_trends.sort(key=lambda x: x['frequency'], reverse=True)
        
        trends = {
            'publication_timeline': year_counts,
            'top_terms': [
                {'term': term, 'count': count} 
                for term, count in term_freq.most_common(10)
            ],
            'growing_trends': growing_trends[:5],
            'research_momentum': 'High' if len(recent_papers) / len(papers) > 0.7 else 'Medium'
        }
        
        logger.info(f"✅ Trend analysis complete")
        return trends
    
    def _build_citation_network(self, papers: List[Dict]) -> Dict:
        """Build citation network (simplified version)"""
        logger.info("Building citation network...")
        
        G = nx.DiGraph()
        
        # Add nodes
        for paper in papers:
            G.add_node(
                paper['id'],
                title=paper['title'][:30] + '...',
                year=paper.get('year', 2023)
            )
        
        # For real citation data, we'd need to scrape or use APIs
        # For now, create a simple network based on year (older papers likely cited by newer)
        papers_sorted = sorted(papers, key=lambda x: x.get('year', 2023))
        
        for i, paper_a in enumerate(papers_sorted):
            for paper_b in papers_sorted[i+1:i+3]:  # Connect to next 2 papers
                if paper_b.get('year', 2023) > paper_a.get('year', 2023):
                    G.add_edge(paper_b['id'], paper_a['id'])
        
        # Calculate metrics
        if len(G.nodes()) > 0:
            try:
                pagerank = nx.pagerank(G)
                influential = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            except:
                influential = []
        else:
            influential = []
        
        network_data = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'influential_papers': [paper_id for paper_id, score in influential],
            'graph': G
        }
        
        logger.info(f"✅ Citation network: {network_data['node_count']} nodes, {network_data['edge_count']} edges")
        return network_data
    
    def _create_visualizations(self, clusters: List[Dict], network_data: Dict, trends: Dict) -> Dict:
        """Create visualization files"""
        logger.info("Creating visualizations...")
        
        viz_paths = {}
        
        try:
            # 1. Cluster Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cluster_themes = [c['theme'][:20] for c in clusters if c['cluster_id'] != -1]
            cluster_sizes = [c['size'] for c in clusters if c['cluster_id'] != -1]
            
            if cluster_themes:
                ax.barh(cluster_themes, cluster_sizes, color='steelblue')
                ax.set_xlabel('Number of Papers')
                ax.set_title('Research Clusters')
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                cluster_path = settings.VISUALIZATION_DIR / 'cluster_distribution.png'
                plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_paths['cluster_distribution'] = str(cluster_path)
                logger.info(f"✅ Saved cluster visualization")
        
        except Exception as e:
            logger.warning(f"Failed to create cluster visualization: {e}")
        
        try:
            # 2. Publication Timeline
            if trends.get('publication_timeline'):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                years = sorted(trends['publication_timeline'].keys())
                counts = [trends['publication_timeline'][y] for y in years]
                
                ax.plot(years, counts, marker='o', linewidth=2, markersize=8, color='darkblue')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Papers')
                ax.set_title('Publication Timeline')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                timeline_path = settings.VISUALIZATION_DIR / 'publication_timeline.png'
                plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_paths['publication_timeline'] = str(timeline_path)
                logger.info(f"✅ Saved timeline visualization")
        
        except Exception as e:
            logger.warning(f"Failed to create timeline visualization: {e}")
        
        try:
            # 3. Citation Network
            G = network_data.get('graph')
            if G and len(G.nodes()) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                nx.draw(
                    G, pos,
                    with_labels=False,
                    node_color='lightblue',
                    node_size=500,
                    edge_color='gray',
                    arrows=True,
                    ax=ax
                )
                
                ax.set_title('Citation Network')
                ax.axis('off')
                
                plt.tight_layout()
                network_path = settings.VISUALIZATION_DIR / 'citation_network.png'
                plt.savefig(network_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_paths['citation_network'] = str(network_path)
                logger.info(f"✅ Saved network visualization")
        
        except Exception as e:
            logger.warning(f"Failed to create network visualization: {e}")
        
        logger.info(f"✅ Created {len(viz_paths)} visualizations")
        return viz_paths
    
    def _generate_recommendations(self, gaps: List[Dict], trends: Dict) -> List[Dict]:
        """Generate research recommendations"""
        logger.info("Generating recommendations...")
        
        recommendations = []
        
        # From high-confidence gaps
        high_confidence_gaps = [g for g in gaps if g.get('confidence', 0) >= 0.75]
        
        for gap in high_confidence_gaps[:3]:
            recommendations.append({
                'recommendation_id': len(recommendations) + 1,
                'title': f"Investigate {gap['title'].lower()}",
                'description': gap['description'],
                'rationale': gap.get('evidence', []),
                'priority': gap.get('impact', 'Medium'),
                'related_gap_id': gap['gap_id']
            })
        
        # From growing trends
        if trends.get('growing_trends'):
            for trend in trends['growing_trends'][:2]:
                recommendations.append({
                    'recommendation_id': len(recommendations) + 1,
                    'title': f"Explore {trend['term']} applications",
                    'description': f"'{trend['term']}' shows high momentum ({trend['momentum']}) in recent research.",
                    'rationale': [f"Appears in {trend['frequency']} papers", "Growing interest in recent publications"],
                    'priority': 'High',
                    'related_gap_id': None
                })
        
        logger.info(f"✅ Generated {len(recommendations)} recommendations")
        return recommendations

# Singleton instance
research_gap_analyzer = ResearchGapAnalyzer()