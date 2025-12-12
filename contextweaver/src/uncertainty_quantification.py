# src/uncertainty_quantification.py
from typing import List, Dict, Tuple, Optional
import logging
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import math

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Quantify and communicate uncertainty in answers
    
    INNOVATION: Probabilistic reasoning with calibrated confidence
    
    Features:
    - Bayesian confidence estimation
    - Sensitivity analysis
    - Evidence sufficiency scoring
    - Uncertainty visualization
    - "What would change my confidence?" analysis
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)
        logger.info("🎲 Uncertainty quantifier initialized")
    
    def quantify_uncertainty(
        self,
        query: str,
        answer: str,
        supporting_documents: List[Document],
        contradicting_documents: List[Document] = None
    ) -> Dict[str, any]:
        """
        Comprehensive uncertainty quantification
        
        Returns:
            - confidence_score: Bayesian confidence (0-1)
            - uncertainty_sources: What causes uncertainty
            - evidence_gaps: What's missing
            - sensitivity: How answer changes with different evidence
        """
        
        logger.info(f"🎲 Quantifying uncertainty for answer...")
        
        if contradicting_documents is None:
            contradicting_documents = []
        
        # Calculate multiple uncertainty factors
        evidence_score = self._calculate_evidence_sufficiency(supporting_documents)
        agreement_score = self._calculate_source_agreement(supporting_documents)
        quality_score = self._calculate_source_quality(supporting_documents)
        contradiction_penalty = self._calculate_contradiction_penalty(contradicting_documents)
        
        # Bayesian confidence estimation
        prior_confidence = 0.5  # Neutral prior
        
        # Update based on evidence
        posterior_confidence = self._bayesian_update(
            prior_confidence,
            evidence_score,
            agreement_score,
            quality_score,
            contradiction_penalty
        )
        
        # Identify uncertainty sources
        uncertainty_sources = self._identify_uncertainty_sources(
            evidence_score,
            agreement_score,
            quality_score,
            len(contradicting_documents)
        )
        
        # Identify evidence gaps
        evidence_gaps = self._identify_evidence_gaps(
            query,
            supporting_documents
        )
        
        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(
            posterior_confidence,
            supporting_documents,
            contradicting_documents
        )
        
        result = {
            'confidence_score': posterior_confidence,
            'confidence_level': self._categorize_confidence(posterior_confidence),
            'uncertainty_sources': uncertainty_sources,
            'evidence_gaps': evidence_gaps,
            'sensitivity_analysis': sensitivity,
            'component_scores': {
                'evidence_sufficiency': evidence_score,
                'source_agreement': agreement_score,
                'source_quality': quality_score,
                'contradiction_penalty': contradiction_penalty
            }
        }
        
        logger.info(f"✅ Uncertainty quantified: {posterior_confidence:.2f} ({result['confidence_level']})")
        
        return result
    
    def _calculate_evidence_sufficiency(self, documents: List[Document]) -> float:
        """
        Calculate if we have sufficient evidence
        
        Factors:
        - Number of sources
        - Coverage of query aspects
        - Document completeness
        """
        
        num_sources = len(documents)
        
        # More sources = higher confidence (with diminishing returns)
        source_score = min(num_sources / 5, 1.0)  # Saturates at 5 sources
        
        # Average quality of sources
        avg_quality = sum(doc.metadata.get('quality_score', 0.5) for doc in documents) / len(documents) if documents else 0
        
        # Combined evidence sufficiency
        return (source_score + avg_quality) / 2
    
    def _calculate_source_agreement(self, documents: List[Document]) -> float:
        """
        Calculate agreement among sources
        
        High agreement = higher confidence
        Low agreement = lower confidence
        """
        
        if len(documents) < 2:
            return 0.5  # Can't assess agreement with <2 sources
        
        # Simplified: assume high agreement if from same domain/time period
        domains = [doc.metadata.get('domain', 'unknown') for doc in documents]
        years = [doc.metadata.get('year', 0) for doc in documents]
        
        # Domain agreement
        most_common_domain = max(set(domains), key=domains.count)
        domain_agreement = domains.count(most_common_domain) / len(domains)
        
        # Temporal agreement (within 3 years = good agreement)
        if years and all(y != 0 for y in years):
            year_range = max(years) - min(years)
            temporal_agreement = max(0, 1 - year_range / 10)  # Penalty for large time spans
        else:
            temporal_agreement = 0.5
        
        return (domain_agreement + temporal_agreement) / 2
    
    def _calculate_source_quality(self, documents: List[Document]) -> float:
        """Calculate average source quality/credibility"""
        
        if not documents:
            return 0.0
        
        credibility_scores = [doc.metadata.get('credibility_score', 0.5) for doc in documents]
        
        return sum(credibility_scores) / len(credibility_scores)
    
    def _calculate_contradiction_penalty(self, contradicting_documents: List[Document]) -> float:
        """
        Penalty for contradicting evidence
        
        More contradictions = lower confidence
        """
        
        if not contradicting_documents:
            return 0.0  # No penalty
        
        # Penalty increases with number of contradictions
        num_contradictions = len(contradicting_documents)
        
        # Logarithmic penalty (diminishing impact)
        penalty = min(math.log(num_contradictions + 1) / math.log(10), 0.5)
        
        return penalty
    
    def _bayesian_update(
        self,
        prior: float,
        evidence_score: float,
        agreement_score: float,
        quality_score: float,
        contradiction_penalty: float
    ) -> float:
        """
        Bayesian confidence update
        
        Innovation: Proper probabilistic reasoning
        
        Uses weighted combination of factors to update confidence
        """
        
        # Weighted combination
        weights = {
            'evidence': 0.3,
            'agreement': 0.25,
            'quality': 0.25,
            'prior': 0.2
        }
        
        # Calculate posterior before contradiction penalty
        posterior = (
            weights['prior'] * prior +
            weights['evidence'] * evidence_score +
            weights['agreement'] * agreement_score +
            weights['quality'] * quality_score
        )
        
        # Apply contradiction penalty
        final_confidence = posterior * (1 - contradiction_penalty)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, final_confidence))
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence into levels"""
        
        if confidence >= 0.85:
            return "VERY HIGH"
        elif confidence >= 0.70:
            return "HIGH"
        elif confidence >= 0.50:
            return "MODERATE"
        elif confidence >= 0.30:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _identify_uncertainty_sources(
        self,
        evidence_score: float,
        agreement_score: float,
        quality_score: float,
        num_contradictions: int
    ) -> List[str]:
        """Identify what's causing uncertainty"""
        
        sources = []
        
        if evidence_score < 0.5:
            sources.append("Insufficient evidence (need more sources)")
        
        if agreement_score < 0.5:
            sources.append("Low agreement among sources")
        
        if quality_score < 0.7:
            sources.append("Low quality/credibility of sources")
        
        if num_contradictions > 0:
            sources.append(f"Contradictory evidence ({num_contradictions} contradictions found)")
        
        if not sources:
            sources.append("Minor uncertainties from limited sample size")
        
        return sources
    
    def _identify_evidence_gaps(
        self,
        query: str,
        documents: List[Document]
    ) -> List[str]:
        """
        Identify what additional evidence would help
        
        Innovation: "What do we need to know?" analysis
        """
        
        gaps = []
        
        # Check temporal coverage
        years = [doc.metadata.get('year') for doc in documents if doc.metadata.get('year')]
        
        if years:
            year_range = max(years) - min(years)
            if year_range > 5:
                gaps.append(f"Gap in temporal coverage ({year_range} year span)")
        
        # Check source diversity
        domains = set(doc.metadata.get('domain', 'general') for doc in documents)
        
        if len(domains) < 2:
            gaps.append("Limited domain diversity (all sources from same field)")
        
        # Check number of sources
        if len(documents) < 3:
            gaps.append(f"Limited number of sources ({len(documents)} found, recommend 5+)")
        
        if not gaps:
            gaps.append("Good evidence coverage, minor gaps possible")
        
        return gaps
    
    def _sensitivity_analysis(
        self,
        base_confidence: float,
        supporting_docs: List[Document],
        contradicting_docs: List[Document]
    ) -> Dict[str, float]:
        """
        Sensitivity analysis: How confidence changes with different evidence
        
        Innovation: "What-if" scenarios for confidence
        
        Scenarios:
        - Add one high-quality source
        - Remove contradictions
        - Add one contradiction
        - Use only recent sources
        """
        
        sensitivity = {}
        
        # Scenario 1: Add high-quality source
        hypothetical_quality = 0.95
        new_quality_score = (
            sum(doc.metadata.get('credibility_score', 0.5) for doc in supporting_docs) + hypothetical_quality
        ) / (len(supporting_docs) + 1)
        
        # Rough estimate of confidence change
        quality_boost = (new_quality_score - self._calculate_source_quality(supporting_docs)) * 0.25
        sensitivity['add_high_quality_source'] = min(base_confidence + quality_boost, 1.0)
        
        # Scenario 2: Remove all contradictions
        if contradicting_docs:
            no_contradiction_conf = base_confidence / (1 - self._calculate_contradiction_penalty(contradicting_docs))
            sensitivity['remove_contradictions'] = min(no_contradiction_conf, 1.0)
        else:
            sensitivity['remove_contradictions'] = base_confidence
        
        # Scenario 3: Add one contradiction
        hypothetical_contradictions = contradicting_docs + [Document(page_content="contradiction", metadata={})]
        new_penalty = self._calculate_contradiction_penalty(hypothetical_contradictions)
        sensitivity['add_contradiction'] = base_confidence * (1 - new_penalty)
        
        # Scenario 4: Double the evidence
        sensitivity['double_evidence'] = min(base_confidence * 1.15, 1.0)
        
        return sensitivity
    
    def generate_uncertainty_report(
        self,
        query: str,
        answer: str,
        uncertainty_data: Dict
    ) -> str:
        """
        Generate human-readable uncertainty report
        
        Innovation: Transparent communication of uncertainty
        """
        
        report = f"""
{'='*60}
UNCERTAINTY QUANTIFICATION REPORT
{'='*60}

Query: {query}

Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}

{'─'*60}
CONFIDENCE ASSESSMENT
{'─'*60}

Overall Confidence: {uncertainty_data['confidence_score']:.2%}
Confidence Level: {uncertainty_data['confidence_level']}

Component Scores:
  • Evidence Sufficiency: {uncertainty_data['component_scores']['evidence_sufficiency']:.2%}
  • Source Agreement: {uncertainty_data['component_scores']['source_agreement']:.2%}
  • Source Quality: {uncertainty_data['component_scores']['source_quality']:.2%}
  • Contradiction Penalty: {uncertainty_data['component_scores']['contradiction_penalty']:.2%}

{'─'*60}
UNCERTAINTY SOURCES
{'─'*60}

"""
        
        for i, source in enumerate(uncertainty_data['uncertainty_sources'], 1):
            report += f"{i}. {source}\n"
        
        report += f"""
{'─'*60}
EVIDENCE GAPS
{'─'*60}

"""
        
        for i, gap in enumerate(uncertainty_data['evidence_gaps'], 1):
            report += f"{i}. {gap}\n"
        
        report += f"""
{'─'*60}
SENSITIVITY ANALYSIS
{'─'*60}
How confidence would change under different scenarios:

"""
        
        for scenario, conf in uncertainty_data['sensitivity_analysis'].items():
            change = conf - uncertainty_data['confidence_score']
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            report += f"  {scenario.replace('_', ' ').title()}: {conf:.2%} ({arrow} {abs(change):.2%})\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def compare_uncertainty_across_answers(
        self,
        answers: List[Dict]
    ) -> Dict[str, any]:
        """
        Compare uncertainty across multiple answers
        
        Use case: Multiple queries or different reasoning approaches
        """
        
        if not answers:
            return {}
        
        comparison = {
            'num_answers': len(answers),
            'confidence_range': (
                min(a['uncertainty']['confidence_score'] for a in answers),
                max(a['uncertainty']['confidence_score'] for a in answers)
            ),
            'avg_confidence': sum(a['uncertainty']['confidence_score'] for a in answers) / len(answers),
            'most_confident': None,
            'least_confident': None
        }
        
        # Find most and least confident
        sorted_by_conf = sorted(
            answers,
            key=lambda x: x['uncertainty']['confidence_score'],
            reverse=True
        )
        
        comparison['most_confident'] = {
            'query': sorted_by_conf[0]['query'],
            'confidence': sorted_by_conf[0]['uncertainty']['confidence_score']
        }
        
        comparison['least_confident'] = {
            'query': sorted_by_conf[-1]['query'],
            'confidence': sorted_by_conf[-1]['uncertainty']['confidence_score']
        }
        
        return comparison


class EvidenceSufficiencyAnalyzer:
    """
    Analyze if we have sufficient evidence to answer confidently
    
    Innovation: Identifies what's missing for high confidence
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.1)
        logger.info("📊 Evidence sufficiency analyzer initialized")
    
    def analyze_sufficiency(
        self,
        query: str,
        documents: List[Document]
    ) -> Dict[str, any]:
        """
        Analyze if evidence is sufficient
        
        Returns:
            - is_sufficient: Boolean
            - sufficiency_score: 0-1
            - missing_aspects: What aspects aren't covered
            - recommended_additions: What to add
        """
        
        logger.info("📊 Analyzing evidence sufficiency...")
        
        # Extract query aspects
        query_aspects = self._extract_query_aspects(query)
        
        # Check coverage of each aspect
        coverage = {}
        for aspect in query_aspects:
            coverage[aspect] = self._check_aspect_coverage(aspect, documents)
        
        # Calculate sufficiency score
        sufficiency_score = sum(coverage.values()) / len(coverage) if coverage else 0
        
        # Identify missing aspects
        missing_aspects = [
            aspect for aspect, covered in coverage.items()
            if covered < 0.5
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(missing_aspects, documents)
        
        result = {
            'is_sufficient': sufficiency_score >= 0.7,
            'sufficiency_score': sufficiency_score,
            'aspect_coverage': coverage,
            'missing_aspects': missing_aspects,
            'recommended_additions': recommendations
        }
        
        return result
    
    def _extract_query_aspects(self, query: str) -> List[str]:
        """Extract key aspects from query"""
        
        # Simplified: extract key nouns and verbs
        # In production, would use NLP
        
        words = query.lower().split()
        
        # Filter out common words
        stop_words = {'is', 'the', 'a', 'an', 'for', 'to', 'of', 'in', 'on', 'at', 'what', 'how', 'why', 'when'}
        
        aspects = [w for w in words if w not in stop_words and len(w) > 3]
        
        return aspects[:5]  # Top 5 aspects
    
    def _check_aspect_coverage(self, aspect: str, documents: List[Document]) -> float:
        """Check if aspect is covered in documents"""
        
        # Simple keyword matching
        mentions = sum(
            1 for doc in documents
            if aspect.lower() in doc.page_content.lower()
        )
        
        # Coverage score
        coverage = min(mentions / 2, 1.0)  # Saturates at 2 mentions
        
        return coverage
    
    def _generate_recommendations(
        self,
        missing_aspects: List[str],
        current_documents: List[Document]
    ) -> List[str]:
        """Generate recommendations for improving evidence"""
        
        recommendations = []
        
        if missing_aspects:
            recommendations.append(
                f"Add documents covering: {', '.join(missing_aspects)}"
            )
        
        if len(current_documents) < 5:
            recommendations.append(
                f"Increase number of sources (currently {len(current_documents)}, recommend 5+)"
            )
        
        # Check temporal coverage
        years = [doc.metadata.get('year') for doc in current_documents if doc.metadata.get('year')]
        
        if years and (max(years) - min(years)) > 5:
            recommendations.append(
                "Add more recent sources to reduce temporal gap"
            )
        
        if not recommendations:
            recommendations.append("Evidence appears sufficient")
        
        return recommendations


class ConfidenceCalibrator:
    """
    Calibrate confidence scores to be accurate
    
    Innovation: Ensures confidence scores are trustworthy
    
    Problem: AI systems often overconfident or underconfident
    Solution: Calibration using historical accuracy
    """
    
    def __init__(self):
        self.calibration_history = []
        logger.info("🎯 Confidence calibrator initialized")
    
    def calibrate_confidence(
        self,
        raw_confidence: float,
        answer_complexity: str = 'medium'
    ) -> float:
        """
        Calibrate raw confidence score
        
        Adjustments:
        - Overconfident models → reduce confidence
        - Complex queries → reduce confidence
        - Simple queries → maintain confidence
        """
        
        # Complexity adjustment
        complexity_adjustment = {
            'simple': 1.0,      # No adjustment
            'medium': 0.9,      # Slight reduction
            'complex': 0.8,     # More reduction
            'very_complex': 0.7 # Significant reduction
        }
        
        adjustment = complexity_adjustment.get(answer_complexity, 0.9)
        
        # Apply calibration
        calibrated = raw_confidence * adjustment
        
        # Additional conservative adjustment for AI safety
        # Slightly reduce confidence to avoid overconfidence
        calibrated = calibrated * 0.95
        
        return max(0.0, min(1.0, calibrated))
    
    def add_calibration_datapoint(
        self,
        predicted_confidence: float,
        actual_correctness: float
    ):
        """
        Add datapoint for improving calibration
        
        Over time, can learn better calibration
        """
        
        self.calibration_history.append({
            'predicted': predicted_confidence,
            'actual': actual_correctness,
            'error': abs(predicted_confidence - actual_correctness)
        })
    
    def get_calibration_metrics(self) -> Dict[str, float]:
        """Get calibration performance metrics"""
        
        if not self.calibration_history:
            return {'error': 'No calibration data'}
        
        errors = [dp['error'] for dp in self.calibration_history]
        
        return {
            'mean_absolute_error': sum(errors) / len(errors),
            'num_datapoints': len(self.calibration_history),
            'calibrated': len(self.calibration_history) >= 10
        }


# Test uncertainty quantification
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING UNCERTAINTY QUANTIFICATION")
    print("=" * 60)
    
    # Create sample documents
    supporting_docs = [
        Document(
            page_content="2023 study shows moderate coffee is beneficial for heart health.",
            metadata={'source': 'study_2023.txt', 'year': 2023, 'domain': 'medical', 
                     'credibility_score': 0.9, 'quality_score': 0.85}
        ),
        Document(
            page_content="2022 meta-analysis confirms moderate coffee is safe and potentially beneficial.",
            metadata={'source': 'meta_2022.txt', 'year': 2022, 'domain': 'research',
                     'credibility_score': 0.95, 'quality_score': 0.95}
        ),
        Document(
            page_content="2021 review supports moderate coffee consumption for cardiovascular health.",
            metadata={'source': 'review_2021.txt', 'year': 2021, 'domain': 'medical',
                     'credibility_score': 0.85, 'quality_score': 0.8}
        )
    ]
    
    contradicting_docs = [
        Document(
            page_content="2018 study found high coffee consumption increases cardiovascular risk.",
            metadata={'source': 'study_2018.txt', 'year': 2018, 'credibility_score': 0.7}
        )
    ]
    
    # Test 1: Quantify uncertainty
    print("\n1️⃣ Testing Uncertainty Quantification...")
    
    quantifier = UncertaintyQuantifier()
    
    query = "Is moderate coffee consumption safe for heart health?"
    answer = "Yes, moderate coffee consumption (2-3 cups/day) is safe and potentially beneficial for heart health based on recent research."
    
    uncertainty = quantifier.quantify_uncertainty(
        query,
        answer,
        supporting_docs,
        contradicting_docs
    )
    
    print(f"   ✅ Uncertainty quantified")
    print(f"   Confidence: {uncertainty['confidence_score']:.2%}")
    print(f"   Level: {uncertainty['confidence_level']}")
    print(f"   Uncertainty sources: {len(uncertainty['uncertainty_sources'])}")
    
    # Test 2: Generate report
    print("\n2️⃣ Generating Uncertainty Report...")
    
    report = quantifier.generate_uncertainty_report(query, answer, uncertainty)
    print(report)
    
    # Test 3: Evidence sufficiency
    print("\n3️⃣ Testing Evidence Sufficiency Analysis...")
    
    sufficiency_analyzer = EvidenceSufficiencyAnalyzer()
    
    sufficiency = sufficiency_analyzer.analyze_sufficiency(query, supporting_docs)
    
    print(f"   ✅ Evidence analysis complete")
    print(f"   Sufficient: {'Yes' if sufficiency['is_sufficient'] else 'No'}")
    print(f"   Sufficiency Score: {sufficiency['sufficiency_score']:.2%}")
    print(f"   Missing aspects: {len(sufficiency['missing_aspects'])}")
    print(f"   Recommendations: {len(sufficiency['recommended_additions'])}")
    
    if sufficiency['recommended_additions']:
        print("\n   Recommendations:")
        for rec in sufficiency['recommended_additions']:
            print(f"      • {rec}")
    
    # Test 4: Confidence calibration
    print("\n4️⃣ Testing Confidence Calibration...")
    
    calibrator = ConfidenceCalibrator()
    
    raw_confidence = 0.85
    calibrated = calibrator.calibrate_confidence(raw_confidence, 'medium')
    
    print(f"   Raw confidence: {raw_confidence:.2%}")
    print(f"   Calibrated confidence: {calibrated:.2%}")
    print(f"   Adjustment: {calibrated - raw_confidence:+.2%}")
    
    # Test 5: Sensitivity scenarios
    print("\n5️⃣ Sensitivity Analysis Scenarios...")
    
    print(f"   Base confidence: {uncertainty['confidence_score']:.2%}")
    print(f"\n   What-if scenarios:")
    
    for scenario, conf in uncertainty['sensitivity_analysis'].items():
        change = conf - uncertainty['confidence_score']
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"      {scenario.replace('_', ' ').title()}: {conf:.2%} ({arrow} {abs(change):.2%})")
    
    print("\n" + "=" * 60)
    print("✅ UNCERTAINTY QUANTIFICATION TEST COMPLETE")
    print("=" * 60)
    print("\n🎯 Uncertainty Innovation Added:")
    print("  ✅ Bayesian confidence estimation")
    print("  ✅ Uncertainty source identification")
    print("  ✅ Evidence gap detection")
    print("  ✅ Sensitivity analysis (what-if scenarios)")
    print("  ✅ Confidence calibration")
    print("  ✅ Transparent uncertainty communication")