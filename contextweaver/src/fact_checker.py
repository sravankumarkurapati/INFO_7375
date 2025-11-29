# src/fact_checker.py
from typing import List, Dict, Tuple, Optional
import logging
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import json
import re

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class AutomatedFactChecker:
    """
    Automated fact verification system
    
    INNOVATION: Multi-source claim verification
    
    Pipeline:
    1. Extract factual claims from text
    2. Cross-verify against available sources
    3. Assign verification scores
    4. Detect unsupported claims
    5. Generate verification report
    
    Use cases:
    - Verify AI-generated answers
    - Fact-check documents
    - Detect misinformation
    - Quality assurance
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.1)
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier()
        logger.info("🔍 Automated fact checker initialized")
    
    def fact_check_answer(
        self,
        answer: str,
        source_documents: List[Document]
    ) -> Dict[str, any]:
        """
        Fact-check an answer against source documents
        
        Innovation: Automated verification pipeline
        
        Returns:
            - claims: Extracted factual claims
            - verifications: Verification status for each claim
            - overall_score: Percentage of verified claims
            - flagged_claims: Potentially false claims
        """
        
        logger.info("🔍 Fact-checking answer against sources...")
        
        # Step 1: Extract claims from answer
        claims = self.claim_extractor.extract_claims(answer)
        
        logger.info(f"   📝 Extracted {len(claims)} factual claims")
        
        # Step 2: Verify each claim
        verifications = []
        
        for claim in claims:
            verification = self.claim_verifier.verify_claim(claim, source_documents)
            verifications.append(verification)
        
        # Step 3: Calculate overall verification score
        verified_count = sum(1 for v in verifications if v['status'] == 'VERIFIED')
        overall_score = verified_count / len(verifications) if verifications else 0
        
        # Step 4: Flag problematic claims
        flagged = [
            v for v in verifications
            if v['status'] in ['CONTRADICTED', 'UNSUPPORTED']
        ]
        
        result = {
            'claims': claims,
            'verifications': verifications,
            'overall_score': overall_score,
            'num_verified': verified_count,
            'num_contradicted': sum(1 for v in verifications if v['status'] == 'CONTRADICTED'),
            'num_unsupported': sum(1 for v in verifications if v['status'] == 'UNSUPPORTED'),
            'flagged_claims': flagged,
            'verification_level': self._categorize_verification(overall_score)
        }
        
        logger.info(f"✅ Fact-check complete: {overall_score:.1%} verified")
        
        return result
    
    def _categorize_verification(self, score: float) -> str:
        """Categorize verification level"""
        
        if score >= 0.9:
            return "HIGHLY VERIFIED"
        elif score >= 0.7:
            return "WELL VERIFIED"
        elif score >= 0.5:
            return "PARTIALLY VERIFIED"
        else:
            return "POORLY VERIFIED"
    
    def fact_check_document(
        self,
        document: Document,
        reference_documents: List[Document]
    ) -> Dict[str, any]:
        """
        Fact-check an entire document against references
        
        Use case: Verify new document before adding to knowledge base
        """
        
        logger.info(f"🔍 Fact-checking document: {document.metadata.get('source', 'unknown')}")
        
        # Extract claims from document
        claims = self.claim_extractor.extract_claims(document.page_content)
        
        # Verify against references
        verifications = []
        
        for claim in claims:
            verification = self.claim_verifier.verify_claim(claim, reference_documents)
            verifications.append(verification)
        
        # Assess document trustworthiness
        verified_ratio = sum(1 for v in verifications if v['status'] == 'VERIFIED') / len(verifications) if verifications else 0
        
        return {
            'document': document.metadata.get('source', 'unknown'),
            'claims_extracted': len(claims),
            'verifications': verifications,
            'verified_ratio': verified_ratio,
            'trustworthiness': 'HIGH' if verified_ratio >= 0.8 else 'MEDIUM' if verified_ratio >= 0.5 else 'LOW'
        }
    
    def generate_fact_check_report(
        self,
        answer: str,
        fact_check_result: Dict
    ) -> str:
        """Generate human-readable fact-check report"""
        
        report = f"""
{'='*60}
FACT-CHECK REPORT
{'='*60}

Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}

{'─'*60}
VERIFICATION SUMMARY
{'─'*60}

Overall Verification: {fact_check_result['overall_score']:.1%}
Verification Level: {fact_check_result['verification_level']}

✅ Verified Claims: {fact_check_result['num_verified']}
⚠️ Unsupported Claims: {fact_check_result['num_unsupported']}
❌ Contradicted Claims: {fact_check_result['num_contradicted']}

{'─'*60}
DETAILED CLAIM VERIFICATION
{'─'*60}

"""
        
        for i, verification in enumerate(fact_check_result['verifications'], 1):
            status_icon = {
                'VERIFIED': '✅',
                'UNSUPPORTED': '⚠️',
                'CONTRADICTED': '❌',
                'UNCERTAIN': '❓'
            }.get(verification['status'], '❓')
            
            report += f"{i}. {status_icon} {verification['claim']}\n"
            report += f"   Status: {verification['status']}\n"
            report += f"   Confidence: {verification['confidence']:.2%}\n"
            
            if verification.get('supporting_sources'):
                report += f"   Sources: {', '.join(verification['supporting_sources'])}\n"
            
            report += "\n"
        
        if fact_check_result['flagged_claims']:
            report += f"{'─'*60}\n"
            report += f"⚠️ FLAGGED CLAIMS (Needs Review)\n"
            report += f"{'─'*60}\n\n"
            
            for flag in fact_check_result['flagged_claims']:
                report += f"• {flag['claim']}\n"
                report += f"  Reason: {flag['status']}\n\n"
        
        report += f"{'='*60}\n"
        
        return report


class ClaimExtractor:
    """
    Extract factual claims from text
    
    Innovation: Structured claim extraction for verification
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.1)
        logger.info("📝 Claim extractor initialized")
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text
        
        Returns list of atomic factual statements
        """
        
        prompt = f"""Extract all factual claims from the following text.

Text:
{text}

Instructions:
- Extract only factual statements (not opinions)
- Make each claim atomic (one fact per claim)
- Include numerical claims if present
- Preserve original wording where possible

Return as JSON array:
["claim 1", "claim 2", ...]

JSON:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            claims = json.loads(content)
            
            return claims if isinstance(claims, list) else []
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting claims: {e}")
            
            # Fallback: sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20][:5]
    
    def categorize_claim_type(self, claim: str) -> str:
        """Categorize type of claim"""
        
        claim_lower = claim.lower()
        
        # Numerical claim
        if re.search(r'\d+%|\d+\.\d+', claim):
            return 'NUMERICAL'
        
        # Causal claim
        if any(word in claim_lower for word in ['causes', 'leads to', 'results in', 'due to']):
            return 'CAUSAL'
        
        # Comparative claim
        if any(word in claim_lower for word in ['more', 'less', 'better', 'worse', 'higher', 'lower']):
            return 'COMPARATIVE'
        
        # Temporal claim
        if any(word in claim_lower for word in ['recent', 'previous', 'earlier', '20']):
            return 'TEMPORAL'
        
        return 'GENERAL'


class ClaimVerifier:
    """
    Verify claims against source documents
    
    Innovation: Multi-source cross-verification
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0.1)
        logger.info("✅ Claim verifier initialized")
    
    def verify_claim(
        self,
        claim: str,
        source_documents: List[Document]
    ) -> Dict[str, any]:
        """
        Verify a single claim against sources
        
        Returns:
            - status: VERIFIED / CONTRADICTED / UNSUPPORTED / UNCERTAIN
            - confidence: 0-1
            - supporting_sources: List of sources that support
            - contradicting_sources: List of sources that contradict
            - evidence: Relevant excerpts
        """
        
        # Search for evidence in documents
        supporting = []
        contradicting = []
        uncertain = []
        
        for doc in source_documents:
            verification = self._verify_against_document(claim, doc)
            
            if verification == 'SUPPORTS':
                supporting.append(doc.metadata.get('source', 'unknown'))
            elif verification == 'CONTRADICTS':
                contradicting.append(doc.metadata.get('source', 'unknown'))
            elif verification == 'UNCERTAIN':
                uncertain.append(doc.metadata.get('source', 'unknown'))
        
        # Determine overall status
        if supporting and not contradicting:
            status = 'VERIFIED'
            confidence = min(0.7 + len(supporting) * 0.1, 0.95)
        elif contradicting and not supporting:
            status = 'CONTRADICTED'
            confidence = min(0.7 + len(contradicting) * 0.1, 0.95)
        elif supporting and contradicting:
            status = 'UNCERTAIN'
            confidence = 0.4
        else:
            status = 'UNSUPPORTED'
            confidence = 0.2
        
        return {
            'claim': claim,
            'status': status,
            'confidence': confidence,
            'supporting_sources': supporting,
            'contradicting_sources': contradicting,
            'num_sources_checked': len(source_documents)
        }
    
    def _verify_against_document(self, claim: str, document: Document) -> str:
        """
        Check if claim is supported/contradicted by document
        
        Returns: SUPPORTS / CONTRADICTS / UNCERTAIN / IRRELEVANT
        """
        
        # Simple keyword matching (production would use semantic similarity)
        claim_lower = claim.lower()
        doc_lower = document.page_content.lower()
        
        # Extract key terms from claim
        claim_terms = set(re.findall(r'\b\w{4,}\b', claim_lower))
        
        # Check overlap
        overlap = sum(1 for term in claim_terms if term in doc_lower)
        overlap_ratio = overlap / len(claim_terms) if claim_terms else 0
        
        if overlap_ratio >= 0.6:
            # High overlap - likely supports or discusses the claim
            
            # Check for negation indicators
            negation_words = ['not', 'no', 'never', 'contradicts', 'opposes', 'against']
            
            # Get context around claim terms
            has_negation = any(neg in doc_lower for neg in negation_words)
            
            if has_negation:
                return 'CONTRADICTS'
            else:
                return 'SUPPORTS'
        
        elif overlap_ratio >= 0.3:
            return 'UNCERTAIN'
        
        else:
            return 'IRRELEVANT'
    
    def multi_source_verification(
        self,
        claim: str,
        sources: List[Document],
        min_sources_required: int = 2
    ) -> Dict[str, any]:
        """
        Require multiple sources to verify a claim
        
        Innovation: Multi-source consensus for higher confidence
        """
        
        verification = self.verify_claim(claim, sources)
        
        num_supporting = len(verification['supporting_sources'])
        
        # Enhanced verification requiring multiple sources
        if num_supporting >= min_sources_required:
            verification['multi_source_verified'] = True
            verification['confidence'] = min(verification['confidence'] * 1.2, 0.98)
        else:
            verification['multi_source_verified'] = False
            if verification['status'] == 'VERIFIED':
                verification['status'] = 'SINGLE_SOURCE_ONLY'
                verification['confidence'] *= 0.7
        
        return verification


class MisinformationDetector:
    """
    Detect potential misinformation patterns
    
    Innovation: Proactive misinformation detection
    
    Red flags:
    - Extreme claims without evidence
    - Cherry-picked data
    - Outdated information presented as current
    - Missing caveats
    """
    
    def __init__(self):
        logger.info("🚨 Misinformation detector initialized")
    
    def detect_red_flags(
        self,
        answer: str,
        fact_check_result: Dict,
        source_documents: List[Document]
    ) -> List[Dict]:
        """
        Detect misinformation red flags
        
        Innovation: Pattern-based misinformation detection
        """
        
        red_flags = []
        
        # Red flag 1: Low verification score
        if fact_check_result['overall_score'] < 0.5:
            red_flags.append({
                'type': 'LOW_VERIFICATION',
                'severity': 'HIGH',
                'description': f"Only {fact_check_result['overall_score']:.1%} of claims verified",
                'recommendation': "Review unsupported claims before trusting answer"
            })
        
        # Red flag 2: Contradicted claims
        if fact_check_result['num_contradicted'] > 0:
            red_flags.append({
                'type': 'CONTRADICTED_CLAIMS',
                'severity': 'HIGH',
                'description': f"{fact_check_result['num_contradicted']} claims contradicted by sources",
                'recommendation': "Investigate contradictions before accepting answer"
            })
        
        # Red flag 3: Extreme language
        extreme_words = ['always', 'never', 'all', 'none', 'definitely', 'certainly', 'proven']
        
        answer_lower = answer.lower()
        extreme_found = [word for word in extreme_words if word in answer_lower]
        
        if extreme_found and fact_check_result['overall_score'] < 0.8:
            red_flags.append({
                'type': 'EXTREME_LANGUAGE',
                'severity': 'MEDIUM',
                'description': f"Extreme language used: {', '.join(extreme_found)}",
                'recommendation': "Be cautious of absolute statements with limited verification"
            })
        
        # Red flag 4: Outdated sources
        if source_documents:
            years = [doc.metadata.get('year', 0) for doc in source_documents if doc.metadata.get('year')]
            
            if years and max(years) < 2020:
                red_flags.append({
                    'type': 'OUTDATED_SOURCES',
                    'severity': 'MEDIUM',
                    'description': f"Most recent source is from {max(years)}",
                    'recommendation': "Seek more recent sources for current information"
                })
        
        # Red flag 5: Single source dependency
        single_source_verifications = [
            v for v in fact_check_result['verifications']
            if len(v.get('supporting_sources', [])) == 1
        ]
        
        if len(single_source_verifications) > len(fact_check_result['verifications']) / 2:
            red_flags.append({
                'type': 'SINGLE_SOURCE_DEPENDENCY',
                'severity': 'MEDIUM',
                'description': "Many claims rely on single sources",
                'recommendation': "Cross-verify with additional sources"
            })
        
        logger.info(f"🚨 Detected {len(red_flags)} red flags")
        
        return red_flags
    
    def calculate_misinformation_risk(
        self,
        fact_check_result: Dict,
        red_flags: List[Dict]
    ) -> Dict[str, any]:
        """
        Calculate overall misinformation risk
        
        Innovation: Composite risk score
        """
        
        # Base risk from verification score
        base_risk = 1 - fact_check_result['overall_score']
        
        # Risk from red flags
        severity_weights = {'HIGH': 0.3, 'MEDIUM': 0.15, 'LOW': 0.05}
        
        red_flag_risk = sum(
            severity_weights.get(flag['severity'], 0.1)
            for flag in red_flags
        )
        
        # Combined risk (capped at 1.0)
        total_risk = min(base_risk + red_flag_risk, 1.0)
        
        # Categorize risk level
        if total_risk >= 0.7:
            risk_level = 'HIGH RISK'
        elif total_risk >= 0.4:
            risk_level = 'MODERATE RISK'
        else:
            risk_level = 'LOW RISK'
        
        return {
            'total_risk': total_risk,
            'risk_level': risk_level,
            'base_risk': base_risk,
            'red_flag_contribution': red_flag_risk,
            'num_red_flags': len(red_flags)
        }


class VerificationScorecard:
    """
    Generate verification scorecards for answers
    
    Innovation: Transparent fact-checking metrics
    """
    
    def __init__(self):
        logger.info("📊 Verification scorecard initialized")
    
    def generate_scorecard(
        self,
        query: str,
        answer: str,
        fact_check: Dict,
        red_flags: List[Dict],
        uncertainty: Dict
    ) -> str:
        """
        Generate comprehensive verification scorecard
        
        Combines:
        - Fact-checking results
        - Uncertainty quantification  
        - Red flag analysis
        """
        
        misinformation_detector = MisinformationDetector()
        risk = misinformation_detector.calculate_misinformation_risk(fact_check, red_flags)
        
        scorecard = f"""
{'='*60}
VERIFICATION SCORECARD
{'='*60}

Query: {query}

{'─'*60}
SCORES
{'─'*60}

Verification Score:     {fact_check['overall_score']:.1%}  {self._get_grade(fact_check['overall_score'])}
Confidence Score:       {uncertainty['confidence_score']:.1%}  {self._get_grade(uncertainty['confidence_score'])}
Misinformation Risk:    {risk['total_risk']:.1%}  {self._get_risk_grade(risk['total_risk'])}

Overall Assessment:     {self._get_overall_assessment(fact_check, uncertainty, risk)}

{'─'*60}
BREAKDOWN
{'─'*60}

✅ Verified Claims:     {fact_check['num_verified']}/{len(fact_check['claims'])}
⚠️ Unsupported:         {fact_check['num_unsupported']}/{len(fact_check['claims'])}
❌ Contradicted:        {fact_check['num_contradicted']}/{len(fact_check['claims'])}

🚨 Red Flags:          {len(red_flags)}
   High Severity:      {sum(1 for f in red_flags if f['severity'] == 'HIGH')}
   Medium Severity:    {sum(1 for f in red_flags if f['severity'] == 'MEDIUM')}

{'─'*60}
RECOMMENDATION
{'─'*60}

{self._get_recommendation(fact_check, uncertainty, risk)}

{'='*60}
"""
        
        return scorecard
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.5:
            return "C"
        else:
            return "D"
    
    def _get_risk_grade(self, risk: float) -> str:
        """Convert risk to grade (inverse)"""
        
        if risk <= 0.2:
            return "A (Low Risk)"
        elif risk <= 0.4:
            return "B (Moderate Risk)"
        elif risk <= 0.6:
            return "C (Elevated Risk)"
        else:
            return "D (High Risk)"
    
    def _get_overall_assessment(
        self,
        fact_check: Dict,
        uncertainty: Dict,
        risk: Dict
    ) -> str:
        """Generate overall assessment"""
        
        verification_score = fact_check['overall_score']
        confidence_score = uncertainty['confidence_score']
        risk_score = risk['total_risk']
        
        if verification_score >= 0.8 and confidence_score >= 0.7 and risk_score <= 0.3:
            return "✅ TRUSTWORTHY - High verification, high confidence, low risk"
        elif verification_score >= 0.6 and risk_score <= 0.5:
            return "⚠️ ACCEPTABLE - Reasonable verification, moderate risk"
        else:
            return "❌ QUESTIONABLE - Low verification or high risk, needs review"
    
    def _get_recommendation(
        self,
        fact_check: Dict,
        uncertainty: Dict,
        risk: Dict
    ) -> str:
        """Generate actionable recommendation"""
        
        recommendations = []
        
        if fact_check['overall_score'] < 0.7:
            recommendations.append("• Verify unsupported claims with additional sources")
        
        if uncertainty['confidence_score'] < 0.6:
            recommendations.append("• Address uncertainty sources to improve confidence")
        
        if risk['total_risk'] > 0.4:
            recommendations.append("• Review red flags before accepting answer")
        
        if fact_check['num_contradicted'] > 0:
            recommendations.append("• Investigate contradicted claims urgently")
        
        if not recommendations:
            recommendations.append("• Answer appears reliable, minimal concerns")
        
        return "\n".join(recommendations)


# Test fact checker
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING FACT CHECKER")
    print("=" * 60)
    
    # Sample answer to fact-check
    answer = """Yes, moderate coffee consumption (2-3 cups/day) is safe and potentially 
beneficial for heart health. Recent research shows a 15% reduction in cardiovascular 
disease risk. The 2022 meta-analysis confirms this finding."""
    
    # Source documents
    source_docs = [
        Document(
            page_content="2023 study found moderate coffee consumption (2-3 cups daily) associated with 15% reduction in cardiovascular risk.",
            metadata={'source': 'study_2023.txt', 'year': 2023, 'credibility_score': 0.9}
        ),
        Document(
            page_content="2022 meta-analysis of 50 studies concluded moderate coffee is safe for heart health.",
            metadata={'source': 'meta_2022.txt', 'year': 2022, 'credibility_score': 0.95}
        ),
        Document(
            page_content="2018 study found high coffee consumption increases cardiovascular risk by 23%.",
            metadata={'source': 'study_2018.txt', 'year': 2018, 'credibility_score': 0.7}
        )
    ]
    
    # Test 1: Fact-check answer
    print("\n1️⃣ Testing Fact-Checking Pipeline...")
    
    fact_checker = AutomatedFactChecker()
    
    fact_check_result = fact_checker.fact_check_answer(answer, source_docs)
    
    print(f"   ✅ Fact-check complete")
    print(f"   Verification: {fact_check_result['overall_score']:.1%}")
    print(f"   Level: {fact_check_result['verification_level']}")
    print(f"   Claims extracted: {len(fact_check_result['claims'])}")
    
    # Test 2: Generate fact-check report
    print("\n2️⃣ Generating Fact-Check Report...")
    
    report = fact_checker.generate_fact_check_report(answer, fact_check_result)
    print(report)
    
    # Test 3: Detect red flags
    print("\n3️⃣ Testing Misinformation Detection...")
    
    detector = MisinformationDetector()
    
    red_flags = detector.detect_red_flags(answer, fact_check_result, source_docs)
    
    print(f"   ✅ Red flag detection complete")
    print(f"   Red flags found: {len(red_flags)}")
    
    for flag in red_flags:
        print(f"      {flag['severity']}: {flag['type']}")
    
    # Test 4: Calculate misinformation risk
    print("\n4️⃣ Calculating Misinformation Risk...")
    
    risk = detector.calculate_misinformation_risk(fact_check_result, red_flags)
    
    print(f"   Total Risk: {risk['total_risk']:.1%}")
    print(f"   Risk Level: {risk['risk_level']}")
    
    # Test 5: Generate scorecard
    print("\n5️⃣ Generating Verification Scorecard...")
    
    # Create dummy uncertainty data
    uncertainty_data = {
        'confidence_score': 0.75,
        'confidence_level': 'HIGH'
    }
    
    scorecard_gen = VerificationScorecard()
    scorecard = scorecard_gen.generate_scorecard(
        "Is moderate coffee consumption safe?",
        answer,
        fact_check_result,
        red_flags,
        uncertainty_data
    )
    
    print(scorecard)
    
    print("=" * 60)
    print("✅ FACT CHECKER TEST COMPLETE")
    print("=" * 60)
    print("\n🎯 Fact-Checking Innovation Added:")
    print("  ✅ Automated claim extraction")
    print("  ✅ Multi-source verification")
    print("  ✅ Contradiction detection")
    print("  ✅ Red flag identification")
    print("  ✅ Misinformation risk scoring")
    print("  ✅ Verification scorecards")