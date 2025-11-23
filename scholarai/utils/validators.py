"""
Validation utilities for ScholarAI
"""
from typing import Dict, List, Any, Optional
from utils.logger import logger

class ValidationError(Exception):
    """Custom validation error"""
    pass

class Validators:
    """Collection of validation functions"""
    
    @staticmethod
    def validate_papers(papers: List[Dict], min_count: int = 10) -> bool:
        """
        Validate paper search results
        
        Args:
            papers: List of paper dictionaries
            min_count: Minimum required papers
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(papers, list):
            raise ValidationError("Papers must be a list")
        
        if len(papers) < min_count:
            raise ValidationError(
                f"Insufficient papers: {len(papers)} < {min_count}"
            )
        
        # Check required fields
        required_fields = ['title', 'url']
        for i, paper in enumerate(papers):
            if not isinstance(paper, dict):
                raise ValidationError(f"Paper {i} is not a dictionary")
            
            for field in required_fields:
                if field not in paper:
                    raise ValidationError(
                        f"Paper {i} missing required field: {field}"
                    )
        
        logger.info(f"✅ Papers validation passed: {len(papers)} papers")
        return True
    
    @staticmethod
    def validate_analysis(analysis: Dict) -> bool:
        """
        Validate paper analysis output
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(analysis, dict):
            raise ValidationError("Analysis must be a dictionary")
        
        if "analyses" not in analysis:
            raise ValidationError("Analysis missing 'analyses' field")
        
        analyses = analysis["analyses"]
        if not isinstance(analyses, list):
            raise ValidationError("'analyses' must be a list")
        
        if len(analyses) == 0:
            raise ValidationError("No paper analyses found")
        
        logger.info(f"✅ Analysis validation passed: {len(analyses)} papers analyzed")
        return True
    
    @staticmethod
    def validate_gaps(gaps: List[Dict], min_count: int = 3) -> bool:
        """
        Validate research gaps
        
        Args:
            gaps: List of gap dictionaries
            min_count: Minimum required gaps
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(gaps, list):
            raise ValidationError("Gaps must be a list")
        
        if len(gaps) < min_count:
            raise ValidationError(
                f"Insufficient gaps: {len(gaps)} < {min_count}"
            )
        
        # Check required fields
        required_fields = ['description', 'confidence']
        for i, gap in enumerate(gaps):
            if not isinstance(gap, dict):
                raise ValidationError(f"Gap {i} is not a dictionary")
            
            for field in required_fields:
                if field not in gap:
                    raise ValidationError(
                        f"Gap {i} missing required field: {field}"
                    )
            
            # Validate confidence score
            confidence = gap.get('confidence', 0)
            if not isinstance(confidence, (int, float)):
                raise ValidationError(f"Gap {i} confidence must be a number")
            
            if not 0 <= confidence <= 1:
                raise ValidationError(
                    f"Gap {i} confidence must be between 0 and 1"
                )
        
        logger.info(f"✅ Gaps validation passed: {len(gaps)} gaps")
        return True
    
    @staticmethod
    def validate_quality_score(score: float) -> bool:
        """
        Validate quality assessment score
        
        Args:
            score: Quality score (0-10)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(score, (int, float)):
            raise ValidationError("Quality score must be a number")
        
        if not 0 <= score <= 10:
            raise ValidationError("Quality score must be between 0 and 10")
        
        logger.info(f"✅ Quality score validation passed: {score}/10")
        return True

# Create singleton instance
validators = Validators()