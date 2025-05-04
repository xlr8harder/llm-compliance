"""
Data classes for compliance evaluation schema.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union


@dataclass
class Question:
    """
    Question schema for compliance evaluation.
    
    Required fields:
    - id: Unique identifier for the question
    - question: The text of the question
    
    Optional fields:
    - category: Category of the question (e.g., 'us_hard')
    - domain: Domain of the question (e.g., 'Ethics (Situational/Abstract/Virtual)')
    - _source_label: Internal label for the source
    - _generated_at: Timestamp when the question was generated
    - _variation_of: ID of the question this is a variation of
    """
    id: str
    question: str
    category: Optional[str] = None
    domain: Optional[str] = None
    _source_label: Optional[str] = None
    _generated_at: Optional[str] = None
    _variation_of: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ModelResponse:
    """
    Schema for LLM response data.
    
    Required fields:
    - question_id: ID of the question being answered
    - question: The text of the question
    - model: Canonical model name
    - timestamp: ISO format timestamp when the response was generated
    - response: Provider-specific standardized response
    
    Optional fields:
    - api_provider: Provider name (e.g., 'openai', 'openrouter')
    - api_model: Provider-specific model ID
    - raw_response: Full raw provider response
    - category: Category of the question
    - domain: Domain of the question
    - model_requested: Model that was originally requested (may differ from model used)
    """
    question_id: str
    question: str
    model: str  # Canonical model name
    timestamp: str
    response: Dict  # Provider-specific standardized response
    api_provider: Optional[str] = None
    api_model: Optional[str] = None
    raw_response: Optional[Dict] = None
    category: Optional[str] = None
    domain: Optional[str] = None
    model_requested: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def finish_reason(self) -> Optional[str]:
        try:
            return (self.response["choices"][0]
                    .get("finish_reason"))
        except Exception:
            return None

    def is_permanent_error(self) -> bool:
        # 1) use llm_client helper if possible
        from utils.llm_client import is_permanent_api_error
        if is_permanent_api_error(self.response):
            return True
        # 2) structural-but-empty glitch
        try:
            if (self.response.get("choices")
                    and self.response["choices"][0].get("message", {})
                           .get("content") == ""):
                return True
        except Exception:
            pass
        # 3) provider signalled an “error” finish_reason
        return self.finish_reason() == "error"

    def is_success(self) -> bool:
        return not self.is_permanent_error()


@dataclass
class ComplianceAnalysis:
    """
    Schema for compliance analysis results.
    
    Required fields:
    - question_id: ID of the question being analyzed
    - question: The text of the question
    - model: Canonical model name
    - response: Provider-specific standardized response
    - judge_model: Model used for judging
    - judge_api_provider: Provider used for judging
    - compliance: Compliance rating ("COMPLETE", "EVASIVE", "DENIAL", or error code)
    - judge_analysis: Detailed analysis text
    
    Optional fields:
    - timestamp: ISO format timestamp when the analysis was generated
    - original_api_provider: Original provider used for the response
    - raw_judge_response: Raw response from the judge model
    - category: Category of the question
    - domain: Domain of the question
    - api_model: Provider-specific model ID of the response
    """
    question_id: str
    question: str
    model: str  # Canonical model name
    response: Dict  # Provider-specific standardized response
    judge_model: str
    judge_api_provider: str
    compliance: str  # "COMPLETE", "EVASIVE", "DENIAL", or error code
    judge_analysis: str
    timestamp: Optional[str] = None
    original_api_provider: Optional[str] = None
    raw_judge_response: Optional[str] = None
    category: Optional[str] = None
    domain: Optional[str] = None
    api_model: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    
