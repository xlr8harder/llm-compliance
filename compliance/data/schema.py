"""
Data classes for compliance evaluation schema.
"""
from dataclasses import dataclass, field, asdict
import re
from typing import Dict, List, Optional, Any, Union


MODERATION_ERROR_TEXT_RE = re.compile(
    r"("
    r"usage policy|"
    r"requires moderation|"
    r"input was flagged|"
    r"prompt was flagged|"
    r"output data may contain inappropriate content|"
    r"input data may contain inappropriate content|"
    r"data_inspection_failed|"
    r"prohibited_content|"
    r"content[_ -]?filter|"
    r"safety policy|"
    r"policy violation|"
    r"blocked by (?:safety|moderation)|"
    r"\bmoderation\b"
    r")",
    re.IGNORECASE,
)


def _collect_strings(value: Any) -> list[str]:
    """Collect strings from nested API error payloads."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for nested in value.values():
            strings.extend(_collect_strings(nested))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for nested in value:
            strings.extend(_collect_strings(nested))
        return strings
    return []


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
    - request_format: Client request format used to make the request
    - raw_response_format: Schema/shape of the stored response payload
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
    request_format: Optional[str] = None
    raw_response_format: Optional[str] = None
    category: Optional[str] = None
    domain: Optional[str] = None
    model_requested: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def _finish_reason(self) -> str | None:
        """Return the provider finish/stop reason when present."""
        try:
            finish_reason = self.response["choices"][0].get("finish_reason")
            if finish_reason is not None:
                return finish_reason
        except Exception:  # noqa: BLE001
            pass
        if isinstance(self.response, dict):
            stop_reason = self.response.get("stop_reason")
            if isinstance(stop_reason, str):
                return stop_reason
        return None

    def _native_finish_reason(self) -> str | None:
        """Return provider-native finish/stop reason when OpenRouter exposes one."""
        try:
            native_finish_reason = self.response["choices"][0].get("native_finish_reason")
            if isinstance(native_finish_reason, str):
                return native_finish_reason
        except Exception:  # noqa: BLE001
            pass
        return None

    def _message_content(self) -> Any:
        """Return assistant content from Chat Completions shape when present."""
        try:
            return self.response["choices"][0].get("message", {}).get("content")
        except Exception:  # noqa: BLE001
            return None

    def _content_blocks(self) -> List[Dict[str, Any]]:
        """Return native Messages content blocks when present."""
        if not isinstance(self.response, dict):
            return []
        content = self.response.get("content")
        if not isinstance(content, list):
            return []
        return [block for block in content if isinstance(block, dict)]

    def final_content_text(self) -> str:
        """Return final assistant text across supported response formats."""
        content = self._message_content()
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "".join(parts)

        parts = [
            block.get("text", "")
            for block in self._content_blocks()
            if block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        return "".join(parts)

    def _has_missing_final_content(self) -> bool:
        """True when the provider returned no final assistant text."""
        return self.final_content_text() == ""

    def _contains_api_error(self) -> bool:
        """
        Tight clone of the old utils.llm_client.is_permanent_api_error
        but local to the schema:

        • Top-level "error" key  -> permanent error  
        • choices[0]["error"]    -> permanent error  
        • HTTP / provider errors are already baked into the blob
        """
        # Top-level
        if isinstance(self.response, dict) and self.response.get("error"):
            return True
        # Inside first choice
        try:
            first_choice = self.response["choices"][0]
            if isinstance(first_choice, dict) and first_choice.get("error"):
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def original_moderation_reason(self) -> str | None:
        """Return the original-provider moderation/refusal stop reason, if any."""
        finish_reason = self._finish_reason()
        native_finish_reason = self._native_finish_reason()

        if finish_reason in {"content_filter", "refusal"}:
            return finish_reason

        if native_finish_reason in {"content_filter", "refusal"}:
            return native_finish_reason

        if isinstance(self.response, dict):
            top_error = self.response.get("error")
            if isinstance(top_error, dict) and top_error.get("type") == "content_filter":
                return "content_filter"
            if any(MODERATION_ERROR_TEXT_RE.search(text) for text in _collect_strings(top_error)):
                return "moderation_error_text"
            try:
                choice_error = self.response["choices"][0].get("error")
                if isinstance(choice_error, dict) and choice_error.get("type") == "content_filter":
                    return "content_filter"
                if any(MODERATION_ERROR_TEXT_RE.search(text) for text in _collect_strings(choice_error)):
                    return "moderation_error_text"
            except Exception:  # noqa: BLE001
                pass

        return None

    def is_original_moderation_error(self) -> bool:
        """True when the original model was stopped by a moderation/classifier gate."""
        return self.original_moderation_reason() is not None

    def is_frpe_retry_candidate(self) -> bool:
        """True when FRPE cleanup should remove the row for retry."""
        return self.is_permanent_error() and not self.is_original_moderation_error()

    def is_permanent_error(self) -> bool:
        if self._contains_api_error():
            return True
        if self.is_original_moderation_error():
            return True
        if self._finish_reason() == "error":
            return True
        if self._finish_reason() == "max_tokens":
            return True
        if self._has_missing_final_content():
            return True
        return False

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
    - compliance: Compliance rating ("COMPLETE", "EVASIVE", "DENIAL", or error code)
    - judge_analysis: Detailed analysis text
    
    Optional fields:
    - timestamp: ISO format timestamp when the analysis was generated
    - original_api_provider: Original provider used for the response
    - raw_judge_response: Raw response from the judge model
    - category: Category of the question
    - domain: Domain of the question
    - api_model: Provider-specific model ID of the response
    - judge_api_provider: Provider used for judging
    """
    question_id: str
    question: str
    model: str  # Canonical model name
    response: Dict  # Provider-specific standardized response
    judge_model: str
    compliance: str  # "COMPLETE", "EVASIVE", "DENIAL", or error code
    judge_analysis: str
    timestamp: Optional[str] = None
    original_api_provider: Optional[str] = None
    raw_judge_response: Optional[str] = None
    category: Optional[str] = None
    domain: Optional[str] = None
    api_model: Optional[str] = None
    judge_api_provider: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    
