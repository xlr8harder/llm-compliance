"""
Data classes for compliance evaluation schema.
"""
from dataclasses import dataclass, field, asdict
import re
from typing import Dict, List, Optional, Any, Union


RESPONSE_STATUS_SUCCESS = "success"
RESPONSE_STATUS_MODERATION = "moderation"
RESPONSE_STATUS_TRUNCATION = "truncation"
RESPONSE_STATUS_PROVIDER_ERROR = "provider_error"
RESPONSE_STATUS_EMPTY_RESPONSE = "empty_response"
RESPONSE_STATUS_MISSING_CONTENT = "missing_content"
RESPONSE_STATUS_METADATA_ERROR = "metadata_error"
RESPONSE_STATUS_UNKNOWN_METADATA = "unknown_metadata"
EMPTY_STOP_SUPPRESSION_MAX_COMPLETION_TOKENS = 8

SUCCESS_FINISH_REASONS = {
    "stop",
    "STOP",
    "completed",
    "end_turn",
    "eos_token",
    "eos",
}
MODERATION_FINISH_REASONS = {
    "content_filter",
    "PROHIBITED_CONTENT",
    "RECITATION",
    "refusal",
    "sensitive",
}
TRUNCATION_FINISH_REASONS = {
    "length",
    "MAX_TOKENS",
    "max_output_tokens",
    "max_tokens",
}
PROVIDER_ERROR_FINISH_REASONS = {
    "engine_overloaded",
    "error",
}
METADATA_ERROR_FINISH_REASONS = {
    "OTHER",
}
KNOWN_FINISH_REASONS = (
    SUCCESS_FINISH_REASONS
    | MODERATION_FINISH_REASONS
    | TRUNCATION_FINISH_REASONS
    | PROVIDER_ERROR_FINISH_REASONS
    | METADATA_ERROR_FINISH_REASONS
)

MODERATION_ERROR_TEXT_RE = re.compile(
    r"("
    r"usage policy|"
    r"requires moderation|"
    r"input was flagged|"
    r"prompt was flagged|"
    r"output data may contain inappropriate content|"
    r"input data may contain inappropriate content|"
    r"(?:input|output) text data may contain inappropriate content|"
    r"data_inspection_failed|"
    r"prohibited_content|"
    r"\brecitation\b|"
    r"content[_ -]?filter|"
    r"content (?:you provided or machine outputted )?is blocked|"
    r"content violates safety guidelines|"
    r"safety policy|"
    r"SAFETY_CHECK_TYPE|"
    r"\bcensorship\b|"
    r"policy violation|"
    r"blocked by (?:safety|moderation)|"
    r"limited access to this content for safety reasons|"
    r"\bmoderation\b"
    r")",
    re.IGNORECASE,
)


class UnknownResponseMetadataError(ValueError):
    """Raised when a response has metadata this pipeline does not classify."""


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
    response_status: Optional[str] = None
    response_status_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def _choice0(self) -> Dict[str, Any] | None:
        """Return first Chat Completions choice when present and well-formed."""
        try:
            choice = self.response["choices"][0]
        except Exception:  # noqa: BLE001
            return None
        return choice if isinstance(choice, dict) else None

    def _chat_finish_reason_is_missing(self) -> bool:
        """True when a Chat Completions row lacks a classified finish reason."""
        choice = self._choice0()
        if choice is None:
            return False
        return choice.get("finish_reason") is None

    def _llm_client_metadata(self) -> Dict[str, Any]:
        """Return llm_client sidecar metadata when present on newly collected rows."""
        if not isinstance(self.response, dict):
            return {}
        metadata = self.response.get("_llm_client")
        return metadata if isinstance(metadata, dict) else {}

    def _llm_client_standardized_response(self) -> Dict[str, Any]:
        metadata = self._llm_client_metadata()
        standardized = metadata.get("standardized_response")
        return standardized if isinstance(standardized, dict) else {}

    def _llm_client_error_info(self) -> Dict[str, Any]:
        metadata = self._llm_client_metadata()
        error_info = metadata.get("error_info")
        return error_info if isinstance(error_info, dict) else {}

    def _primary_finish_reason_with_source(self) -> tuple[str, str] | None:
        """Return the primary terminal reason with its provider metadata field."""
        choice = self._choice0()
        if choice is not None:
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                return "finish_reason", finish_reason
        if isinstance(self.response, dict):
            stop_reason = self.response.get("stop_reason")
            if isinstance(stop_reason, str):
                return "stop_reason", stop_reason
        standardized = self._llm_client_standardized_response()
        finish_reason = standardized.get("finish_reason")
        if isinstance(finish_reason, str):
            return "_llm_client.standardized_response.finish_reason", finish_reason
        return None

    def _finish_reason(self) -> str | None:
        """Return the provider finish/stop reason when present."""
        reason = self._primary_finish_reason_with_source()
        return reason[1] if reason is not None else None

    def _native_finish_reason(self) -> str | None:
        """Return provider-native finish/stop reason when OpenRouter exposes one."""
        choice = self._choice0()
        if choice is not None:
            native_finish_reason = choice.get("native_finish_reason")
            if isinstance(native_finish_reason, str):
                return native_finish_reason
        standardized = self._llm_client_standardized_response()
        native_finish_reason = standardized.get("native_finish_reason")
        if isinstance(native_finish_reason, str):
            return native_finish_reason
        error_info = self._llm_client_error_info()
        native_finish_reason = error_info.get("native_finish_reason")
        if isinstance(native_finish_reason, str):
            return native_finish_reason
        return None

    def _message_content(self) -> Any:
        """Return assistant content from Chat Completions shape when present."""
        choice = self._choice0()
        if choice is None:
            return None
        message = choice.get("message")
        if not isinstance(message, dict):
            return None
        return message.get("content")

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
        if parts:
            return "".join(parts)

        standardized_content = self._llm_client_standardized_response().get("content")
        if isinstance(standardized_content, str):
            return standardized_content
        return ""

    def _has_missing_final_content(self) -> bool:
        """True when the provider returned no final assistant text."""
        return self.final_content_text() == ""

    def _completion_tokens(self) -> int | None:
        """Return completion-token usage when available."""
        if not isinstance(self.response, dict):
            return None
        usage = self.response.get("usage")
        if not isinstance(usage, dict):
            return None
        completion_tokens = usage.get("completion_tokens")
        return completion_tokens if isinstance(completion_tokens, int) else None

    def _is_anthropic_model(self) -> bool:
        return self.model.startswith("anthropic/")

    def _has_empty_success_stop(self) -> bool:
        """True when provider reports success/stop but delivered no final text."""
        if not self._has_missing_final_content():
            return False
        if self._contains_api_error():
            return False

        primary_finish_reason = self._primary_finish_reason_with_source()
        native_finish_reason = self._native_finish_reason()
        finish_reasons = [
            reason
            for reason in (
                primary_finish_reason[1] if primary_finish_reason else None,
                native_finish_reason,
            )
            if isinstance(reason, str)
        ]
        if not finish_reasons or any(reason not in SUCCESS_FINISH_REASONS for reason in finish_reasons):
            return False

        completion_tokens = self._completion_tokens()
        return self._is_anthropic_model() or (
            completion_tokens is not None
            and completion_tokens <= EMPTY_STOP_SUPPRESSION_MAX_COMPLETION_TOKENS
        )

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
        first_choice = self._choice0()
        if first_choice is not None and first_choice.get("error"):
            return True
        return False

    def _provider_error_reason(self) -> str | None:
        """Return provider error detail when a structured API error is present."""
        if not isinstance(self.response, dict):
            return "response_not_dict"
        top_error = self.response.get("error")
        if top_error:
            if any(MODERATION_ERROR_TEXT_RE.search(text) for text in _collect_strings(top_error)):
                return None
            if isinstance(top_error, dict) and top_error.get("code") is not None:
                return f"top_error_code:{top_error['code']}"
            if isinstance(top_error, dict) and isinstance(top_error.get("type"), str):
                return f"top_error:{top_error['type']}"
            return "top_error"

        first_choice = self._choice0()
        if first_choice is not None:
            choice_error = first_choice.get("error")
            if choice_error:
                if any(MODERATION_ERROR_TEXT_RE.search(text) for text in _collect_strings(choice_error)):
                    return None
                if isinstance(choice_error, dict) and choice_error.get("code") is not None:
                    return f"choice_error_code:{choice_error['code']}"
                if isinstance(choice_error, dict) and isinstance(choice_error.get("type"), str):
                    return f"choice_error:{choice_error['type']}"
                return "choice_error"

        error_info = self._llm_client_error_info()
        if error_info:
            if any(MODERATION_ERROR_TEXT_RE.search(text) for text in _collect_strings(error_info)):
                return None
            if error_info.get("status_code") is not None:
                return f"llm_client_error_status:{error_info['status_code']}"
            if error_info.get("code") is not None:
                return f"llm_client_error_code:{error_info['code']}"
            if isinstance(error_info.get("type"), str):
                return f"llm_client_error:{error_info['type']}"
            return "llm_client_error"
        return None

    def _finish_status_for_reason(self, source: str, reason: str) -> tuple[str, str]:
        """Map one known finish/stop reason into an internal response status."""
        detail = f"{source}:{reason}"
        if reason in MODERATION_FINISH_REASONS:
            return RESPONSE_STATUS_MODERATION, detail
        if reason in TRUNCATION_FINISH_REASONS:
            return RESPONSE_STATUS_TRUNCATION, detail
        if reason in PROVIDER_ERROR_FINISH_REASONS:
            return RESPONSE_STATUS_PROVIDER_ERROR, detail
        if reason in METADATA_ERROR_FINISH_REASONS:
            return RESPONSE_STATUS_METADATA_ERROR, detail
        if reason in SUCCESS_FINISH_REASONS:
            return RESPONSE_STATUS_SUCCESS, detail
        return RESPONSE_STATUS_UNKNOWN_METADATA, f"unknown_{detail}"

    def classify_response_status(self) -> tuple[str, str]:
        """
        Return the internal terminal status for this response.

        This is intentionally closed-world for finish/stop metadata: a response
        with an unfamiliar or missing terminal reason is not treated as success.
        """
        if not isinstance(self.response, dict) or not self.response:
            return RESPONSE_STATUS_EMPTY_RESPONSE, "empty_response"

        moderation_reason = self.original_moderation_reason()
        if moderation_reason is not None:
            return RESPONSE_STATUS_MODERATION, moderation_reason

        provider_error_reason = self._provider_error_reason()
        if provider_error_reason is not None:
            return RESPONSE_STATUS_PROVIDER_ERROR, provider_error_reason

        choice = self._choice0()
        if "choices" in self.response and choice is None:
            return RESPONSE_STATUS_UNKNOWN_METADATA, "malformed_choices"

        reason_statuses: list[tuple[str, str]] = []
        primary_finish_reason = self._primary_finish_reason_with_source()
        native_finish_reason = self._native_finish_reason()

        if choice is not None and self._chat_finish_reason_is_missing():
            return RESPONSE_STATUS_METADATA_ERROR, "missing_finish_reason"

        if primary_finish_reason is not None:
            source, finish_reason = primary_finish_reason
            reason_statuses.append(self._finish_status_for_reason(source, finish_reason))

        if native_finish_reason is not None:
            reason_statuses.append(
                self._finish_status_for_reason("native_finish_reason", native_finish_reason)
            )

        if not reason_statuses:
            if "content" in self.response:
                return RESPONSE_STATUS_UNKNOWN_METADATA, "missing_stop_reason"
            return RESPONSE_STATUS_UNKNOWN_METADATA, "unrecognized_response_shape"

        unknown = [detail for status, detail in reason_statuses if status == RESPONSE_STATUS_UNKNOWN_METADATA]
        if unknown:
            return RESPONSE_STATUS_UNKNOWN_METADATA, ";".join(unknown)

        for target_status in (
            RESPONSE_STATUS_MODERATION,
            RESPONSE_STATUS_TRUNCATION,
            RESPONSE_STATUS_PROVIDER_ERROR,
            RESPONSE_STATUS_METADATA_ERROR,
        ):
            for status, detail in reason_statuses:
                if status == target_status:
                    return status, detail

        if self._has_missing_final_content():
            return RESPONSE_STATUS_MISSING_CONTENT, "missing_final_content"

        return RESPONSE_STATUS_SUCCESS, ";".join(detail for _status, detail in reason_statuses)

    def annotate_response_status(self) -> None:
        """Populate persisted internal response-status fields."""
        status, reason = self.classify_response_status()
        self.response_status = status
        self.response_status_reason = reason

    def ensure_known_response_status(self) -> None:
        """Raise if the row has terminal metadata that has not been classified."""
        self.annotate_response_status()
        if self.response_status == RESPONSE_STATUS_UNKNOWN_METADATA:
            raise UnknownResponseMetadataError(
                "unknown response metadata "
                f"for question_id={self.question_id!r} model={self.model!r}: "
                f"{self.response_status_reason}"
            )

    def original_moderation_reason(self) -> str | None:
        """Return the original-provider moderation/refusal stop reason, if any."""
        if self._provider_error_reason() is not None:
            return None

        finish_reason = self._finish_reason()
        native_finish_reason = self._native_finish_reason()

        if finish_reason in MODERATION_FINISH_REASONS:
            return finish_reason

        if native_finish_reason in MODERATION_FINISH_REASONS:
            return native_finish_reason

        error_info = self._llm_client_error_info()
        if error_info.get("type") == "content_filter":
            native_reason = error_info.get("native_finish_reason")
            if isinstance(native_reason, str):
                return native_reason
            finish_reason = error_info.get("finish_reason")
            if isinstance(finish_reason, str):
                return finish_reason
            return "content_filter"

        if self._has_empty_success_stop():
            return "empty_stop_content_suppressed"

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
        status, _reason = self.classify_response_status()
        return status in {
            RESPONSE_STATUS_PROVIDER_ERROR,
            RESPONSE_STATUS_EMPTY_RESPONSE,
            RESPONSE_STATUS_MISSING_CONTENT,
        }

    def is_truncation_error(self) -> bool:
        """True when generation hit a model/provider output limit."""
        status, _reason = self.classify_response_status()
        return status == RESPONSE_STATUS_TRUNCATION

    def is_permanent_error(self) -> bool:
        status, _reason = self.classify_response_status()
        return status != RESPONSE_STATUS_SUCCESS

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
    failed_primary_judge: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    
