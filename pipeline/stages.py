"""Guardrails and NLU stages for evidence-grounded research-paper QA."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

GUARDRAIL_ACTIONS = frozenset({"allow", "clarify", "refuse"})
INTENTS = frozenset({"new_search", "follow_up", "clarification", "meta", "out_of_scope"})
RETRIEVAL_INTENTS = frozenset({"new_search", "follow_up"})
INJECTION_SIGNALS = (
    "ignore previous instructions",
    "reveal system prompt",
    "execute shell",
    "忽略之前的指令",
    "泄露系统提示",
    "调用 bash",
)


class CompletionClient(Protocol):
    """Minimal completion interface implemented by Provider adapters and mocks."""

    def complete(self, stage: str, prompt: str) -> str: ...


class StageParseError(ValueError):
    """Raised when structured model output remains invalid after one repair."""


class StageCompletionError(RuntimeError):
    """Raised when the underlying Provider fails before producing stage output."""

    def __init__(self, stage: str, attempt: int, cause: Exception):
        super().__init__(f"{stage} completion failed on attempt {attempt}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.attempt = attempt


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain, fenced, or lightly wrapped model output."""
    if not isinstance(text, str):
        raise StageParseError("model output must be a string")
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise StageParseError(str(exc)) from exc
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as nested:
            raise StageParseError(str(nested)) from nested
    if not isinstance(value, dict):
        raise StageParseError("model output must be a JSON object")
    return value


def structured_completion(
    client: CompletionClient,
    stage: str,
    prompt: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Complete and validate once, allowing exactly one repair request."""
    attempts: list[str] = []
    request = prompt
    last_error = "unknown validation error"
    for attempt in range(1, 3):
        try:
            raw = client.complete(stage, request)
        except Exception as exc:
            raise StageCompletionError(stage, attempt, exc) from exc
        attempts.append(raw)
        try:
            return validator(parse_json_object(raw)), attempts
        except (StageParseError, ValueError, TypeError, KeyError) as exc:
            last_error = str(exc)
            if attempt == 1:
                request = (
                    f"Your previous output was invalid: {exc}. Return ONLY a corrected JSON object.\n"
                    f"Original task:\n{prompt}\nInvalid output:\n{raw}"
                )
    raise StageParseError(f"structured output remained invalid after one repair: {last_error}")


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise TypeError(f"{field} must be a list of non-empty strings")
    return list(dict.fromkeys(item.strip() for item in value))


def _guardrail_schema(value: dict[str, Any]) -> dict[str, Any]:
    action = value.get("action")
    if action not in GUARDRAIL_ACTIONS:
        raise ValueError("action must be allow, clarify, or refuse")
    return {"action": action, "reason": _required_string(value.get("reason"), "reason")}


def _nlu_schema(value: dict[str, Any]) -> dict[str, Any]:
    intent = value.get("intent")
    if intent not in INTENTS:
        raise ValueError(f"intent must be one of {sorted(INTENTS)}")
    query = value.get("standalone_query", value.get("rewritten_query"))
    standalone_query = _required_string(query, "standalone_query")
    needs_retrieval = value.get("needs_retrieval")
    if not isinstance(needs_retrieval, bool):
        raise TypeError("needs_retrieval must be boolean")
    expected_retrieval = intent in RETRIEVAL_INTENTS
    if needs_retrieval != expected_retrieval:
        raise ValueError(f"intent {intent!r} requires needs_retrieval={expected_retrieval}")
    return {
        "intent": intent,
        "standalone_query": standalone_query,
        "resolved_references": _string_list(value.get("resolved_references", []), "resolved_references"),
        "needs_retrieval": needs_retrieval,
        "reason": _required_string(value.get("reason"), "reason"),
        "constraints": _string_list(value.get("constraints", []), "constraints"),
        "comparison_targets": _string_list(value.get("comparison_targets", []), "comparison_targets"),
    }


def _normalize_for_traceability(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def _traceable_context(
    user_message: str,
    history: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
) -> str:
    source = json.dumps(
        {"user_message": user_message, "history": list(history), "memory": dict(memory)},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return _normalize_for_traceability(source)


def _validate_nlu_traceability(value: dict[str, Any], source_context: str) -> dict[str, Any]:
    declared = {
        "resolved_references": value["resolved_references"],
        "constraints": value["constraints"],
        "comparison_targets": value["comparison_targets"],
    }
    query_context = _normalize_for_traceability(value["standalone_query"])
    for field, items in declared.items():
        for item in items:
            normalized = _normalize_for_traceability(item)
            if not normalized or normalized not in source_context:
                raise ValueError(f"{field} contains untraceable value: {item!r}")
            if normalized not in query_context:
                raise ValueError(f"standalone_query does not preserve {field} value: {item!r}")
    return value


@dataclass
class GuardrailsStage:
    """Apply deterministic safety rules, then optional semantic scope routing."""

    client: CompletionClient | None
    max_input_chars: int = 8000

    def __post_init__(self) -> None:
        if self.max_input_chars <= 0:
            raise ValueError("max_input_chars must be positive")

    def run(self, user_message: str, memory: Mapping[str, Any]) -> dict[str, Any]:
        message = (user_message or "").strip()
        if not message:
            return {"action": "clarify", "reason": "empty input", "source": "rule"}
        if len(message) > self.max_input_chars:
            return {
                "action": "refuse",
                "reason": "input exceeds the configured length limit",
                "source": "rule",
            }
        folded = message.casefold()
        if any(signal in folded for signal in INJECTION_SIGNALS):
            return {
                "action": "refuse",
                "reason": "prompt-injection or tool-escalation request",
                "source": "rule",
            }
        if self.client is None:
            return {
                "action": "allow",
                "reason": "semantic classifier unavailable; defer to NLU",
                "source": "fallback",
            }
        prompt = (
            "Classify whether this request belongs to research-paper retrieval/QA. "
            "Use allow for paper questions and questions about this agent, clarify for genuinely ambiguous "
            "requests, and refuse for clearly out-of-domain requests. Return ONLY JSON with non-empty "
            "action and reason fields. Guardrails handles obvious malformed, injection, and coarse domain "
            "boundaries; NLU retains out_of_scope as a semantic fallback.\n"
            f"Verified memory: {json.dumps(dict(memory), ensure_ascii=False, sort_keys=True, default=str)}\n"
            f"Request: {message}"
        )
        try:
            value, attempts = structured_completion(self.client, "Guardrails", prompt, _guardrail_schema)
            return {**value, "source": "model", "parse_attempts": len(attempts)}
        except StageParseError as exc:
            return {
                "action": "clarify",
                "reason": f"guardrail parsing failed: {exc}",
                "source": "safe_fallback",
                "parse_attempts": 2,
            }
        except StageCompletionError as exc:
            return {
                "action": "clarify",
                "reason": str(exc),
                "source": "safe_fallback",
                "completion_attempts": exc.attempt,
            }


@dataclass
class NLUStage:
    """Classify intent and produce a traceable standalone retrieval query."""

    client: CompletionClient | None
    max_history_messages: int = 6

    def __post_init__(self) -> None:
        if self.max_history_messages < 0:
            raise ValueError("max_history_messages cannot be negative")

    def _fallback(self, message: str, reason: str, **metadata: Any) -> dict[str, Any]:
        return {
            "intent": "clarification",
            "standalone_query": message or "clarify request",
            "resolved_references": [],
            "needs_retrieval": False,
            "reason": reason,
            "constraints": [],
            "comparison_targets": [],
            "source": "safe_fallback",
            **metadata,
        }

    def run(
        self,
        user_message: str,
        history: Sequence[Mapping[str, Any]],
        memory: Mapping[str, Any],
    ) -> dict[str, Any]:
        message = (user_message or "").strip()
        if not message:
            return self._fallback("", "NLU received empty input", source="rule")
        if self.client is None:
            return {
                "intent": "new_search",
                "standalone_query": message,
                "resolved_references": [],
                "needs_retrieval": True,
                "reason": "NLU model unavailable; use literal query",
                "constraints": [],
                "comparison_targets": [],
                "source": "fallback",
            }
        recent_history = list(history[-self.max_history_messages :]) if self.max_history_messages else []
        source_context = _traceable_context(message, recent_history, memory)
        prompt = (
            "Classify exactly one intent: new_search, follow_up, clarification, meta, or out_of_scope. "
            "Resolve references using ONLY the current user message, supplied history, and verified memory. "
            "Return ONLY JSON with intent, standalone_query, resolved_references, needs_retrieval, reason, "
            "constraints, and comparison_targets. new_search/follow_up require retrieval; all other intents do not. "
            "Use exact source phrases for every declared reference, constraint, and comparison target, and preserve "
            "each phrase in standalone_query. Guardrails handles obvious malformed/injection/coarse-domain requests; "
            "out_of_scope here is the semantic fallback.\n"
            f"History: {json.dumps(recent_history, ensure_ascii=False, sort_keys=True, default=str)}\n"
            f"Verified memory: {json.dumps(dict(memory), ensure_ascii=False, sort_keys=True, default=str)}\n"
            f"User: {message}"
        )

        def validate(value: dict[str, Any]) -> dict[str, Any]:
            return _validate_nlu_traceability(_nlu_schema(value), source_context)

        try:
            value, attempts = structured_completion(self.client, "NLU", prompt, validate)
            return {**value, "source": "model", "parse_attempts": len(attempts)}
        except StageParseError as exc:
            return self._fallback(
                message,
                f"NLU parsing failed: {exc}",
                parse_attempts=2,
            )
        except StageCompletionError as exc:
            return self._fallback(
                message,
                str(exc),
                completion_attempts=exc.attempt,
            )
