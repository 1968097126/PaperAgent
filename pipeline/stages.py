"""Independently testable stages for evidence-grounded research-paper QA."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pipeline.memory import ResearchMemory
from pipeline.trace import Timer, TurnTrace

REFLECT_ACTIONS = frozenset({"answer", "retrieve_again", "clarify", "refuse"})
CITATION_PATTERN = re.compile(r"\[([^\[\]\/\s]+)/([^\[\]\/\s]+)\]")
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


class StageExecutionError(RuntimeError):
    """Expose retrieval-stage failures without leaking them as empty evidence."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{stage} execution failed: {type(cause).__name__}: {cause}")
        self.stage = stage


def _validated_records(value: Any, stage: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{stage} output must be a list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{stage} output item {index} must be an object")
        record = dict(item)
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"{stage} output item {index} requires a non-empty chunk_id"
            )
        chunk_id = chunk_id.strip()
        if chunk_id in seen:
            raise ValueError(f"{stage} output contains duplicate chunk_id {chunk_id!r}")
        seen.add(chunk_id)
        record["chunk_id"] = chunk_id
        records.append(record)
    return records


def _reflect_schema(value: dict[str, Any]) -> dict[str, Any]:
    action = value.get("action")
    if action not in REFLECT_ACTIONS:
        raise ValueError(f"action must be one of {sorted(REFLECT_ACTIONS)}")
    sufficient = value.get("sufficient")
    if not isinstance(sufficient, bool):
        raise TypeError("sufficient must be boolean")
    if sufficient != (action == "answer"):
        raise ValueError("sufficient must be true exactly when action is answer")
    missing = value.get("missing_information")
    rewritten_query = value.get("rewritten_query")
    if not isinstance(missing, str):
        raise TypeError("missing_information must be a string")
    if not isinstance(rewritten_query, str):
        raise TypeError("rewritten_query must be a string")
    missing = missing.strip()
    rewritten_query = rewritten_query.strip()
    if action != "answer" and not missing:
        raise ValueError(f"{action} requires missing_information")
    if action == "retrieve_again" and not rewritten_query:
        raise ValueError("retrieve_again requires rewritten_query")
    if action != "retrieve_again" and rewritten_query:
        raise ValueError("rewritten_query is only allowed for retrieve_again")
    return {
        "sufficient": sufficient,
        "action": action,
        "missing_information": missing,
        "rewritten_query": rewritten_query,
        "reason": _required_string(value.get("reason"), "reason"),
    }


def _reflect_refusal(
    reason: str, missing_information: str, **metadata: Any
) -> dict[str, Any]:
    return {
        "sufficient": False,
        "action": "refuse",
        "missing_information": missing_information,
        "rewritten_query": "",
        "reason": reason,
        "source": "safe_fallback",
        **metadata,
    }


@dataclass
class RetrieveStage:
    """Delegate deterministic candidate retrieval to the Phase 2 searcher."""

    searcher: Any

    def run(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = _required_string(query, "query")
        if filters is not None and not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping or None")
        normalized_filters = dict(filters) if filters is not None else None
        try:
            candidates = self.searcher.candidates(
                normalized_query,
                mode="hybrid",
                filters=normalized_filters,
            )
            return _validated_records(candidates, "Retrieve")
        except Exception as exc:
            raise StageExecutionError("Retrieve", exc) from exc


@dataclass
class RerankStage:
    """Delegate candidate ordering to the Phase 2 Cross-Encoder service."""

    searcher: Any

    def run(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        normalized_query = _required_string(query, "query")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        validated_candidates = _validated_records(candidates, "Rerank input")
        candidate_ids = {item["chunk_id"] for item in validated_candidates}
        try:
            ranked = self.searcher.rerank_candidates(
                normalized_query,
                validated_candidates,
                top_k,
            )
            evidence = _validated_records(ranked, "Rerank")
            unexpected = [
                item["chunk_id"]
                for item in evidence
                if item["chunk_id"] not in candidate_ids
            ]
            if unexpected:
                raise ValueError(f"Rerank returned unknown chunk_ids: {unexpected}")
            return evidence
        except Exception as exc:
            raise StageExecutionError("Rerank", exc) from exc


@dataclass
class ReflectStage:
    """Make a validated evidence decision under an explicit retrieval budget."""

    client: CompletionClient | None

    def run(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        attempt: int,
        max_second_retrievals: int,
    ) -> dict[str, Any]:
        normalized_query = _required_string(query, "query")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        if (
            not isinstance(max_second_retrievals, int)
            or isinstance(max_second_retrievals, bool)
            or max_second_retrievals < 0
        ):
            raise ValueError("max_second_retrievals must be a non-negative integer")
        validated_evidence = _validated_records(evidence, "Reflect input")
        if not validated_evidence:
            if attempt <= max_second_retrievals:
                return {
                    "sufficient": False,
                    "action": "retrieve_again",
                    "missing_information": "no evidence was retrieved",
                    "rewritten_query": normalized_query,
                    "reason": "retry the deterministic retrieval once within budget",
                    "source": "rule",
                }
            return _reflect_refusal(
                "retrieval budget exhausted without evidence",
                "supporting evidence",
                source="rule",
            )
        if self.client is None:
            return {
                "sufficient": True,
                "action": "answer",
                "missing_information": "",
                "rewritten_query": "",
                "reason": "evidence is present; semantic reflection is unavailable",
                "source": "fallback",
            }
        compact_evidence = [
            {
                "chunk_id": item["chunk_id"],
                "paper_id": item.get("paper_id", ""),
                "title": item.get("title", ""),
                "section": item.get("section", ""),
                "text": str(item.get("text", ""))[:1200],
            }
            for item in validated_evidence
        ]
        prompt = (
            "Judge whether the supplied evidence is sufficient to answer the query using no outside knowledge. "
            "Return ONLY JSON with sufficient:boolean, missing_information:string, "
            "action:answer|retrieve_again|clarify|refuse, rewritten_query:string, and a non-empty reason. "
            "sufficient must be true exactly for answer. A non-answer requires missing_information. "
            "Only retrieve_again may provide rewritten_query, which must materially differ from the current query.\n"
            f"Retrieval attempt: {attempt} of {max_second_retrievals + 1}\n"
            f"Query: {normalized_query}\n"
            f"Evidence: {json.dumps(compact_evidence, ensure_ascii=False, sort_keys=True)}"
        )

        def validate(value: dict[str, Any]) -> dict[str, Any]:
            decision = _reflect_schema(value)
            if decision["action"] == "retrieve_again" and _normalize_for_traceability(
                decision["rewritten_query"]
            ) == _normalize_for_traceability(normalized_query):
                raise ValueError("retrieve_again requires a materially rewritten query")
            return decision

        try:
            decision, attempts = structured_completion(
                self.client, "Reflect", prompt, validate
            )
        except StageParseError as exc:
            return _reflect_refusal(
                f"Reflect parsing failed: {exc}",
                "a validated evidence-sufficiency decision",
                parse_attempts=2,
            )
        except StageCompletionError as exc:
            return _reflect_refusal(
                str(exc),
                "a validated evidence-sufficiency decision",
                completion_attempts=exc.attempt,
            )
        if decision["action"] == "retrieve_again" and attempt > max_second_retrievals:
            return _reflect_refusal(
                "retrieval budget exhausted after the final evidence review",
                decision["missing_information"],
                source="policy",
                parse_attempts=len(attempts),
            )
        return {**decision, "source": "model", "parse_attempts": len(attempts)}


@dataclass
class RetrievalAttempt:
    """One query-to-candidate-to-evidence lineage record."""

    attempt: int
    query: str
    candidates: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    reflection: dict[str, Any]
    status: str = "ok"
    error: str | None = None

    @property
    def candidate_ids(self) -> list[str]:
        return [item["chunk_id"] for item in self.candidates]

    @property
    def evidence_ids(self) -> list[str]:
        return [item["chunk_id"] for item in self.evidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "query": self.query,
            "candidate_ids": self.candidate_ids,
            "evidence_ids": self.evidence_ids,
            "candidates": [dict(item) for item in self.candidates],
            "evidence": [dict(item) for item in self.evidence],
            "reflection": dict(self.reflection),
            "status": self.status,
            "error": self.error,
        }


@dataclass
class EvidenceRefinementResult:
    """Final evidence plus all bounded retrieval attempts that produced it."""

    initial_query: str
    final_query: str
    candidates: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    reflection: dict[str, Any]
    attempts: list[RetrievalAttempt]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_query": self.initial_query,
            "final_query": self.final_query,
            "candidates": [dict(item) for item in self.candidates],
            "evidence": [dict(item) for item in self.evidence],
            "reflection": dict(self.reflection),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def _record_refinement_trace(
    trace: TurnTrace | None,
    stage: str,
    stage_input: Any,
    stage_output: Any,
    timer: Timer,
    attempt: int,
    *,
    status: str = "ok",
    error: str | None = None,
    **metadata: Any,
) -> None:
    if trace is None:
        return
    trace.record(
        stage,
        stage_input,
        stage_output,
        timer.elapsed_ms,
        attempt=attempt,
        status=status,
        error=error,
        started_at=timer.started_at,
        ended_at=timer.ended_at,
        **metadata,
    )


@dataclass
class EvidenceRefiner:
    """Run bounded evidence refinement and optionally write StageTrace lineage."""

    retrieve: RetrieveStage
    rerank: RerankStage
    reflect: ReflectStage
    top_k: int = 5
    max_second_retrievals: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")
        if (
            not isinstance(self.max_second_retrievals, int)
            or isinstance(self.max_second_retrievals, bool)
            or self.max_second_retrievals < 0
        ):
            raise ValueError("max_second_retrievals must be a non-negative integer")

    def run(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        *,
        trace: TurnTrace | None = None,
    ) -> EvidenceRefinementResult:
        initial_query = _required_string(query, "query")
        current_query = initial_query
        attempts: list[RetrievalAttempt] = []
        candidates: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        reflection = _reflect_refusal(
            "evidence refinement did not run", "supporting evidence"
        )

        for attempt_number in range(1, self.max_second_retrievals + 2):
            candidates = []
            evidence = []
            retrieve_input = {
                "query": current_query,
                "filters": dict(filters) if filters is not None else None,
            }
            try:
                with Timer() as retrieve_timer:
                    candidates = self.retrieve.run(current_query, filters)
            except StageExecutionError as exc:
                _record_refinement_trace(
                    trace,
                    "Retrieve",
                    retrieve_input,
                    None,
                    retrieve_timer,
                    attempt_number,
                    status="error",
                    error=str(exc),
                    candidate_ids=[],
                )
                reflection = _reflect_refusal(
                    str(exc), "supporting evidence", failed_stage=exc.stage
                )
                attempts.append(
                    RetrievalAttempt(
                        attempt=attempt_number,
                        query=current_query,
                        candidates=[],
                        evidence=[],
                        reflection=reflection,
                        status="error",
                        error=str(exc),
                    )
                )
                break
            _record_refinement_trace(
                trace,
                "Retrieve",
                retrieve_input,
                candidates,
                retrieve_timer,
                attempt_number,
                candidate_ids=[item["chunk_id"] for item in candidates],
            )

            rerank_input = {
                "query": current_query,
                "candidate_ids": [item["chunk_id"] for item in candidates],
                "top_k": self.top_k,
            }
            try:
                with Timer() as rerank_timer:
                    evidence = self.rerank.run(current_query, candidates, self.top_k)
            except StageExecutionError as exc:
                _record_refinement_trace(
                    trace,
                    "Rerank",
                    rerank_input,
                    None,
                    rerank_timer,
                    attempt_number,
                    status="error",
                    error=str(exc),
                    evidence_ids=[],
                )
                reflection = _reflect_refusal(
                    str(exc), "supporting evidence", failed_stage=exc.stage
                )
                attempts.append(
                    RetrievalAttempt(
                        attempt=attempt_number,
                        query=current_query,
                        candidates=candidates,
                        evidence=[],
                        reflection=reflection,
                        status="error",
                        error=str(exc),
                    )
                )
                break
            _record_refinement_trace(
                trace,
                "Rerank",
                rerank_input,
                evidence,
                rerank_timer,
                attempt_number,
                evidence_ids=[item["chunk_id"] for item in evidence],
            )

            reflect_input = {
                "query": current_query,
                "evidence_ids": [item["chunk_id"] for item in evidence],
                "attempt": attempt_number,
                "max_second_retrievals": self.max_second_retrievals,
            }
            try:
                with Timer() as reflect_timer:
                    reflection = self.reflect.run(
                        current_query,
                        evidence,
                        attempt_number,
                        self.max_second_retrievals,
                    )
            except (KeyError, TypeError, ValueError) as cause:
                error = StageExecutionError("Reflect", cause)
                _record_refinement_trace(
                    trace,
                    "Reflect",
                    reflect_input,
                    None,
                    reflect_timer,
                    attempt_number,
                    status="error",
                    error=str(error),
                )
                reflection = _reflect_refusal(
                    str(error),
                    "a validated evidence-sufficiency decision",
                    failed_stage="Reflect",
                )
                attempts.append(
                    RetrievalAttempt(
                        attempt=attempt_number,
                        query=current_query,
                        candidates=candidates,
                        evidence=evidence,
                        reflection=reflection,
                        status="error",
                        error=str(error),
                    )
                )
                break
            _record_refinement_trace(
                trace,
                "Reflect",
                reflect_input,
                reflection,
                reflect_timer,
                attempt_number,
                action=reflection["action"],
                source=reflection.get("source", "unknown"),
            )
            attempts.append(
                RetrievalAttempt(
                    attempt=attempt_number,
                    query=current_query,
                    candidates=candidates,
                    evidence=evidence,
                    reflection=reflection,
                )
            )
            if reflection["action"] != "retrieve_again":
                break
            current_query = reflection["rewritten_query"]

        return EvidenceRefinementResult(
            initial_query=initial_query,
            final_query=current_query,
            candidates=candidates,
            evidence=evidence,
            reflection=reflection,
            attempts=attempts,
        )


def validate_citations(
    answer: Any,
    evidence: Sequence[Mapping[str, Any]],
    *,
    require_citation: bool = True,
) -> dict[str, Any]:
    """Validate stable ``[paper_id/chunk_id]`` tokens against retrieved evidence."""
    evidence_pairs: set[tuple[str, str]] = set()
    evidence_errors: list[str] = []
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        evidence_errors.append("evidence must be a sequence")
        evidence = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            evidence_errors.append(f"evidence item {index} must be an object")
            continue
        paper_id = item.get("paper_id")
        chunk_id = item.get("chunk_id")
        if not isinstance(paper_id, str) or not paper_id.strip():
            evidence_errors.append(f"evidence item {index} requires paper_id")
            continue
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            evidence_errors.append(f"evidence item {index} requires chunk_id")
            continue
        pair = (paper_id.strip(), chunk_id.strip())
        if pair in evidence_pairs:
            evidence_errors.append(f"duplicate evidence identity: {pair}")
            continue
        evidence_pairs.add(pair)

    errors = list(evidence_errors)
    text = answer if isinstance(answer, str) else ""
    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer must be a non-empty string")
    matches = CITATION_PATTERN.findall(text)
    duplicates = sorted({pair for pair in matches if matches.count(pair) > 1})
    invalid = sorted(pair for pair in set(matches) if pair not in evidence_pairs)
    if require_citation and not evidence_pairs:
        errors.append("no evidence is available for a cited answer")
    if require_citation and not matches:
        errors.append("answer has no evidence citation")
    if invalid:
        errors.append(f"unknown citations: {invalid}")
    if duplicates:
        errors.append(f"duplicate citations: {duplicates}")
    return {
        "valid": not errors,
        "citations": [chunk_id for _, chunk_id in matches],
        "citation_pairs": matches,
        "valid_citations": [
            chunk_id
            for paper_id, chunk_id in matches
            if (paper_id, chunk_id) in evidence_pairs
        ],
        "invalid": invalid,
        "duplicates": duplicates,
        "evidence_errors": evidence_errors,
        "errors": errors,
    }


def _non_evidence_response(answer: str, route: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": [],
        "citation_validation": validate_citations(
            answer,
            [],
            require_citation=False,
        ),
        "source": "rule",
        "route": route,
        "generation_attempts": 0,
    }


def _generation_failure(
    reason: str,
    *,
    attempts: int,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failed_validation = dict(validation or {})
    if not failed_validation:
        failed_validation = {
            "valid": False,
            "citations": [],
            "citation_pairs": [],
            "valid_citations": [],
            "invalid": [],
            "duplicates": [],
            "evidence_errors": [],
            "errors": [reason],
        }
    return {
        "answer": "现有证据无法生成引用可验证的答案。",
        "citations": [],
        "citation_validation": failed_validation,
        "source": "safe_fallback",
        "fallback": True,
        "generation_attempts": attempts,
        "generation_error": reason,
    }


def _complete_generation(
    client: CompletionClient,
    prompt: str,
    attempt: int,
) -> str:
    try:
        return client.complete("Generate", prompt)
    except Exception as cause:
        raise StageCompletionError("Generate", attempt, cause) from cause


def _generation_evidence(
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            errors.append(f"evidence item {index} must be an object")
            continue
        required: dict[str, str] = {}
        for field in ("paper_id", "chunk_id", "text"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"evidence item {index} requires {field}")
                break
            required[field] = value.strip()
        else:
            pair = (required["paper_id"], required["chunk_id"])
            if pair in seen:
                errors.append(f"duplicate evidence identity: {pair}")
                continue
            seen.add(pair)
            records.append(
                {
                    **dict(item),
                    **required,
                    "title": str(item.get("title", "")).strip(),
                    "section": str(item.get("section", "")).strip(),
                }
            )
    return records, errors


@dataclass
class GenerateStage:
    """Generate only from final evidence and enforce citation integrity."""

    client: CompletionClient | None
    max_evidence_chars: int = 2400

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_evidence_chars, int)
            or isinstance(self.max_evidence_chars, bool)
            or self.max_evidence_chars <= 0
        ):
            raise ValueError("max_evidence_chars must be a positive integer")

    def run(
        self,
        user_message: str,
        nlu: Mapping[str, Any],
        reflect: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        intent = nlu.get("intent")
        action = reflect.get("action")
        if intent == "meta":
            return _non_evidence_response(
                "我是一个仅检索本地科研论文语料、展示证据引用和执行轨迹的问答 Agent。",
                "meta",
            )
        if action == "clarify" or intent == "clarification":
            return _non_evidence_response(
                "请补充要检索的论文、方法或比较条件。",
                "clarify",
            )
        if action != "answer" or intent == "out_of_scope":
            return _non_evidence_response(
                "现有论文证据不足，无法仅依据语料库可靠回答。",
                "refuse",
            )
        if reflect.get("sufficient") is not True:
            return _generation_failure(
                "Reflect did not mark the evidence sufficient",
                attempts=0,
            )
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            return _generation_failure("evidence must be a sequence", attempts=0)
        validated_evidence, evidence_errors = _generation_evidence(evidence)
        if evidence_errors or not validated_evidence:
            reason = "; ".join(evidence_errors) or "no evidence was provided"
            return _generation_failure(reason, attempts=0)

        if self.client is None:
            item = validated_evidence[0]
            prefix = (
                "根据检索证据："
                if re.search(r"[\u3400-\u9fff]", user_message or "")
                else "Based on the retrieved evidence: "
            )
            answer = (
                f"{prefix}{item['text'][:300]} [{item['paper_id']}/{item['chunk_id']}]"
            )
            validation = validate_citations(answer, validated_evidence)
            return {
                "answer": answer,
                "citations": validation["valid_citations"],
                "citation_validation": validation,
                "source": "fallback",
                "generation_attempts": 0,
            }

        query = nlu.get("standalone_query")
        if not isinstance(query, str) or not query.strip():
            return _generation_failure(
                "NLU did not provide a standalone query", attempts=0
            )
        labelled = [
            {
                "citation": f"[{item['paper_id']}/{item['chunk_id']}]",
                "title": item["title"],
                "section": item["section"],
                "text": item["text"][: self.max_evidence_chars],
            }
            for item in validated_evidence
        ]
        language = (
            "Chinese by default, unless the user explicitly requested another language"
            if re.search(r"[\u3400-\u9fff]", user_message or "")
            else "the language used by the user"
        )
        prompt = (
            f"Answer in {language}. Use ONLY the supplied evidence; do not add model knowledge. "
            "Every factual claim must carry an exact citation token shown below. Do not invent or repeat "
            "citation tokens. Return only the answer text.\n"
            f"Question: {query.strip()}\n"
            f"Evidence: {json.dumps(labelled, ensure_ascii=False, sort_keys=True)}"
        )
        try:
            answer = _complete_generation(self.client, prompt, 1)
        except StageCompletionError as error:
            return _generation_failure(str(error), attempts=1)
        validation = validate_citations(answer, validated_evidence)
        if validation["valid"]:
            return {
                "answer": answer.strip(),
                "citations": validation["valid_citations"],
                "citation_validation": validation,
                "source": "model",
                "generation_attempts": 1,
            }

        repair_prompt = (
            "Repair the answer using ONLY the supplied evidence and exact citation tokens. "
            "Remove unsupported claims, unknown citations, and duplicate citation occurrences. "
            "Return only the corrected answer text.\n"
            f"Validation errors: {json.dumps(validation['errors'], ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(labelled, ensure_ascii=False, sort_keys=True)}\n"
            f"Invalid answer: {answer if isinstance(answer, str) else repr(answer)}"
        )
        try:
            repaired = _complete_generation(self.client, repair_prompt, 2)
        except StageCompletionError as error:
            return _generation_failure(
                str(error),
                attempts=2,
                validation=validation,
            )
        repaired_validation = validate_citations(repaired, validated_evidence)
        if not repaired_validation["valid"]:
            return _generation_failure(
                "citation validation failed after one repair",
                attempts=2,
                validation=repaired_validation,
            )
        return {
            "answer": repaired.strip(),
            "citations": repaired_validation["valid_citations"],
            "citation_validation": repaired_validation,
            "source": "model_repair",
            "generation_attempts": 2,
        }


@dataclass
class MemoryStage:
    """Apply an evidence-only domain-memory update after generation."""

    def run(
        self,
        memory: ResearchMemory,
        *,
        user_message: str,
        nlu: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        generated: Mapping[str, Any],
        reflect: Mapping[str, Any],
        traceable_context: Sequence[Any] = (),
    ) -> dict[str, Any]:
        if not isinstance(memory, ResearchMemory):
            raise TypeError("memory must be a ResearchMemory instance")
        citations = generated.get("citations", [])
        if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
            raise TypeError("generated citations must be a sequence")
        memory.update(
            user_message=user_message,
            nlu=nlu,
            evidence=evidence,
            citations=citations,
            unresolved=reflect.get("action") in {"clarify", "refuse"},
            traceable_context=traceable_context,
        )
        return memory.to_dict()
