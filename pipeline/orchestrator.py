"""Seven-stage research-paper QA orchestration with bounded degradation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from pipeline.client import NativeProviderClient
from pipeline.memory import MEMORY_MODES, ResearchMemory
from pipeline.stages import (
    CompletionClient,
    EvidenceRefinementResult,
    EvidenceRefiner,
    GenerateStage,
    GuardrailsStage,
    MemoryStage,
    NLUStage,
    ReflectStage,
    RerankStage,
    RetrievalAttempt,
    RetrieveStage,
)
from pipeline.trace import Timer, TurnTrace
from retrieval.config import RetrievalConfig
from retrieval.hybrid_v2 import HybridSearcher

if TYPE_CHECKING:
    from agent import AgentState


STAGE_NAMES = (
    "Guardrails",
    "NLU",
    "Retrieve",
    "Rerank",
    "Reflect",
    "Generate",
    "Memory",
)
T = TypeVar("T")


def _nlu_fallback(user_message: str, *, intent: str, reason: str) -> dict[str, Any]:
    return {
        "intent": intent,
        "standalone_query": user_message.strip() or "clarify request",
        "resolved_references": [],
        "needs_retrieval": False,
        "reason": reason,
        "constraints": [],
        "comparison_targets": [],
        "source": "safe_fallback",
    }


def _reflect_route(intent: str, reason: str) -> dict[str, Any]:
    action = {
        "meta": "answer",
        "clarification": "clarify",
        "out_of_scope": "refuse",
    }.get(intent, "refuse")
    return {
        "sufficient": action == "answer",
        "action": action,
        "missing_information": "" if action == "answer" else reason,
        "rewritten_query": "",
        "reason": reason,
        "source": "deterministic_route",
    }


def _refinement_failure(query: str, error: str) -> EvidenceRefinementResult:
    reflection = {
        "sufficient": False,
        "action": "refuse",
        "missing_information": "supporting evidence",
        "rewritten_query": "",
        "reason": error,
        "source": "safe_fallback",
        "failed_stage": "Orchestrator",
    }
    attempt = RetrievalAttempt(
        attempt=1,
        query=query,
        candidates=[],
        evidence=[],
        reflection=reflection,
        status="error",
        error=error,
    )
    return EvidenceRefinementResult(query, query, [], [], reflection, [attempt])


@dataclass
class PipelineResult:
    """User-visible result plus the evidence and trace that produced it."""

    answer: str
    citations: list[str]
    intent: str
    action: str
    evidence: list[dict[str, Any]]
    trace: TurnTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "intent": self.intent,
            "action": self.action,
            "evidence": [dict(item) for item in self.evidence],
            "trace": self.trace.to_dict(),
        }


class ResearchPipeline:
    """Run one research turn and commit a coherent update to ``AgentState``."""

    def __init__(
        self,
        config: Mapping[str, Any] | RetrievalConfig | None = None,
        *,
        client: CompletionClient | None = None,
        searcher: Any | None = None,
    ):
        self.runtime_config = dict(config or {}) if isinstance(config, Mapping) else {}
        self.config = (
            config
            if isinstance(config, RetrievalConfig)
            else RetrievalConfig.from_mapping(config)
        )
        self.client = client
        self.searcher = (
            searcher if searcher is not None else HybridSearcher(self.config)
        )
        retrieve = RetrieveStage(self.searcher)
        rerank = RerankStage(self.searcher)
        reflect = ReflectStage(client)
        self.guardrails = GuardrailsStage(client)
        self.nlu = NLUStage(client)
        self.refiner = EvidenceRefiner(
            retrieve,
            rerank,
            reflect,
            top_k=self.config.result_top_k,
            max_second_retrievals=self.config.max_second_retrievals,
        )
        self.generate = GenerateStage(client)
        self.memory_stage = MemoryStage()

    @classmethod
    def with_native_provider(
        cls,
        config: Mapping[str, Any],
        *,
        searcher: Any | None = None,
    ) -> ResearchPipeline:
        """Build the production pipeline against the inherited Provider registry."""
        return cls(config, client=NativeProviderClient(config), searcher=searcher)

    @staticmethod
    def _execute(
        trace: TurnTrace,
        name: str,
        stage_input: Any,
        operation: Callable[[], T],
        fallback: T,
        **metadata: Any,
    ) -> T:
        timer = Timer()
        try:
            with timer:
                output = operation()
        except Exception as exc:  # noqa: BLE001 - stage boundary must degrade safely
            error = f"{type(exc).__name__}: {exc}"
            trace.record(
                name,
                stage_input,
                fallback,
                timer.elapsed_ms,
                status="error",
                error=error,
                started_at=timer.started_at,
                ended_at=timer.ended_at,
                **metadata,
            )
            return fallback
        status = (
            "degraded"
            if isinstance(output, Mapping)
            and (
                output.get("source") == "safe_fallback"
                or output.get("fallback") is True
            )
            else "ok"
        )
        trace.record(
            name,
            stage_input,
            output,
            timer.elapsed_ms,
            status=status,
            started_at=timer.started_at,
            ended_at=timer.ended_at,
            **metadata,
        )
        return output

    @staticmethod
    def _mark_refinement_degradation(trace: TurnTrace, start: int) -> None:
        for stage in trace.stages[start:]:
            if (
                stage.status == "ok"
                and isinstance(stage.output, Mapping)
                and stage.output.get("source") == "safe_fallback"
            ):
                stage.status = "degraded"

    @staticmethod
    def _finish_failed_refinement_trace(
        trace: TurnTrace,
        start: int,
        result: EvidenceRefinementResult,
    ) -> None:
        if not result.attempts or result.attempts[-1].status != "error":
            return
        attempt = result.attempts[-1]
        present = {
            stage.stage
            for stage in trace.stages[start:]
            if stage.meta.get("attempt") == attempt.attempt
        }
        reason = attempt.error or "evidence refinement failed"
        for name in ("Retrieve", "Rerank", "Reflect"):
            if name not in present:
                trace.skip(name, reason).meta["attempt"] = attempt.attempt

    def _refine(
        self,
        trace: TurnTrace,
        query: str,
    ) -> EvidenceRefinementResult:
        start = len(trace.stages)
        try:
            result = self.refiner.run(query, trace=trace)
        except Exception as exc:  # noqa: BLE001 - preserve a complete turn on contract bugs
            error = f"Evidence refinement failed: {type(exc).__name__}: {exc}"
            result = _refinement_failure(query, error)
            if not any(stage.stage == "Retrieve" for stage in trace.stages[start:]):
                trace.record(
                    "Retrieve",
                    {"query": query, "filters": None},
                    None,
                    0.0,
                    status="error",
                    error=error,
                    attempt=1,
                )
        self._finish_failed_refinement_trace(trace, start, result)
        self._mark_refinement_degradation(trace, start)
        return result

    def run(self, user_message: str, state: AgentState) -> PipelineResult:
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string")
        trace = TurnTrace(len(state.turn_traces), user_message)
        history = list(state.messages)
        token_before = (
            int(getattr(self.client, "input_tokens", 0) or 0),
            int(getattr(self.client, "output_tokens", 0) or 0),
        )

        memory_error: str | None = None
        try:
            memory = ResearchMemory.from_dict(state.research_memory, state.memory_mode)
        except Exception as exc:  # noqa: BLE001 - corrupted persisted state is recoverable
            memory_error = (
                f"Invalid research memory was reset: {type(exc).__name__}: {exc}"
            )
            recovered_mode = (
                state.memory_mode if state.memory_mode in MEMORY_MODES else "compressed"
            )
            memory = ResearchMemory(mode=recovered_mode)
        memory_context = memory.context()

        guard_fallback = {
            "action": "clarify",
            "reason": "Guardrails failed before producing a safe decision",
            "source": "safe_fallback",
        }
        guard = self._execute(
            trace,
            "Guardrails",
            {"user_message": user_message, "memory": memory_context},
            lambda: self.guardrails.run(user_message, memory_context),
            guard_fallback,
        )

        evidence: list[dict[str, Any]] = []
        if guard["action"] != "allow":
            intent = "clarification" if guard["action"] == "clarify" else "out_of_scope"
            nlu = _nlu_fallback(user_message, intent=intent, reason=guard["reason"])
            for name in ("NLU", "Retrieve", "Rerank", "Reflect"):
                trace.skip(name, f"Guardrails routed request to {guard['action']}")
            reflection = _reflect_route(intent, guard["reason"])
            refinement = None
        else:
            nlu = self._execute(
                trace,
                "NLU",
                {
                    "user_message": user_message,
                    "history": history,
                    "memory": memory_context,
                },
                lambda: self.nlu.run(user_message, history, memory_context),
                _nlu_fallback(
                    user_message,
                    intent="clarification",
                    reason="NLU failed before producing a safe decision",
                ),
            )
            if nlu["needs_retrieval"]:
                refinement = self._refine(trace, nlu["standalone_query"])
                evidence = refinement.evidence
                reflection = refinement.reflection
            else:
                trace.skip("Retrieve", "NLU marked retrieval unnecessary")
                trace.skip("Rerank", "NLU marked retrieval unnecessary")
                reflection = _reflect_route(nlu["intent"], "NLU deterministic route")
                trace.record(
                    "Reflect",
                    nlu,
                    reflection,
                    0.0,
                    source="deterministic_route",
                )
                refinement = None

        generated_fallback = {
            "answer": "现有证据无法生成引用可验证的答案。",
            "citations": [],
            "citation_validation": {"valid": False, "errors": ["Generate failed"]},
            "source": "safe_fallback",
            "fallback": True,
        }
        generated: dict[str, Any] = self._execute(
            trace,
            "Generate",
            {
                "user_message": user_message,
                "nlu": nlu,
                "reflection": reflection,
                "evidence_ids": [item["chunk_id"] for item in evidence],
            },
            lambda: self.generate.run(user_message, nlu, reflection, evidence),
            generated_fallback,
        )

        memory_value = self._execute(
            trace,
            "Memory",
            {"mode": memory.mode, "citations": generated.get("citations", [])},
            lambda: self.memory_stage.run(
                memory,
                user_message=user_message,
                nlu=nlu,
                evidence=evidence,
                generated=generated,
                reflect=reflection,
                traceable_context=history,
            ),
            memory.to_dict(),
        )
        if memory_error is not None:
            memory_trace = trace.stages[-1]
            if memory_trace.status == "ok":
                memory_trace.status = "degraded"
            memory_trace.error = memory_error
            memory_trace.meta["recovered_from_invalid_state"] = True

        retrieval_state: dict[str, Any]
        if refinement is None:
            retrieval_state = {
                "needs_retrieval": False,
                "initial_query": nlu["standalone_query"],
                "final_query": nlu["standalone_query"],
                "candidates": [],
                "evidence": [],
                "attempts": [],
                "reflection": dict(reflection),
            }
        else:
            retrieval_state = {"needs_retrieval": True, **refinement.to_dict()}

        citations = list(generated.get("citations", []))
        state.semantic_state = dict(nlu)
        state.retrieval_state = retrieval_state
        state.citations = citations
        state.research_memory = memory_value
        state.memory_mode = memory.mode
        state.messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": generated["answer"]},
            ]
        )
        state.turn_count += 1
        state.total_input_tokens += max(
            0, int(getattr(self.client, "input_tokens", 0) or 0) - token_before[0]
        )
        state.total_output_tokens += max(
            0, int(getattr(self.client, "output_tokens", 0) or 0) - token_before[1]
        )
        final_status = (
            "degraded"
            if any(stage.status not in {"ok", "skipped"} for stage in trace.stages)
            else "ok"
        )
        trace.finish(final_status)
        state.turn_traces.append(trace)
        return PipelineResult(
            answer=generated["answer"],
            citations=citations,
            intent=nlu["intent"],
            action=reflection["action"],
            evidence=evidence,
            trace=trace,
        )
