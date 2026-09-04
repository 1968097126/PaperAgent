from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline.stages import (
    EvidenceRefiner,
    ReflectStage,
    RerankStage,
    RetrieveStage,
    StageExecutionError,
)
from pipeline.trace import TurnTrace


def record(chunk_id: str, text: str = "evidence") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": f"paper-{chunk_id}",
        "title": f"Paper {chunk_id}",
        "section": "Results",
        "text": text,
    }


def decision(
    action: str,
    *,
    sufficient: bool,
    rewritten_query: str = "",
    missing_information: str = "",
) -> str:
    return json.dumps(
        {
            "sufficient": sufficient,
            "action": action,
            "missing_information": missing_information,
            "rewritten_query": rewritten_query,
            "reason": "evidence decision",
        }
    )


class ScriptedClient:
    def __init__(self, *responses: str | Exception):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, stage: str, prompt: str) -> str:
        self.calls.append((stage, prompt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeSearcher:
    def __init__(
        self,
        batches: list[list[dict[str, Any]]] | None = None,
        *,
        fail_retrieve: Exception | None = None,
        fail_retrieve_on_call: int | None = None,
        fail_rerank: Exception | None = None,
        invented_evidence: list[dict[str, Any]] | None = None,
    ):
        self.batches = batches if batches is not None else [[record("c1")]]
        self.fail_retrieve = fail_retrieve
        self.fail_retrieve_on_call = fail_retrieve_on_call
        self.fail_rerank = fail_rerank
        self.invented_evidence = invented_evidence
        self.candidate_calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.rerank_calls: list[tuple[str, list[str], int]] = []

    def candidates(
        self,
        query: str,
        *,
        mode: str,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        self.candidate_calls.append((query, mode, filters))
        if self.fail_retrieve is not None and (
            self.fail_retrieve_on_call is None
            or len(self.candidate_calls) == self.fail_retrieve_on_call
        ):
            raise self.fail_retrieve
        index = min(len(self.candidate_calls) - 1, len(self.batches) - 1)
        return self.batches[index]

    def rerank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        self.rerank_calls.append(
            (query, [item["chunk_id"] for item in candidates], top_k)
        )
        if self.fail_rerank is not None:
            raise self.fail_rerank
        if self.invented_evidence is not None:
            return self.invented_evidence
        return list(reversed(candidates))[:top_k]


def refiner(
    searcher: FakeSearcher,
    client: ScriptedClient | None = None,
    *,
    max_second_retrievals: int = 1,
) -> EvidenceRefiner:
    return EvidenceRefiner(
        RetrieveStage(searcher),
        RerankStage(searcher),
        ReflectStage(client),
        top_k=2,
        max_second_retrievals=max_second_retrievals,
    )


class TestRetrieveAndRerankStages:
    def test_delegates_to_phase2_components_without_reimplementing_algorithms(self):
        searcher = FakeSearcher([[record("c1"), record("c2")]])

        candidates = RetrieveStage(searcher).run("  query  ", {"sections": ["Results"]})
        evidence = RerankStage(searcher).run("query", candidates, 1)

        assert searcher.candidate_calls == [
            ("query", "hybrid", {"sections": ["Results"]})
        ]
        assert searcher.rerank_calls == [("query", ["c1", "c2"], 1)]
        assert [item["chunk_id"] for item in evidence] == ["c2"]

    @pytest.mark.parametrize(
        ("stage", "expected"),
        [
            (
                RetrieveStage(FakeSearcher(fail_retrieve=OSError("index unavailable"))),
                "Retrieve",
            ),
            (
                RerankStage(FakeSearcher(fail_rerank=RuntimeError("model failed"))),
                "Rerank",
            ),
        ],
    )
    def test_backend_exceptions_are_explicit(self, stage, expected):
        with pytest.raises(
            StageExecutionError, match=f"{expected} execution failed"
        ) as raised:
            if expected == "Retrieve":
                stage.run("query")
            else:
                stage.run("query", [record("c1")], 1)
        assert raised.value.stage == expected

    def test_rejects_invalid_inputs_duplicate_ids_and_invented_evidence(self):
        with pytest.raises(ValueError, match="query"):
            RetrieveStage(FakeSearcher()).run("  ")
        with pytest.raises(StageExecutionError, match="duplicate chunk_id"):
            RetrieveStage(FakeSearcher([[record("c1"), record("c1")]])).run("query")
        with pytest.raises(ValueError, match="positive integer"):
            RerankStage(FakeSearcher()).run("query", [record("c1")], 0)
        stage = RerankStage(FakeSearcher(invented_evidence=[record("invented")]))
        with pytest.raises(StageExecutionError, match="unknown chunk_ids"):
            stage.run("query", [record("c1")], 1)


class TestReflectStage:
    def test_validates_schema_and_allows_exactly_one_repair(self):
        inconsistent = decision("answer", sufficient=False)
        repaired = decision("answer", sufficient=True)
        client = ScriptedClient(inconsistent, repaired)

        result = ReflectStage(client).run("query", [record("c1")], 1, 1)

        assert result["action"] == "answer"
        assert result["parse_attempts"] == 2
        assert "previous output was invalid" in client.calls[1][1]

    def test_retry_query_must_be_nonempty_and_materially_rewritten(self):
        same_query = decision(
            "retrieve_again",
            sufficient=False,
            rewritten_query="query",
            missing_information="method details",
        )
        repaired = decision(
            "retrieve_again",
            sufficient=False,
            rewritten_query="query method details",
            missing_information="method details",
        )

        result = ReflectStage(ScriptedClient(same_query, repaired)).run(
            "query", [record("c1")], 1, 1
        )

        assert result["action"] == "retrieve_again"
        assert result["rewritten_query"] == "query method details"
        assert result["parse_attempts"] == 2

    def test_invalid_json_and_provider_failure_have_safe_refusals(self):
        invalid = ReflectStage(ScriptedClient("bad", "still bad")).run(
            "query", [record("c1")], 1, 1
        )
        failed = ReflectStage(ScriptedClient(RuntimeError("offline"))).run(
            "query", [record("c1")], 1, 1
        )

        assert invalid["action"] == "refuse"
        assert invalid["source"] == "safe_fallback"
        assert invalid["parse_attempts"] == 2
        assert failed["action"] == "refuse"
        assert failed["completion_attempts"] == 1
        assert "RuntimeError: offline" in failed["reason"]

    def test_empty_evidence_retries_only_while_budget_remains(self):
        first = ReflectStage(None).run("query", [], 1, 1)
        second = ReflectStage(None).run("query", [], 2, 1)

        assert first["action"] == "retrieve_again"
        assert first["source"] == "rule"
        assert second["action"] == "refuse"
        assert second["source"] == "rule"

    def test_model_cannot_exceed_retrieval_budget(self):
        retry = decision(
            "retrieve_again",
            sufficient=False,
            rewritten_query="query details",
            missing_information="details",
        )

        result = ReflectStage(ScriptedClient(retry)).run("query", [record("c1")], 1, 0)

        assert result["action"] == "refuse"
        assert result["source"] == "policy"
        assert "budget exhausted" in result["reason"]


class TestEvidenceRefiner:
    def test_first_retrieval_success_stops_after_one_attempt(self):
        searcher = FakeSearcher([[record("c1")]])

        result = refiner(searcher).run("query")

        assert result.reflection["action"] == "answer"
        assert len(result.attempts) == 1
        assert len(searcher.candidate_calls) == 1
        assert result.attempts[0].candidate_ids == ["c1"]
        assert result.attempts[0].evidence_ids == ["c1"]

    def test_second_retrieval_success_preserves_complete_lineage(self):
        first = record("c1", "partial")
        second = record("c2", "complete")
        searcher = FakeSearcher([[first], [second]])
        client = ScriptedClient(
            decision(
                "retrieve_again",
                sufficient=False,
                rewritten_query="query missing experiment",
                missing_information="experiment",
            ),
            decision("answer", sufficient=True),
        )

        trace = TurnTrace(turn_index=1, user_message="query")
        result = refiner(searcher, client).run(
            "query", {"sections": ["Results"]}, trace=trace
        )

        assert result.reflection["action"] == "answer"
        assert result.initial_query == "query"
        assert result.final_query == "query missing experiment"
        assert [attempt.query for attempt in result.attempts] == [
            "query",
            "query missing experiment",
        ]
        assert result.attempts[0].candidates == [first]
        assert result.attempts[1].evidence == [second]
        assert result.to_dict()["attempts"][1]["evidence_ids"] == ["c2"]
        assert all(
            call[2] == {"sections": ["Results"]} for call in searcher.candidate_calls
        )
        assert [stage.stage for stage in trace.stages] == [
            "Retrieve",
            "Rerank",
            "Reflect",
            "Retrieve",
            "Rerank",
            "Reflect",
        ]
        assert [stage.meta["attempt"] for stage in trace.stages] == [1, 1, 1, 2, 2, 2]
        assert all(stage.started_at and stage.ended_at for stage in trace.stages)

    def test_empty_results_stop_after_one_configured_second_retrieval(self):
        searcher = FakeSearcher([[], []])

        result = refiner(searcher).run("query")

        assert result.reflection["action"] == "refuse"
        assert len(searcher.candidate_calls) == 2
        assert [attempt.attempt for attempt in result.attempts] == [1, 2]
        assert all(attempt.candidate_ids == [] for attempt in result.attempts)

    def test_zero_second_retrieval_budget_makes_only_one_call(self):
        searcher = FakeSearcher([[]])

        result = refiner(searcher, max_second_retrievals=0).run("query")

        assert result.reflection["action"] == "refuse"
        assert len(searcher.candidate_calls) == 1

    def test_repeated_model_retry_is_bounded_to_two_total_attempts(self):
        searcher = FakeSearcher([[record("c1")], [record("c2")]])
        first_retry = decision(
            "retrieve_again",
            sufficient=False,
            rewritten_query="query detail one",
            missing_information="detail one",
        )
        second_retry = decision(
            "retrieve_again",
            sufficient=False,
            rewritten_query="query detail two",
            missing_information="detail two",
        )

        result = refiner(searcher, ScriptedClient(first_retry, second_retry)).run(
            "query"
        )

        assert result.reflection["action"] == "refuse"
        assert result.reflection["source"] == "policy"
        assert len(searcher.candidate_calls) == 2
        assert len(result.attempts) == 2

    @pytest.mark.parametrize(
        ("searcher", "failed_stage", "candidate_ids"),
        [
            (FakeSearcher(fail_retrieve=OSError("missing index")), "Retrieve", []),
            (
                FakeSearcher(fail_rerank=RuntimeError("reranker offline")),
                "Rerank",
                ["c1"],
            ),
        ],
    )
    def test_execution_failure_becomes_traced_safe_refusal(
        self,
        searcher,
        failed_stage,
        candidate_ids,
    ):
        trace = TurnTrace(turn_index=1, user_message="query")
        result = refiner(searcher).run("query", trace=trace)

        assert result.reflection["action"] == "refuse"
        assert result.reflection["failed_stage"] == failed_stage
        assert len(result.attempts) == 1
        assert result.attempts[0].status == "error"
        assert result.attempts[0].candidate_ids == candidate_ids
        assert result.attempts[0].error is not None
        assert trace.stages[-1].stage == failed_stage
        assert trace.stages[-1].status == "error"
        assert trace.stages[-1].error == result.attempts[0].error

    def test_second_retrieval_failure_does_not_reuse_first_attempt_candidates(self):
        first = record("c1")
        searcher = FakeSearcher(
            [[first]],
            fail_retrieve=OSError("second lookup failed"),
            fail_retrieve_on_call=2,
        )
        retry = decision(
            "retrieve_again",
            sufficient=False,
            rewritten_query="query more detail",
            missing_information="more detail",
        )

        result = refiner(searcher, ScriptedClient(retry)).run("query")

        assert result.reflection["action"] == "refuse"
        assert result.reflection["failed_stage"] == "Retrieve"
        assert len(result.attempts) == 2
        assert result.attempts[0].candidate_ids == ["c1"]
        assert result.attempts[1].candidate_ids == []
        assert result.attempts[1].evidence_ids == []
        assert result.candidates == []
        assert result.evidence == []

    @pytest.mark.parametrize(
        ("top_k", "max_second_retrievals"),
        [(0, 1), (1, -1), (True, 1), (1, False)],
    )
    def test_invalid_budgets_are_rejected(self, top_k, max_second_retrievals):
        with pytest.raises(ValueError):
            EvidenceRefiner(
                RetrieveStage(FakeSearcher()),
                RerankStage(FakeSearcher()),
                ReflectStage(None),
                top_k=top_k,
                max_second_retrievals=max_second_retrievals,
            )
