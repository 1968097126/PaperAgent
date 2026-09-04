from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline.memory import ResearchMemory
from pipeline.stages import GenerateStage, MemoryStage, validate_citations

EVIDENCE = [
    {
        "paper_id": "p1",
        "chunk_id": "c1",
        "title": "Paper One",
        "section": "Method",
        "text": "The method uses verified evidence.",
    },
    {
        "paper_id": "p2",
        "chunk_id": "c2",
        "title": "Paper Two",
        "section": "Results",
        "text": "The experiment reports a verified result.",
    },
]


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


def nlu(**values: Any) -> dict[str, Any]:
    return {
        "intent": "new_search",
        "standalone_query": "verified method",
        "resolved_references": [],
        "constraints": [],
        "comparison_targets": [],
        **values,
    }


def reflect(**values: Any) -> dict[str, Any]:
    return {
        "sufficient": True,
        "action": "answer",
        "missing_information": "",
        "rewritten_query": "",
        "reason": "supported",
        **values,
    }


class TestCitationValidation:
    def test_valid_citations_return_stable_pairs_and_chunk_ids(self):
        result = validate_citations("One [p1/c1]. Two [p2/c2].", EVIDENCE)

        assert result["valid"] is True
        assert result["citations"] == ["c1", "c2"]
        assert result["valid_citations"] == ["c1", "c2"]
        assert result["citation_pairs"] == [("p1", "c1"), ("p2", "c2")]

    def test_missing_unknown_duplicate_and_no_evidence_are_rejected(self):
        missing = validate_citations("unsupported answer", EVIDENCE)
        unknown = validate_citations("claim [p9/c9]", EVIDENCE)
        duplicate = validate_citations("a [p1/c1], b [p1/c1]", EVIDENCE)
        empty = validate_citations("claim [p1/c1]", [])

        assert "answer has no evidence citation" in missing["errors"]
        assert unknown["invalid"] == [("p9", "c9")]
        assert duplicate["duplicates"] == [("p1", "c1")]
        assert "no evidence is available for a cited answer" in empty["errors"]
        assert all(
            not result["valid"] for result in (missing, unknown, duplicate, empty)
        )

    def test_malformed_or_duplicate_evidence_is_reported_not_trusted(self):
        malformed = [{"paper_id": "p1"}, EVIDENCE[0], dict(EVIDENCE[0])]

        result = validate_citations("claim [p1/c1]", malformed)

        assert result["valid"] is False
        assert "evidence item 0 requires chunk_id" in result["evidence_errors"]
        assert any("duplicate evidence identity" in error for error in result["errors"])

    def test_non_evidence_text_can_explicitly_disable_citation_requirement(self):
        result = validate_citations(
            "Please clarify the paper.", [], require_citation=False
        )

        assert result["valid"] is True
        assert result["citations"] == []


class TestGenerateStage:
    @pytest.mark.parametrize(
        ("nlu_value", "reflect_value", "route"),
        [
            (nlu(intent="meta"), reflect(), "meta"),
            (
                nlu(intent="clarification"),
                reflect(action="clarify", sufficient=False),
                "clarify",
            ),
            (
                nlu(intent="out_of_scope"),
                reflect(action="refuse", sufficient=False),
                "refuse",
            ),
        ],
    )
    def test_non_answer_routes_are_deterministic_and_need_no_citations(
        self,
        nlu_value,
        reflect_value,
        route,
    ):
        client = ScriptedClient(AssertionError("provider must not be called"))

        result = GenerateStage(client).run("question", nlu_value, reflect_value, [])

        assert result["route"] == route
        assert result["citations"] == []
        assert result["citation_validation"]["valid"] is True
        assert result["generation_attempts"] == 0
        assert client.calls == []

    def test_answer_route_requires_sufficient_nonempty_valid_evidence(self):
        client = ScriptedClient(AssertionError("provider must not be called"))
        insufficient = GenerateStage(client).run(
            "question", nlu(), reflect(sufficient=False), EVIDENCE
        )
        empty = GenerateStage(client).run("question", nlu(), reflect(), [])
        malformed = GenerateStage(client).run(
            "question", nlu(), reflect(), [{"paper_id": "p1", "chunk_id": "c1"}]
        )

        assert insufficient["fallback"] is True
        assert empty["fallback"] is True
        assert "no evidence" in empty["generation_error"]
        assert "text" in malformed["generation_error"]
        assert client.calls == []

    def test_no_client_has_evidence_extract_fallback_with_valid_citation(self):
        result = GenerateStage(None).run("请概括方法", nlu(), reflect(), EVIDENCE)

        assert result["source"] == "fallback"
        assert result["answer"].startswith("根据检索证据")
        assert result["answer"].endswith("[p1/c1]")
        assert result["citations"] == ["c1"]
        assert result["citation_validation"]["valid"] is True

    def test_model_generation_is_evidence_only_and_uses_chinese_default(self):
        client = ScriptedClient("该方法使用了证据 [p1/c1]")

        result = GenerateStage(client, max_evidence_chars=12).run(
            "请解释这个方法", nlu(), reflect(), EVIDENCE
        )

        assert result["source"] == "model"
        assert result["generation_attempts"] == 1
        assert result["citations"] == ["c1"]
        assert "Chinese by default" in client.calls[0][1]
        assert "Use ONLY the supplied evidence" in client.calls[0][1]
        prompt_evidence = json.loads(client.calls[0][1].split("Evidence: ", 1)[1])
        assert all(len(item["text"]) <= 12 for item in prompt_evidence)

    def test_invalid_citation_is_repaired_exactly_once(self):
        client = ScriptedClient(
            "unsupported [p9/c9]",
            "supported [p1/c1]",
        )

        result = GenerateStage(client).run("question", nlu(), reflect(), EVIDENCE)

        assert result["source"] == "model_repair"
        assert result["generation_attempts"] == 2
        assert result["citations"] == ["c1"]
        assert len(client.calls) == 2
        assert "unknown citations" in client.calls[1][1]

    def test_failed_repair_never_returns_the_unsupported_model_answer(self):
        client = ScriptedClient("unsupported", "still unsupported")

        result = GenerateStage(client).run("question", nlu(), reflect(), EVIDENCE)

        assert result["fallback"] is True
        assert result["generation_attempts"] == 2
        assert result["citation_validation"]["valid"] is False
        assert result["answer"] != "still unsupported"
        assert len(client.calls) == 2

    @pytest.mark.parametrize(
        ("responses", "attempt"),
        [
            ((RuntimeError("offline"),), 1),
            (("unsupported", RuntimeError("offline")), 2),
        ],
    )
    def test_provider_failures_have_safe_fallback(self, responses, attempt):
        result = GenerateStage(ScriptedClient(*responses)).run(
            "question", nlu(), reflect(), EVIDENCE
        )

        assert result["fallback"] is True
        assert result["generation_attempts"] == attempt
        assert f"attempt {attempt}" in result["generation_error"]
        assert "RuntimeError: offline" in result["generation_error"]

    def test_invalid_configuration_and_duplicate_evidence_are_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            GenerateStage(None, max_evidence_chars=0)
        result = GenerateStage(None).run(
            "question", nlu(), reflect(), [EVIDENCE[0], dict(EVIDENCE[0])]
        )
        assert result["fallback"] is True
        assert "duplicate evidence identity" in result["generation_error"]


class TestResearchMemory:
    def test_modes_and_configuration_are_validated(self):
        with pytest.raises(ValueError, match="memory mode"):
            ResearchMemory("unknown")
        with pytest.raises(ValueError, match="positive integer"):
            ResearchMemory(compression_token_threshold=0)
        with pytest.raises(TypeError, match="mapping"):
            ResearchMemory.from_dict([])  # type: ignore[arg-type]

    def test_off_mode_does_not_mutate_or_consume_tokens(self):
        memory = ResearchMemory("off")

        memory.update(
            user_message="Paper One",
            nlu={"resolved_references": ["Paper One"]},
            evidence=EVIDENCE,
            citations=["c1"],
        )

        assert memory.update_count == 0
        assert memory.context() == {}
        assert memory.estimated_tokens() == 0

    def test_on_mode_writes_only_traceable_fields_and_cited_evidence(self):
        memory = ResearchMemory("on", compression_token_threshold=1)
        value = nlu(
            resolved_references=["Paper One", "Invented Entity"],
            constraints=["recent results", "fabricated constraint"],
            comparison_targets=["Paper One", "Paper Two", "Unknown Paper"],
        )

        memory.update(
            user_message="Compare Paper One and Paper Two using recent results",
            nlu=value,
            evidence=EVIDENCE,
            citations=["c1", "unknown"],
        )

        assert memory.entities == ["Paper One"]
        assert memory.constraints == ["recent results"]
        assert memory.comparison_targets == ["Paper One", "Paper Two"]
        assert memory.confirmed_papers == [{"paper_id": "p1", "title": "Paper One"}]
        assert memory.citations == ["c1"]
        assert memory.verified_facts[0]["chunk_id"] == "c1"
        assert memory.compressed is False

    def test_history_and_prior_memory_are_valid_traceability_sources(self):
        memory = ResearchMemory("on", entities=["PPO paper"])

        memory.update(
            user_message="What about it?",
            nlu={
                "resolved_references": ["PPO paper", "TRPO paper"],
                "constraints": ["training stability"],
            },
            evidence=[],
            citations=[],
            traceable_context=["TRPO paper discusses training stability"],
        )

        assert memory.entities == ["PPO paper", "TRPO paper"]
        assert memory.constraints == ["training stability"]

    def test_compressed_mode_uses_threshold_instead_of_every_turn(self):
        below = ResearchMemory("compressed", compression_token_threshold=10_000)
        below.update(
            user_message="Paper One",
            nlu={"resolved_references": ["Paper One"]},
            evidence=EVIDENCE,
            citations=["c1"],
        )
        above = ResearchMemory("compressed", compression_token_threshold=1)
        above.update(
            user_message="Paper One",
            nlu={"resolved_references": ["Paper One"]},
            evidence=EVIDENCE,
            citations=["c1"],
        )

        assert below.compressed is False
        assert below.compression_count == 0
        assert above.compressed is True
        assert above.compression_count == 1

    def test_explicit_compression_bounds_every_field_and_preserves_recent_keys(self):
        memory = ResearchMemory(
            "compressed",
            confirmed_papers=[
                {"paper_id": f"p{i}", "title": f"Paper {i}"} for i in range(3)
            ],
            entities=[f"entity {i}" for i in range(3)],
            constraints=[f"constraint {i}" for i in range(3)],
            comparison_targets=[f"target {i}" for i in range(3)],
            unresolved_questions=[f"question {i}" for i in range(3)],
            citations=[f"c{i}" for i in range(3)],
            verified_facts=[
                {"chunk_id": f"c{i}", "excerpt": "x" * 300} for i in range(3)
            ],
        )

        changed = memory.compress(
            max_papers=2,
            max_entities=2,
            max_constraints=2,
            max_comparison_targets=2,
            max_unresolved=2,
            max_citations=2,
            max_facts=2,
        )

        assert changed is True
        assert memory.entities == ["entity 1", "entity 2"]
        assert memory.constraints == ["constraint 1", "constraint 2"]
        assert memory.confirmed_papers[-1]["paper_id"] == "p2"
        assert memory.citations == ["c1", "c2"]
        assert len(memory.verified_facts) == 2
        assert all(len(fact["excerpt"]) <= 160 for fact in memory.verified_facts)

    def test_hard_limit_prevents_unbounded_uncompressed_lists(self):
        memory = ResearchMemory("on", hard_item_limit=2)

        memory.update(
            user_message="entity zero entity one entity two",
            nlu={"resolved_references": ["entity zero", "entity one", "entity two"]},
            evidence=[],
            citations=[],
        )

        assert memory.entities == ["entity one", "entity two"]

    def test_from_dict_restores_known_fields_and_overrides_mode(self):
        memory = ResearchMemory.from_dict(
            {"entities": ["PPO"], "update_count": 2, "unknown": "ignored"},
            mode="on",
        )

        assert memory.mode == "on"
        assert memory.entities == ["PPO"]
        assert memory.update_count == 2


class TestMemoryStage:
    def test_updates_unresolved_state_and_clears_same_question_after_answer(self):
        memory = ResearchMemory("on")
        stage = MemoryStage()

        first = stage.run(
            memory,
            user_message="Paper One limitation?",
            nlu=nlu(resolved_references=["Paper One"]),
            evidence=EVIDENCE,
            generated={"citations": []},
            reflect={"action": "clarify"},
        )
        second = stage.run(
            memory,
            user_message="Paper One limitation?",
            nlu=nlu(resolved_references=["Paper One"]),
            evidence=EVIDENCE,
            generated={"citations": ["c1"]},
            reflect={"action": "answer"},
        )

        assert first["unresolved_questions"] == ["Paper One limitation?"]
        assert second["unresolved_questions"] == []
        assert second["confirmed_papers"] == [{"paper_id": "p1", "title": "Paper One"}]

    def test_rejects_invalid_generated_citation_shape(self):
        with pytest.raises(TypeError, match="citations"):
            MemoryStage().run(
                ResearchMemory("on"),
                user_message="question",
                nlu=nlu(),
                evidence=EVIDENCE,
                generated={"citations": "c1"},
                reflect={"action": "answer"},
            )
