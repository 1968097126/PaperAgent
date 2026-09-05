from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from eval.datasets import (
    BOUNDARY_COUNTS,
    E_INTENT_COUNTS,
    EXPECTED_SIZES,
    I_INTENT_COUNTS,
    INTENT_COUNTS,
    PROPERTY_COUNTS,
    PROVENANCE,
    DatasetValidationError,
    build_all_datasets,
    read_jsonl,
    validate_datasets,
    write_jsonl,
)
from eval.metrics import (
    citation_metrics,
    classification_accuracy,
    intent_metrics,
    memory_consistency,
    pipeline_success,
    recall_at_k,
    rewrite_metrics,
    status_counts,
    token_overhead,
)
from eval.run_evaluation import (
    cli,
    run_intent,
    run_retrieval,
    summarize_predictions,
    write_results,
)
from retrieval.chunking_v2 import write_chunks
from retrieval.config import RetrievalConfig


def chunks(count: int = 120) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": f"c{i}",
            "paper_id": f"p{i}",
            "title": f"Paper {i}",
            "section": "Abstract" if i % 2 == 0 else "Method",
            "source": f"fixture://{i}",
            "chunk_index": 0,
            "text": f"Evidence about policy method number {i} and robust learning.",
        }
        for i in range(count)
    ]


def _build(tmp_path) -> tuple[Any, Any, dict[str, Any]]:
    chunks_path = tmp_path / "chunks.jsonl"
    data_dir = tmp_path / "data"
    write_chunks(chunks(), chunks_path)
    return chunks_path, data_dir, build_all_datasets(chunks_path, data_dir)


def test_dataset_builder_exact_counts_provenance_and_determinism(tmp_path) -> None:
    chunks_path, data_dir, stats = _build(tmp_path)

    assert stats["sizes"] == EXPECTED_SIZES
    assert stats["intent_counts"] == INTENT_COUNTS
    assert stats["intent_set_counts"] == I_INTENT_COUNTS
    assert stats["boundary_counts"] == BOUNDARY_COUNTS
    assert stats["e2e_intent_counts"] == E_INTENT_COUNTS
    assert {key: stats[key] for key in PROPERTY_COUNTS} == PROPERTY_COUNTS
    assert stats["refusals"] == {"out_of_scope": 18, "insufficient_evidence": 6}

    first = {
        name: read_jsonl(data_dir / f"{name}.jsonl")
        for name in EXPECTED_SIZES
    }
    assert all(
        row["synthetic"] is True and row["provenance"] == PROVENANCE
        for rows in first.values()
        for row in rows
    )
    assert len({row["conversation_id"] for row in first["M"]}) == 30
    assert len({row["source_turn_id"] for row in first["E"]}) == 70
    assert len(
        {
            json.dumps([row["query"], row["history"]], sort_keys=True)
            for row in first["M"]
        }
    ) == 120
    assert len(
        {
            json.dumps([row["query"], row["history"]], sort_keys=True)
            for row in first["I"]
        }
    ) == 100
    assert len({row["query"] for row in first["B"]}) == 55
    assert all(
        all(
            item.casefold() in row["standalone_query"].casefold()
            for item in (
                row["required_entities"]
                + row["required_constraints"]
                + row["resolved_references"]
            )
        )
        for row in first["M"]
        if row["requires_rewrite"]
    )

    second_dir = tmp_path / "second"
    build_all_datasets(chunks_path, second_dir)
    assert first == {
        name: read_jsonl(second_dir / f"{name}.jsonl")
        for name in EXPECTED_SIZES
    }


def test_validator_rejects_unknown_chunks_and_e2e_drift(tmp_path) -> None:
    chunks_path, data_dir, _stats = _build(tmp_path)
    retrieval_rows = read_jsonl(data_dir / "R.jsonl")
    retrieval_rows[0]["ground_truth_chunk_ids"] = ["not-in-corpus"]
    write_jsonl(data_dir / "R.jsonl", retrieval_rows)
    with pytest.raises(DatasetValidationError, match="unknown chunks"):
        validate_datasets(data_dir, chunks_path)

    build_all_datasets(chunks_path, data_dir)
    e2e_rows = read_jsonl(data_dir / "E.jsonl")
    e2e_rows[0]["query"] = "changed after sampling"
    write_jsonl(data_dir / "E.jsonl", e2e_rows)
    with pytest.raises(DatasetValidationError, match="differs from source"):
        validate_datasets(data_dir, chunks_path)


def test_read_jsonl_reports_file_and_line_for_invalid_json(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{}\n{broken}\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=r"broken\.jsonl:2"):
        read_jsonl(path)


def test_intent_rewrite_and_classification_metrics_expose_coverage() -> None:
    intent_rows = [
        {"status": "ok", "expected": "a", "predicted": "a"},
        {"status": "ok", "expected": "a", "predicted": "b"},
        {"status": "skipped", "expected": "b", "predicted": "b"},
        {"status": "failed", "expected": "b"},
    ]
    intent = intent_metrics(intent_rows, ["a", "b"])
    assert intent["accuracy"] == 0.5
    assert intent["evaluated"] == 2
    assert intent["failed"] == 1
    assert intent["skipped"] == 1
    assert intent["confusion_matrix"]["a"] == {"a": 1, "b": 1, "__unknown__": 0}

    rewrite = rewrite_metrics(
        [
            {
                "id": "1",
                "status": "ok",
                "rewritten_query": "Compare PPO with a recent baseline",
                "required_entities": ["PPO"],
                "required_constraints": ["recent"],
            },
            {
                "id": "2",
                "status": "ok",
                "rewritten_query": "PPO only",
                "required_entities": ["PPO"],
                "required_constraints": ["recent"],
            },
        ]
    )
    assert rewrite["accuracy"] == 0.5
    assert rewrite["details"][1]["missing"] == ["recent"]
    decision = classification_accuracy(intent_rows, "expected", "predicted")
    assert decision["accuracy"] == 0.5


def test_recall_marks_empty_truth_unevaluable_and_validates_k() -> None:
    result = recall_at_k(
        [
            {
                "status": "ok",
                "ground_truth_chunk_ids": ["g"],
                "retrieved_chunk_ids": ["x", "g"],
            },
            {
                "status": "ok",
                "ground_truth_chunk_ids": [],
                "retrieved_chunk_ids": [],
            },
            {
                "status": "failed",
                "ground_truth_chunk_ids": ["g"],
                "retrieved_chunk_ids": [],
            },
        ]
    )
    assert result["recall@1"] == 0
    assert result["recall@5"] == 1
    assert result["evaluated"] == 1
    assert result["unevaluable"] == 1
    assert result["failed"] == 1
    with pytest.raises(ValueError, match="positive integers"):
        recall_at_k([], (0,))


def test_citation_pipeline_memory_and_token_metrics() -> None:
    citations = citation_metrics(
        [
            {
                "status": "ok",
                "predicted_citations": ["g", "g", "x"],
                "ground_truth_chunk_ids": ["g"],
                "valid_evidence_chunk_ids": ["g", "x"],
            }
        ]
    )
    assert citations["validity"] == pytest.approx(2 / 3)
    assert citations["precision"] == pytest.approx(1 / 3)
    assert citations["recall"] == 1

    pipeline = pipeline_success(
        [
            {
                "id": "1",
                "status": "ok",
                "routing_ok": True,
                "retrieval_ok": True,
                "answer_or_refusal_ok": True,
                "citations_ok": False,
            }
        ]
    )
    assert pipeline["success_rate"] == 0

    rows = [
        {
            "mode": mode,
            "status": "ok",
            "required_items": ["paper", "constraint"],
            "retained_items": retained,
            "input_tokens": input_tokens,
            "output_tokens": 2,
            "context_chars": context_chars,
        }
        for mode, retained, input_tokens, context_chars in (
            ("off", [], 8, 0),
            ("on", ["paper", "constraint"], 12, 100),
            ("compressed", ["paper", "constraint"], 10, 50),
        )
    ]
    memory = memory_consistency(rows)
    assert memory["off"]["consistency"] == 0
    assert memory["on"]["consistency"] == 1
    assert memory["compressed"]["consistency"] == 1
    tokens = token_overhead(rows)
    assert tokens["on"]["total_token_delta_vs_off"] == 4
    assert tokens["compressed"]["total_token_ratio_vs_off"] == 1.2


class FakeSearcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def search(self, query: str, top_k: int, *, mode: str) -> list[dict[str, Any]]:
        self.calls.append((query, top_k, mode))
        return [{"chunk_id": "c0", "score": 0.75}]


class IntentClient:
    input_tokens = 0
    output_tokens = 0

    def complete(self, stage: str, prompt: str) -> str:
        assert stage == "NLU"
        self.input_tokens += 8
        self.output_tokens += 4
        return json.dumps(
            {
                "intent": "new_search",
                "standalone_query": "paper method",
                "resolved_references": [],
                "needs_retrieval": True,
                "reason": "explicit paper request",
                "constraints": [],
                "comparison_targets": [],
            }
        )


class BrokenIntentClient:
    input_tokens = 0
    output_tokens = 0

    def complete(self, stage: str, prompt: str) -> str:
        self.input_tokens += 1
        self.output_tokens += 1
        return "not json"


def test_runner_components_and_summary_are_mockable(tmp_path) -> None:
    _chunks_path, data_dir, _stats = _build(tmp_path)
    searcher = FakeSearcher()
    retrieval = run_retrieval(
        RetrievalConfig(corpus_dir=tmp_path), data_dir, searcher=searcher
    )
    assert len(retrieval) == 300
    assert {row["mode"] for row in retrieval} == {
        "dense",
        "hybrid",
        "hybrid_rerank",
    }
    assert len(searcher.calls) == 300

    write_jsonl(
        data_dir / "I.jsonl",
        [
            {
                "id": "I-001",
                "query": "paper method",
                "history": [],
                "intent": "new_search",
            }
        ],
    )
    intents = run_intent(IntentClient(), data_dir)
    assert intents[0]["status"] == "ok"
    assert intents[0]["input_tokens"] == 8
    failed_intent = run_intent(BrokenIntentClient(), data_dir)[0]
    assert failed_intent["status"] == "failed"
    assert "structured output remained invalid" in failed_intent["error"]

    summary = summarize_predictions([*retrieval, *intents])
    assert summary["intent"]["accuracy"] == 1
    assert summary["retrieval_dense"]["evaluated"] == 100
    assert summary["prediction_statuses"]["failed"] == 0


def test_result_writer_persists_json_csv_markdown_and_config(tmp_path) -> None:
    predictions = [
        {
            "id": "R-001",
            "task": "retrieval",
            "mode": "dense",
            "status": "ok",
            "ground_truth_chunk_ids": ["c1"],
            "retrieved_chunk_ids": ["c1"],
        }
    ]
    summary = summarize_predictions(predictions)
    run_config = {"random_seed": 42, "dataset_provenance": PROVENANCE}
    output = tmp_path / "results"

    write_results(output, predictions, summary, run_config)

    assert read_jsonl(output / "per_sample.jsonl") == predictions
    assert json.loads((output / "summary.json").read_text())["retrieval_dense"][
        "recall@1"
    ] == 1
    assert "experiment,metric,value" in (output / "summary.csv").read_text()
    assert PROVENANCE in (output / "summary.md").read_text()
    assert json.loads((output / "run_config.json").read_text())["random_seed"] == 42


def test_dataset_only_cli_needs_chunks_but_not_index_or_provider(tmp_path, capsys) -> None:
    corpus_dir = tmp_path / "corpus"
    write_chunks(chunks(), corpus_dir / "chunks.jsonl")
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "results"

    result = cli(
        [
            "--suite",
            "datasets",
            "--build-datasets",
            "--corpus-dir",
            str(corpus_dir),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert json.loads((output_dir / "run_config.json").read_text())[
        "index_manifest"
    ] == {}
    assert json.loads(capsys.readouterr().out)["summary"]["prediction_statuses"] == {
        "ok": 0,
        "failed": 0,
        "skipped": 0,
        "unevaluable": 0,
    }


def test_provider_cli_stops_cleanly_when_credentials_are_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    corpus_dir = tmp_path / "corpus"
    write_chunks(chunks(), corpus_dir / "chunks.jsonl")
    data_dir = tmp_path / "data"
    build_all_datasets(corpus_dir / "chunks.jsonl", data_dir)
    output_dir = tmp_path / "results"
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    monkeypatch.setattr(
        "eval.run_evaluation.load_config", lambda: {"model": "custom/test-model"}
    )

    result = cli(
        [
            "--suite",
            "provider",
            "--corpus-dir",
            str(corpus_dir),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert "requires credentials for custom" in error["error"]
    assert not output_dir.exists()


def test_status_counts_treats_missing_and_unknown_as_failures() -> None:
    rows = [{"status": "ok"}, {"status": "mystery"}, {}, {"status": "skipped"}]
    original = deepcopy(rows)

    assert status_counts(rows) == {
        "ok": 1,
        "failed": 2,
        "skipped": 1,
        "unevaluable": 0,
    }
    assert rows == original
