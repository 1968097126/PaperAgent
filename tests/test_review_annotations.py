from __future__ import annotations

import csv
import json
from copy import deepcopy

import pytest

from eval.review_annotations import (
    ANNOTATION_COLUMNS,
    ReviewValidationError,
    build_evidence_rows,
    build_review_rows,
    check_review_package,
    export_review_package,
    validate_review_rows,
)


def _values():
    main = {
        "id": "M-001",
        "conversation_id": "M-C01",
        "turn": 1,
        "query": "What does the paper report?",
        "history": [],
        "intent": "new_search",
        "standalone_query": "What does the paper report?",
        "resolved_references": [],
        "required_entities": [],
        "required_constraints": [],
        "comparison_targets": [],
        "needs_retrieval": True,
        "ground_truth_chunk_ids": ["c1"],
        "requires_rewrite": False,
        "requires_second_retrieval": False,
        "should_refuse": False,
        "refusal_reason": "",
        "expected_action": "answer",
        "cross_paper_comparison": False,
        "unanswerable_marker": "",
        "synthetic": True,
        "provenance": "synthetic_grounded_in_real_corpus",
    }
    datasets = {
        "M": [main],
        "I": [
            {
                "id": "I-001",
                "source_turn_id": "M-001",
                "query": main["query"],
                "history": [],
                "intent": "new_search",
            }
        ],
        "R": [
            {
                "id": "R-001",
                "query": "paper finding",
                "ground_truth_chunk_ids": ["c1"],
            }
        ],
        "B": [
            {
                "id": "B-001",
                "source_turn_id": "M-001",
                "query": main["query"],
                "expected_action": "allow",
            }
        ],
        "E": [{"id": "E-001", "source_turn_id": "M-001"}],
    }
    chunks = [
        {
            "chunk_id": "c1",
            "paper_id": "p1",
            "title": "Paper One",
            "section": "Results",
            "source": "fixture://p1",
            "chunk_index": 0,
            "text": "The full evidence text.",
        }
    ]
    return datasets, chunks


def _approve(rows):
    approved = deepcopy(rows)
    for row in approved:
        row.update(
            {
                "decision": "approve",
                "reviewer": "reviewer-1",
                "reviewed_at": "2026-09-05T16:30:00+08:00",
            }
        )
    return approved


def test_build_review_rows_covers_independent_sets_and_e2e_links():
    datasets, chunks = _values()
    rows = build_review_rows(datasets)

    assert [row["dataset"] for row in rows] == ["M", "I", "R", "B"]
    assert rows[0]["e2e_id"] == "E-001"
    assert all(row["decision"] == "pending" for row in rows)
    assert set(rows[0]) == set(ANNOTATION_COLUMNS)
    evidence = build_evidence_rows(rows, chunks)
    assert len(evidence) == 1
    assert evidence[0]["text"] == "The full evidence text."


def test_pending_review_is_valid_but_not_complete():
    datasets, _ = _values()
    rows = build_review_rows(datasets)

    result = validate_review_rows(rows, rows)
    assert result["total"] == 4
    assert result["decisions"]["pending"] == 4
    assert result["complete"] is False
    with pytest.raises(ReviewValidationError, match="not complete"):
        validate_review_rows(rows, rows, require_complete=True)


def test_all_approved_rows_are_complete():
    datasets, _ = _values()
    expected = build_review_rows(datasets)

    result = validate_review_rows(_approve(expected), expected, require_complete=True)
    assert result["complete"] is True
    assert result["decisions"]["approve"] == 4


def test_machine_columns_and_review_contract_cannot_be_bypassed():
    datasets, _ = _values()
    expected = build_review_rows(datasets)
    changed = _approve(expected)
    changed[0]["intent"] = "follow_up"
    with pytest.raises(ReviewValidationError, match="machine-owned columns"):
        validate_review_rows(changed, expected)

    revised = _approve(expected)
    revised[0].update(
        {
            "decision": "revise",
            "issue_codes": "intent",
            "proposed_correction_json": "",
        }
    )
    with pytest.raises(ReviewValidationError, match="requires issue_codes"):
        validate_review_rows(revised, expected)


def test_csv_round_trip_preserves_multiline_review_cells(tmp_path):
    datasets, _ = _values()
    rows = _approve(build_review_rows(datasets))
    rows[0]["notes"] = "first line\nsecond line, with comma"
    path = tmp_path / "annotations.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["notes"] == rows[0]["notes"]
    assert json.loads(loaded[0]["ground_truth_chunk_ids_json"]) == ["c1"]


def test_package_refuses_overwrite_and_detects_source_drift(tmp_path, monkeypatch):
    datasets, chunks = _values()
    data_dir = tmp_path / "data"
    review_dir = tmp_path / "review"
    chunks_path = tmp_path / "chunks.jsonl"
    data_dir.mkdir()
    for name, rows in datasets.items():
        (data_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    chunks_path.write_text(
        "".join(json.dumps(row) + "\n" for row in chunks), encoding="utf-8"
    )
    monkeypatch.setattr("eval.review_annotations.validate_datasets", lambda *_args: {})

    result = export_review_package(data_dir, chunks_path, review_dir)
    assert result["independent_rows"] == 4
    assert check_review_package(data_dir, chunks_path, review_dir)["total"] == 4
    with pytest.raises(ReviewValidationError, match="already exists"):
        export_review_package(data_dir, chunks_path, review_dir)

    with (data_dir / "I.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ReviewValidationError, match="manifest is stale"):
        check_review_package(data_dir, chunks_path, review_dir)
