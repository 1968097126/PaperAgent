"""Export and validate a human-review package for evaluation annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.datasets import DATASET_NAMES, read_jsonl, validate_datasets
from retrieval.chunking import load_chunks

REVIEW_SCHEMA_VERSION = 1
INDEPENDENT_DATASETS = ("M", "I", "R", "B")
DECISIONS = frozenset({"pending", "approve", "revise", "reject"})
ISSUE_CODES = frozenset(
    {
        "query",
        "history",
        "intent",
        "standalone_query",
        "reference_resolution",
        "retrieval_flag",
        "ground_truth",
        "expected_action",
        "rewrite_flag",
        "second_retrieval_flag",
        "refusal_label",
        "comparison_label",
        "e2e_selection",
        "other",
    }
)
MACHINE_COLUMNS = (
    "dataset",
    "record_id",
    "conversation_id",
    "turn",
    "source_turn_id",
    "e2e_id",
    "query",
    "history_json",
    "intent",
    "standalone_query",
    "resolved_references_json",
    "required_entities_json",
    "required_constraints_json",
    "comparison_targets_json",
    "needs_retrieval",
    "ground_truth_chunk_ids_json",
    "expected_action",
    "requires_rewrite",
    "requires_second_retrieval",
    "should_refuse",
    "refusal_reason",
    "cross_paper_comparison",
    "unanswerable_marker",
    "record_sha256",
)
REVIEW_COLUMNS = (
    "decision",
    "issue_codes",
    "proposed_correction_json",
    "reviewer",
    "reviewed_at",
    "notes",
)
ANNOTATION_COLUMNS = (*MACHINE_COLUMNS, *REVIEW_COLUMNS)
EVIDENCE_COLUMNS = (
    "chunk_id",
    "paper_id",
    "title",
    "section",
    "source",
    "text",
    "chunk_sha256",
)
CORRECTABLE_FIELDS = frozenset(
    {
        "query",
        "history",
        "intent",
        "standalone_query",
        "resolved_references",
        "required_entities",
        "required_constraints",
        "comparison_targets",
        "needs_retrieval",
        "ground_truth_chunk_ids",
        "expected_action",
        "requires_rewrite",
        "requires_second_retrieval",
        "should_refuse",
        "refusal_reason",
        "cross_paper_comparison",
        "unanswerable_marker",
        "e2e_included",
    }
)


class ReviewValidationError(ValueError):
    """Raised when a review package is incomplete, stale, or malformed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _json_cell(value: Any) -> str:
    return _canonical_json(value)


def _scalar_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _machine_row(
    dataset: str,
    record: Mapping[str, Any],
    e2e_by_source: Mapping[str, str],
) -> dict[str, str]:
    record_id = str(record["id"])
    return {
        "dataset": dataset,
        "record_id": record_id,
        "conversation_id": _scalar_cell(record.get("conversation_id")),
        "turn": _scalar_cell(record.get("turn")),
        "source_turn_id": _scalar_cell(record.get("source_turn_id")),
        "e2e_id": e2e_by_source.get(record_id, "") if dataset == "M" else "",
        "query": _scalar_cell(record.get("query")),
        "history_json": _json_cell(record.get("history", [])),
        "intent": _scalar_cell(record.get("intent")),
        "standalone_query": _scalar_cell(record.get("standalone_query")),
        "resolved_references_json": _json_cell(record.get("resolved_references", [])),
        "required_entities_json": _json_cell(record.get("required_entities", [])),
        "required_constraints_json": _json_cell(record.get("required_constraints", [])),
        "comparison_targets_json": _json_cell(record.get("comparison_targets", [])),
        "needs_retrieval": _scalar_cell(record.get("needs_retrieval")),
        "ground_truth_chunk_ids_json": _json_cell(
            record.get("ground_truth_chunk_ids", [])
        ),
        "expected_action": _scalar_cell(record.get("expected_action")),
        "requires_rewrite": _scalar_cell(record.get("requires_rewrite")),
        "requires_second_retrieval": _scalar_cell(
            record.get("requires_second_retrieval")
        ),
        "should_refuse": _scalar_cell(record.get("should_refuse")),
        "refusal_reason": _scalar_cell(record.get("refusal_reason")),
        "cross_paper_comparison": _scalar_cell(record.get("cross_paper_comparison")),
        "unanswerable_marker": _scalar_cell(record.get("unanswerable_marker")),
        "record_sha256": _sha256_value(record),
    }


def build_review_rows(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, str]]:
    """Create deterministic rows for the 375 independent review cases."""
    missing = set(DATASET_NAMES) - set(datasets)
    if missing:
        raise ReviewValidationError(f"missing datasets: {sorted(missing)}")
    e2e_by_source: dict[str, str] = {}
    for record in datasets["E"]:
        source_id = str(record.get("source_turn_id", ""))
        if not source_id or source_id in e2e_by_source:
            raise ReviewValidationError(
                "E source_turn_id values must be non-empty and unique"
            )
        e2e_by_source[source_id] = str(record["id"])

    rows: list[dict[str, str]] = []
    for dataset in INDEPENDENT_DATASETS:
        for record in datasets[dataset]:
            machine = _machine_row(dataset, record, e2e_by_source)
            rows.append(
                {
                    **machine,
                    "decision": "pending",
                    "issue_codes": "",
                    "proposed_correction_json": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "notes": "",
                }
            )
    return rows


def build_evidence_rows(
    review_rows: Sequence[Mapping[str, str]],
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Create a deduplicated catalog for every ground-truth chunk under review."""
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise ReviewValidationError("corpus chunk IDs are not unique")
    referenced: set[str] = set()
    for row in review_rows:
        ground_truth_ids = json.loads(row["ground_truth_chunk_ids_json"])
        if not isinstance(ground_truth_ids, list):
            raise ReviewValidationError("ground-truth cell must contain a JSON list")
        referenced.update(str(value) for value in ground_truth_ids)
    unknown = referenced - set(chunks_by_id)
    if unknown:
        raise ReviewValidationError(
            f"review rows reference unknown chunks: {sorted(unknown)}"
        )

    evidence_rows: list[dict[str, str]] = []
    for chunk_id in sorted(referenced):
        chunk = chunks_by_id[chunk_id]
        machine = {
            "chunk_id": chunk_id,
            "paper_id": _scalar_cell(chunk.get("paper_id")),
            "title": _scalar_cell(chunk.get("title")),
            "section": _scalar_cell(chunk.get("section")),
            "source": _scalar_cell(chunk.get("source")),
            "text": _scalar_cell(chunk.get("text")),
        }
        evidence_rows.append({**machine, "chunk_sha256": _sha256_value(machine)})
    return evidence_rows


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_columns):
            raise ReviewValidationError(
                f"{path} columns differ from review schema {REVIEW_SCHEMA_VERSION}"
            )
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ReviewValidationError(
                    f"{path}:{line_number} has missing or extra CSV cells"
                )
            rows.append(dict(row))
        return rows


def _source_manifest(
    data_dir: Path,
    chunks_path: Path,
    review_rows: Sequence[Mapping[str, str]],
    evidence_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "independent_rows": len(review_rows),
        "dataset_rows": dict(Counter(row["dataset"] for row in review_rows)),
        "e2e_links": sum(bool(row["e2e_id"]) for row in review_rows),
        "evidence_chunks": len(evidence_rows),
        "source_sha256": {
            **{
                f"eval/data/{name}.jsonl": _sha256_bytes(
                    (data_dir / f"{name}.jsonl").read_bytes()
                )
                for name in DATASET_NAMES
            },
            "corpus/chunks.jsonl": _sha256_bytes(chunks_path.read_bytes()),
        },
    }


def export_review_package(
    data_dir: Path,
    chunks_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Export review CSVs after strict source validation."""
    validate_datasets(data_dir, chunks_path)
    output_dir = Path(output_dir)
    paths = {
        "annotations": output_dir / "annotations.csv",
        "evidence": output_dir / "evidence.csv",
        "manifest": output_dir / "review_manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not force:
        raise ReviewValidationError(
            "review output already exists; preserve manual work or pass --force: "
            + ", ".join(existing)
        )
    datasets = {
        name: read_jsonl(Path(data_dir) / f"{name}.jsonl") for name in DATASET_NAMES
    }
    chunks = load_chunks(chunks_path)
    review_rows = build_review_rows(datasets)
    evidence_rows = build_evidence_rows(review_rows, chunks)
    manifest = _source_manifest(data_dir, chunks_path, review_rows, evidence_rows)
    _write_csv(paths["annotations"], ANNOTATION_COLUMNS, review_rows)
    _write_csv(paths["evidence"], EVIDENCE_COLUMNS, evidence_rows)
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"status": "pending", **manifest, "output_dir": str(output_dir)}


def _parse_correction(row: Mapping[str, str]) -> dict[str, Any]:
    raw = row["proposed_correction_json"].strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewValidationError(
            f"{row['dataset']}/{row['record_id']} correction is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ReviewValidationError(
            f"{row['dataset']}/{row['record_id']} correction must be a JSON object"
        )
    unknown = set(value) - CORRECTABLE_FIELDS
    if unknown:
        raise ReviewValidationError(
            f"{row['dataset']}/{row['record_id']} correction has unknown fields: "
            f"{sorted(unknown)}"
        )
    return value


def _validate_timestamp(value: str, identity: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewValidationError(f"{identity} reviewed_at must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ReviewValidationError(f"{identity} reviewed_at must include a timezone")


def validate_review_rows(
    actual: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, str]],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate immutable source columns and human decision fields."""
    expected_by_key = {(row["dataset"], row["record_id"]): row for row in expected}
    actual_by_key = {(row["dataset"], row["record_id"]): row for row in actual}
    if len(actual_by_key) != len(actual):
        raise ReviewValidationError(
            "review sheet contains duplicate dataset/record IDs"
        )
    if set(actual_by_key) != set(expected_by_key):
        raise ReviewValidationError(
            "review sheet row identities differ from source datasets"
        )

    decisions: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = {
        name: Counter() for name in INDEPENDENT_DATASETS
    }
    for key, expected_row in expected_by_key.items():
        row = actual_by_key[key]
        identity = f"{key[0]}/{key[1]}"
        changed = [
            column for column in MACHINE_COLUMNS if row[column] != expected_row[column]
        ]
        if changed:
            raise ReviewValidationError(
                f"{identity} machine-owned columns changed: {changed}"
            )
        decision = row["decision"].strip().casefold()
        if decision not in DECISIONS:
            raise ReviewValidationError(f"{identity} has invalid decision {decision!r}")
        issues = {
            value.strip() for value in row["issue_codes"].split(";") if value.strip()
        }
        unknown_issues = issues - ISSUE_CODES
        if unknown_issues:
            raise ReviewValidationError(
                f"{identity} has unknown issue codes: {sorted(unknown_issues)}"
            )
        correction = _parse_correction(row)
        if decision != "pending":
            if not row["reviewer"].strip():
                raise ReviewValidationError(f"{identity} reviewer is required")
            _validate_timestamp(row["reviewed_at"].strip(), identity)
        if decision == "approve" and (issues or correction):
            raise ReviewValidationError(
                f"{identity} approved rows cannot contain issues or corrections"
            )
        if decision == "revise" and (not issues or not correction):
            raise ReviewValidationError(
                f"{identity} revise requires issue_codes and a correction object"
            )
        if decision == "reject" and (not issues or not row["notes"].strip()):
            raise ReviewValidationError(
                f"{identity} reject requires issue_codes and explanatory notes"
            )
        decisions[decision] += 1
        by_dataset[key[0]][decision] += 1

    report = {
        "total": len(actual),
        "decisions": {name: decisions[name] for name in sorted(DECISIONS)},
        "by_dataset": {
            dataset: {name: counts[name] for name in sorted(DECISIONS)}
            for dataset, counts in by_dataset.items()
        },
        "complete": decisions["approve"] == len(actual),
    }
    if require_complete and not report["complete"]:
        raise ReviewValidationError(
            "review is not complete: every independent row must be approved after corrections"
        )
    return report


def check_review_package(
    data_dir: Path,
    chunks_path: Path,
    review_dir: Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Check review progress and reject stale or edited machine-owned data."""
    validate_datasets(data_dir, chunks_path)
    datasets = {
        name: read_jsonl(Path(data_dir) / f"{name}.jsonl") for name in DATASET_NAMES
    }
    expected_rows = build_review_rows(datasets)
    expected_evidence = build_evidence_rows(expected_rows, load_chunks(chunks_path))
    review_dir = Path(review_dir)
    annotations = _read_csv(review_dir / "annotations.csv", ANNOTATION_COLUMNS)
    evidence = _read_csv(review_dir / "evidence.csv", EVIDENCE_COLUMNS)
    if evidence != expected_evidence:
        raise ReviewValidationError("evidence.csv differs from the current corpus")
    expected_manifest = _source_manifest(
        data_dir, chunks_path, expected_rows, expected_evidence
    )
    manifest = json.loads(
        (review_dir / "review_manifest.json").read_text(encoding="utf-8")
    )
    if manifest != expected_manifest:
        raise ReviewValidationError("review manifest is stale or was modified")
    return {
        "status": "complete" if require_complete else "valid",
        **validate_review_rows(
            annotations, expected_rows, require_complete=require_complete
        ),
        "evidence_chunks": len(evidence),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "check"):
        child = subparsers.add_parser(command)
        child.add_argument("--data-dir", type=Path, default=Path("eval/data"))
        child.add_argument("--chunks", type=Path, default=Path("corpus/chunks.jsonl"))
        child.add_argument("--review-dir", type=Path, default=Path("eval/review/work"))
        if command == "export":
            child.add_argument("--force", action="store_true")
        else:
            child.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            result = export_review_package(
                args.data_dir, args.chunks, args.review_dir, force=args.force
            )
        else:
            result = check_review_package(
                args.data_dir,
                args.chunks,
                args.review_dir,
                require_complete=args.require_complete,
            )
    except (OSError, ValueError, csv.Error) as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
