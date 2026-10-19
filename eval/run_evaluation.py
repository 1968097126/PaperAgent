"""Run reproducible component/E2E evaluations and persist per-sample evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent import AgentState
from config import has_api_key, load_config
from eval.datasets import (
    DATASET_NAMES,
    INTENT_COUNTS,
    PROVENANCE,
    SCHEMA_VERSION,
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
from pipeline.client import NativeProviderClient
from pipeline.orchestrator import ResearchPipeline
from pipeline.stages import GuardrailsStage, NLUStage
from providers import detect_provider
from retrieval.config import RetrievalConfig
from retrieval.hybrid_search import HybridSearcher
from retrieval.manifest import load_manifest

RETRIEVAL_MODES = ("dense", "hybrid", "hybrid_rerank")
MEMORY_MODES = ("off", "on", "compressed")


class EvaluationConfigurationError(RuntimeError):
    """Raised before a run when required external configuration is absent."""


def _failure(
    row: Mapping[str, Any], task: str, exc: Exception, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "task": task,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        **metadata,
    }


def _token_snapshot(client: Any) -> tuple[int, int]:
    return (
        int(getattr(client, "input_tokens", 0) or 0),
        int(getattr(client, "output_tokens", 0) or 0),
    )


def _token_delta(client: Any, before: tuple[int, int]) -> dict[str, int]:
    after = _token_snapshot(client)
    return {
        "input_tokens": max(0, after[0] - before[0]),
        "output_tokens": max(0, after[1] - before[1]),
    }


def _runtime_config(
    provider_config: Mapping[str, Any], retrieval: RetrievalConfig, **overrides: Any
) -> dict[str, Any]:
    result = dict(provider_config)
    result.update(overrides)
    result["retrieval"] = {
        field.name: getattr(retrieval, field.name) for field in fields(RetrievalConfig)
    }
    return result


def run_retrieval(
    config: RetrievalConfig,
    data_dir: Path,
    *,
    searcher: Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate dense, hybrid, and hybrid+rerank with the same R rows."""
    rows = read_jsonl(Path(data_dir) / "R.jsonl")
    active_searcher = searcher or HybridSearcher(config)
    predictions: list[dict[str, Any]] = []
    for mode in RETRIEVAL_MODES:
        for row in rows:
            try:
                hits = active_searcher.search(row["query"], top_k=10, mode=mode)
                predictions.append(
                    {
                        "id": row["id"],
                        "task": "retrieval",
                        "mode": mode,
                        "status": "ok",
                        "ground_truth_chunk_ids": row["ground_truth_chunk_ids"],
                        "retrieved_chunk_ids": [hit["chunk_id"] for hit in hits],
                        "scores": [hit.get("score") for hit in hits],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-sample failures
                predictions.append(
                    _failure(
                        row,
                        "retrieval",
                        exc,
                        mode=mode,
                        ground_truth_chunk_ids=row["ground_truth_chunk_ids"],
                        retrieved_chunk_ids=[],
                    )
                )
    return predictions


def run_intent(client: Any, data_dir: Path) -> list[dict[str, Any]]:
    stage = NLUStage(client)
    predictions: list[dict[str, Any]] = []
    for row in read_jsonl(Path(data_dir) / "I.jsonl"):
        before = _token_snapshot(client)
        try:
            result = stage.run(row["query"], row.get("history", []), {})
            predictions.append(
                {
                    "id": row["id"],
                    "task": "intent",
                    "status": (
                        "failed" if result.get("source") == "safe_fallback" else "ok"
                    ),
                    "error": (
                        result.get("reason")
                        if result.get("source") == "safe_fallback"
                        else None
                    ),
                    "expected": row["intent"],
                    "predicted": result["intent"],
                    "source": result.get("source"),
                    **_token_delta(client, before),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-sample failures
            predictions.append(
                _failure(
                    row,
                    "intent",
                    exc,
                    expected=row["intent"],
                    **_token_delta(client, before),
                )
            )
    return predictions


def run_rewrite(client: Any, data_dir: Path) -> list[dict[str, Any]]:
    stage = NLUStage(client)
    predictions: list[dict[str, Any]] = []
    rows = [
        row
        for row in read_jsonl(Path(data_dir) / "M.jsonl")
        if row["requires_rewrite"]
    ]
    for row in rows:
        before = _token_snapshot(client)
        try:
            result = stage.run(row["query"], row["history"], {})
            predictions.append(
                {
                    "id": row["id"],
                    "task": "rewrite",
                    "status": (
                        "failed" if result.get("source") == "safe_fallback" else "ok"
                    ),
                    "error": (
                        result.get("reason")
                        if result.get("source") == "safe_fallback"
                        else None
                    ),
                    "rewritten_query": result["standalone_query"],
                    "required_entities": row["required_entities"],
                    "required_references": row["resolved_references"],
                    "required_constraints": row["required_constraints"],
                    "source": result.get("source"),
                    **_token_delta(client, before),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-sample failures
            predictions.append(
                _failure(row, "rewrite", exc, **_token_delta(client, before))
            )
    return predictions


def run_boundary(client: Any, data_dir: Path) -> list[dict[str, Any]]:
    stage = GuardrailsStage(client)
    predictions: list[dict[str, Any]] = []
    for row in read_jsonl(Path(data_dir) / "B.jsonl"):
        before = _token_snapshot(client)
        try:
            result = stage.run(row["query"], {}, row.get("history", []))
            predictions.append(
                {
                    "id": row["id"],
                    "task": "boundary",
                    "status": (
                        "failed" if result.get("source") == "safe_fallback" else "ok"
                    ),
                    "error": (
                        result.get("reason")
                        if result.get("source") == "safe_fallback"
                        else None
                    ),
                    "expected_action": row["expected_action"],
                    "predicted_action": result["action"],
                    "source": result.get("source"),
                    **_token_delta(client, before),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-sample failures
            predictions.append(
                _failure(
                    row,
                    "boundary",
                    exc,
                    expected_action=row["expected_action"],
                    **_token_delta(client, before),
                )
            )
    return predictions


def _pipeline_checks(
    row: Mapping[str, Any], result: Any, state: AgentState
) -> dict[str, Any]:
    truth = {str(item) for item in row.get("ground_truth_chunk_ids", [])}
    evidence_ids = {str(item["chunk_id"]) for item in result.evidence}
    citations = {str(item) for item in result.citations}
    expected_action = str(row["expected_action"])
    routing_ok = result.intent == row["intent"]
    if row["needs_retrieval"] and truth:
        retrieval_ok = truth.issubset(evidence_ids)
    elif row["needs_retrieval"]:
        retrieval_ok = bool(state.retrieval_state.get("attempts"))
    else:
        retrieval_ok = not state.retrieval_state.get("needs_retrieval", False)
    answer_or_refusal_ok = result.action == expected_action
    if expected_action == "answer" and row["needs_retrieval"]:
        citations_ok = bool(citations) and truth.issubset(citations) and citations.issubset(
            evidence_ids
        )
    else:
        citations_ok = not citations
    return {
        "routing_ok": routing_ok,
        "retrieval_ok": retrieval_ok,
        "answer_or_refusal_ok": answer_or_refusal_ok,
        "citations_ok": citations_ok,
        "ground_truth_chunk_ids": sorted(truth),
        "valid_evidence_chunk_ids": sorted(evidence_ids),
        "predicted_citations": list(result.citations),
        "predicted_intent": result.intent,
        "predicted_action": result.action,
        "answer": result.answer,
        "trace": result.trace.to_dict(),
        "trace_status": result.trace.status,
    }


def run_e2e(
    provider_config: Mapping[str, Any],
    retrieval: RetrievalConfig,
    client: Any,
    data_dir: Path,
    *,
    searcher: Any | None = None,
) -> list[dict[str, Any]]:
    runtime = _runtime_config(provider_config, retrieval)
    pipeline = ResearchPipeline(runtime, client=client, searcher=searcher)
    predictions: list[dict[str, Any]] = []
    for row in read_jsonl(Path(data_dir) / "E.jsonl"):
        state = AgentState(messages=list(row["history"]), memory_mode="compressed")
        before = _token_snapshot(client)
        try:
            result = pipeline.run(row["query"], state)
            predictions.append(
                {
                    "id": row["id"],
                    "task": "e2e",
                    "status": "ok",
                    **_pipeline_checks(row, result, state),
                    **_token_delta(client, before),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-sample failures
            predictions.append(
                _failure(row, "e2e", exc, **_token_delta(client, before))
            )
    return predictions


def run_reflect_ablation(
    provider_config: Mapping[str, Any],
    retrieval: RetrievalConfig,
    client: Any,
    data_dir: Path,
    *,
    searcher: Any | None = None,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(Path(data_dir) / "M.jsonl")
        if row["requires_second_retrieval"]
        or row["refusal_reason"] == "insufficient_evidence"
    ]
    predictions: list[dict[str, Any]] = []
    for mode, budget in (("off", 0), ("on", 1)):
        mode_config = RetrievalConfig(
            **{
                field.name: getattr(retrieval, field.name)
                for field in fields(RetrievalConfig)
                if field.name != "max_second_retrievals"
            },
            max_second_retrievals=budget,
        )
        runtime = _runtime_config(provider_config, mode_config)
        pipeline = ResearchPipeline(runtime, client=client, searcher=searcher)
        for row in rows:
            state = AgentState(messages=list(row["history"]))
            before = _token_snapshot(client)
            try:
                result = pipeline.run(row["query"], state)
                predictions.append(
                    {
                        "id": row["id"],
                        "task": "reflect",
                        "mode": mode,
                        "status": "ok",
                        "expected_refuse": bool(row["should_refuse"]),
                        "predicted_refuse": result.action == "refuse",
                        "predicted_action": result.action,
                        "trace_status": result.trace.status,
                        "retrieval_attempts": len(
                            state.retrieval_state.get("attempts", [])
                        ),
                        **_token_delta(client, before),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-sample failures
                predictions.append(
                    _failure(
                        row,
                        "reflect",
                        exc,
                        mode=mode,
                        expected_refuse=bool(row["should_refuse"]),
                        **_token_delta(client, before),
                    )
                )
    return predictions


def _retained_memory_items(memory: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("entities", "constraints", "comparison_targets", "citations"):
        values.extend(str(item) for item in memory.get(field, []))
    for paper in memory.get("confirmed_papers", []):
        if isinstance(paper, Mapping):
            values.extend(
                str(paper[key]) for key in ("paper_id", "title") if paper.get(key)
            )
    return list(dict.fromkeys(values))


def run_memory_ablation(
    provider_config: Mapping[str, Any],
    retrieval: RetrievalConfig,
    client: Any,
    data_dir: Path,
    *,
    searcher: Any | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(data_dir) / "M.jsonl")
    predictions: list[dict[str, Any]] = []
    for mode in MEMORY_MODES:
        states: dict[str, AgentState] = {}
        runtime = _runtime_config(provider_config, retrieval)
        pipeline = ResearchPipeline(runtime, client=client, searcher=searcher)
        for row in rows:
            conversation_id = str(row["conversation_id"])
            state = states.setdefault(conversation_id, AgentState(memory_mode=mode))
            before = _token_snapshot(client)
            try:
                result = pipeline.run(row["query"], state)
                required = list(
                    dict.fromkeys(
                        [
                            *row["required_entities"],
                            *row["required_constraints"],
                            *row["comparison_targets"],
                            *row["ground_truth_chunk_ids"],
                        ]
                    )
                )
                predictions.append(
                    {
                        "id": row["id"],
                        "task": "memory",
                        "mode": mode,
                        "status": "ok",
                        "required_items": required,
                        "retained_items": _retained_memory_items(state.research_memory),
                        "context_chars": len(
                            json.dumps(state.research_memory, ensure_ascii=False)
                        ),
                        "trace_status": result.trace.status,
                        **_token_delta(client, before),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-sample failures
                predictions.append(
                    _failure(
                        row,
                        "memory",
                        exc,
                        mode=mode,
                        **_token_delta(client, before),
                    )
                )
    return predictions


def run_provider_suites(
    provider_config: Mapping[str, Any],
    retrieval: RetrievalConfig,
    data_dir: Path,
    *,
    client: Any | None = None,
    searcher: Any | None = None,
) -> list[dict[str, Any]]:
    active_client = client or NativeProviderClient(provider_config)
    active_searcher = searcher or HybridSearcher(retrieval)
    predictions: list[dict[str, Any]] = []
    predictions.extend(run_intent(active_client, data_dir))
    predictions.extend(run_rewrite(active_client, data_dir))
    predictions.extend(run_boundary(active_client, data_dir))
    predictions.extend(
        run_reflect_ablation(
            provider_config,
            retrieval,
            active_client,
            data_dir,
            searcher=active_searcher,
        )
    )
    predictions.extend(
        run_e2e(
            provider_config,
            retrieval,
            active_client,
            data_dir,
            searcher=active_searcher,
        )
    )
    predictions.extend(
        run_memory_ablation(
            provider_config,
            retrieval,
            active_client,
            data_dir,
            searcher=active_searcher,
        )
    )
    return predictions


def summarize_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate every metric only from persisted per-sample prediction fields."""
    summary: dict[str, Any] = {}
    for mode in RETRIEVAL_MODES:
        rows = [
            row
            for row in predictions
            if row.get("task") == "retrieval" and row.get("mode") == mode
        ]
        if rows:
            summary[f"retrieval_{mode}"] = recall_at_k(rows)
    intent_rows = [row for row in predictions if row.get("task") == "intent"]
    if intent_rows:
        summary["intent"] = intent_metrics(intent_rows, list(INTENT_COUNTS))
    rewrite_rows = [row for row in predictions if row.get("task") == "rewrite"]
    if rewrite_rows:
        summary["rewrite"] = rewrite_metrics(rewrite_rows)
    boundary_rows = [row for row in predictions if row.get("task") == "boundary"]
    if boundary_rows:
        summary["guardrails"] = classification_accuracy(
            boundary_rows, "expected_action", "predicted_action"
        )
    for mode in ("off", "on"):
        rows = [
            row
            for row in predictions
            if row.get("task") == "reflect" and row.get("mode") == mode
        ]
        if rows:
            summary[f"reflect_{mode}"] = classification_accuracy(
                rows, "expected_refuse", "predicted_refuse"
            )
    e2e_rows = [row for row in predictions if row.get("task") == "e2e"]
    if e2e_rows:
        summary["citations"] = citation_metrics(e2e_rows)
        summary["pipeline"] = pipeline_success(e2e_rows)
    memory_rows = [row for row in predictions if row.get("task") == "memory"]
    if memory_rows:
        summary["memory"] = memory_consistency(memory_rows)
        summary["token_overhead"] = token_overhead(memory_rows)
    summary["prediction_statuses"] = status_counts(predictions)
    return summary


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _flatten_scalars(
    value: Mapping[str, Any], prefix: str = ""
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            rows.extend(_flatten_scalars(item, name))
        elif not isinstance(item, list):
            rows.append((name, item))
    return rows


def write_results(
    output_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> None:
    """Write JSONL predictions plus machine-readable and Markdown summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_sample.jsonl", predictions)
    _json(output_dir / "summary.json", summary)
    _json(output_dir / "run_config.json", run_config)
    scalar_rows: list[tuple[str, str, Any]] = []
    for experiment, values in summary.items():
        if isinstance(values, Mapping):
            scalar_rows.extend(
                (experiment, metric, value)
                for metric, value in _flatten_scalars(values)
            )
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "metric", "value"])
        writer.writerows(scalar_rows)
    lines = [
        "# Evaluation results",
        "",
        f"Dataset provenance: `{run_config.get('dataset_provenance', 'unknown')}`",
        "",
        "| Experiment | Metric | Value |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| {experiment} | {metric} | {value} |"
        for experiment, metric, value in scalar_rows
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the research-paper agent on corpus-grounded JSONL data"
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--data-dir", type=Path, default=Path("eval/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results/latest"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model")
    parser.add_argument(
        "--suite",
        choices=("datasets", "retrieval", "provider", "all"),
        default="retrieval",
    )
    parser.add_argument("--build-datasets", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    retrieval = RetrievalConfig(
        corpus_dir=args.corpus_dir, device=args.device, random_seed=args.seed
    )
    if args.build_datasets or any(
        not (args.data_dir / f"{name}.jsonl").exists() for name in DATASET_NAMES
    ):
        dataset_stats = build_all_datasets(
            retrieval.chunks_path, args.data_dir, args.seed
        )
    else:
        dataset_stats = validate_datasets(args.data_dir, retrieval.chunks_path)
    provider_config = load_config()
    if args.model:
        provider_config["model"] = args.model.replace(":", "/", 1)

    predictions: list[dict[str, Any]] = []
    if args.suite in {"retrieval", "all"}:
        predictions.extend(run_retrieval(retrieval, args.data_dir))
    if args.suite in {"provider", "all"}:
        if not has_api_key(provider_config):
            provider = detect_provider(provider_config["model"])
            raise EvaluationConfigurationError(
                f"provider suite requires credentials for {provider}"
            )
        predictions.extend(
            run_provider_suites(provider_config, retrieval, args.data_dir)
        )

    summary = summarize_predictions(predictions)
    manifest: dict[str, Any] = {}
    if retrieval.manifest_path.exists() or args.suite != "datasets":
        manifest = load_manifest(retrieval.manifest_path, retrieval.index_version)
    run_config = {
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "random_seed": args.seed,
        "suite": args.suite,
        "model": provider_config.get("model"),
        "provider": detect_provider(str(provider_config.get("model", ""))),
        "python": sys.version,
        "platform": platform.platform(),
        "retrieval_config": retrieval.to_dict(),
        "index_manifest": manifest,
        "dataset_stats": dataset_stats,
        "dataset_provenance": PROVENANCE,
    }
    write_results(args.output_dir, predictions, summary, run_config)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )
    failures = status_counts(predictions)["failed"]
    return 1 if failures else 0


def cli(argv: list[str] | None = None) -> int:
    """Render expected configuration/data failures without a traceback."""
    try:
        return main(argv)
    except (EvaluationConfigurationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
