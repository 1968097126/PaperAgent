"""Pure evaluation metrics with explicit failure, skip, and unevaluable counts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

EVALUATION_STATUSES = ("ok", "failed", "skipped", "unevaluable")


def status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count all declared statuses; absent/unknown values are failures."""
    counts = {status: 0 for status in EVALUATION_STATUSES}
    for row in rows:
        status = row.get("status", "failed")
        if status not in counts:
            status = "failed"
        counts[str(status)] += 1
    return counts


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = status_counts(rows)
    return {
        "evaluated": counts["ok"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "unevaluable": counts["unevaluable"],
        "status_counts": counts,
    }


def _ok_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def intent_metrics(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    """Compute exact intent accuracy and a gold-by-prediction confusion matrix."""
    label_list = list(dict.fromkeys(labels))
    predicted_labels = [*label_list, "__unknown__"]
    matrix = {
        gold: {predicted: 0 for predicted in predicted_labels} for gold in label_list
    }
    correct = 0
    for row in _ok_rows(rows):
        gold = str(row.get("expected", ""))
        predicted = str(row.get("predicted", ""))
        if gold not in matrix:
            matrix[gold] = {item: 0 for item in predicted_labels}
        bucket = predicted if predicted in label_list else "__unknown__"
        matrix[gold][bucket] += 1
        correct += gold == predicted
    coverage = _coverage(rows)
    evaluated = coverage["evaluated"]
    return {
        "accuracy": correct / evaluated if evaluated else None,
        "confusion_matrix": matrix,
        **coverage,
    }


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", (text or "").casefold()).strip()


def _contains_requirement(output: str, requirement: str) -> bool:
    normalized_requirement = _normalize(requirement)
    return bool(
        normalized_requirement
        and f" {normalized_requirement} " in f" {output} "
    )


def rewrite_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score whether a rewrite preserves every annotated entity/reference/constraint."""
    details: list[dict[str, Any]] = []
    for row in rows:
        status = row.get("status", "failed")
        if status != "ok":
            details.append({"id": row.get("id"), "status": status})
            continue
        output = _normalize(str(row.get("rewritten_query", "")))
        required = list(
            dict.fromkeys(
                str(item)
                for field in (
                    "required_entities",
                    "required_references",
                    "required_constraints",
                )
                for item in row.get(field, [])
            )
        )
        missing = [item for item in required if not _contains_requirement(output, item)]
        details.append(
            {
                "id": row.get("id"),
                "status": "ok",
                "correct": not missing,
                "missing": missing,
            }
        )
    evaluated_details = [item for item in details if item["status"] == "ok"]
    coverage = _coverage(rows)
    return {
        "accuracy": (
            sum(bool(item["correct"]) for item in evaluated_details)
            / len(evaluated_details)
            if evaluated_details
            else None
        ),
        "details": details,
        **coverage,
    }


def recall_at_k(
    rows: Sequence[Mapping[str, Any]], ks: Iterable[int] = (1, 5, 10)
) -> dict[str, Any]:
    """Return query-level hit recall: any relevant chunk in the first K is a hit."""
    k_values = tuple(dict.fromkeys(ks))
    if not k_values or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in k_values):
        raise ValueError("ks must contain positive integers")
    totals = {k: 0 for k in k_values}
    eligible: list[Mapping[str, Any]] = []
    adjusted: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        truth = {
            str(item) for item in row.get("ground_truth_chunk_ids", []) if str(item)
        }
        if row.get("status") == "ok" and not truth:
            copy["status"] = "unevaluable"
        adjusted.append(copy)
        if copy.get("status") != "ok":
            continue
        eligible.append(copy)
        ranked = [str(item) for item in copy.get("retrieved_chunk_ids", [])]
        for k in k_values:
            totals[k] += bool(truth.intersection(ranked[:k]))
    coverage = _coverage(adjusted)
    return {
        **{
            f"recall@{k}": totals[k] / len(eligible) if eligible else None
            for k in k_values
        },
        **coverage,
    }


def classification_accuracy(
    rows: Sequence[Mapping[str, Any]], expected_key: str, predicted_key: str
) -> dict[str, Any]:
    """Accuracy for boundary/refusal decisions with standard coverage fields."""
    valid = _ok_rows(rows)
    coverage = _coverage(rows)
    return {
        "accuracy": (
            sum(row.get(expected_key) == row.get(predicted_key) for row in valid)
            / len(valid)
            if valid
            else None
        ),
        **coverage,
    }


def citation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute citation validity plus ground-truth precision and recall.

    Duplicate occurrences remain in the denominator and only their first occurrence
    can be valid/correct, so repeating a citation cannot improve the scores.
    """
    valid_count = 0
    predicted_count = 0
    true_positive = 0
    relevant_count = 0
    for row in _ok_rows(rows):
        predicted = [str(item) for item in row.get("predicted_citations", [])]
        truth = {str(item) for item in row.get("ground_truth_chunk_ids", [])}
        valid = {str(item) for item in row.get("valid_evidence_chunk_ids", [])}
        seen: set[str] = set()
        for citation in predicted:
            predicted_count += 1
            if citation in seen:
                continue
            seen.add(citation)
            valid_count += citation in valid
            true_positive += citation in truth
        relevant_count += len(truth)
    coverage = _coverage(rows)
    return {
        "validity": valid_count / predicted_count if predicted_count else None,
        "precision": true_positive / predicted_count if predicted_count else None,
        "recall": true_positive / relevant_count if relevant_count else None,
        **coverage,
    }


def pipeline_success(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Conjunction of routing, retrieval, answer/refusal, and citation correctness."""
    details: list[dict[str, Any]] = []
    keys = ("routing_ok", "retrieval_ok", "answer_or_refusal_ok", "citations_ok")
    for row in _ok_rows(rows):
        missing = [key for key in keys if key not in row]
        details.append(
            {
                "id": row.get("id"),
                "success": not missing and all(bool(row[key]) for key in keys),
                "missing_components": missing,
            }
        )
    coverage = _coverage(rows)
    return {
        "success_rate": (
            sum(bool(item["success"]) for item in details) / len(details)
            if details
            else None
        ),
        "details": details,
        **coverage,
    }


def memory_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure required-item retention independently for all three memory modes."""
    results: dict[str, Any] = {}
    for mode in ("off", "on", "compressed"):
        mode_rows = [row for row in rows if row.get("mode") == mode]
        valid = _ok_rows(mode_rows)
        scores: list[float] = []
        for row in valid:
            required = {str(item) for item in row.get("required_items", [])}
            retained = {str(item) for item in row.get("retained_items", [])}
            scores.append(
                len(required.intersection(retained)) / len(required)
                if required
                else 1.0
            )
        results[mode] = {
            "consistency": sum(scores) / len(scores) if scores else None,
            **_coverage(mode_rows),
        }
    return results


def token_overhead(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average provider tokens/context by mode and deltas versus memory-off."""
    results: dict[str, Any] = {}
    for mode in ("off", "on", "compressed"):
        mode_rows = [row for row in rows if row.get("mode") == mode]
        valid = _ok_rows(mode_rows)
        input_values = [int(row.get("input_tokens", 0) or 0) for row in valid]
        output_values = [int(row.get("output_tokens", 0) or 0) for row in valid]
        context_values = [int(row.get("context_chars", 0) or 0) for row in valid]
        totals = [left + right for left, right in zip(input_values, output_values)]
        results[mode] = {
            "avg_input_tokens": sum(input_values) / len(valid) if valid else None,
            "avg_output_tokens": sum(output_values) / len(valid) if valid else None,
            "avg_total_tokens": sum(totals) / len(valid) if valid else None,
            "avg_context_chars": sum(context_values) / len(valid) if valid else None,
            **_coverage(mode_rows),
        }
    baseline = results["off"]["avg_total_tokens"]
    for mode in ("on", "compressed"):
        current = results[mode]["avg_total_tokens"]
        results[mode]["total_token_delta_vs_off"] = (
            current - baseline if current is not None and baseline is not None else None
        )
        results[mode]["total_token_ratio_vs_off"] = (
            current / baseline
            if current is not None and baseline not in {None, 0}
            else None
        )
    return results
