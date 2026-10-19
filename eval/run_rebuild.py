"""Resumable real-model experiments for the frozen 20-paper corpus.

Independent dialogues run concurrently; turns within a dialogue remain ordered.
Only public paper excerpts and test questions are sent to the selected provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from agent import AgentState
from eval.datasets import read_jsonl, validate_datasets, write_jsonl
from eval.metrics import classification_accuracy
from eval.run_evaluation import (
    _pipeline_checks,
    _retained_memory_items,
    _runtime_config,
    _token_delta,
    _token_snapshot,
    run_boundary,
    run_intent,
    run_rewrite,
    summarize_predictions,
    write_results,
)
from local_env import load_local_env
from pipeline.client import NativeProviderClient
from pipeline.memory import ResearchMemory
from pipeline.orchestrator import ResearchPipeline
from retrieval.config import RetrievalConfig
from retrieval.hybrid_search import HybridSearcher
from retrieval.model_versions import MODEL_REVISIONS

MODEL_CONFIG = {
    "model": "deepseek-flash",
    "max_tokens": 2048,
    "temperature": 0,
    "thinking": False,
}


class CachedClient(NativeProviderClient):
    """Persist successful stage calls so interruptions do not repeat paid work."""

    def __init__(self, directory):
        super().__init__(MODEL_CONFIG)
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def complete(self, stage, prompt):
        key = hashlib.sha256(
            json.dumps([MODEL_CONFIG, stage, prompt], ensure_ascii=False).encode()
        ).hexdigest()
        path = self.directory / f"{key}.json"
        if path.exists():
            value = json.loads(path.read_text())
            self.input_tokens += value["input_tokens"]
            self.output_tokens += value["output_tokens"]
            return value["text"]
        before = _token_snapshot(self)
        text = super().complete(stage, prompt)
        path.write_text(
            json.dumps({"text": text, **_token_delta(self, before)}, ensure_ascii=False)
        )
        return text


class CachedSearcher:
    """Serialize CPU model inference and reuse identical retrieval requests."""

    def __init__(self, config):
        self.base = HybridSearcher(config)
        self.lock = threading.RLock()
        self.cache = {}

    def candidates(self, query, **kwargs):
        key = ("candidates", query, json.dumps(kwargs, sort_keys=True))
        with self.lock:
            if key not in self.cache:
                self.cache[key] = self.base.candidates(query, **kwargs)
            return [dict(x) for x in self.cache[key]]

    def rerank_candidates(self, query, candidates, top_k=None):
        key = ("rerank", query, tuple(x["chunk_id"] for x in candidates), top_k)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = self.base.rerank_candidates(query, candidates, top_k)
            return [dict(x) for x in self.cache[key]]

    def search(self, query, top_k=10, mode="hybrid_rerank"):
        candidates = self.candidates(
            query, mode="dense" if mode == "dense" else "hybrid"
        )
        return (
            self.rerank_candidates(query, candidates, top_k)
            if mode == "hybrid_rerank"
            else candidates[:top_k]
        )


class FirstPassRemoval:
    """Controlled retry probe: hide the gold paper only in the first retrieval."""

    def __init__(self, searcher, excluded):
        self.searcher = searcher
        self.excluded = set(excluded)
        self.calls = 0

    def candidates(self, query, **kwargs):
        self.calls += 1
        hits = self.searcher.candidates(query, **kwargs)
        return [
            h for h in hits if self.calls != 1 or h["paper_id"] not in self.excluded
        ]

    def rerank_candidates(self, *args, **kwargs):
        return self.searcher.rerank_candidates(*args, **kwargs)


class NoReflection:
    def run(self, query, evidence, attempt=1, max_second_retrievals=0):
        sufficient = bool(evidence)
        return {
            "sufficient": sufficient,
            "action": "answer" if sufficient else "refuse",
            "missing_information": "" if sufficient else "no evidence",
            "rewritten_query": "",
            "reason": "Reflect disabled for ablation",
            "source": "ablation",
        }


class ComponentGuard:
    """Hold routing fixed when isolating retrieval and reflection effects."""

    def run(self, user_message, memory, history=()):
        return {
            "action": "allow",
            "reason": "controlled component experiment",
            "source": "experiment_control",
        }


class ComponentNLU:
    def __init__(self, row):
        self.row = row

    def run(self, user_message, history, memory):
        return {
            "intent": self.row["intent"],
            "standalone_query": self.row["standalone_query"],
            "resolved_references": self.row["resolved_references"],
            "constraints": self.row["required_constraints"],
            "comparison_targets": self.row["comparison_targets"],
            "needs_retrieval": True,
            "reason": "fixed annotated Query for component ablation",
            "source": "experiment_control",
        }


def dialogue(rows, mode, client, searcher, config, e_by_source):
    state = AgentState(memory_mode=mode)
    pipeline = ResearchPipeline(
        _runtime_config(MODEL_CONFIG, config), client=client, searcher=searcher
    )
    predictions = []
    for row in rows:
        before = _token_snapshot(client)
        result = pipeline.run(row["query"], state)
        checks = _pipeline_checks(row, result, state)
        required = list(
            dict.fromkeys(
                row["required_entities"]
                + row["required_constraints"]
                + row["comparison_targets"]
                + row["ground_truth_chunk_ids"]
            )
        )
        predictions.append(
            dict(
                id=row["id"],
                task="memory",
                mode=mode,
                status="ok",
                required_items=required,
                retained_items=_retained_memory_items(state.research_memory),
                context_chars=len(
                    json.dumps(
                        ResearchMemory.from_dict(state.research_memory, mode).context(),
                        ensure_ascii=False,
                    )
                ),
                compression_count=state.research_memory.get("compression_count", 0),
                **checks,
                **_token_delta(client, before),
            )
        )
        if mode == "compressed" and row["id"] in e_by_source:
            predictions.append(
                dict(
                    id=e_by_source[row["id"]]["id"],
                    source_turn_id=row["id"],
                    task="e2e",
                    status="ok",
                    **checks,
                    **_token_delta(client, before),
                )
            )
    return predictions


def execute_unit(unit, output, config, searcher, e_by_source):
    name, kind, rows, mode = unit
    target = output / "checkpoints" / f"{name}.json"
    if target.exists():
        return json.loads(target.read_text())
    client = CachedClient(output / "calls" / name)
    if kind in {"intent", "rewrite", "boundary"}:
        data = output / "inputs" / name
        filename = {"intent": "I", "rewrite": "M", "boundary": "B"}[kind]
        write_jsonl(data / f"{filename}.jsonl", rows)
        predictions = {
            "intent": run_intent,
            "rewrite": run_rewrite,
            "boundary": run_boundary,
        }[kind](client, data)
    elif kind == "dialogue":
        predictions = dialogue(rows, mode, client, searcher, config, e_by_source)
    elif kind == "retrieval":
        row = rows[0]
        hits = searcher.search(row["query"], top_k=10, mode=mode)
        predictions = [
            {
                "id": row["id"],
                "task": "retrieval",
                "mode": mode,
                "status": "ok",
                "ground_truth_chunk_ids": row["ground_truth_chunk_ids"],
                "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            }
        ]
    else:
        row = rows[0]
        excluded = (
            [c.split(":")[0] for c in row["ground_truth_chunk_ids"]]
            if row["requires_second_retrieval"]
            else []
        )
        probe = FirstPassRemoval(searcher, excluded)
        pipeline = ResearchPipeline(
            _runtime_config(MODEL_CONFIG, config), client=client, searcher=probe
        )
        pipeline.guardrails = ComponentGuard()
        pipeline.nlu = ComponentNLU(row)
        if mode == "off":
            pipeline.refiner.reflect = NoReflection()
            pipeline.refiner.max_second_retrievals = 0
        elif mode == "no_retry":
            pipeline.refiner.max_second_retrievals = 0
        state = AgentState(messages=list(row["history"]))
        before = _token_snapshot(client)
        result = pipeline.run(row["query"], state)
        predictions = [
            dict(
                id=row["id"],
                task="reflect",
                mode=mode,
                status="ok",
                expected_refuse=row["should_refuse"],
                predicted_refuse=result.action == "refuse",
                retrieval_attempts=len(state.retrieval_state.get("attempts", [])),
                intervention="first_pass_gold_paper_removal" if excluded else "none",
                query_control="fixed_gold_annotation",
                **_pipeline_checks(row, result, state),
                **_token_delta(client, before),
            )
        ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(predictions, ensure_ascii=False))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=["components", "retrieval", "pipeline", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval/results/rebuild-20261019")
    )
    args = parser.parse_args()
    load_local_env()
    stats = validate_datasets(Path("eval/data"), Path("corpus/chunks.jsonl"))
    config = RetrievalConfig(
        cache_dir=Path(".cache/models"), batch_size=8, device=args.device
    )
    searcher = CachedSearcher(config) if args.suite != "components" else None
    datasets = {
        n: read_jsonl(Path("eval/data") / f"{n}.jsonl")
        for n in ["M", "I", "R", "B", "E"]
    }
    e_by_source = {r["source_turn_id"]: r for r in datasets["E"]}
    units = []
    if args.suite in {"components", "all"}:
        for name, rows in [
            ("intent", datasets["I"]),
            ("rewrite", [r for r in datasets["M"] if r["requires_rewrite"]]),
            ("boundary", datasets["B"]),
        ]:
            units.extend((f"{name}-{r['id']}", name, [r], None) for r in rows)
    if args.suite in {"retrieval", "all"}:
        for mode in ["dense", "hybrid", "hybrid_rerank"]:
            units.extend(
                (f"retrieval-{mode}-{r['id']}", "retrieval", [r], mode)
                for r in datasets["R"]
            )
    if args.suite in {"pipeline", "all"}:
        groups = defaultdict(list)
        for row in datasets["M"]:
            groups[row["conversation_id"]].append(row)
        for mode in ["off", "on", "compressed"]:
            units.extend(
                (f"dialogue-{mode}-{key}", "dialogue", rows, mode)
                for key, rows in groups.items()
            )
        probes = [
            r
            for r in datasets["M"]
            if r["requires_second_retrieval"]
            or r["refusal_reason"] == "insufficient_evidence"
        ]
        for mode in ["off", "no_retry", "on"]:
            units.extend(
                (f"reflect-{mode}-{r['id']}", "reflect", [r], mode) for r in probes
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                execute_unit, u, args.output_dir, config, searcher, e_by_source
            ): u[0]
            for u in units
        }
        for index, future in enumerate(as_completed(futures), 1):
            predictions.extend(future.result())
            if index % 10 == 0 or index == len(units):
                print(f"{args.suite}: {index}/{len(units)} units", flush=True)
    # Include completed suites when invoked separately, retaining one row per task/mode/id.
    all_rows = []
    for path in sorted((args.output_dir / "checkpoints").glob("*.json")):
        all_rows.extend(json.loads(path.read_text()))
    summary = summarize_predictions(all_rows)
    # Report requested-sample accuracy as well, so parse/API failures cannot inflate results.
    for metrics in summary.values():
        if isinstance(metrics, dict) and metrics.get("accuracy") is not None:
            requested = sum(metrics.get("status_counts", {}).values())
            if requested:
                metrics["accuracy_all_requested"] = (
                    metrics["accuracy"] * metrics.get("evaluated", 0) / requested
                )
    no_retry = [
        r for r in all_rows if r["task"] == "reflect" and r["mode"] == "no_retry"
    ]
    if no_retry:
        summary["reflect_no_retry"] = classification_accuracy(
            no_retry, "expected_refuse", "predicted_refuse"
        )
    for mode in ["off", "on", "compressed"]:
        rows = [r for r in all_rows if r["task"] == "memory" and r["mode"] == mode]
        if rows:
            summary[f"memory_answer_{mode}"] = {
                "turns": len(rows),
                "action_accuracy": sum(r["answer_or_refusal_ok"] for r in rows)
                / len(rows),
                "pipeline_success": sum(
                    all(
                        r[k]
                        for k in [
                            "routing_ok",
                            "retrieval_ok",
                            "answer_or_refusal_ok",
                            "citations_ok",
                        ]
                    )
                    and r["trace_status"] == "ok"
                    for r in rows
                )
                / len(rows),
                "turns_after_compression": sum(
                    r["compression_count"] > 0 for r in rows
                ),
            }
    memory_tokens = {
        mode: sum(
            r.get("input_tokens", 0) + r.get("output_tokens", 0)
            for r in all_rows
            if r["task"] == "memory" and r["mode"] == mode
        )
        for mode in ["on", "compressed"]
    }
    if memory_tokens["on"] and memory_tokens["compressed"]:
        summary["compression_tokens"] = {
            **memory_tokens,
            "saving_vs_uncompressed": 1
            - memory_tokens["compressed"] / memory_tokens["on"],
        }
    run_config = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL_CONFIG,
        "dataset_stats": stats,
        "dataset_provenance": "model-assisted; verbatim spans checked; human review pending",
        "retrieval_config": config.to_dict(),
        "model_revisions": MODEL_REVISIONS,
        "dataset_sha256": {
            name: hashlib.sha256(
                (Path("eval/data") / f"{name}.jsonl").read_bytes()
            ).hexdigest()
            for name in datasets
        },
        "corpus_selection_sha256": hashlib.sha256(
            Path("corpus/selection.json").read_bytes()
        ).hexdigest(),
        "e2e_protocol": "70 scored turns in 30 continuous four-turn dialogues",
        "reflect_protocol": "fixed annotated Query, bypass Guardrails/NLU only for component ablation; off / reflect without retry / reflect with one retry; first-pass gold-paper removal on 24 probes",
        "token_accounting": "logical input/output tokens, replayed calls included; call files retain provider usage",
    }
    write_results(args.output_dir, all_rows, summary, run_config)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
