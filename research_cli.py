"""Research-specific slash commands independent from terminal rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pipeline.memory import MEMORY_MODES, ResearchMemory
from retrieval.chunking import load_chunks
from retrieval.config import RetrievalConfig
from retrieval.manifest import index_status


def _trace_dict(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else None
    return dict(value) if isinstance(value, Mapping) else None


def trace_payload(state: Any, selector: str = "") -> dict[str, Any]:
    """Return the latest or selected complete turn trace."""
    traces = list(getattr(state, "turn_traces", []) or [])
    if not traces:
        return {"error": "no turn traces are available"}
    requested = (selector or "").strip()
    if not requested:
        payload = _trace_dict(traces[-1])
        return payload or {"error": "latest turn trace is invalid"}

    selected = None
    if requested.isdigit():
        turn_index = int(requested)
        selected = next(
            (
                item
                for item in traces
                if getattr(item, "turn_index", None) == turn_index
                or (
                    isinstance(item, Mapping)
                    and item.get("turn_index") == turn_index
                )
            ),
            None,
        )
    if selected is None:
        selected = next(
            (
                item
                for item in traces
                if getattr(item, "turn_id", None) == requested
                or (isinstance(item, Mapping) and item.get("turn_id") == requested)
            ),
            None,
        )
    if selected is None:
        return {"error": f"turn '{requested}' was not found"}
    return _trace_dict(selected) or {"error": f"turn '{requested}' is invalid"}


def _empty_corpus_payload(config: Mapping[str, Any] | None, error: str) -> dict[str, Any]:
    retrieval_values = dict(config or {}).get("retrieval", {})
    return {
        "status": "error",
        "paper_count": 0,
        "chunk_count": 0,
        "index_ready": False,
        "index_stale": True,
        "index_files": {},
        "error": error,
        "dense_model": (
            retrieval_values.get("dense_model")
            if isinstance(retrieval_values, Mapping)
            else None
        ),
        "reranker_model": (
            retrieval_values.get("reranker_model")
            if isinstance(retrieval_values, Mapping)
            else None
        ),
        "device": (
            retrieval_values.get("device")
            if isinstance(retrieval_values, Mapping)
            else None
        ),
        "corpus_dir": None,
    }


def corpus_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize corpus counts, index health, and retrieval model configuration."""
    try:
        retrieval = RetrievalConfig.from_mapping(config)
        health = index_status(retrieval)
    except (OSError, TypeError, ValueError) as exc:
        return _empty_corpus_payload(config, f"{type(exc).__name__}: {exc}")

    manifest = health.get("manifest", {})
    paper_count = manifest.get("paper_count", 0)
    chunk_count = manifest.get("chunk_count", 0)
    error = health.get("error")
    if retrieval.chunks_path.exists() and not chunk_count:
        try:
            chunks = load_chunks(retrieval.chunks_path)
            chunk_count = len(chunks)
            paper_count = len({str(item["paper_id"]) for item in chunks})
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"

    files = dict(health.get("files", {}))
    if error:
        status = "error"
    elif health.get("ready"):
        status = "ready"
    elif any(files.values()):
        status = "stale" if health.get("stale") else "incomplete"
    else:
        status = "missing"
    portable = retrieval.to_dict()
    return {
        "status": status,
        "paper_count": int(paper_count or 0),
        "chunk_count": int(chunk_count or 0),
        "index_ready": bool(health.get("ready")),
        "index_stale": bool(health.get("stale")),
        "index_files": files,
        "error": error,
        "index_version": manifest.get("index_version", retrieval.index_version),
        "dense_model": retrieval.dense_model,
        "reranker_model": retrieval.reranker_model,
        "device": retrieval.device,
        "corpus_dir": portable["corpus_dir"],
    }


def memory_payload(state: Any) -> dict[str, Any]:
    """Return structured memory together with compression diagnostics."""
    mode = getattr(state, "memory_mode", "compressed")
    raw = getattr(state, "research_memory", {})
    try:
        memory = ResearchMemory.from_dict(raw, mode)
    except (TypeError, ValueError) as exc:
        return {
            "mode": mode,
            "compressed": False,
            "estimated_tokens": 0,
            "update_count": 0,
            "compression_count": 0,
            "memory": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    value = memory.to_dict()
    return {
        "mode": memory.mode,
        "compressed": memory.compressed,
        "estimated_tokens": memory.estimated_tokens(),
        "update_count": memory.update_count,
        "compression_count": memory.compression_count,
        "memory": value,
        "error": None,
    }


def _print_payload(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True))


def cmd_trace(args: str, state: Any, _config: Mapping[str, Any]) -> bool:
    _print_payload(trace_payload(state, args))
    return True


def cmd_corpus(_args: str, _state: Any, config: Mapping[str, Any]) -> bool:
    _print_payload(corpus_payload(config))
    return True


def cmd_memory(args: str, state: Any, _config: Mapping[str, Any]) -> bool:
    requested = (args or "").strip().lower()
    if requested:
        if requested not in MEMORY_MODES:
            print("Error: memory mode must be off, on, or compressed")
            return True
        state.memory_mode = requested
        if requested == "off":
            state.research_memory = {}
    _print_payload(memory_payload(state))
    return True
