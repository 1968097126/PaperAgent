"""Research-domain ToolDefs; only search/retrieve are exposed to conversations."""
from __future__ import annotations

import json

from retrieval.config import RetrievalConfig
from tool_registry import ToolDef, register_tool

SEARCH_SCHEMA = {
    "name": "search_papers",
    "description": "Search local papers with dense + BM25 + RRF + Cross-Encoder retrieval.",
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            "mode": {"type": "string", "enum": ["dense", "hybrid", "hybrid_rerank"]},
            "filters": {"type": "object"},
        },
        "required": ["query"],
    },
}
RETRIEVE_SCHEMA = {
    "name": "retrieve_chunk",
    "description": "Retrieve one real chunk and optional neighboring chunks.",
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "chunk_id": {"type": "string", "minLength": 1},
            "context": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "required": ["chunk_id"],
    },
}
RERANK_SCHEMA = {
    "name": "rerank",
    "description": "Rerank explicit chunk candidates for offline diagnostics.",
    "input_schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "candidates": {"type": "array"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["query", "candidates"],
    },
}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _search(params: dict, runtime: dict) -> str:
    from retrieval.hybrid_v2 import search_papers
    config = RetrievalConfig.from_mapping(runtime)
    results = search_papers(
        params["query"].strip(), params.get("top_k", config.result_top_k),
        config=config, mode=params.get("mode", "hybrid_rerank"), filters=params.get("filters"),
    )
    return _json({"query": params["query"].strip(), "results": results})


def _retrieve(params: dict, runtime: dict) -> str:
    from retrieval.hybrid_v2 import retrieve_chunk
    result = retrieve_chunk(
        params["chunk_id"].strip(), params.get("context", 1),
        config=RetrievalConfig.from_mapping(runtime),
    )
    if result is None:
        return f"Error: chunk_id '{params['chunk_id']}' not found"
    return _json(result)


def _rerank(params: dict, runtime: dict) -> str:
    from retrieval.rerank_v2 import rerank
    candidates = params["candidates"]
    if not all(isinstance(item, dict) and {"chunk_id", "text"} <= set(item) for item in candidates):
        return "Error: each candidate must contain chunk_id and text"
    config = RetrievalConfig.from_mapping(runtime)
    ranked = rerank(
        params["query"], [(item["chunk_id"], item["text"]) for item in candidates],
        params.get("top_k", config.result_top_k), model_name=config.reranker_model,
        device=config.device, cache_dir=config.cache_dir,
    )
    return _json({"results": [{"chunk_id": chunk_id, "score": score} for chunk_id, score in ranked]})


def register_domain_tools() -> None:
    register_tool(ToolDef("search_papers", SEARCH_SCHEMA, _search, read_only=True, concurrent_safe=True))
    register_tool(ToolDef("retrieve_chunk", RETRIEVE_SCHEMA, _retrieve, read_only=True, concurrent_safe=True))
    register_tool(ToolDef("rerank", RERANK_SCHEMA, _rerank, read_only=True, concurrent_safe=True))
