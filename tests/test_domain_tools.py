from __future__ import annotations

import json

import retrieval.hybrid_search as hybrid
import tools  # noqa: F401 - importing registers built-in and domain ToolDefs
from tool_registry import execute_tool, get_tool_schemas


def test_domain_tool_schemas_and_invalid_arguments():
    schemas = get_tool_schemas(["search_papers", "retrieve_chunk"])
    assert [schema["name"] for schema in schemas] == ["search_papers", "retrieve_chunk"]
    assert execute_tool("search_papers", {}, {}).startswith("Error: invalid arguments")
    assert execute_tool("search_papers", {"query": "q", "top_k": True}, {}).startswith("Error: invalid arguments")
    assert execute_tool("retrieve_chunk", {"chunk_id": "x", "context": 11}, {}).startswith("Error: invalid arguments")


def test_search_tool_returns_stable_json(monkeypatch):
    monkeypatch.setattr(hybrid, "search_papers", lambda *args, **kwargs: [
        {"chunk_id": "c1", "paper_id": "p1", "title": "T", "section": "A",
         "source": "s", "chunk_index": 0, "text": "evidence", "score": 1.0}
    ])
    raw = execute_tool("search_papers", {"query": " q ", "top_k": 1}, {})
    value = json.loads(raw)
    assert value["query"] == "q"
    assert value["results"][0]["chunk_id"] == "c1"
    assert raw == json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_tool_exception_is_an_error_string(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(hybrid, "search_papers", broken)
    assert execute_tool("search_papers", {"query": "q"}, {}).startswith(
        "Error: tool 'search_papers' failed: RuntimeError: offline"
    )
