from __future__ import annotations

import json
from typing import Any

from agent import AgentState
from nano_claude import cmd_clear, cmd_config, cmd_load, cmd_save, handle_slash, repl
from pipeline.trace import TurnTrace
from research_cli import corpus_payload, memory_payload, trace_payload
from retrieval.chunking_v2 import write_chunks

EVIDENCE = {
    "chunk_id": "c1",
    "paper_id": "p1",
    "title": "Paper One",
    "section": "Method",
    "source": "fixture://p1",
    "chunk_index": 0,
    "text": "Verified evidence about a method.",
}


class ScriptedClient:
    def __init__(self):
        self.responses = {
            "Guardrails": [json.dumps({"action": "allow", "reason": "paper question"})],
            "NLU": [
                json.dumps(
                    {
                        "intent": "new_search",
                        "standalone_query": "method",
                        "resolved_references": [],
                        "needs_retrieval": True,
                        "reason": "new search",
                        "constraints": [],
                        "comparison_targets": [],
                    }
                )
            ],
            "Reflect": [
                json.dumps(
                    {
                        "sufficient": True,
                        "action": "answer",
                        "missing_information": "",
                        "rewritten_query": "",
                        "reason": "supported",
                    }
                )
            ],
            "Generate": ["The method is supported [p1/c1]"],
        }
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, stage: str, prompt: str) -> str:
        self.input_tokens += 10
        response = self.responses[stage].pop(0)
        self.output_tokens += 4
        return response


class FakeSearcher:
    def candidates(
        self,
        _query: str,
        *,
        mode: str,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        assert mode == "hybrid"
        assert filters is None
        return [dict(EVIDENCE)]

    def rerank_candidates(
        self,
        _query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return candidates[:top_k]


def _complete_trace(index: int) -> TurnTrace:
    trace = TurnTrace(index, f"question {index}", turn_id=f"turn-{index}")
    for stage in (
        "Guardrails",
        "NLU",
        "Retrieve",
        "Rerank",
        "Reflect",
        "Generate",
        "Memory",
    ):
        trace.record(stage, {"input": index}, {"output": stage}, 1.0)
    trace.finish()
    return trace


def test_trace_payload_selects_latest_index_and_turn_id() -> None:
    state = AgentState(turn_traces=[_complete_trace(0), _complete_trace(1)])

    assert trace_payload(state)["turn_id"] == "turn-1"
    assert trace_payload(state, "0")["turn_id"] == "turn-0"
    selected = trace_payload(state, "turn-1")
    assert [item["stage"] for item in selected["stages"]] == [
        "Guardrails",
        "NLU",
        "Retrieve",
        "Rerank",
        "Reflect",
        "Generate",
        "Memory",
    ]
    assert trace_payload(state, "missing") == {"error": "turn 'missing' was not found"}
    assert trace_payload(AgentState()) == {"error": "no turn traces are available"}


def test_corpus_payload_handles_missing_partial_and_corrupt_indexes(tmp_path) -> None:
    config = {"retrieval": {"corpus_dir": str(tmp_path), "dense_model": "dense-x"}}
    missing = corpus_payload(config)
    assert missing["status"] == "missing"
    assert missing["paper_count"] == 0
    assert missing["chunk_count"] == 0
    assert missing["index_ready"] is False
    assert missing["dense_model"] == "dense-x"
    assert not str(missing["corpus_dir"]).startswith("/")

    write_chunks([EVIDENCE], tmp_path / "chunks.jsonl")
    partial = corpus_payload(config)
    assert partial["status"] == "stale"
    assert partial["paper_count"] == 1
    assert partial["chunk_count"] == 1

    (tmp_path / "chunks.jsonl").write_text("{bad json}\n", encoding="utf-8")
    corrupt = corpus_payload(config)
    assert corrupt["status"] == "error"
    assert "JSONDecodeError" in corrupt["error"]


def test_corpus_payload_reports_ready_manifest(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "research_cli.index_status",
        lambda _config: {
            "ready": True,
            "stale": False,
            "files": {
                "chunks": True,
                "dense": True,
                "dense_ids": True,
                "sparse": True,
                "manifest": True,
            },
            "manifest": {"paper_count": 30, "chunk_count": 1800, "index_version": 2},
            "error": None,
        },
    )

    payload = corpus_payload({"retrieval": {"corpus_dir": str(tmp_path)}})

    assert payload["status"] == "ready"
    assert payload["paper_count"] == 30
    assert payload["chunk_count"] == 1800
    assert payload["index_version"] == 2


def test_memory_payload_and_command_cover_all_modes(capsys) -> None:
    state = AgentState(
        research_memory={"entities": ["DQN"], "update_count": 2},
        memory_mode="compressed",
    )

    payload = memory_payload(state)
    assert payload["mode"] == "compressed"
    assert payload["memory"]["entities"] == ["DQN"]
    assert payload["update_count"] == 2
    assert payload["estimated_tokens"] > 0

    assert handle_slash("/memory on", state, {}) is True
    assert state.memory_mode == "on"
    assert json.loads(capsys.readouterr().out)["mode"] == "on"

    assert handle_slash("/memory off", state, {}) is True
    assert state.memory_mode == "off"
    assert state.research_memory == {}
    assert json.loads(capsys.readouterr().out)["estimated_tokens"] == 0

    assert handle_slash("/memory invalid", state, {}) is True
    assert "Error: memory mode" in capsys.readouterr().out


def test_slash_commands_render_json_and_unknown_turn(capsys, tmp_path) -> None:
    state = AgentState(turn_traces=[_complete_trace(0)])
    config = {"retrieval": {"corpus_dir": str(tmp_path)}}

    assert handle_slash("/trace 0", state, config) is True
    assert json.loads(capsys.readouterr().out)["turn_id"] == "turn-0"
    assert handle_slash("/trace 99", state, config) is True
    assert "was not found" in json.loads(capsys.readouterr().out)["error"]
    assert handle_slash("/corpus", state, config) is True
    assert json.loads(capsys.readouterr().out)["status"] == "missing"


def test_mock_provider_repl_runs_pipeline_without_double_counting_tokens(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr("nano_claude.setup_readline", lambda _path: None)
    client = ScriptedClient()

    state = repl(
        {"model": "mock", "max_tokens": 1000, "research_mode": True},
        initial_prompt="请解释方法",
        research_client_override=client,
        research_searcher=FakeSearcher(),
    )

    output = capsys.readouterr().out
    assert "The method is supported [p1/c1]" in output
    assert state.turn_count == 1
    assert state.total_input_tokens == client.input_tokens == 40
    assert state.total_output_tokens == client.output_tokens == 16
    assert len(state.turn_traces) == 1
    assert state.citations == ["c1"]


def test_clear_resets_all_research_conversation_state() -> None:
    state = AgentState(
        messages=[{"role": "user", "content": "question"}],
        turn_count=1,
        turn_traces=[_complete_trace(0)],
        semantic_state={"intent": "new_search"},
        retrieval_state={"evidence": [EVIDENCE]},
        citations=["c1"],
        research_memory={"entities": ["DQN"]},
    )

    assert cmd_clear("", state, {}) is True
    assert state.messages == []
    assert state.turn_count == 0
    assert state.turn_traces == []
    assert state.semantic_state == {}
    assert state.retrieval_state == {}
    assert state.citations == []
    assert state.research_memory == {}


def test_session_round_trip_preserves_research_state(tmp_path, capsys) -> None:
    path = tmp_path / "research-session.json"
    source = AgentState(
        messages=[{"role": "user", "content": "compare methods"}],
        turn_count=2,
        total_input_tokens=123,
        total_output_tokens=45,
        semantic_state={"intent": "comparison"},
        retrieval_state={"evidence": [EVIDENCE]},
        citations=["c1"],
        research_memory={"entities": ["DQN"], "update_count": 1},
        memory_mode="on",
    )

    assert cmd_save(str(path), source, {}) is True
    restored = AgentState()
    assert cmd_load(str(path), restored, {}) is True
    capsys.readouterr()

    assert restored.messages == source.messages
    assert restored.turn_count == 2
    assert restored.total_input_tokens == 123
    assert restored.total_output_tokens == 45
    assert restored.semantic_state == source.semantic_state
    assert restored.retrieval_state == source.retrieval_state
    assert restored.citations == ["c1"]
    assert restored.research_memory == source.research_memory
    assert restored.memory_mode == "on"


def test_config_command_never_echoes_provider_keys(monkeypatch, capsys) -> None:
    monkeypatch.setattr("config.save_config", lambda _config: None)
    config = {"model": "gpt-5.6-luna", "openai_api_key": "top-secret"}

    cmd_config("", None, config)
    cmd_config("openai_api_key", None, config)
    cmd_config("deepseek_api_key=another-secret", None, config)
    output = capsys.readouterr().out

    assert "top-secret" not in output
    assert "another-secret" not in output
    assert output.count("<redacted>") == 3


def test_help_documents_research_commands(capsys) -> None:
    assert handle_slash("/help", AgentState(), {}) is True
    output = capsys.readouterr().out
    assert "/trace [id]" in output
    assert "/corpus" in output
    assert "/memory [mode]" in output
