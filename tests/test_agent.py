"""Phase 3 tests for Agent state, tool scoping, permissions, and tracing."""
from __future__ import annotations

import pytest

import agent
from agent import (
    RESEARCH_TOOL_NAMES,
    AgentState,
    ToolEnd,
    _check_permission,
    run,
    run_research,
)
from providers import AssistantTurn


def _fake_stream_sequence(turns: list[AssistantTurn]):
    iterator = iter(turns)

    def _stream(**_kwargs):
        turn = next(
            iterator,
            AssistantTurn(text="done", tool_calls=[], in_tokens=1, out_tokens=1),
        )
        yield turn

    return _stream


BASE_CONFIG = {"model": "claude-opus-4-6", "permission_mode": "accept-all"}


class TestAgentState:
    def test_research_fields_have_isolated_defaults(self):
        first = AgentState()
        second = AgentState()

        first.semantic_state["intent"] = "new_search"
        first.citations.append("chunk-1")

        assert second.semantic_state == {}
        assert second.citations == []
        assert first.memory_mode == "compressed"


class TestCheckPermissionWhitelist:
    @pytest.mark.parametrize("name", RESEARCH_TOOL_NAMES)
    def test_research_read_tools_are_auto_approved(self, name):
        assert _check_permission(
            {"name": name, "input": {"query": "x"}},
            {"permission_mode": "auto"},
        ) is True

    def test_write_still_requires_confirmation_in_auto_mode(self):
        tool_call = {"name": "Write", "input": {"file_path": "x", "content": "y"}}
        assert _check_permission(tool_call, {"permission_mode": "auto"}) is False


class TestToolNameWhitelist:
    def test_research_entry_exposes_only_domain_tool_schemas(self, monkeypatch):
        observed = {}

        def capture_stream(**kwargs):
            observed["schemas"] = kwargs["tool_schemas"]
            yield AssistantTurn(text="done", tool_calls=[], in_tokens=1, out_tokens=1)

        monkeypatch.setattr(agent, "stream", capture_stream)
        list(run_research("paper question", AgentState(), BASE_CONFIG, "system"))

        assert tuple(schema["name"] for schema in observed["schemas"]) == RESEARCH_TOOL_NAMES

    def test_call_outside_whitelist_is_blocked_before_dispatch(self, monkeypatch):
        executed = []
        monkeypatch.setattr(
            agent,
            "execute_tool",
            lambda *args, **_kwargs: executed.append(args) or "should not run",
        )
        monkeypatch.setattr(
            agent,
            "stream",
            _fake_stream_sequence(
                [
                    AssistantTurn(
                        text="",
                        tool_calls=[{"id": "1", "name": "Bash", "input": {"command": "echo hi"}}],
                        in_tokens=10,
                        out_tokens=5,
                    )
                ]
            ),
        )

        state = AgentState()
        events = list(run_research("hi", state, BASE_CONFIG, "system"))
        tool_ends = [event for event in events if isinstance(event, ToolEnd)]

        assert len(tool_ends) == 1
        assert tool_ends[0].permitted is False
        assert "not available" in tool_ends[0].result
        assert executed == []
        assert state.turn_traces[0].status == "degraded"
        assert state.turn_traces[0].stages[1].status == "blocked"

    def test_call_inside_whitelist_executes_normally(self, monkeypatch):
        monkeypatch.setattr(agent, "execute_tool", lambda *args, **kwargs: "ok result")
        monkeypatch.setattr(
            agent,
            "stream",
            _fake_stream_sequence(
                [
                    AssistantTurn(
                        text="",
                        tool_calls=[{"id": "1", "name": "search_papers", "input": {"query": "x"}}],
                        in_tokens=10,
                        out_tokens=5,
                    )
                ]
            ),
        )

        events = list(run_research("hi", AgentState(), BASE_CONFIG, "system"))
        tool_ends = [event for event in events if isinstance(event, ToolEnd)]

        assert tool_ends[0].result == "ok result"
        assert tool_ends[0].permitted is True

    def test_unknown_tool_returns_error_without_crashing(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "stream",
            _fake_stream_sequence(
                [
                    AssistantTurn(
                        text="",
                        tool_calls=[{"id": "1", "name": "unknown_tool", "input": {}}],
                        in_tokens=1,
                        out_tokens=1,
                    )
                ]
            ),
        )

        events = list(run("hi", AgentState(), BASE_CONFIG, "system"))
        tool_end = next(event for event in events if isinstance(event, ToolEnd))

        assert tool_end.result == "Error: tool 'unknown_tool' not found."

    def test_invalid_domain_arguments_return_stable_error(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "stream",
            _fake_stream_sequence(
                [
                    AssistantTurn(
                        text="",
                        tool_calls=[{"id": "1", "name": "search_papers", "input": {}}],
                        in_tokens=1,
                        out_tokens=1,
                    )
                ]
            ),
        )

        events = list(run_research("hi", AgentState(), BASE_CONFIG, "system"))
        tool_end = next(event for event in events if isinstance(event, ToolEnd))

        assert tool_end.result.startswith("Error: invalid arguments for tool 'search_papers'")

    def test_tool_names_none_preserves_coding_mode(self, monkeypatch):
        monkeypatch.setattr(agent, "execute_tool", lambda *args, **kwargs: "ok")
        monkeypatch.setattr(
            agent,
            "stream",
            _fake_stream_sequence(
                [
                    AssistantTurn(
                        text="",
                        tool_calls=[{"id": "1", "name": "Bash", "input": {"command": "echo hi"}}],
                        in_tokens=10,
                        out_tokens=5,
                    )
                ]
            ),
        )

        events = list(run("hi", AgentState(), BASE_CONFIG, "system"))
        tool_end = next(event for event in events if isinstance(event, ToolEnd))

        assert tool_end.result == "ok"

    def test_invalid_scope_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            list(run("hi", AgentState(), BASE_CONFIG, "system", tool_names=[""]))


class TestTraceRecording:
    def test_run_appends_finished_turn_trace(self, monkeypatch):
        monkeypatch.setattr(
            agent,
            "stream",
            _fake_stream_sequence(
                [AssistantTurn(text="hello", tool_calls=[], in_tokens=3, out_tokens=2)]
            ),
        )

        state = AgentState()
        list(run("hi there", state, BASE_CONFIG, "system"))

        trace = state.turn_traces[0]
        assert trace.user_message == "hi there"
        assert trace.status == "ok"
        assert trace.ended_at is not None
        assert trace.stages[0].stage == "assistant_turn"
        assert trace.stages[0].output == "hello"

    def test_tool_call_is_recorded_as_stage(self, monkeypatch):
        monkeypatch.setattr(agent, "execute_tool", lambda *args, **kwargs: "search result")
        monkeypatch.setattr(
            agent,
            "stream",
            _fake_stream_sequence(
                [
                    AssistantTurn(
                        text="",
                        tool_calls=[{"id": "1", "name": "search_papers", "input": {"query": "x"}}],
                        in_tokens=10,
                        out_tokens=5,
                    )
                ]
            ),
        )

        state = AgentState()
        list(run("hi", state, BASE_CONFIG, "system"))

        assert "tool:search_papers" in [stage.stage for stage in state.turn_traces[0].stages]

    def test_multiple_runs_accumulate_separate_traces(self, monkeypatch):
        state = AgentState()
        for message in ("first", "second"):
            monkeypatch.setattr(
                agent,
                "stream",
                _fake_stream_sequence(
                    [AssistantTurn(text=message, tool_calls=[], in_tokens=1, out_tokens=1)]
                ),
            )
            list(run(message, state, BASE_CONFIG, "system"))

        assert [trace.turn_index for trace in state.turn_traces] == [0, 1]
        assert state.turn_traces[1].user_message == "second"

    def test_provider_exception_is_traced_then_reraised(self, monkeypatch):
        def failing_stream(**_kwargs):
            raise RuntimeError("provider unavailable")
            yield

        monkeypatch.setattr(agent, "stream", failing_stream)
        state = AgentState()

        with pytest.raises(RuntimeError, match="provider unavailable"):
            list(run("hi", state, BASE_CONFIG, "system"))

        assert len(state.turn_traces) == 1
        assert state.turn_traces[0].status == "error"
        assert state.turn_traces[0].stages[0].status == "error"
        assert "provider unavailable" in state.turn_traces[0].stages[0].error
