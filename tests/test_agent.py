"""Tests for agent.py — permission whitelist fix, tool_names restriction, trace recording.

用 monkeypatch 替换掉 agent.stream（原本指向 providers.stream，真实调用LLM API）和
agent.execute_tool，让测试不依赖网络/API key，只验证 run() 循环本身的控制流逻辑。
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent
from agent import AgentState, run, ToolEnd, _check_permission
from providers import AssistantTurn


def _fake_stream_sequence(turns: list[AssistantTurn]):
    """构造一个假的 providers.stream()：按顺序依次产出turns列表里的AssistantTurn，
    每次调用消耗一个；列表用完之后再被调用就产出一个空tool_calls的收尾轮次
    （模拟"模型不再调用工具、直接给最终回答"），避免测试因为漏算run()内部
    会循环几轮而导致while True死循环。
    """
    it = iter(turns)

    def _stream(**kwargs):
        turn = next(it, AssistantTurn(text="done", tool_calls=[], in_tokens=1, out_tokens=1))
        yield turn

    return _stream


BASE_CONFIG = {"model": "claude-opus-4-6", "permission_mode": "accept-all"}


class TestCheckPermissionWhitelist:
    """对应Phase 3步骤5：search_papers/retrieve_chunk之前漏加进auto模式的只读白名单，
    导致每次调用都会被误判成需要用户确认——这里验证修复生效。"""

    def test_search_papers_auto_approved_in_auto_mode(self):
        tc = {"name": "search_papers", "input": {"query": "x"}}
        assert _check_permission(tc, {"permission_mode": "auto"}) is True

    def test_retrieve_chunk_auto_approved_in_auto_mode(self):
        tc = {"name": "retrieve_chunk", "input": {"chunk_id": "x"}}
        assert _check_permission(tc, {"permission_mode": "auto"}) is True

    def test_write_still_requires_confirmation_in_auto_mode(self):
        # 回归保护：确认这次修复只加了白名单条目，没有意外放宽Write/Edit的权限门槛
        tc = {"name": "Write", "input": {"file_path": "x", "content": "y"}}
        assert _check_permission(tc, {"permission_mode": "auto"}) is False


class TestToolNameWhitelist:
    """对应Phase 3步骤3/4：run()的tool_names参数不仅要让LLM看不到白名单外的schema，
    真的调用了也要在执行前被拦下。"""

    def test_call_outside_whitelist_is_blocked_not_executed(self, monkeypatch):
        executed = []
        monkeypatch.setattr(agent, "execute_tool",
                             lambda *a, **k: executed.append(a) or "should not run")
        monkeypatch.setattr(agent, "stream", _fake_stream_sequence([
            AssistantTurn(text="", tool_calls=[
                {"id": "1", "name": "Bash", "input": {"command": "echo hi"}},
            ], in_tokens=10, out_tokens=5),
        ]))

        state = AgentState()
        events = list(run("hi", state, BASE_CONFIG, "system", tool_names=["search_papers"]))

        tool_ends = [e for e in events if isinstance(e, ToolEnd)]
        assert len(tool_ends) == 1
        assert tool_ends[0].permitted is False
        assert "not available" in tool_ends[0].result
        assert executed == []   # execute_tool从未被真正调用到

    def test_call_inside_whitelist_executes_normally(self, monkeypatch):
        monkeypatch.setattr(agent, "execute_tool", lambda *a, **k: "ok result")
        monkeypatch.setattr(agent, "stream", _fake_stream_sequence([
            AssistantTurn(text="", tool_calls=[
                {"id": "1", "name": "search_papers", "input": {"query": "x"}},
            ], in_tokens=10, out_tokens=5),
        ]))

        state = AgentState()
        events = list(run("hi", state, BASE_CONFIG, "system", tool_names=["search_papers"]))

        tool_ends = [e for e in events if isinstance(e, ToolEnd)]
        assert len(tool_ends) == 1
        assert tool_ends[0].result == "ok result"

    def test_tool_names_none_keeps_backward_compatible_behavior(self, monkeypatch):
        # 不传tool_names（默认None）时，任何工具名都不该被白名单机制拦下——
        # 这保证了nano_claude.py现有的编程助手用法不受Phase 3改动影响
        monkeypatch.setattr(agent, "execute_tool", lambda *a, **k: "ok")
        monkeypatch.setattr(agent, "stream", _fake_stream_sequence([
            AssistantTurn(text="", tool_calls=[
                {"id": "1", "name": "Bash", "input": {"command": "echo hi"}},
            ], in_tokens=10, out_tokens=5),
        ]))

        state = AgentState()
        events = list(run("hi", state, BASE_CONFIG, "system"))

        tool_ends = [e for e in events if isinstance(e, ToolEnd)]
        assert tool_ends[0].result == "ok"


class TestTraceRecording:
    """对应Phase 3步骤1/2：run()应该把这一轮的LLM调用和工具调用都记进state.turn_traces。"""

    def test_run_appends_one_turn_trace(self, monkeypatch):
        monkeypatch.setattr(agent, "stream", _fake_stream_sequence([
            AssistantTurn(text="hello", tool_calls=[], in_tokens=3, out_tokens=2),
        ]))

        state = AgentState()
        list(run("hi there", state, BASE_CONFIG, "system"))

        assert len(state.turn_traces) == 1
        assert state.turn_traces[0].user_message == "hi there"
        assert state.turn_traces[0].stages[0].stage == "assistant_turn"
        assert state.turn_traces[0].stages[0].output == "hello"

    def test_tool_call_recorded_as_stage(self, monkeypatch):
        monkeypatch.setattr(agent, "execute_tool", lambda *a, **k: "search result")
        monkeypatch.setattr(agent, "stream", _fake_stream_sequence([
            AssistantTurn(text="", tool_calls=[
                {"id": "1", "name": "search_papers", "input": {"query": "x"}},
            ], in_tokens=10, out_tokens=5),
        ]))

        state = AgentState()
        list(run("hi", state, BASE_CONFIG, "system"))

        stage_names = [s.stage for s in state.turn_traces[0].stages]
        assert "tool:search_papers" in stage_names

    def test_multiple_run_calls_accumulate_separate_traces(self, monkeypatch):
        monkeypatch.setattr(agent, "stream", _fake_stream_sequence([
            AssistantTurn(text="first", tool_calls=[], in_tokens=1, out_tokens=1),
        ]))
        state = AgentState()
        list(run("first message", state, BASE_CONFIG, "system"))

        monkeypatch.setattr(agent, "stream", _fake_stream_sequence([
            AssistantTurn(text="second", tool_calls=[], in_tokens=1, out_tokens=1),
        ]))
        list(run("second message", state, BASE_CONFIG, "system"))

        assert len(state.turn_traces) == 2
        assert state.turn_traces[0].turn_index == 0
        assert state.turn_traces[1].turn_index == 1
        assert state.turn_traces[1].user_message == "second message"
