"""Tests for pipeline/trace.py — StageTrace/TurnTrace/Timer."""
from __future__ import annotations

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.trace import StageTrace, TurnTrace, Timer


class TestTurnTraceRecord:
    def test_record_appends_stage_with_all_fields(self):
        trace = TurnTrace(turn_index=0, user_message="hello")
        stage = trace.record("assistant_turn", input="hello", output="hi there",
                              duration_ms=12.5, model="claude-opus-4-6")

        assert len(trace.stages) == 1
        assert trace.stages[0] is stage
        assert stage.stage == "assistant_turn"
        assert stage.input == "hello"
        assert stage.output == "hi there"
        assert stage.duration_ms == 12.5
        assert stage.meta == {"model": "claude-opus-4-6"}

    def test_multiple_records_preserve_order(self):
        trace = TurnTrace(turn_index=0, user_message="q")
        trace.record("assistant_turn", input="q", output="a", duration_ms=1)
        trace.record("tool:search_papers", input={"query": "q"}, output="[...]", duration_ms=2)
        trace.record("assistant_turn", input="q", output="final answer", duration_ms=1)

        assert [s.stage for s in trace.stages] == [
            "assistant_turn", "tool:search_papers", "assistant_turn",
        ]

    def test_meta_defaults_to_empty_dict_without_kwargs(self):
        trace = TurnTrace(turn_index=0, user_message="q")
        stage = trace.record("assistant_turn", input="q", output="a", duration_ms=1)
        assert stage.meta == {}


class TestTurnTraceSerialization:
    def test_to_dict_is_json_serializable(self):
        trace = TurnTrace(turn_index=2, user_message="what is PPO?")
        trace.record("tool:search_papers", input={"query": "PPO"},
                      output="[1] ...", duration_ms=340.789, permitted=True)

        d = trace.to_dict()
        # 能被json.dumps()序列化不抛异常，是这个函数存在的意义
        # （以后/trace命令或者落盘存档都要靠这个）
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)

        assert reloaded["turn_index"] == 2
        assert reloaded["user_message"] == "what is PPO?"
        assert reloaded["stages"][0]["stage"] == "tool:search_papers"
        assert reloaded["stages"][0]["meta"] == {"permitted": True}

    def test_duration_ms_rounded_to_two_decimals(self):
        trace = TurnTrace(turn_index=0, user_message="q")
        trace.record("s", input=None, output=None, duration_ms=1.23456789)
        d = trace.to_dict()
        assert d["stages"][0]["duration_ms"] == 1.23

    def test_empty_trace_serializes_to_empty_stage_list(self):
        trace = TurnTrace(turn_index=0, user_message="q")
        assert trace.to_dict()["stages"] == []


class TestTimer:
    def test_measures_elapsed_time(self):
        with Timer() as t:
            time.sleep(0.02)
        assert t.elapsed_ms >= 20   # 至少20ms（sleep(0.02)理论上是20ms，允许系统调度误差往上浮动）

    def test_elapsed_ms_available_after_context_exits(self):
        with Timer() as t:
            pass
        # 退出with块之后elapsed_ms应该是一个具体数值（>=0），不是初始占位或者未定义
        assert isinstance(t.elapsed_ms, float)
        assert t.elapsed_ms >= 0
