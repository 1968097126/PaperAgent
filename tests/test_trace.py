"""Tests for the Phase 3 TurnTrace, StageTrace, and Timer contract."""
from __future__ import annotations

import json
import time

from pipeline.trace import Timer, TurnTrace, summarize


class TestTurnTraceRecord:
    def test_record_appends_stage_with_all_fields(self):
        trace = TurnTrace(turn_index=0, user_message="hello")
        stage = trace.record(
            "assistant_turn",
            input="hello",
            output="hi there",
            duration_ms=12.5,
            model="claude-opus-4-6",
        )

        assert trace.stages == [stage]
        assert stage.stage == "assistant_turn"
        assert stage.input == "hello"
        assert stage.output == "hi there"
        assert stage.duration_ms == 12.5
        assert stage.status == "ok"
        assert stage.error is None
        assert stage.started_at
        assert stage.ended_at
        assert stage.input_summary == "hello"
        assert stage.output_summary == "hi there"
        assert stage.meta == {"model": "claude-opus-4-6"}

    def test_multiple_records_preserve_order(self):
        trace = TurnTrace(turn_index=0, user_message="q")
        trace.record("assistant_turn", input="q", output="a", duration_ms=1)
        trace.record(
            "tool:search_papers",
            input={"query": "q"},
            output="[...]",
            duration_ms=2,
        )
        trace.record("assistant_turn", input="q", output="final answer", duration_ms=1)

        assert [stage.stage for stage in trace.stages] == [
            "assistant_turn",
            "tool:search_papers",
            "assistant_turn",
        ]

    def test_error_and_skip_status_are_structured(self):
        trace = TurnTrace(turn_index=0, user_message="q")
        failed = trace.record(
            "tool:search_papers",
            input={},
            output="Error: invalid arguments",
            duration_ms=1,
            status="error",
            error="Error: invalid arguments",
            permitted=True,
        )
        skipped = trace.skip("Generate", "guardrail refused")
        trace.finish("degraded")

        assert failed.status == "error"
        assert failed.error == "Error: invalid arguments"
        assert failed.meta == {"permitted": True}
        assert skipped.status == "skipped"
        assert skipped.meta == {"reason": "guardrail refused"}
        assert trace.status == "degraded"
        assert trace.ended_at is not None


class TestTurnTraceSerialization:
    def test_to_dict_is_json_serializable(self):
        trace = TurnTrace(turn_index=2, user_message="what is PPO?")
        trace.record(
            "tool:search_papers",
            input={"query": "PPO"},
            output="[1] ...",
            duration_ms=340.789,
            permitted=True,
        )
        trace.finish()

        reloaded = json.loads(json.dumps(trace.to_dict()))

        assert reloaded["turn_id"].startswith("turn-0002-")
        assert reloaded["turn_index"] == 2
        assert reloaded["user_message"] == "what is PPO?"
        assert reloaded["status"] == "ok"
        assert reloaded["started_at"]
        assert reloaded["ended_at"]
        assert reloaded["stages"][0]["stage"] == "tool:search_papers"
        assert reloaded["stages"][0]["meta"] == {"permitted": True}

    def test_duration_is_rounded_and_empty_trace_is_supported(self):
        trace = TurnTrace(turn_index=0, user_message="q")
        trace.record("s", input=None, output=None, duration_ms=1.23456789)

        assert trace.to_dict()["stages"][0]["duration_ms"] == 1.23
        assert TurnTrace(turn_index=1, user_message="q").to_dict()["stages"] == []

    def test_summaries_are_stable_and_bounded(self):
        assert summarize({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
        assert summarize("x" * 20, limit=10) == "x" * 9 + "…"


class TestTimer:
    def test_measures_elapsed_time_and_timestamps(self):
        with Timer() as timer:
            time.sleep(0.02)

        assert timer.elapsed_ms >= 20
        assert timer.started_at
        assert timer.ended_at

    def test_elapsed_time_is_available_after_exception(self):
        try:
            with Timer() as timer:
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert timer.elapsed_ms >= 0
        assert timer.ended_at
