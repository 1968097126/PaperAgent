"""Turn/stage tracing with timings, summaries, status, and errors."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Self


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            text = repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


@dataclass
class StageTrace:
    stage: str
    input: Any
    output: Any
    duration_ms: float
    meta: dict = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)
    ended_at: str = field(default_factory=utc_now)
    status: str = "ok"
    error: str | None = None
    input_summary: str = ""
    output_summary: str = ""

    def __post_init__(self) -> None:
        if not self.input_summary:
            self.input_summary = summarize(self.input)
        if not self.output_summary:
            self.output_summary = summarize(self.output)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage, "status": self.status,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "duration_ms": round(self.duration_ms, 2),
            "input": self.input, "output": self.output,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error": self.error, "meta": self.meta,
        }


@dataclass
class TurnTrace:
    turn_index: int
    user_message: str
    stages: list[StageTrace] = field(default_factory=list)
    turn_id: str = ""
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    status: str = "running"

    def __post_init__(self) -> None:
        if not self.turn_id:
            self.turn_id = f"turn-{self.turn_index:04d}-{uuid.uuid4().hex[:8]}"

    def record(self, stage: str, input: Any, output: Any,
               duration_ms: float, **meta) -> StageTrace:
        status = str(meta.pop("status", "ok"))
        error = meta.pop("error", None)
        started_at = str(meta.pop("started_at", utc_now()))
        ended_at = str(meta.pop("ended_at", utc_now()))
        value = StageTrace(
            stage=stage, input=input, output=output, duration_ms=duration_ms,
            meta=meta, status=status, error=error,
            started_at=started_at, ended_at=ended_at,
        )
        self.stages.append(value)
        return value

    def skip(self, stage: str, reason: str) -> StageTrace:
        return self.record(stage, None, None, 0.0, status="skipped", reason=reason)

    def finish(self, status: str = "ok") -> None:
        self.ended_at = utc_now()
        self.status = status

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id, "turn_index": self.turn_index,
            "user_message": self.user_message, "status": self.status,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "stages": [stage.to_dict() for stage in self.stages],
        }


class Timer:
    def __enter__(self) -> Self:
        self._start = time.monotonic()
        self.started_at = utc_now()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
        self.ended_at = utc_now()
