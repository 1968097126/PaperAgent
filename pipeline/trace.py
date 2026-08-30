"""执行轨迹（trace）数据结构：记录一轮对话内部每一步做了什么，供事后调试/评测复盘。

一次用户输入触发的完整处理过程是一个 TurnTrace（对应"一次trace"）；
过程中每一个可观测的动作（调一次LLM、跑一次工具）是一个 StageTrace（对应
分布式追踪里"span"的概念）——记录这一步的输入、输出、耗时。两者是
"一个trace由若干span组成"的标准关系，只是这里的"span"不是网络请求，
是Agent流水线里的一步。

Phase 3 阶段这里先把数据结构和记录机制建好，接进 agent.py 现有的循环里验证；
Phase 4 的 Guardrails/NLU/Reflect 等阶段会复用同一套 TurnTrace.record() 接口，
让"七阶段显式执行链路"里的每一步都自然落进同一份可追溯记录里。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTrace:
    """一步可观测动作的记录（比如一次LLM调用、一次工具执行）。"""
    stage:       str            # 阶段名，如 "assistant_turn"、"tool:search_papers"
    input:       Any            # 这一步的输入（不同stage含义不同，故意用Any不强制类型）
    output:      Any            # 这一步的输出/结果
    duration_ms: float          # 耗时（毫秒）
    meta:        dict = field(default_factory=dict)   # 额外的结构化信息，比如permitted/scores等


@dataclass
class TurnTrace:
    """一次用户输入触发的完整处理过程，由若干StageTrace按发生顺序组成。"""
    turn_index:   int           # 第几轮对话（0-indexed），对应 state.turn_traces 里的下标
    user_message: str
    stages:       list[StageTrace] = field(default_factory=list)

    def record(self, stage: str, input: Any, output: Any,
               duration_ms: float, **meta) -> StageTrace:
        """追加一条StageTrace记录，返回这条记录本身（方便调用方需要时再查看）。

        用 **meta 而不是让调用方自己构造StageTrace对象，是因为调用方（agent.py里的
        run()循环）本来就是按"stage名 + 几个具体字段"的方式在用，写成关键字参数
        更贴近调用现场的样子，比如 trace.record("tool:search_papers", input=...,
        output=..., duration_ms=..., permitted=True) 这种。
        """
        s = StageTrace(stage=stage, input=input, output=output,
                        duration_ms=duration_ms, meta=meta)
        self.stages.append(s)
        return s

    def to_dict(self) -> dict:
        """转成纯字典（可以直接json.dumps），供以后 /trace 命令展示或落盘留存用。"""
        return {
            "turn_index":   self.turn_index,
            "user_message": self.user_message,
            "stages": [
                {
                    "stage": s.stage,
                    "input": s.input,
                    "output": s.output,
                    "duration_ms": round(s.duration_ms, 2),
                    "meta": s.meta,
                }
                for s in self.stages
            ],
        }


class Timer:
    """小工具：`with Timer() as t: ...` 之后 t.elapsed_ms 就是代码块的耗时（毫秒）。

    用contextmanager风格封装time.monotonic()的起止差值计算，避免在agent.py里
    每处需要计时的地方都重复写"记录起始时间->算差值->转毫秒"这三行样板代码。
    time.monotonic()而不是time.time()：前者不受系统时钟被人为调整（如NTP校时）
    影响，专门用来测量"经过了多久"这种场景，是标准实践。
    """
    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
