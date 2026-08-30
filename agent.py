"""Core agent loop: neutral message format, multi-provider streaming."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Generator

from tool_registry import get_tool_schemas
from tools import execute_tool
import tools as _tools_init  # ensure built-in tools are registered on import
from providers import stream, AssistantTurn, TextChunk, ThinkingChunk, detect_provider
from compaction import maybe_compact
from pipeline.trace import TurnTrace, Timer

# ── Re-export event types (used by nano_claude.py) ────────────────────────
__all__ = [
    "AgentState", "run",
    "TextChunk", "ThinkingChunk",
    "ToolStart", "ToolEnd", "TurnDone", "PermissionRequest",
]


@dataclass # 自动生成常用方法，如init
class AgentState:
    """Mutable session state. messages use the neutral provider-independent format."""
    messages: list = field(default_factory=list)
    total_input_tokens:  int = 0
    total_output_tokens: int = 0
    turn_count: int = 0
    # 每次调用run()（对应一次用户输入的完整处理）产出一个TurnTrace，按发生顺序追加在这里；
    # 是messages之外单独的一份记录——messages是要发给LLM的对话历史（受压缩/裁剪影响），
    # turn_traces是给人/评测脚本看的调试记录，两者用途不同，故意不合并成一份数据
    turn_traces: list = field(default_factory=list)


@dataclass
class ToolStart:  # 事件类
    name:   str
    inputs: dict

@dataclass
class ToolEnd:
    name:      str
    result:    str
    permitted: bool = True

@dataclass
class TurnDone:
    input_tokens:  int
    output_tokens: int

@dataclass
class PermissionRequest:
    description: str
    granted: bool = False


# ── Agent loop ─────────────────────────────────────────────────────────────

def run(
    user_message: str,
    state: AgentState,
    config: dict,
    system_prompt: str,
    tool_names: list[str] | None = None,
) -> Generator:  #返回一个生成器，边执行边产出事件
    """
    Multi-turn agent loop (generator).
    Yields: TextChunk | ThinkingChunk | ToolStart | ToolEnd |
            PermissionRequest | TurnDone

    Args:
        tool_names: 只把这些工具的schema发给LLM，且LLM试图调用不在这个列表里的
                    工具时直接拒绝执行（不只是"隐藏schema"这种表面限制，是真的
                    不会被execute_tool执行到）。不传（默认None）= 原有行为，
                    发送全部已注册工具，兼容nano_claude.py现有的编程助手用法。
                    Phase 4会用这个参数把研究agent的对话限制在
                    ["search_papers", "retrieve_chunk"]范围内。
    """
    # Append user turn in neutral format
    state.messages.append({"role": "user", "content": user_message})

    # 本次run()调用（对应一次用户输入的完整处理，哪怕内部要经过好几轮LLM<->工具往返）
    # 产出一个TurnTrace；turn_index用当前已有多少条trace记录来编号，从0开始
    trace = TurnTrace(turn_index=len(state.turn_traces), user_message=user_message)

    while True:
        state.turn_count += 1
        assistant_turn: AssistantTurn | None = None # 为none或AssistanTurn类型，先设置为none

        # Compact context if approaching window limit
        maybe_compact(state, config)  #若历史过长则进行压缩

        # Stream from provider (auto-detected from model name)
        with Timer() as timer:
            for event in stream(
                model=config["model"],
                system=system_prompt,
                messages=state.messages,
                tool_schemas=get_tool_schemas(names=tool_names),
                config=config,
            ):
                # 判断event是否为TextChunk或ThinkingChunk的实例
                if isinstance(event, (TextChunk, ThinkingChunk)):
                    yield event
                    # yield:每次产出一个事件，调用方可以逐个处理，不必等整个回复结束
                    # 下一次调用从上次停止位置继续
                    # 前文Generator的典型用法
                elif isinstance(event, AssistantTurn):
                    assistant_turn = event
        trace.record(
            "assistant_turn",
            input=user_message,
            output=assistant_turn.text if assistant_turn else None,
            duration_ms=timer.elapsed_ms,
            tool_calls=[tc["name"] for tc in assistant_turn.tool_calls] if assistant_turn else [],
        )

        if assistant_turn is None:
            break

        # Record assistant turn in neutral format
        state.messages.append({
            "role":       "assistant",
            "content":    assistant_turn.text,
            "tool_calls": assistant_turn.tool_calls,
        })

        state.total_input_tokens  += assistant_turn.in_tokens
        state.total_output_tokens += assistant_turn.out_tokens
        # 事件对象TurnDone，标志生成结束
        yield TurnDone(assistant_turn.in_tokens, assistant_turn.out_tokens)

        if not assistant_turn.tool_calls:
            break   # No tools → conversation turn complete

        # ── Execute tools ────────────────────────────────────────────────
        for tc in assistant_turn.tool_calls:
            yield ToolStart(tc["name"], tc["input"])

            # 工具白名单：tool_names非None时，模型理论上不会看到白名单外的schema，
            # 但"看不到"不等于"调用不了"——万一模型凭训练记忆猜出了一个工具名并尝试调用，
            # 这里是真正拦截执行的地方，不依赖"隐藏schema"这一层表面限制
            if tool_names is not None and tc["name"] not in tool_names:
                result = f"Error: tool '{tc['name']}' is not available in this context"
                trace.record(f"tool:{tc['name']}", input=tc["input"], output=result,
                             duration_ms=0.0, permitted=False, blocked_by_whitelist=True)
                yield ToolEnd(tc["name"], result, permitted=False)
                state.messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": tc["name"], "content": result,
                })
                continue

            # Permission gate
            # 根据配置判断工具是否允许
            permitted = _check_permission(tc, config)
            # 若不允许则进行权限请求
            if not permitted:
                req = PermissionRequest(description=_permission_desc(tc))
                yield req
                permitted = req.granted

            with Timer() as timer:
                if not permitted:
                    result = "Denied: user rejected this operation"
                else:
                    result = execute_tool(
                        tc["name"], tc["input"],
                        permission_mode="accept-all",  # already gate-checked above
                    )
            trace.record(f"tool:{tc['name']}", input=tc["input"], output=result,
                         duration_ms=timer.elapsed_ms, permitted=permitted)

            yield ToolEnd(tc["name"], result, permitted)

            # Append tool result in neutral format
            state.messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         tc["name"],
                "content":      result,
            })

    state.turn_traces.append(trace)


# ── Helpers ───────────────────────────────────────────────────────────────

# 权限系统
def _check_permission(tc: dict, config: dict) -> bool:
    """Return True if operation is auto-approved (no need to ask user)."""
    perm_mode = config.get("permission_mode", "auto")
    # 第一种，全部接受
    if perm_mode == "accept-all":
        return True
        # 第二种，全部手动确认
    if perm_mode == "manual":
        return False   # always ask

    # "auto" mode: only ask for writes and non-safe bash
    # 第三种，只对危险操作询问
    name = tc["name"]
    # search_papers/retrieve_chunk是Phase 2新加的领域工具，本质和Read/Grep一样是
    # 只读检索、不修改任何状态，理应享受同样的自动放行待遇——这里漏加过一次，
    # 效果是"auto"模式下这两个工具会走到函数末尾的`return False`，每次调用都弹权限确认，
    # 对一个只读检索工具来说是不必要的打扰，故修复
    if name in ("Read", "Glob", "Grep", "WebFetch", "WebSearch",
                "search_papers", "retrieve_chunk"):
        return True
    if name == "Bash":
        from tools import _is_safe_bash
        return _is_safe_bash(tc["input"].get("command", ""))
    return False   # Write, Edit → ask

# 把工具请求描述给用户看
def _permission_desc(tc: dict) -> str:
    name = tc["name"]
    inp  = tc["input"]
    if name == "Bash":   return f"Run: {inp.get('command', '')}"
    if name == "Write":  return f"Write to: {inp.get('file_path', '')}"
    if name == "Edit":   return f"Edit: {inp.get('file_path', '')}"
    return f"{name}({list(inp.values())[:1]})"
