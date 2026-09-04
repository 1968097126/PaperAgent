"""Core agent loop: neutral message format, multi-provider streaming."""
from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import Any

from compaction import maybe_compact
from pipeline.trace import Timer, TurnTrace
from providers import AssistantTurn, TextChunk, ThinkingChunk, stream
from tool_registry import get_tool_schemas
from tools import execute_tool

RESEARCH_TOOL_NAMES = ("search_papers", "retrieve_chunk")

__all__ = [
    "RESEARCH_TOOL_NAMES",
    "AgentState",
    "PermissionRequest",
    "TextChunk",
    "ThinkingChunk",
    "ToolEnd",
    "ToolStart",
    "TurnDone",
    "run",
    "run_research",
]


@dataclass
class AgentState:
    """Mutable provider-independent state shared by coding and research modes."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_count: int = 0
    # Traces are kept outside messages so compaction cannot remove diagnostic history.
    turn_traces: list[TurnTrace] = field(default_factory=list)
    semantic_state: dict[str, Any] = field(default_factory=dict)
    retrieval_state: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    research_memory: dict[str, Any] = field(default_factory=dict)
    memory_mode: str = "compressed"


@dataclass
class ToolStart:
    name: str
    inputs: dict[str, Any]


@dataclass
class ToolEnd:
    name: str
    result: str
    permitted: bool = True


@dataclass
class TurnDone:
    input_tokens: int
    output_tokens: int


@dataclass
class PermissionRequest:
    description: str
    granted: bool = False


def run(
    user_message: str,
    state: AgentState,
    config: dict[str, Any],
    system_prompt: str,
    tool_names: Sequence[str] | None = None,
) -> Generator[object, None, None]:
    """Run one inherited agent turn with an optional execution-level tool scope.

    ``tool_names=None`` preserves the original coding-agent behavior. When a scope
    is supplied, only those schemas are sent to the provider and every returned
    tool call is checked against the same immutable scope before dispatch.
    """
    allowed_tool_names = None if tool_names is None else tuple(dict.fromkeys(tool_names))
    if allowed_tool_names is not None and any(
        not isinstance(name, str) or not name for name in allowed_tool_names
    ):
        raise ValueError("tool_names must contain non-empty strings")

    state.messages.append({"role": "user", "content": user_message})
    trace = TurnTrace(turn_index=len(state.turn_traces), user_message=user_message)

    while True:
        state.turn_count += 1
        assistant_turn: AssistantTurn | None = None
        maybe_compact(state, config)

        try:
            with Timer() as timer:
                for event in stream(
                    model=config["model"],
                    system=system_prompt,
                    messages=state.messages,
                    tool_schemas=get_tool_schemas(
                        names=list(allowed_tool_names) if allowed_tool_names is not None else None
                    ),
                    config=config,
                ):
                    if isinstance(event, (TextChunk, ThinkingChunk)):
                        yield event
                    elif isinstance(event, AssistantTurn):
                        assistant_turn = event
        except Exception as exc:
            trace.record(
                "assistant_turn",
                input=user_message,
                output=None,
                duration_ms=timer.elapsed_ms,
                started_at=timer.started_at,
                ended_at=timer.ended_at,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            trace.finish("error")
            state.turn_traces.append(trace)
            raise

        trace.record(
            "assistant_turn",
            input=user_message,
            output=assistant_turn.text if assistant_turn else None,
            duration_ms=timer.elapsed_ms,
            started_at=timer.started_at,
            ended_at=timer.ended_at,
            tool_calls=[call["name"] for call in assistant_turn.tool_calls] if assistant_turn else [],
        )

        if assistant_turn is None:
            break

        state.messages.append(
            {
                "role": "assistant",
                "content": assistant_turn.text,
                "tool_calls": assistant_turn.tool_calls,
            }
        )
        state.total_input_tokens += assistant_turn.in_tokens
        state.total_output_tokens += assistant_turn.out_tokens
        yield TurnDone(assistant_turn.in_tokens, assistant_turn.out_tokens)

        if not assistant_turn.tool_calls:
            break

        for tool_call in assistant_turn.tool_calls:
            name = str(tool_call.get("name", ""))
            inputs = tool_call.get("input", {})
            call_id = str(tool_call.get("id", ""))
            if not isinstance(inputs, dict):
                inputs = {"__invalid_input__": inputs}
            yield ToolStart(name, inputs)

            if allowed_tool_names is not None and name not in allowed_tool_names:
                result = f"Error: tool '{name}' is not available in this context"
                trace.record(
                    f"tool:{name}",
                    input=inputs,
                    output=result,
                    duration_ms=0.0,
                    permitted=False,
                    blocked_by_whitelist=True,
                    status="blocked",
                    error=result,
                )
                yield ToolEnd(name, result, permitted=False)
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result,
                    }
                )
                continue

            normalized_call = {"name": name, "input": inputs}
            permitted = _check_permission(normalized_call, config)
            if not permitted:
                request = PermissionRequest(description=_permission_desc(normalized_call))
                yield request
                permitted = request.granted

            with Timer() as timer:
                if permitted:
                    result = execute_tool(name, inputs, permission_mode="accept-all")
                else:
                    result = "Denied: user rejected this operation"
            stage_status = "blocked" if not permitted else (
                "error" if result.startswith("Error:") else "ok"
            )
            trace.record(
                f"tool:{name}",
                input=inputs,
                output=result,
                duration_ms=timer.elapsed_ms,
                started_at=timer.started_at,
                ended_at=timer.ended_at,
                permitted=permitted,
                status=stage_status,
                error=result if stage_status != "ok" else None,
            )
            yield ToolEnd(name, result, permitted)
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result,
                }
            )

    trace.finish("degraded" if any(stage.status != "ok" for stage in trace.stages) else "ok")
    state.turn_traces.append(trace)


def run_research(
    user_message: str,
    state: AgentState,
    config: dict[str, Any],
    system_prompt: str,
) -> Generator[object, None, None]:
    """Run the inherited loop with the fixed research-domain tool allowlist."""
    return run(
        user_message,
        state,
        config,
        system_prompt,
        tool_names=RESEARCH_TOOL_NAMES,
    )


def _check_permission(tool_call: dict[str, Any], config: dict[str, Any]) -> bool:
    """Return whether a tool is auto-approved under the inherited permission mode."""
    permission_mode = config.get("permission_mode", "auto")
    if permission_mode == "accept-all":
        return True
    if permission_mode == "manual":
        return False

    name = tool_call["name"]
    if name in (
        "Read",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
        *RESEARCH_TOOL_NAMES,
    ):
        return True
    if name == "Bash":
        from tools import _is_safe_bash

        return _is_safe_bash(tool_call["input"].get("command", ""))
    return False


def _permission_desc(tool_call: dict[str, Any]) -> str:
    name = tool_call["name"]
    inputs = tool_call["input"]
    if name == "Bash":
        return f"Run: {inputs.get('command', '')}"
    if name == "Write":
        return f"Write to: {inputs.get('file_path', '')}"
    if name == "Edit":
        return f"Edit: {inputs.get('file_path', '')}"
    return f"{name}({list(inputs.values())[:1]})"
