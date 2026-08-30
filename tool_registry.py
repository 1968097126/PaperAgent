"""Tool plugin registry for nano-claude-code.

Provides a central registry for tool definitions, lookup, schema export,
and dispatch with output truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDef:
    """Definition of a single tool plugin.

    Attributes:
        name: unique tool identifier
        schema: JSON-schema dict sent to the API (name, description, input_schema)
        func: callable(params: dict, config: dict) -> str
        read_only: True if the tool never mutates state
        concurrent_safe: True if safe to run in parallel with other tools
    """
    name: str
    schema: Dict[str, Any]
    func: Callable[[Dict[str, Any], Dict[str, Any]], str]
    read_only: bool = False
    concurrent_safe: bool = False


# --------------- internal state ---------------

_registry: Dict[str, ToolDef] = {}


# --------------- public API ---------------

def register_tool(tool_def: ToolDef) -> None:
    """Register a tool, overwriting any existing tool with the same name."""
    _registry[tool_def.name] = tool_def


def get_tool(name: str) -> Optional[ToolDef]:
    """Look up a tool by name. Returns None if not found."""
    return _registry.get(name)


def get_all_tools() -> List[ToolDef]:
    """Return all registered tools (insertion order)."""
    return list(_registry.values())


def get_tool_schemas(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Return the schemas of registered tools (for API tool parameter).

    Args:
        names: 只返回这些名字对应的schema，且按names给定的顺序输出；
               不传（默认None）时保持原行为——返回全部已注册工具的schema。
               这个参数是Phase 3加的：研究agent场景下，对话里只想让LLM看到
               search_papers/retrieve_chunk这类领域工具的schema，而不是连
               Read/Write/Bash这些编程工具的schema也一起发给模型——但这些
               编程工具本身仍然保留在registry里（供离线脚本/开发调试用），
               不是被删除，只是"这次不放进发给模型的schema列表"。
    """
    if names is None:
        return [t.schema for t in _registry.values()]
    # 用names给定的顺序取，而不是遍历_registry再过滤——这样调用方传入的顺序
    # 会体现在最终发给模型的schema列表顺序里；名字在registry里不存在就跳过，
    # 不抛异常（比如白名单里手误写错名字，静默跳过好过整个请求失败）
    return [_registry[n].schema for n in names if n in _registry]


def execute_tool(
    name: str,
    params: Dict[str, Any],
    config: Dict[str, Any],
    max_output: int = 32000,
) -> str:
    """Dispatch a tool call by name.

    Args:
        name: tool name
        params: tool input parameters dict
        config: runtime configuration dict
        max_output: maximum allowed output length in characters

    Returns:
        Tool result string, possibly truncated.
    """
    tool = get_tool(name)
    if tool is None:
        return f"Error: tool '{name}' not found."

    result = tool.func(params, config)

    if len(result) > max_output:
        first_half = max_output // 2
        last_quarter = max_output // 4
        truncated = len(result) - first_half - last_quarter
        result = (
            result[:first_half]
            + f"\n[... {truncated} chars truncated ...]\n"
            + result[-last_quarter:]
        )

    return result


def clear_registry() -> None:
    """Remove all registered tools. Intended for testing."""
    _registry.clear()
