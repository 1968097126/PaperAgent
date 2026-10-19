from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from providers import COSTS, PROVIDERS, AssistantTurn, stream


def _install_openai_stub(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class Completions:
        @staticmethod
        def create(**kwargs: Any):
            calls.append(kwargs)
            text_delta = SimpleNamespace(content='{"status":"ok"}', tool_calls=[])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=text_delta)],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            )

    class OpenAI:
        def __init__(self, **_kwargs: Any):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    return calls


def _complete(model: str, config: dict[str, Any]) -> list[Any]:
    return list(
        stream(
            model=model,
            system="system",
            messages=[{"role": "user", "content": "prompt"}],
            tool_schemas=[],
            config=config,
        )
    )


def test_current_provider_models_and_environment_key_contract() -> None:
    assert PROVIDERS["openai"]["models"] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ]
    assert PROVIDERS["deepseek"]["models"] == [
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash-vision-exp",
    ]
    assert PROVIDERS["openai"]["api_key_env"] == "OPENAI_API_KEY"
    assert PROVIDERS["deepseek"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert COSTS["gpt-5.6-luna"] == (0.2, 1.2)


def test_gpt_5_uses_completion_token_parameter_and_stream_usage(monkeypatch) -> None:
    calls = _install_openai_stub(monkeypatch)

    events = _complete("gpt-5.6-luna", {"openai_api_key": "test", "max_tokens": 256})

    assert calls[0]["max_completion_tokens"] == 256
    assert "max_tokens" not in calls[0]
    assert calls[0]["stream_options"] == {"include_usage": True}
    completed = next(event for event in events if isinstance(event, AssistantTurn))
    assert (completed.in_tokens, completed.out_tokens) == (7, 3)


@pytest.mark.parametrize(
    ("thinking", "effort", "expected"),
    [
        (False, "max", {"thinking": {"type": "disabled"}}),
        (True, "max", {"thinking": {"type": "enabled"}}),
    ],
)
def test_deepseek_v4_explicit_thinking_contract(
    monkeypatch: pytest.MonkeyPatch,
    thinking: bool,
    effort: str,
    expected: dict[str, Any],
) -> None:
    calls = _install_openai_stub(monkeypatch)

    _complete(
        "deepseek-v4-flash",
        {
            "deepseek_api_key": "test",
            "max_tokens": 512,
            "thinking": thinking,
            "reasoning_effort": effort,
        },
    )

    assert calls[0]["max_tokens"] == 512
    assert calls[0]["extra_body"] == expected
    if thinking:
        assert calls[0]["reasoning_effort"] == "max"
    else:
        assert "reasoning_effort" not in calls[0]
