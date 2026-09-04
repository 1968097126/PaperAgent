"""Adapter from the inherited Provider stream API to stage completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from providers import AssistantTurn, stream


class NativeProviderClient:
    """Expose the repository's streaming Provider as a stage completion client."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        model = self.config.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Provider config requires a non-empty model")
        self.model = model.strip()
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, stage: str, prompt: str) -> str:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        completed: AssistantTurn | None = None
        system = (
            "You are one stage inside a research-paper QA pipeline. "
            f"Execute only the {stage.strip()} contract in the user message. "
            "Never use tools and never claim facts absent from supplied evidence."
        )
        for event in stream(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tool_schemas=[],
            config=self.config,
        ):
            if isinstance(event, AssistantTurn):
                completed = event
        if completed is None:
            raise RuntimeError(f"Provider returned no AssistantTurn for {stage}")
        if not isinstance(completed.text, str):
            raise TypeError(f"Provider returned non-string text for {stage}")
        self.input_tokens += int(completed.in_tokens or 0)
        self.output_tokens += int(completed.out_tokens or 0)
        return completed.text
