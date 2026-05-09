"""LLM backends for the orchestrator.

`AnthropicBackend` is the default; it caches the system prompt + the tool
block (last tool gets cache_control), which is the standard pattern for
multi-turn tool-use loops.

`MockLLM` returns a scripted sequence of responses so the agent loop can be
tested without any network calls.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]


class LLMBackend(Protocol):
    def step(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolSpec],
    ) -> dict[str, Any]:
        ...


class AnthropicBackend:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(max_retries=10)
        return self._client

    def step(self, system, messages, tools):
        client = self._client_lazy()
        sys_blocks = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
        tool_defs = []
        for i, t in enumerate(tools):
            entry = {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema,
            }
            if i == len(tools) - 1:
                entry["cache_control"] = {"type": "ephemeral"}
            tool_defs.append(entry)
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=sys_blocks,
            tools=tool_defs,
            messages=messages,
        )
        return {
            "stop_reason": resp.stop_reason,
            "content": [block.model_dump() for block in resp.content],
            "usage": getattr(resp, "usage", None),
        }


class MockLLM:
    """Deterministic backend: returns scripted responses one per `step` call."""
    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.i = 0

    def step(self, system, messages, tools):
        if self.i >= len(self.script):
            return {"stop_reason": "end_turn", "content": []}
        out = self.script[self.i]
        self.i += 1
        return out
