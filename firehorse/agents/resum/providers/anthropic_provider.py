from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, TYPE_CHECKING

from firehorse.agents.resum.providers.base import (
    LLMResponse,
    ProviderClient,
    ToolCallInfo,
)

if TYPE_CHECKING:
    from openreward.api.environments.types import ToolSpec


KNOWN_CONTEXT_WINDOWS = {
    "claude-opus-4-6": 1000000,
    "claude-sonnet-4-6": 1000000,
    "claude-sonnet-4-5-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
}


def _format_tool(tool: ToolSpec) -> dict:
    from openreward.api.environments.client import _strip_titles
    schema = _strip_titles(dict(tool.input_schema)) if tool.input_schema else {"type": "object", "properties": {}}
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }


class AnthropicProvider(ProviderClient):
    """Anthropic Messages API provider."""

    def __init__(self, model: str, api_key: str, context_window: int | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install 'firehorse-cli[resum]'"
            )
        self.model = model
        self._context_window_override = context_window
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        # Store system prompt separately since Anthropic takes it as a parameter
        self._system_prompt: str = ""

    @property
    def context_window(self) -> int | None:
        if self._context_window_override is not None:
            return self._context_window_override
        return KNOWN_CONTEXT_WINDOWS.get(self.model)

    def format_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [_format_tool(t) for t in tools]

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        self._system_prompt = system_prompt
        return [
            {"role": "user", "content": user_prompt},
        ]

    async def call(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 16384,
        effort: str | None = None,
    ) -> LLMResponse:
        import anthropic as anthropic_mod

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "system": self._system_prompt,
        }
        if tools:
            kwargs["tools"] = tools
        if effort:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                response = await self._client.messages.create(**kwargs)
                break
            except anthropic_mod.BadRequestError as e:
                err_msg = str(e).lower()
                if "too long" in err_msg or "context" in err_msg or "token" in err_msg:
                    return LLMResponse(raw_message=None, context_overflow=True)
                raise
            except (anthropic_mod.RateLimitError, anthropic_mod.APITimeoutError, anthropic_mod.InternalServerError) as e:
                last_err = e
                wait = min(2 ** attempt, 60)
                print(f"[resum/anthropic] Retry {attempt + 1}/5 after {type(e).__name__}, waiting {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
        else:
            raise last_err  # type: ignore[misc]

        tool_calls = []
        text_parts = []
        reasoning_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallInfo(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
            elif block.type == "thinking":
                reasoning_parts.append(block.thinking)

        return LLMResponse(
            raw_message=response,
            tool_calls=tool_calls,
            text_content="\n".join(text_parts) if text_parts else None,
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            input_tokens=response.usage.input_tokens if response.usage else None,
            output_tokens=response.usage.output_tokens if response.usage else None,
        )

    def append_assistant(self, messages: list[dict], response: LLMResponse) -> None:
        anthropic_response = response.raw_message
        # Build content blocks in the format Anthropic expects
        content: list[dict] = []
        for block in anthropic_response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "thinking":
                entry = {
                    "type": "thinking",
                    "thinking": block.thinking,
                }
                if hasattr(block, "signature") and block.signature:
                    entry["signature"] = block.signature
                content.append(entry)
        messages.append({"role": "assistant", "content": content})

    def append_tool_result(
        self,
        messages: list[dict],
        call_id: str,
        tool_name: str,
        output: str,
    ) -> None:
        # Anthropic groups tool results as user messages
        # Check if the last message is already a user message with tool_results
        if messages and messages[-1].get("role") == "user":
            last_content = messages[-1].get("content", [])
            if isinstance(last_content, list) and last_content and last_content[0].get("type") == "tool_result":
                last_content.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output,
                })
                return
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output,
                }
            ],
        })

    def append_user_message(self, messages: list[dict], content: str) -> None:
        messages.append({"role": "user", "content": content})

    def messages_to_text(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                pieces = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        pieces.append(block.get("text", ""))
                    elif btype == "thinking":
                        pieces.append(f"[Thinking] {block.get('thinking', '')}")
                    elif btype == "tool_use":
                        pieces.append(f"[Tool Call] {block.get('name', '')}({json.dumps(block.get('input', {}))})")
                    elif btype == "tool_result":
                        result_content = block.get("content", "")
                        if len(result_content) > 1000:
                            result_content = result_content[:500] + "\n... [truncated] ...\n" + result_content[-500:]
                        pieces.append(f"[Tool Result {block.get('tool_use_id', '')}]: {result_content}")
                parts.append(f"{role}: {' '.join(pieces)}")
        return "\n\n".join(parts)

    def rebuild_after_compaction(
        self,
        system_prompt: str,
        original_prompt: str,
        summary: str,
    ) -> list[dict]:
        self._system_prompt = system_prompt
        return [
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": "I'll continue working on this task."},
            {"role": "user", "content": f"[COMPACTED CONVERSATION SUMMARY]\n\n{summary}\n\nPlease continue working on the task based on the summary above. Resume from where you left off."},
        ]

    async def call_for_compaction(
        self,
        conversation_text: str,
        compaction_prompt: str,
        max_tokens: int,
    ) -> str:
        response = await self._client.messages.create(
            model=self.model,
            messages=[
                {"role": "user", "content": f"Here is the conversation history:\n\n{conversation_text}\n\n{compaction_prompt}"},
            ],
            max_tokens=max_tokens,
        )
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts)
