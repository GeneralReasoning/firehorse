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
    # Current (as of April 2026)
    "gpt-5.4": 1000000,
    "gpt-5.4-mini": 400000,
    "gpt-5.4-pro": 1000000,
    "gpt-5.2": 400000,
    "gpt-5.1": 400000,
    "o3": 200000,
    "o3-mini": 200000,
    # Deprecated but may still be used
    "gpt-4.1": 1047576,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1-nano": 1047576,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o4-mini": 200000,
}


def _sanitize_schema(schema: dict | None) -> dict | None:
    """Remove keys unsupported by OpenAI function calling."""
    if schema is None:
        return None
    from openreward.api.environments.client import _sanitize_openai_schema, _strip_titles
    return _sanitize_openai_schema(_strip_titles(schema))


def _format_tool(tool: ToolSpec) -> dict:
    params = _sanitize_schema(dict(tool.input_schema) if tool.input_schema else None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


class OpenAIProvider(ProviderClient):
    """OpenAI Responses API provider."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        context_window: int | None = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI provider. "
                "Install with: pip install 'firehorse[resum]'"
            )
        self.model = model
        self._context_window_override = context_window
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        self._system_prompt: str = ""

    @property
    def context_window(self) -> int | None:
        if self._context_window_override is not None:
            return self._context_window_override
        return KNOWN_CONTEXT_WINDOWS.get(self.model)

    def format_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [_format_tool(t) for t in tools]

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> list[Any]:
        self._system_prompt = system_prompt
        return [
            {"role": "user", "content": user_prompt},
        ]

    async def call(
        self,
        messages: list[Any],
        tools: list[dict],
        max_tokens: int = 16384,
        effort: str | None = None,
    ) -> LLMResponse:
        import openai as openai_mod

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "instructions": self._system_prompt,
        }
        if tools:
            kwargs["tools"] = tools
        if effort:
            mapped = "high" if effort == "max" else effort
            kwargs["reasoning"] = {"effort": mapped}

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                response = await self._client.responses.create(**kwargs)
            except openai_mod.BadRequestError as e:
                err_msg = str(e).lower()
                if "context length" in err_msg or ("maximum" in err_msg and "token" in err_msg):
                    return LLMResponse(raw_message=None, context_overflow=True)
                raise
            except (openai_mod.RateLimitError, openai_mod.APITimeoutError, openai_mod.InternalServerError) as e:
                last_err = e
                wait = min(2 ** attempt, 60)
                print(f"[resum/openai] Retry {attempt + 1}/5 after {type(e).__name__}, waiting {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
                continue

            if not response.output:
                last_err = RuntimeError("LLM returned empty output")
                wait = min(2 ** attempt, 60)
                print(f"[resum/openai] Retry {attempt + 1}/5: response.output is empty, waiting {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
                continue

            break
        else:
            raise last_err  # type: ignore[misc]

        # Parse output items
        tool_calls = []
        text_parts = []
        reasoning_parts = []

        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                try:
                    args = json.loads(item.arguments) if item.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCallInfo(
                    id=item.call_id,
                    name=item.name,
                    arguments=args,
                ))
            elif item_type == "message":
                for content_block in getattr(item, "content", []):
                    block_type = getattr(content_block, "type", None)
                    if block_type == "output_text":
                        text_parts.append(content_block.text)
            elif item_type == "reasoning":
                for content_block in getattr(item, "content", []):
                    block_type = getattr(content_block, "type", None)
                    if block_type == "reasoning_text":
                        reasoning_parts.append(content_block.text)

        return LLMResponse(
            raw_message=response,
            tool_calls=tool_calls,
            text_content="\n".join(text_parts) if text_parts else None,
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            input_tokens=response.usage.input_tokens if response.usage else None,
            output_tokens=response.usage.output_tokens if response.usage else None,
        )

    def append_assistant(self, messages: list[Any], response: LLMResponse) -> None:
        """Append output items as serializable dicts for caching/compaction."""
        raw = response.raw_message

        for item in raw.output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                messages.append({
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                })
            elif item_type == "message":
                text = ""
                for cb in getattr(item, "content", []):
                    if getattr(cb, "type", None) == "output_text":
                        text += cb.text
                if text:
                    messages.append({"role": "assistant", "content": text})

    def append_tool_result(
        self,
        messages: list[Any],
        call_id: str,
        tool_name: str,
        output: str,
    ) -> None:
        messages.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        })

    def append_user_message(self, messages: list[Any], content: str) -> None:
        messages.append({"role": "user", "content": content})

    def messages_to_text(self, messages: list[Any]) -> str:
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            role = msg.get("role", "").upper()

            if msg_type == "function_call":
                parts.append(f"ASSISTANT: [Tool Call] {msg.get('name', '')}({msg.get('arguments', '{}')})")
            elif msg_type == "function_call_output":
                output = msg.get("output", "")
                if len(output) > 1000:
                    output = output[:500] + "\n... [truncated] ...\n" + output[-500:]
                parts.append(f"TOOL OUTPUT [{msg.get('call_id', '')}]: {output}")
            elif role == "ASSISTANT":
                parts.append(f"ASSISTANT: {msg.get('content', '')}")
            elif role:
                parts.append(f"{role}: {msg.get('content', '')}")
        return "\n\n".join(parts)

    def rebuild_after_compaction(
        self,
        system_prompt: str,
        original_prompt: str,
        summary: str,
    ) -> list[Any]:
        self._system_prompt = system_prompt
        return [
            {"role": "user", "content": original_prompt},
            {"role": "user", "content": f"[COMPACTED CONVERSATION SUMMARY]\n\n{summary}\n\nPlease continue working on the task based on the summary above. Resume from where you left off."},
        ]

    async def call_for_compaction(
        self,
        conversation_text: str,
        compaction_prompt: str,
        max_tokens: int,
    ) -> str:
        response = await self._client.responses.create(
            model=self.model,
            input=[
                {"role": "user", "content": f"Here is the conversation history:\n\n{conversation_text}\n\n{compaction_prompt}"},
            ],
        )
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for cb in getattr(item, "content", []):
                    if getattr(cb, "type", None) == "output_text":
                        return cb.text
        return ""
