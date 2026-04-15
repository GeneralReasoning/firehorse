from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, TYPE_CHECKING

try:
    import openai
except ImportError:
    openai = None

from openreward.api.environments.client import _sanitize_openai_schema, _strip_titles

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
    """OpenAI Chat Completions provider."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        context_window: int | None = None,
    ):
        if openai is None:
            raise ImportError(
                "openai package required for OpenAI provider. "
                "Install with: pip install 'firehorse-cli[resum]'"
            )
        self.model = model
        self._context_window_override = context_window
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    @property
    def context_window(self) -> int | None:
        if self._context_window_override is not None:
            return self._context_window_override
        return KNOWN_CONTEXT_WINDOWS.get(self.model)

    def format_tools(self, tools: list[ToolSpec]) -> list[dict]:
        return [_format_tool(t) for t in tools]

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def call(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 16384,
        effort: str | None = None,
    ) -> LLMResponse:
        # Newer OpenAI models (gpt-5.x, o-series) require max_completion_tokens
        uses_completion_tokens = any(self.model.startswith(p) for p in ("gpt-5", "o3", "o4", "o1"))
        token_key = "max_completion_tokens" if uses_completion_tokens else "max_tokens"
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            token_key: max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if effort:
            mapped = "xhigh" if effort == "max" else effort
            kwargs["reasoning_effort"] = mapped
            # Also send OpenRouter's unified reasoning format via extra_body.
            # Direct OpenAI ignores extra_body fields it doesn't recognize.
            kwargs["extra_body"] = {"reasoning": {"effort": mapped}}

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except openai.BadRequestError as e:
                err_msg = str(e).lower()
                if "context length" in err_msg or "maximum" in err_msg and "token" in err_msg:
                    return LLMResponse(raw_message=None, context_overflow=True)
                raise
            except (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError) as e:
                last_err = e
                wait = min(2 ** attempt, 60)
                print(f"[resum/openai] Retry {attempt + 1}/5 after {type(e).__name__}, waiting {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
                continue

            if not response.choices:
                last_err = RuntimeError("LLM returned empty choices")
                wait = min(2 ** attempt, 60)
                print(f"[resum/openai] Retry {attempt + 1}/5: response.choices is None/empty, waiting {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
                continue

            break
        else:
            raise last_err  # type: ignore[misc]

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCallInfo(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        # Extract reasoning content from model_extra (OpenRouter returns this
        # for models with thinking/reasoning, e.g. Gemini, DeepSeek-R1, Grok).
        reasoning_content = None
        if hasattr(msg, "model_extra") and msg.model_extra:
            reasoning_content = (
                msg.model_extra.get("reasoning")
                or msg.model_extra.get("reasoning_content")
            )

        return LLMResponse(
            raw_message=msg,
            tool_calls=tool_calls,
            text_content=msg.content,
            reasoning_content=reasoning_content,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )

    def append_assistant(self, messages: list[dict], response: LLMResponse) -> None:
        msg = response.raw_message
        entry: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            entry["content"] = msg.content
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(entry)

    def append_tool_result(
        self,
        messages: list[dict],
        call_id: str,
        tool_name: str,
        output: str,
    ) -> None:
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": output,
        })

    def append_user_message(self, messages: list[dict], content: str) -> None:
        messages.append({"role": "user", "content": content})

    def messages_to_text(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if role == "ASSISTANT":
                pieces = []
                if content:
                    pieces.append(content)
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    pieces.append(f"[Tool Call] {func.get('name', '')}({func.get('arguments', '{}')})")
                parts.append(f"ASSISTANT: {' '.join(pieces)}")
            elif role == "TOOL":
                tool_content = content
                if len(tool_content) > 1000:
                    tool_content = tool_content[:500] + "\n... [truncated] ...\n" + tool_content[-500:]
                parts.append(f"TOOL OUTPUT [{msg.get('tool_call_id', '')}]: {tool_content}")
            else:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def rebuild_after_compaction(
        self,
        system_prompt: str,
        original_prompt: str,
        summary: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_prompt},
            {"role": "user", "content": f"[COMPACTED CONVERSATION SUMMARY]\n\n{summary}\n\nPlease continue working on the task based on the summary above. Resume from where you left off."},
        ]

    async def call_for_compaction(
        self,
        conversation_text: str,
        compaction_prompt: str,
        max_tokens: int,
    ) -> str:
        messages = [
            {"role": "user", "content": f"Here is the conversation history:\n\n{conversation_text}\n\n{compaction_prompt}"},
        ]
        uses_completion_tokens = any(self.model.startswith(p) for p in ("gpt-5", "o3", "o4", "o1"))
        token_key = "max_completion_tokens" if uses_completion_tokens else "max_tokens"
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **{token_key: max_tokens},
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""
