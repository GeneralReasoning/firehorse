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
    "gemini-2.5-pro": 2000000,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-flash-lite": 1048576,
    "gemini-2.0-flash": 1048576,
}


def _sanitize_schema(schema: dict | None) -> dict | None:
    if schema is None:
        return None
    from openreward.api.environments.client import _sanitize_google_schema, _strip_titles
    return _sanitize_google_schema(_strip_titles(schema))


class GoogleProvider(ProviderClient):
    """Google Gemini provider."""

    def __init__(self, model: str, api_key: str, context_window: int | None = None):
        try:
            from google import genai
            from google.genai import types  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-genai package required for Google provider. "
                "Install with: pip install 'firehorse[resum]'"
            )
        self.model = model
        self._context_window_override = context_window
        self._client = genai.Client(api_key=api_key)
        self._system_prompt: str = ""
        # Track tool call ID mapping since Gemini doesn't use IDs
        self._call_counter = 0

    @property
    def context_window(self) -> int | None:
        if self._context_window_override is not None:
            return self._context_window_override
        return KNOWN_CONTEXT_WINDOWS.get(self.model)

    def _next_call_id(self) -> str:
        self._call_counter += 1
        return f"call_{self._call_counter}"

    def format_tools(self, tools: list[ToolSpec]) -> list[Any]:
        from google.genai import types

        declarations = []
        for tool in tools:
            params = _sanitize_schema(dict(tool.input_schema) if tool.input_schema else None)
            declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=params or {"type": "object", "properties": {}},
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> list[Any]:
        from google.genai import types
        self._system_prompt = system_prompt
        return [
            types.Content(role="user", parts=[types.Part(text=user_prompt)]),
        ]

    def _build_safety_settings(self) -> list[Any]:
        from google.genai import types
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    def _build_thinking_config(self, effort: str | None) -> Any:
        """Build Gemini thinking config based on effort level and model version."""
        if not effort:
            return None
        from google.genai import types

        # Gemini 3.x uses thinking_level
        if any(v in self.model for v in ("3.0", "3.1", "3.2")):
            level = "high" if effort == "max" else effort
            return types.ThinkingConfig(thinking_level=level)

        # Gemini 2.5 uses thinking_budget_tokens
        budgets = {"low": 1024, "medium": 5000, "high": 16000, "max": 24576}
        return types.ThinkingConfig(thinking_budget_tokens=budgets.get(effort, 16000))

    async def call(
        self,
        messages: list[Any],
        tools: list[Any],
        max_tokens: int = 16384,
        effort: str | None = None,
    ) -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            tools=tools if tools else None,
            max_output_tokens=max_tokens,
            safety_settings=self._build_safety_settings(),
            system_instruction=self._system_prompt,
            thinking_config=self._build_thinking_config(effort),
        )

        last_err: Exception | None = None
        for attempt in range(5):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=config,
                )
                break
            except Exception as e:
                err_msg = str(e).lower()
                if any(kw in err_msg for kw in ("context", "too long", "token limit", "exceeds the maximum")):
                    return LLMResponse(raw_message=None, context_overflow=True)
                if attempt < 4 and any(kw in err_msg for kw in ("429", "500", "503", "rate", "overloaded")):
                    last_err = e
                    wait = min(2 ** attempt, 60)
                    print(f"[resum/google] Retry {attempt + 1}/5 after {type(e).__name__}, waiting {wait}s", file=sys.stderr)
                    await asyncio.sleep(wait)
                else:
                    raise
        else:
            raise last_err  # type: ignore[misc]

        if not response.candidates:
            return LLMResponse(raw_message=None, context_overflow=True)

        candidate = response.candidates[0]
        parts = candidate.content.parts if candidate.content else []

        tool_calls = []
        text_parts = []
        reasoning_parts = []

        # Build a native Content object to store as raw_message
        response_parts = []
        # Map from function call name+args to our generated call_id
        call_id_map: dict[str, str] = {}

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                call_id = self._next_call_id()
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCallInfo(
                    id=call_id,
                    name=fc.name,
                    arguments=args,
                ))
                call_id_map[fc.name] = call_id
                response_parts.append(part)
            elif hasattr(part, "thought") and part.thought:
                reasoning_parts.append(part.text if hasattr(part, "text") and part.text else "")
                response_parts.append(part)
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)
                response_parts.append(part)

        raw_content = types.Content(role="model", parts=response_parts) if response_parts else None

        input_tokens = None
        output_tokens = None
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        return LLMResponse(
            raw_message=raw_content,
            tool_calls=tool_calls,
            text_content="\n".join(text_parts) if text_parts else None,
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def append_assistant(self, messages: list[Any], response: LLMResponse) -> None:
        if response.raw_message is not None:
            messages.append(response.raw_message)

    def append_tool_result(
        self,
        messages: list[Any],
        call_id: str,
        tool_name: str,
        output: str,
    ) -> None:
        from google.genai import types

        part = types.Part.from_function_response(
            name=tool_name,
            response={"result": output},
        )
        # Check if last message is already a user/function_response content we can append to
        if messages and hasattr(messages[-1], "role") and messages[-1].role == "user":
            existing_parts = messages[-1].parts or []
            if existing_parts and any(hasattr(p, "function_response") and p.function_response for p in existing_parts):
                messages[-1] = types.Content(role="user", parts=list(existing_parts) + [part])
                return
        messages.append(types.Content(role="user", parts=[part]))

    def append_user_message(self, messages: list[Any], content: str) -> None:
        from google.genai import types
        messages.append(types.Content(role="user", parts=[types.Part(text=content)]))

    def messages_to_text(self, messages: list[Any]) -> str:
        parts = []
        for msg in messages:
            role = getattr(msg, "role", "unknown").upper()
            if role == "MODEL":
                role = "ASSISTANT"
            pieces = []
            for part in getattr(msg, "parts", []):
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    pieces.append(f"[Tool Call] {fc.name}({json.dumps(dict(fc.args) if fc.args else {})})")
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    result_str = str(fr.response.get("result", "")) if fr.response else ""
                    if len(result_str) > 1000:
                        result_str = result_str[:500] + "\n... [truncated] ...\n" + result_str[-500:]
                    pieces.append(f"[Tool Result {fr.name}]: {result_str}")
                elif hasattr(part, "text") and part.text:
                    pieces.append(part.text)
            parts.append(f"{role}: {' '.join(pieces)}")
        return "\n\n".join(parts)

    def rebuild_after_compaction(
        self,
        system_prompt: str,
        original_prompt: str,
        summary: str,
    ) -> list[Any]:
        from google.genai import types
        self._system_prompt = system_prompt
        return [
            types.Content(role="user", parts=[types.Part(text=original_prompt)]),
            types.Content(role="model", parts=[types.Part(text="I'll continue working on this task.")]),
            types.Content(role="user", parts=[
                types.Part(text=f"[COMPACTED CONVERSATION SUMMARY]\n\n{summary}\n\nPlease continue working on the task based on the summary above. Resume from where you left off."),
            ]),
        ]

    async def call_for_compaction(
        self,
        conversation_text: str,
        compaction_prompt: str,
        max_tokens: int,
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            safety_settings=self._build_safety_settings(),
        )
        messages = [
            types.Content(role="user", parts=[
                types.Part(text=f"Here is the conversation history:\n\n{conversation_text}\n\n{compaction_prompt}"),
            ]),
        ]
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=messages,
            config=config,
        )
        if response.candidates and response.candidates[0].content:
            text_parts = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
            return "\n".join(text_parts)
        return ""
