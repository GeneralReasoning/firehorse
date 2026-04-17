"""Prefix cache for LLM responses.

Intercepts LLM calls and caches responses keyed by the full conversation
prefix (messages + model + tools + system prompt + effort). On a cache hit,
the stored response is returned without an API call. Tool calls are still
executed against the live environment — only the LLM cost is saved.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from firehorse.agents.resum.providers.base import (
    LLMResponse,
    ProviderClient,
    ToolCallInfo,
)

if TYPE_CHECKING:
    from openreward.api.environments.types import ToolSpec


# ---------------------------------------------------------------------------
# Message serialization (deterministic, per-provider)
# ---------------------------------------------------------------------------

def _serialize_google_content(content: Any) -> list[dict]:
    """Convert a Google Content object to a JSON-serializable list of part dicts."""
    parts = []
    for part in content.parts:
        if part.text is not None:
            parts.append({"text": part.text})
        elif part.function_call:
            fc = part.function_call
            parts.append({
                "function_call": {
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                },
            })
        elif part.function_response:
            fr = part.function_response
            parts.append({
                "function_response": {
                    "name": fr.name,
                    "response": dict(fr.response) if fr.response else {},
                },
            })
        elif getattr(part, "thought", None):
            parts.append({"thought": True, "text": part.text or ""})
        else:
            parts.append({"raw": str(part)})
    return [{"role": content.role, "parts": parts}]


def serialize_messages(messages: list[Any], provider_name: str) -> str:
    """Serialize a provider-native message list to a deterministic JSON string."""
    if provider_name == "google":
        serialized = []
        for msg in messages:
            serialized.extend(_serialize_google_content(msg))
        return json.dumps(serialized, sort_keys=True, ensure_ascii=False, default=str)
    # Anthropic, OpenAI, OpenRouter — messages are already dicts
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Response serialization (provider-agnostic normalized format)
# ---------------------------------------------------------------------------

def serialize_response(response: LLMResponse) -> dict:
    """Serialize an LLMResponse to a JSON-storable dict (without raw_message)."""
    return {
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        ],
        "text_content": response.text_content,
        "reasoning_content": response.reasoning_content,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }


def deserialize_response(data: dict) -> LLMResponse:
    """Reconstruct an LLMResponse from a cached dict.

    The raw_message is set to None — callers that need it (e.g.
    append_assistant) must handle the cache_hit case specially.
    """
    tool_calls = [
        ToolCallInfo(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
        for tc in data.get("tool_calls", [])
    ]
    return LLMResponse(
        raw_message=None,
        tool_calls=tool_calls,
        text_content=data.get("text_content"),
        reasoning_content=data.get("reasoning_content"),
        input_tokens=data.get("input_tokens"),
        output_tokens=data.get("output_tokens"),
        cache_hit=True,
    )


# ---------------------------------------------------------------------------
# PrefixCache
# ---------------------------------------------------------------------------

class PrefixCache:
    """On-disk SHA-256-keyed cache for LLM responses."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def compute_key(
        self,
        model: str,
        system_prompt: str,
        tools: Any,
        messages: list[Any],
        effort: str | None,
        provider_name: str,
    ) -> str:
        """Compute a SHA-256 cache key from the full request prefix."""
        blob = json.dumps(
            {
                "provider": provider_name,
                "model": model,
                "system_prompt": system_prompt,
                "tools": tools,
                "effort": effort,
                "messages": serialize_messages(messages, provider_name),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path_for(self, key: str) -> Path:
        shard = key[:8]
        return self.cache_dir / shard / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Look up a cached response. Returns the response dict on hit, None on miss."""
        path = self._path_for(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text())
            self.hits += 1
            return data["response"]
        except (json.JSONDecodeError, KeyError):
            self.misses += 1
            return None

    def put(self, key: str, response: dict, model: str = "", provider: str = "") -> None:
        """Store a response in the cache (atomic write)."""
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "version": 1,
            "cache_key": key,
            "model": model,
            "provider": provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": response,
        }
        content = json.dumps(entry, indent=2)
        # Atomic write: write to temp file in the same dir, then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            os.write(fd, content.encode())
            os.close(fd)
            fd = -1  # mark as closed
            os.rename(tmp_path, str(path))
        except Exception:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# CachingProviderClient — wraps a ProviderClient for ReSum agent caching
# ---------------------------------------------------------------------------

class CachingProviderClient(ProviderClient):
    """Transparent caching wrapper around any ProviderClient.

    Intercepts ``call()`` with prefix-cache lookup/store.
    All other methods delegate directly to the inner provider.
    """

    def __init__(self, inner: ProviderClient, cache: PrefixCache, provider_name: str):
        self.inner = inner
        self.cache = cache
        self.provider_name = provider_name
        self._system_prompt: str = ""

    # -- Delegated methods --------------------------------------------------

    @property
    def context_window(self) -> int | None:
        return self.inner.context_window

    def format_tools(self, tools: list[ToolSpec]) -> Any:
        return self.inner.format_tools(tools)

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> list[Any]:
        self._system_prompt = system_prompt
        return self.inner.build_initial_messages(system_prompt, user_prompt)

    def append_assistant(self, messages: list[Any], response: LLMResponse) -> None:
        if response.cache_hit:
            # raw_message is None for cached responses — build the message
            # from the normalized fields instead.
            self._append_assistant_from_cache(messages, response)
        else:
            self.inner.append_assistant(messages, response)

    def append_tool_result(
        self, messages: list[Any], call_id: str, tool_name: str, output: str,
    ) -> None:
        self.inner.append_tool_result(messages, call_id, tool_name, output)

    def append_user_message(self, messages: list[Any], content: str) -> None:
        self.inner.append_user_message(messages, content)

    def messages_to_text(self, messages: list[Any]) -> str:
        return self.inner.messages_to_text(messages)

    def rebuild_after_compaction(
        self, system_prompt: str, original_prompt: str, summary: str,
    ) -> list[Any]:
        self._system_prompt = system_prompt
        return self.inner.rebuild_after_compaction(system_prompt, original_prompt, summary)

    async def call_for_compaction(
        self, conversation_text: str, compaction_prompt: str, max_tokens: int,
    ) -> str:
        # Compaction calls are not cached — they're internal summarization
        return await self.inner.call_for_compaction(
            conversation_text, compaction_prompt, max_tokens,
        )

    # -- Cached call --------------------------------------------------------

    async def call(
        self,
        messages: list[Any],
        tools: Any,
        max_tokens: int = 16384,
        effort: str | None = None,
    ) -> LLMResponse:
        model = getattr(self.inner, "model", "unknown")

        key = self.cache.compute_key(
            model=model,
            system_prompt=self._system_prompt,
            tools=tools,
            messages=messages,
            effort=effort,
            provider_name=self.provider_name,
        )

        cached = self.cache.get(key)
        if cached is not None:
            print(f"[cache] HIT for prefix (key={key[:12]}...)", file=sys.stderr)
            return deserialize_response(cached)

        response = await self.inner.call(messages, tools, max_tokens, effort)

        # Don't cache overflow responses
        if not response.context_overflow:
            self.cache.put(
                key,
                serialize_response(response),
                model=model,
                provider=self.provider_name,
            )

        return response

    # -- Helpers ------------------------------------------------------------

    def _append_assistant_from_cache(
        self, messages: list[Any], response: LLMResponse,
    ) -> None:
        """Append a cached response to the message list using provider-native format."""
        if self.provider_name == "anthropic":
            content: list[dict] = []
            if response.reasoning_content:
                content.append({
                    "type": "thinking",
                    "thinking": response.reasoning_content,
                    "signature": "",  # placeholder — not needed for cache replay
                })
            if response.text_content:
                content.append({"type": "text", "text": response.text_content})
            for tc in response.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            messages.append({"role": "assistant", "content": content})

        elif self.provider_name == "openai":
            # Responses API format: separate items for text and function calls
            if response.text_content:
                messages.append({"role": "assistant", "content": response.text_content})
            for tc in response.tool_calls:
                messages.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                })

        elif self.provider_name == "openrouter":
            # Chat Completions format
            entry: dict[str, Any] = {"role": "assistant"}
            if response.text_content:
                entry["content"] = response.text_content
            if response.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
            messages.append(entry)

        elif self.provider_name == "google":
            from google.genai import types
            parts = []
            if response.text_content:
                parts.append(types.Part(text=response.text_content))
            for tc in response.tool_calls:
                parts.append(types.Part.from_function_call(
                    name=tc.name, args=tc.arguments,
                ))
            messages.append(types.Content(role="model", parts=parts))

        else:
            raise ValueError(f"Unknown provider for cache replay: {self.provider_name}")


# ---------------------------------------------------------------------------
# Trajectory warming
# ---------------------------------------------------------------------------

def warm_from_trajectories(
    cache: PrefixCache,
    trajectory_dir: str | Path,
    model: str,
    provider_url: str | None = None,
    context_window: int | None = None,
) -> int:
    """Warm the prefix cache from existing JSONL trajectory files.

    Replays each trajectory through the actual provider's message-building
    methods so that cache keys match what a live run would produce.

    Returns the number of entries added to the cache.
    """
    from firehorse.agents.resum.providers import parse_provider, get_provider

    trajectory_dir = Path(trajectory_dir)
    jsonl_files = sorted(trajectory_dir.glob("trial_*.jsonl"))

    if not jsonl_files:
        print(f"[cache] No trajectory files found in {trajectory_dir}", file=sys.stderr)
        return 0

    provider_name, model_id = parse_provider(model, provider_url)

    # Instantiate provider with dummy key — only used for message formatting
    provider = get_provider(provider_name, model_id, api_key="cache-warming-dummy", provider_url=provider_url, context_window=context_window)

    added = 0
    for jsonl_path in jsonl_files:
        try:
            added += _warm_single_trajectory(cache, jsonl_path, provider, provider_name, model_id)
        except Exception as e:
            print(f"[cache] Failed to warm from {jsonl_path.name}: {e}", file=sys.stderr)

    print(f"[cache] Warmed {added} entries from {len(jsonl_files)} trajectory files", file=sys.stderr)
    return added


def _warm_single_trajectory(
    cache: PrefixCache,
    jsonl_path: Path,
    provider: ProviderClient,
    provider_name: str,
    model_id: str,
) -> int:
    """Replay a single trajectory JSONL and populate the cache."""
    events: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Find the prompt event to get system_prompt and environment_prompt
    system_prompt = ""
    environment_prompt = ""
    tools_json: Any = None

    for ev in events:
        if ev.get("type") == "openreward_prompt":
            system_prompt = ev.get("system_prompt", "")
            environment_prompt = ev.get("environment_prompt", "")
            break

    if not environment_prompt:
        return 0

    # Build initial messages
    messages = provider.build_initial_messages(system_prompt, environment_prompt)

    # We need the formatted tools for the cache key. Since we don't have
    # the ToolSpec objects during warming, we use an empty list — this means
    # warmed cache entries will only match if the live run also uses the same
    # tools. For ReSum trajectories the tools don't change mid-run, so this
    # is safe as long as the environment's tools haven't changed.
    #
    # TODO: Store formatted tools in trajectory for exact matching.
    if tools_json is None:
        tools_json = []

    added = 0
    effort = None

    # Extract effort from summary event if present
    for ev in events:
        if ev.get("type") == "openreward_summary":
            # Try to find effort in the metadata
            break

    for ev in events:
        etype = ev.get("type")

        if etype == "assistant":
            # Compute cache key from the current prefix
            key = cache.compute_key(
                model=model_id,
                system_prompt=system_prompt,
                tools=tools_json,
                messages=messages,
                effort=effort,
                provider_name=provider_name,
            )

            # Build an LLMResponse from the event data
            tool_calls_data = ev.get("tool_calls", [])
            tool_calls = [
                ToolCallInfo(
                    id=tc.get("id", f"warm_{i}"),
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                )
                for i, tc in enumerate(tool_calls_data)
            ]

            response = LLMResponse(
                raw_message=None,
                tool_calls=tool_calls,
                text_content=ev.get("text_content"),
                reasoning_content=ev.get("reasoning_content"),
                input_tokens=ev.get("input_tokens"),
                output_tokens=ev.get("output_tokens"),
                cache_hit=True,
            )

            # Store in cache
            cache.put(
                key,
                serialize_response(response),
                model=model_id,
                provider=provider_name,
            )
            added += 1

            # Append to messages using provider-native format
            _append_assistant_for_warming(messages, response, provider_name)

        elif etype == "tool_result":
            call_id = ev.get("tool_call_id", "")
            tool_name = ev.get("tool_name", "")
            output = ev.get("output", "")
            provider.append_tool_result(messages, call_id, tool_name, output)

        elif etype in ("compaction", "micro_compaction"):
            # After compaction the message list is rebuilt — we can't
            # continue replaying accurately, so stop here.
            break

    return added


def _append_assistant_for_warming(
    messages: list[Any],
    response: LLMResponse,
    provider_name: str,
) -> None:
    """Append a cached response to messages during warming (no raw_message)."""
    if provider_name == "anthropic":
        content: list[dict] = []
        if response.reasoning_content:
            content.append({
                "type": "thinking",
                "thinking": response.reasoning_content,
                "signature": "",
            })
        if response.text_content:
            content.append({"type": "text", "text": response.text_content})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        messages.append({"role": "assistant", "content": content})

    elif provider_name == "openai":
        # Responses API format
        if response.text_content:
            messages.append({"role": "assistant", "content": response.text_content})
        for tc in response.tool_calls:
            messages.append({
                "type": "function_call",
                "call_id": tc.id,
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            })

    elif provider_name == "openrouter":
        # Chat Completions format
        entry: dict[str, Any] = {"role": "assistant"}
        if response.text_content:
            entry["content"] = response.text_content
        if response.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        messages.append(entry)

    elif provider_name == "google":
        from google.genai import types
        parts = []
        if response.text_content:
            parts.append(types.Part(text=response.text_content))
        for tc in response.tool_calls:
            parts.append(types.Part.from_function_call(
                name=tc.name, args=tc.arguments,
            ))
        messages.append(types.Content(role="model", parts=parts))
