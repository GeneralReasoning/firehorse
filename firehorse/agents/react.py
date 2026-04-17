"""ReAct agent — direct LLM tool-use loop over OpenReward sessions.

Supports: anthropic, openai, google, openrouter providers.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, IO

from openreward import SystemMessage, UserMessage
from openreward.models import RolloutInfo

from firehorse.agents.base import BaseAgent, AgentResult, TrialContext

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from google import genai as google_genai
    from google.genai import types as google_types
except ImportError:
    google_genai = None
    google_types = None


def _require(module: Any, package: str, provider: str) -> None:
    if module is None:
        raise RuntimeError(
            f"install '{package}' to use the {provider} provider "
            f"(pip install {package})"
        )


KNOWN_PROVIDERS = ("anthropic", "openai", "google", "openrouter")

SYSTEM_PROMPT = (
    "You are an agent solving a task in an OpenReward environment. "
    "Use the tools provided to complete the task."
)


def _parse_provider(model: str) -> tuple[str, str]:
    """Split 'provider/model-name' into (provider, model_name)."""
    for prefix in KNOWN_PROVIDERS:
        if model.startswith(prefix + "/"):
            return prefix, model[len(prefix) + 1:]
    raise ValueError(
        f"Unknown provider in model {model!r}. "
        f"Expected one of: {', '.join(p + '/...' for p in KNOWN_PROVIDERS)}"
    )


def _format_tool_output(output: Any) -> str:
    """Extract text from ToolOutput blocks. Images become placeholders."""
    parts: list[str] = []
    for block in getattr(output, "blocks", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        elif getattr(block, "type", None) == "image":
            mime = getattr(block, "mimeType", "image")
            parts.append(f"[Image: {mime}]")
        else:
            parts.append(str(block))
    return "".join(parts)


def _format_tool_output_anthropic(output: Any) -> list[dict]:
    """Return Anthropic-native content blocks (text + base64 images)."""
    blocks: list[dict] = []
    for block in getattr(output, "blocks", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            blocks.append({"type": "text", "text": text})
        elif getattr(block, "type", None) == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": getattr(block, "mimeType", "image/png"),
                    "data": getattr(block, "data", ""),
                },
            })
    return blocks or [{"type": "text", "text": ""}]


def _jsonl_write(log_file: IO | None, event: dict) -> None:
    if log_file is not None:
        log_file.write(json.dumps(event, default=str) + "\n")


def _format_openrouter_tool(tool: Any) -> dict:
    # Worked around here because openreward's built-in format="openrouter"
    # nests `parameters` outside `function`, which Chat Completions silently
    # ignores — models then see no schema and emit empty `{}` args.
    from openreward.api.environments.client import _sanitize_openai_schema, _strip_titles

    schema = dict(tool.input_schema) if getattr(tool, "input_schema", None) else None
    params = _sanitize_openai_schema(_strip_titles(schema)) if schema else None
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


class ReactAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "react"

    async def setup(self) -> None:
        pass

    async def run(self, ctx: TrialContext) -> AgentResult:
        provider, model_name = _parse_provider(ctx.model)
        trial_id = ctx.task_spec.get("id", ctx.task_spec.get("index", ctx.task_index))
        log_dir = Path(ctx.output_dir) if ctx.output_dir else None
        start = time.monotonic()

        # --- Rollout ---
        rollout = None
        if ctx.logging and ctx.rollout_client:
            try:
                rollout = ctx.rollout_client.rollout.create(
                    run_name=ctx.run_name,
                    rollout_name=f"trial_{trial_id}",
                    environment=ctx.env_name,
                    variant=ctx.variant,
                    split=ctx.split,
                    task_spec=ctx.task_spec,
                    metadata={},
                )
                print(
                    f"[react] Rollout: https://openreward.ai/rollout/{rollout.event_id}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[react] Failed to create rollout: {e}", file=sys.stderr)

        # --- JSONL log ---
        log_file = None
        if log_dir:
            log_file = open(log_dir / f"trial_{trial_id}.jsonl", "w")

        # Log initial prompt
        _jsonl_write(log_file, {"type": "system", "content": SYSTEM_PROMPT})
        _jsonl_write(log_file, {"type": "user", "content": ctx.prompt_text})

        if rollout:
            rollout.log(
                SystemMessage(content=SYSTEM_PROMPT),
                rollout_info=RolloutInfo(task_index=ctx.task_index, harness="react"),
            )
            rollout.log(UserMessage(content=ctx.prompt_text))

        print(f"[react] Launching with provider={provider} model={model_name}", file=sys.stderr)

        # --- Run provider loop ---
        result: AgentResult | None = None
        try:
            if provider == "anthropic":
                result = await self._run_anthropic(ctx, model_name, rollout, log_file, start)
            elif provider == "openai":
                result = await self._run_openai(ctx, model_name, rollout, log_file, start)
            elif provider == "google":
                result = await self._run_google(ctx, model_name, rollout, log_file, start)
            elif provider == "openrouter":
                result = await self._run_openrouter(ctx, model_name, rollout, log_file, start)
            else:
                result = AgentResult(success=False, error=f"Unsupported provider: {provider}")
        except Exception as e:
            result = AgentResult(success=False, error=str(e))

        duration_ms = int((time.monotonic() - start) * 1000)
        if result.duration_ms is None:
            result.duration_ms = duration_ms

        # --- Write summary + result JSON ---
        try:
            _jsonl_write(log_file, {
                "type": "openreward_summary",
                "task_spec": ctx.task_spec,
                "env": ctx.env_name,
                "model": ctx.model,
                "agent": "react",
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "duration_ms": result.duration_ms,
                },
                "reward": result.reward,
                "finished": result.finished,
            })

            if log_dir:
                trial_result = {
                    "task_id": trial_id,
                    "task_spec": ctx.task_spec,
                    "environment": ctx.env_name,
                    "agent": "react",
                    "model": ctx.model,
                    "split": ctx.split,
                    "final_reward": result.reward,
                    "finished": result.finished,
                    "total_reward": result.reward,
                    "tool_calls": result.turns_used,
                    "duration_seconds": (result.duration_ms / 1000) if result.duration_ms else None,
                    "usage": {
                        "cost_usd": result.cost_usd,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    },
                    "error": result.error,
                    "rollout_url": (
                        f"https://openreward.ai/rollout/{rollout.event_id}"
                        if rollout else None
                    ),
                }
                (log_dir / f"trial_{trial_id}_result.json").write_text(
                    json.dumps(trial_result, indent=2)
                )
        finally:
            if log_file is not None:
                log_file.close()

        return result

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    async def _run_anthropic(
        self,
        ctx: TrialContext,
        model_name: str,
        rollout: Any,
        log_file: IO | None,
        trial_start: float,
    ) -> AgentResult:
        _require(anthropic, "anthropic", "anthropic")

        def _final_info() -> RolloutInfo:
            return RolloutInfo(
                task_index=ctx.task_index,
                duration_ms=int((time.monotonic() - trial_start) * 1000),
            )

        kwargs: dict[str, Any] = {}
        if ctx.provider_url:
            kwargs["base_url"] = ctx.provider_url
        client = anthropic.AsyncAnthropic(**kwargs)

        tools = await ctx.session.list_tools(format="anthropic")
        messages: list[dict] = [{"role": "user", "content": ctx.prompt_text}]

        turns_used = 0
        finished = False
        last_reward: float | None = None
        thinking_supported = True
        total_input = 0
        total_output = 0

        while not finished:
            if ctx.max_turns and turns_used >= ctx.max_turns:
                break

            create_kwargs: dict[str, Any] = {
                "model": model_name,
                "max_tokens": 16384,
                "system": SYSTEM_PROMPT,
                "tools": tools,
                "messages": messages,
            }
            if ctx.effort and thinking_supported:
                create_kwargs["thinking"] = {"type": "adaptive"}
                create_kwargs["output_config"] = {"effort": ctx.effort}

            # --- Prefix cache check ---
            cache_hit = False
            cached_response = None
            if ctx.prefix_cache:
                from firehorse.cache import serialize_response as _sr, deserialize_response
                cache_key = ctx.prefix_cache.compute_key(
                    model=model_name, system_prompt=SYSTEM_PROMPT,
                    tools=tools, messages=messages,
                    effort=ctx.effort, provider_name="anthropic",
                )
                cached_data = ctx.prefix_cache.get(cache_key)
                if cached_data is not None:
                    cached_response = deserialize_response(cached_data)
                    cache_hit = True
                    print(f"[react/anthropic] Cache HIT (key={cache_key[:12]}...)", file=sys.stderr)

            if not cache_hit:
                try:
                    message = await client.messages.create(**create_kwargs)
                except Exception as e:
                    if "thinking" in str(e).lower() and "not supported" in str(e).lower():
                        thinking_supported = False
                        create_kwargs.pop("thinking", None)
                        create_kwargs.pop("output_config", None)
                        print(f"\n⚠  WARNING: {model_name} does not support thinking. --effort will be ignored.\n", file=sys.stderr)
                        message = await client.messages.create(**create_kwargs)
                    else:
                        raise
                total_input += message.usage.input_tokens
                total_output += message.usage.output_tokens

            # Serialize content for logging
            assistant_content = []
            if cache_hit:
                if cached_response.reasoning_content:
                    assistant_content.append({
                        "type": "thinking",
                        "thinking": cached_response.reasoning_content,
                    })
                if cached_response.text_content:
                    assistant_content.append({"type": "text", "text": cached_response.text_content})
                for tc in cached_response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
            else:
                for block in message.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "thinking":
                        assistant_content.append({
                            "type": "thinking",
                            "thinking": getattr(block, "thinking", ""),
                        })
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                    else:
                        assistant_content.append({"type": block.type})

            assistant_msg = {"role": "assistant", "content": assistant_content}
            log_event: dict[str, Any] = {"type": "assistant", "provider": "anthropic", "raw": assistant_msg}
            if cache_hit:
                log_event["cache_hit"] = True
            _jsonl_write(log_file, log_event)
            if rollout:
                rollout.log_anthropic_message(assistant_msg)

            # Append to messages
            if cache_hit:
                messages.append({"role": "assistant", "content": assistant_content})
            else:
                messages.append({"role": "assistant", "content": message.content})
                # Store in cache
                if ctx.prefix_cache:
                    from firehorse.agents.resum.providers.base import LLMResponse, ToolCallInfo
                    tc_infos = []
                    text_parts = []
                    reasoning_parts = []
                    for block in message.content:
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tc_infos.append(ToolCallInfo(id=block.id, name=block.name, arguments=block.input if isinstance(block.input, dict) else {}))
                        elif block.type == "thinking":
                            reasoning_parts.append(block.thinking)
                    resp = LLMResponse(
                        raw_message=None, tool_calls=tc_infos,
                        text_content="\n".join(text_parts) if text_parts else None,
                        reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
                        input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens,
                    )
                    ctx.prefix_cache.put(cache_key, _sr(resp), model=model_name, provider="anthropic")

            # Check stop reason
            has_tool_use = any(b.get("type") == "tool_use" for b in assistant_content) if cache_hit else (message.stop_reason == "tool_use")
            if not has_tool_use:
                break

            tool_result_blocks: list[dict] = []
            for block in assistant_content:
                if block.get("type") != "tool_use":
                    continue
                turns_used += 1
                try:
                    tr = await ctx.session.call_tool(block["name"], block["input"])
                    content = _format_tool_output_anthropic(tr)

                    if tr.finished:
                        finished = True
                    if tr.reward is not None:
                        last_reward = tr.reward

                    _jsonl_write(log_file, {
                        "type": "tool_result",
                        "provider": "anthropic",
                        "tool_use_id": block["id"],
                        "tool_name": block["name"],
                        "reward": tr.reward,
                        "finished": tr.finished,
                    })

                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": content,
                    })
                except Exception as e:
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": f"Error: {e}",
                        "is_error": True,
                    })

            user_msg = {"role": "user", "content": tool_result_blocks}
            _jsonl_write(log_file, {"type": "tool_results", "provider": "anthropic", "raw": user_msg})
            if rollout:
                rollout.log_anthropic_message(
                    user_msg,
                    rollout_info=_final_info() if finished else None,
                )
            messages.append(user_msg)

        return AgentResult(
            success=True,
            reward=last_reward,
            finished=finished,
            turns_used=turns_used,
            input_tokens=total_input,
            output_tokens=total_output,
        )

    # ------------------------------------------------------------------
    # OpenAI (responses API)
    # ------------------------------------------------------------------

    async def _run_openai(
        self,
        ctx: TrialContext,
        model_name: str,
        rollout: Any,
        log_file: IO | None,
        trial_start: float,
    ) -> AgentResult:
        _require(AsyncOpenAI, "openai", "openai")

        def _final_info() -> RolloutInfo:
            return RolloutInfo(
                task_index=ctx.task_index,
                duration_ms=int((time.monotonic() - trial_start) * 1000),
            )

        kwargs: dict[str, Any] = {}
        if ctx.provider_url:
            kwargs["base_url"] = ctx.provider_url
        client = AsyncOpenAI(**kwargs)

        tools = await ctx.session.list_tools(format="openai")
        input_list: list[Any] = [{"role": "user", "content": ctx.prompt_text}]

        turns_used = 0
        finished = False
        last_reward: float | None = None
        total_input = 0
        total_output = 0
        reasoning_supported = True

        while not finished:
            if ctx.max_turns and turns_used >= ctx.max_turns:
                break

            resp_kwargs: dict[str, Any] = {
                "model": model_name,
                "tools": tools,
                "input": input_list,
                "instructions": SYSTEM_PROMPT,
            }
            if ctx.effort and reasoning_supported:
                mapped = "high" if ctx.effort == "max" else ctx.effort
                resp_kwargs["reasoning"] = {"effort": mapped}

            # --- Prefix cache check ---
            cache_hit = False
            cached_response = None
            if ctx.prefix_cache:
                from firehorse.cache import serialize_response as _sr, deserialize_response
                cache_key = ctx.prefix_cache.compute_key(
                    model=model_name, system_prompt=SYSTEM_PROMPT,
                    tools=tools, messages=input_list,
                    effort=ctx.effort, provider_name="openai",
                )
                cached_data = ctx.prefix_cache.get(cache_key)
                if cached_data is not None:
                    cached_response = deserialize_response(cached_data)
                    cache_hit = True
                    print(f"[react/openai] Cache HIT (key={cache_key[:12]}...)", file=sys.stderr)

            if not cache_hit:
                try:
                    response = await client.responses.create(**resp_kwargs)
                except Exception as e:
                    if "reasoning" in str(e).lower() and "not supported" in str(e).lower():
                        reasoning_supported = False
                        resp_kwargs.pop("reasoning", None)
                        print(f"\n⚠  WARNING: {model_name} does not support reasoning. --effort will be ignored.\n", file=sys.stderr)
                        response = await client.responses.create(**resp_kwargs)
                    else:
                        raise

                if response.usage:
                    total_input += response.usage.input_tokens
                    total_output += response.usage.output_tokens

            # Convert output to serializable dicts
            output_items: list[dict] = []
            if cache_hit:
                if cached_response.text_content:
                    output_items.append({"role": "assistant", "content": cached_response.text_content})
                for tc in cached_response.tool_calls:
                    output_items.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    })
            else:
                for item in response.output:
                    item_type = getattr(item, "type", None)
                    if item_type == "function_call":
                        output_items.append({
                            "type": "function_call",
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        })
                    elif item_type == "message":
                        text = ""
                        for cb in getattr(item, "content", None) or []:
                            if getattr(cb, "type", None) == "output_text":
                                text += cb.text
                        if text:
                            output_items.append({"role": "assistant", "content": text})

            log_event_oi: dict[str, Any] = {"type": "assistant", "provider": "openai", "raw": output_items}
            if cache_hit:
                log_event_oi["cache_hit"] = True
            _jsonl_write(log_file, log_event_oi)
            if not cache_hit and rollout:
                rollout.log_openai_response(response)

            input_list.extend(output_items)

            # Store in cache on miss
            if not cache_hit and ctx.prefix_cache:
                from firehorse.cache import serialize_response as _sr
                from firehorse.agents.resum.providers.base import LLMResponse, ToolCallInfo
                tc_infos = []
                text_parts = []
                for oi in output_items:
                    if oi.get("type") == "function_call":
                        try:
                            args = json.loads(oi["arguments"]) if isinstance(oi["arguments"], str) else oi["arguments"]
                        except json.JSONDecodeError:
                            args = {}
                        tc_infos.append(ToolCallInfo(id=oi["call_id"], name=oi["name"], arguments=args))
                    elif oi.get("role") == "assistant":
                        text_parts.append(oi.get("content", ""))
                resp = LLMResponse(
                    raw_message=None, tool_calls=tc_infos,
                    text_content="\n".join(text_parts) if text_parts else None,
                    input_tokens=response.usage.input_tokens if response.usage else None,
                    output_tokens=response.usage.output_tokens if response.usage else None,
                )
                ctx.prefix_cache.put(cache_key, _sr(resp), model=model_name, provider="openai")

            # Execute tool calls
            has_tool_call = False
            for oi in output_items:
                if oi.get("type") != "function_call":
                    continue
                has_tool_call = True
                turns_used += 1
                try:
                    args = json.loads(oi["arguments"]) if isinstance(oi["arguments"], str) else oi["arguments"]
                    tr = await ctx.session.call_tool(oi["name"], args)
                    text = _format_tool_output(tr)
                    if tr.finished:
                        finished = True
                    if tr.reward is not None:
                        last_reward = tr.reward

                    output_item = {
                        "type": "function_call_output",
                        "call_id": oi["call_id"],
                        "output": text,
                    }
                    input_list.append(output_item)
                    _jsonl_write(log_file, {
                        "type": "tool_result",
                        "provider": "openai",
                        "raw": output_item,
                        "reward": tr.reward,
                        "finished": tr.finished,
                    })
                    if rollout:
                        rollout.log_openai_response(
                            output_item,
                            rollout_info=_final_info() if finished else None,
                        )
                except Exception as e:
                    output_item = {
                        "type": "function_call_output",
                        "call_id": oi["call_id"],
                        "output": f"Error: {e}",
                    }
                    input_list.append(output_item)

                if finished:
                    break

            if not has_tool_call:
                break

        return AgentResult(
            success=True,
            reward=last_reward,
            finished=finished,
            turns_used=turns_used,
            input_tokens=total_input,
            output_tokens=total_output,
        )

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------

    async def _run_google(
        self,
        ctx: TrialContext,
        model_name: str,
        rollout: Any,
        log_file: IO | None,
        trial_start: float,
    ) -> AgentResult:
        _require(google_genai, "google-genai", "google")
        genai = google_genai
        types = google_types

        def _final_info() -> RolloutInfo:
            return RolloutInfo(
                task_index=ctx.task_index,
                duration_ms=int((time.monotonic() - trial_start) * 1000),
            )

        client = genai.Client()
        tools_spec = await ctx.session.list_tools(format="google")
        genai_tools = [types.Tool(function_declarations=tools_spec)]
        thinking_config = None
        if ctx.effort:
            if any(v in model_name for v in ("3.0", "3.1", "3.2")):
                level = "high" if ctx.effort == "max" else ctx.effort
                thinking_config = types.ThinkingConfig(thinking_level=level)
            else:
                budgets = {"low": 1024, "medium": 5000, "high": 16000, "max": 24576}
                thinking_config = types.ThinkingConfig(
                    thinking_budget_tokens=budgets.get(ctx.effort, 16000)
                )
        config = types.GenerateContentConfig(
            tools=genai_tools,
            system_instruction=SYSTEM_PROMPT,
            thinking_config=thinking_config,
        )
        contents: list[Any] = [
            types.Content(role="user", parts=[types.Part(text=ctx.prompt_text)])
        ]

        turns_used = 0
        finished = False
        last_reward: float | None = None
        total_input = 0
        total_output = 0

        while not finished:
            if ctx.max_turns and turns_used >= ctx.max_turns:
                break

            # --- Prefix cache check ---
            cache_hit = False
            cached_response = None
            if ctx.prefix_cache:
                from firehorse.cache import serialize_response as _sr, deserialize_response
                cache_key = ctx.prefix_cache.compute_key(
                    model=model_name, system_prompt=SYSTEM_PROMPT,
                    tools=str(genai_tools), messages=contents,
                    effort=ctx.effort, provider_name="google",
                )
                cached_data = ctx.prefix_cache.get(cache_key)
                if cached_data is not None:
                    cached_response = deserialize_response(cached_data)
                    cache_hit = True
                    print(f"[react/google] Cache HIT (key={cache_key[:12]}...)", file=sys.stderr)

            if not cache_hit:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    config=config,
                    contents=contents,
                )

                if response.usage_metadata:
                    total_input += response.usage_metadata.prompt_token_count or 0
                    total_output += response.usage_metadata.candidates_token_count or 0

            if cache_hit:
                cached_parts = []
                if cached_response.text_content:
                    cached_parts.append(types.Part(text=cached_response.text_content))
                for tc in cached_response.tool_calls:
                    cached_parts.append(types.Part.from_function_call(
                        name=tc.name, args=tc.arguments,
                    ))
                candidate_content = types.Content(role="model", parts=cached_parts)
            else:
                candidate_content = response.candidates[0].content

            log_event_g: dict[str, Any] = {"type": "assistant", "provider": "google", "raw": str(candidate_content)}
            if cache_hit:
                log_event_g["cache_hit"] = True
            _jsonl_write(log_file, log_event_g)
            if rollout:
                rollout.log_gdm_message(candidate_content)
            contents.append(candidate_content)

            # Store in cache on miss
            if not cache_hit and ctx.prefix_cache:
                from firehorse.agents.resum.providers.base import LLMResponse, ToolCallInfo
                tc_infos = []
                text_parts_g = []
                for part in candidate_content.parts:
                    if part.function_call:
                        fc = part.function_call
                        tc_infos.append(ToolCallInfo(
                            id=f"google_{fc.name}_{turns_used}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        ))
                    elif part.text:
                        text_parts_g.append(part.text)
                resp = LLMResponse(
                    raw_message=None, tool_calls=tc_infos,
                    text_content="\n".join(text_parts_g) if text_parts_g else None,
                    input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else None,
                    output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else None,
                )
                ctx.prefix_cache.put(cache_key, _sr(resp), model=model_name, provider="google")

            tool_response_parts: list[Any] = []
            for part in candidate_content.parts:
                if not part.function_call:
                    continue
                turns_used += 1
                fc = part.function_call
                try:
                    tr = await ctx.session.call_tool(
                        fc.name,
                        dict(fc.args) if fc.args else {},
                    )
                    text = _format_tool_output(tr)
                    if tr.finished:
                        finished = True
                    if tr.reward is not None:
                        last_reward = tr.reward

                    _jsonl_write(log_file, {
                        "type": "tool_result",
                        "provider": "google",
                        "tool_name": fc.name,
                        "reward": tr.reward,
                        "finished": tr.finished,
                    })

                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"result": text},
                        )
                    )
                except Exception as e:
                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"error": str(e)},
                        )
                    )

                if finished:
                    break

            if not tool_response_parts:
                break

            tool_content = types.Content(role="user", parts=tool_response_parts)
            _jsonl_write(log_file, {"type": "tool_results", "provider": "google", "raw": str(tool_content)})
            if rollout:
                rollout.log_gdm_message(
                    tool_content,
                    rollout_info=_final_info() if finished else None,
                )
            contents.append(tool_content)

        return AgentResult(
            success=True,
            reward=last_reward,
            finished=finished,
            turns_used=turns_used,
            input_tokens=total_input,
            output_tokens=total_output,
        )

    # ------------------------------------------------------------------
    # OpenRouter (chat completions API)
    # ------------------------------------------------------------------

    async def _run_openrouter(
        self,
        ctx: TrialContext,
        model_name: str,
        rollout: Any,
        log_file: IO | None,
        trial_start: float,
    ) -> AgentResult:
        _require(AsyncOpenAI, "openai", "openrouter")

        def _final_info() -> RolloutInfo:
            return RolloutInfo(
                task_index=ctx.task_index,
                duration_ms=int((time.monotonic() - trial_start) * 1000),
            )

        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = ctx.provider_url or "https://openrouter.ai/api/v1"
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        raw_tools = await ctx.session.list_tools()
        tools = [_format_openrouter_tool(t) for t in raw_tools]
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx.prompt_text},
        ]

        turns_used = 0
        finished = False
        last_reward: float | None = None
        total_input = 0
        total_output = 0
        empty_choice_retries = 0
        reasoning_supported = True

        while not finished:
            if ctx.max_turns and turns_used >= ctx.max_turns:
                break

            or_kwargs: dict[str, Any] = {
                "model": model_name,
                "tools": tools,
                "messages": messages,
            }
            if ctx.effort and reasoning_supported:
                or_kwargs["reasoning_effort"] = "xhigh" if ctx.effort == "max" else ctx.effort

            # --- Prefix cache check ---
            cache_hit = False
            cached_response = None
            if ctx.prefix_cache:
                from firehorse.cache import serialize_response as _sr, deserialize_response
                cache_key = ctx.prefix_cache.compute_key(
                    model=model_name, system_prompt=SYSTEM_PROMPT,
                    tools=tools, messages=messages,
                    effort=ctx.effort, provider_name="openrouter",
                )
                cached_data = ctx.prefix_cache.get(cache_key)
                if cached_data is not None:
                    cached_response = deserialize_response(cached_data)
                    cache_hit = True
                    print(f"[react/openrouter] Cache HIT (key={cache_key[:12]}...)", file=sys.stderr)

            if not cache_hit:
                try:
                    response = await client.chat.completions.create(**or_kwargs)
                except Exception as e:
                    if "reasoning" in str(e).lower() and "not supported" in str(e).lower():
                        reasoning_supported = False
                        or_kwargs.pop("reasoning_effort", None)
                        print(f"\n⚠  WARNING: {model_name} does not support reasoning. --effort will be ignored.\n", file=sys.stderr)
                        response = await client.chat.completions.create(**or_kwargs)
                    else:
                        raise

                if not response.choices:
                    empty_choice_retries += 1
                    if empty_choice_retries >= 3:
                        raise RuntimeError(f"OpenRouter returned empty choices {empty_choice_retries} times consecutively")
                    print(f"[react/openrouter] Warning: response.choices is None/empty, retrying ({empty_choice_retries}/3)...", file=sys.stderr)
                    await asyncio.sleep(min(2 ** empty_choice_retries, 30))
                    continue
                empty_choice_retries = 0

                choice = response.choices[0]
                msg = choice.message

                if response.usage:
                    total_input += response.usage.prompt_tokens
                    total_output += response.usage.completion_tokens

            # Build serializable assistant message
            if cache_hit:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": cached_response.text_content,
                }
                if cached_response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in cached_response.tool_calls
                    ]
            else:
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content,
                }
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
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

            log_event_or: dict[str, Any] = {"type": "assistant", "provider": "openrouter", "raw": assistant_msg}
            if cache_hit:
                log_event_or["cache_hit"] = True
            _jsonl_write(log_file, log_event_or)
            if rollout:
                rollout.log_openai_completions(assistant_msg)
            messages.append(assistant_msg)

            # Store in cache on miss
            if not cache_hit and ctx.prefix_cache:
                from firehorse.cache import serialize_response as _sr
                from firehorse.agents.resum.providers.base import LLMResponse, ToolCallInfo
                tc_infos = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            a = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except json.JSONDecodeError:
                            a = {}
                        tc_infos.append(ToolCallInfo(id=tc.id, name=tc.function.name, arguments=a))
                resp = LLMResponse(
                    raw_message=None, tool_calls=tc_infos,
                    text_content=msg.content,
                    input_tokens=response.usage.prompt_tokens if response.usage else None,
                    output_tokens=response.usage.completion_tokens if response.usage else None,
                )
                ctx.prefix_cache.put(cache_key, _sr(resp), model=model_name, provider="openrouter")

            has_tool_calls = bool(assistant_msg.get("tool_calls"))
            if not has_tool_calls:
                break

            # Build unified tool call items
            tool_call_items = []
            if cache_hit:
                for tc in cached_response.tool_calls:
                    tool_call_items.append({"id": tc.id, "name": tc.name, "arguments": tc.arguments})
            else:
                for tc in msg.tool_calls:
                    try:
                        a = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        a = {}
                    tool_call_items.append({"id": tc.id, "name": tc.function.name, "arguments": a})

            for tc in tool_call_items:
                turns_used += 1
                try:
                    tr = await ctx.session.call_tool(tc["name"], tc["arguments"])
                    text = _format_tool_output(tr)
                    if tr.finished:
                        finished = True
                    if tr.reward is not None:
                        last_reward = tr.reward

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": text,
                    }
                    _jsonl_write(log_file, {
                        "type": "tool_result",
                        "provider": "openrouter",
                        "raw": tool_msg,
                        "reward": tr.reward,
                        "finished": tr.finished,
                    })
                    if rollout:
                        rollout.log_openai_completions(
                            tool_msg,
                            rollout_info=_final_info() if finished else None,
                        )
                    messages.append(tool_msg)
                except Exception as e:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"Error: {e}",
                    }
                    messages.append(tool_msg)

                if finished:
                    break

        return AgentResult(
            success=True,
            reward=last_reward,
            finished=finished,
            turns_used=turns_used,
            input_tokens=total_input,
            output_tokens=total_output,
        )
