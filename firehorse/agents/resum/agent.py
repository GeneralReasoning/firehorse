"""ReSum agent — direct LLM API agentic loop with multi-provider support."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from openreward import (
    AssistantMessage,
    ReasoningItem,
    SystemMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from openreward.models import RolloutInfo

from firehorse.agents.base import AgentResult, BaseAgent, TrialContext
from firehorse.agents.resum.compaction import CompactionResult, compact_conversation, micro_compact, should_compact_proactively
from firehorse.agents.resum.providers import get_provider, parse_provider, resolve_api_key
from firehorse.agents.resum.providers.base import LLMResponse

if TYPE_CHECKING:
    from firehorse.agents.resum.providers.base import ProviderClient

SYSTEM_PROMPT = (
    "You are an agent solving a task in an OpenReward environment. "
    "Use the tools provided to complete the task. "
    "When a tool result indicates the episode is complete, stop working immediately — the task is done. "
    "Do not make any more tool calls after the episode is complete."
)


def _extract_tool_output_text(output: Any) -> str:
    """Extract text from a ToolOutput's blocks."""
    parts = []
    for block in output.blocks:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif hasattr(block, "data"):
            parts.append(f"[Image: {getattr(block, 'mimeType', 'unknown')}]")
    return "\n".join(parts) if parts else ""


class ReSumAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "resum"

    async def setup(self) -> None:
        pass  # No external binary needed

    async def run(self, ctx: TrialContext) -> AgentResult:
        # --- Resolve provider ---
        provider_name, model_id = parse_provider(ctx.model, ctx.provider_url)
        api_key = resolve_api_key(provider_name, ctx.secrets)
        provider = get_provider(provider_name, model_id, api_key, ctx.provider_url, ctx.context_window)

        # Resolve context window dynamically for providers that support it
        # (e.g., OpenRouter models not in the hardcoded lookup table).
        if hasattr(provider, "resolve_context_window"):
            await provider.resolve_context_window()

        # Wrap provider with caching if prefix cache is enabled
        if ctx.prefix_cache:
            from firehorse.cache import CachingProviderClient
            provider = CachingProviderClient(provider, ctx.prefix_cache, provider_name)

        print(f"[resum] Provider={provider_name} model={model_id}", file=sys.stderr)
        if provider.context_window:
            print(f"[resum] Context window: {provider.context_window:,} tokens", file=sys.stderr)
        if ctx.effort:
            print(f"[resum] Thinking effort: {ctx.effort}", file=sys.stderr)

        # --- Format tools ---
        tools = provider.format_tools(ctx.tools)

        # --- Build initial messages ---
        messages = provider.build_initial_messages(SYSTEM_PROMPT, ctx.prompt_text)
        original_prompt = ctx.prompt_text

        # --- Setup logging ---
        trial_id = ctx.task_spec.get("id", ctx.task_spec.get("index", "unknown"))
        log_dir = Path(ctx.output_dir) if ctx.output_dir else None
        log_file = None
        if log_dir:
            log_file = open(log_dir / f"trial_{trial_id}.jsonl", "w")

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
                    metadata={
                        "provider": provider_name,
                        "effort": ctx.effort,
                    },
                )
                print(
                    f"[resum] Rollout: https://openreward.ai/rollout/{rollout.event_id}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[resum] Failed to create rollout: {e}", file=sys.stderr)

        # --- Log initial prompt ---
        self._log_jsonl(log_file, {
            "type": "openreward_prompt",
            "system_prompt": SYSTEM_PROMPT,
            "environment_prompt": ctx.prompt_text,
        })
        self._log_rollout_system_and_prompt(rollout, SYSTEM_PROMPT, ctx.prompt_text, ctx.task_index)

        # --- Core loop ---
        max_turns = ctx.max_turns
        turns_used = 0
        total_input_tokens = 0
        total_output_tokens = 0
        compaction_count = 0
        last_reward: float | None = None
        finished = False
        start_ms = time.monotonic()

        try:
            micro_compacted = False  # Track whether micro-compaction was already tried

            step = 0
            while max_turns is None or step < max_turns:
                step += 1
                # Call the LLM
                response = await provider.call(messages, tools, effort=ctx.effort)

                # Handle context overflow — 3-layer strategy:
                # 1. Micro-compact (clear old tool results, no API call)
                # 2. Full compaction (LLM-based summarization)
                # 3. Simple prune (fallback, handled inside compact_conversation)
                if response.context_overflow:
                    if not micro_compacted:
                        # Layer 1: try micro-compaction first
                        messages, cleared = micro_compact(messages, provider_name)
                        if cleared > 0:
                            micro_compacted = True
                            print(
                                f"[resum] Context overflow at step {step}, "
                                f"micro-compacted {cleared} tool results",
                                file=sys.stderr,
                            )
                            self._log_jsonl(log_file, {
                                "type": "micro_compaction",
                                "cleared_count": cleared,
                                "message_count": len(messages),
                            })
                            continue

                    # Layer 2: full compaction (layer 3 / simple_prune is inside compact_conversation)
                    print(f"[resum] Context overflow at step {step}, attempting full compaction", file=sys.stderr)
                    cr = await compact_conversation(
                        provider=provider,
                        messages=messages,
                        system_prompt=SYSTEM_PROMPT,
                        original_prompt=original_prompt,
                        compaction_count=compaction_count,
                    )
                    messages = cr.new_messages
                    micro_compacted = False  # Reset for next overflow cycle
                    if cr.success:
                        compaction_count += 1
                    self._log_compaction(log_file, rollout, cr, compaction_count, proactive=False)
                    continue

                # Successful LLM call — reset micro-compaction flag
                micro_compacted = False

                # Accumulate tokens (skip for cache hits — no API call was made)
                if not response.cache_hit:
                    if response.input_tokens:
                        total_input_tokens += response.input_tokens
                    if response.output_tokens:
                        total_output_tokens += response.output_tokens

                # Append assistant message
                provider.append_assistant(messages, response)

                # Log assistant response
                self._log_assistant(log_file, rollout, response)

                # If no tool calls, nudge the model
                if not response.tool_calls:
                    if response.text_content:
                        print(f"[resum] Assistant (no tools): {response.text_content[:200]}...", file=sys.stderr)
                    provider.append_user_message(messages, "Continue working on the task. Use tools to make progress.")
                    continue

                # Execute tool calls
                for tc in response.tool_calls:
                    turns_used += 1
                    print(f"[resum] Tool call: {tc.name}", file=sys.stderr)

                    self._log_tool_call(log_file, rollout, tc)

                    try:
                        result = await ctx.session.call_tool(tc.name, tc.arguments)
                    except Exception as e:
                        error_output = f"ERROR: Tool call failed: {e}"
                        provider.append_tool_result(messages, tc.id, tc.name, error_output)
                        self._log_tool_result(log_file, rollout, tc.id, error_output, None, False)
                        continue

                    output_text = _extract_tool_output_text(result)
                    provider.append_tool_result(messages, tc.id, tc.name, output_text)

                    reward = result.reward
                    is_finished = result.finished
                    if reward is not None:
                        last_reward = reward

                    final_info = None
                    if is_finished:
                        final_info = RolloutInfo(
                            task_index=ctx.task_index,
                            duration_ms=int((time.monotonic() - start_ms) * 1000),
                            num_compactions=compaction_count,
                        )
                    self._log_tool_result(log_file, rollout, tc.id, output_text, reward, is_finished, rollout_info=final_info)

                    print(
                        f"[resum]   -> reward={reward} finished={is_finished} "
                        f"output={output_text[:200]}{'...' if len(output_text) > 200 else ''}",
                        file=sys.stderr,
                    )

                    if is_finished:
                        finished = True
                        break

                if finished:
                    break

                # Proactive compaction: check after processing this step's tool calls
                if should_compact_proactively(response.input_tokens, provider.context_window):
                    print(
                        f"[resum] Proactive compaction at {response.input_tokens} tokens "
                        f"({response.input_tokens / provider.context_window * 100:.0f}% of "
                        f"{provider.context_window} context window)",
                        file=sys.stderr,
                    )
                    # Try micro-compaction first
                    messages, cleared = micro_compact(messages, provider_name)
                    if cleared > 0:
                        self._log_jsonl(log_file, {
                            "type": "micro_compaction",
                            "cleared_count": cleared,
                            "message_count": len(messages),
                            "proactive": True,
                        })
                    else:
                        # Full compaction needed
                        cr = await compact_conversation(
                            provider=provider,
                            messages=messages,
                            system_prompt=SYSTEM_PROMPT,
                            original_prompt=original_prompt,
                            compaction_count=compaction_count,
                        )
                        messages = cr.new_messages
                        if cr.success:
                            compaction_count += 1
                        self._log_compaction(log_file, rollout, cr, compaction_count, proactive=True)

        except Exception as e:
            duration_ms = int((time.monotonic() - start_ms) * 1000)
            self._log_summary(log_file, ctx, last_reward, finished, turns_used, total_input_tokens, total_output_tokens, duration_ms)
            self._close_log(log_file)
            return AgentResult(
                success=False,
                error=str(e),
                reward=last_reward,
                finished=finished,
                turns_used=turns_used,
                input_tokens=total_input_tokens or None,
                output_tokens=total_output_tokens or None,
                duration_ms=duration_ms,
            )

        duration_ms = int((time.monotonic() - start_ms) * 1000)

        # --- Write summary and close ---
        self._log_summary(log_file, ctx, last_reward, finished, turns_used, total_input_tokens, total_output_tokens, duration_ms)

        if log_dir:
            trial_result = {
                "task_id": trial_id,
                "task_spec": ctx.task_spec,
                "environment": ctx.env_name,
                "agent": "resum",
                "model": ctx.model,
                "provider": provider_name,
                "split": ctx.split,
                "final_reward": last_reward,
                "finished": finished,
                "tool_calls": turns_used,
                "compactions": compaction_count,
                "duration_seconds": duration_ms / 1000 if duration_ms else None,
                "usage": {
                    "input_tokens": total_input_tokens or None,
                    "output_tokens": total_output_tokens or None,
                },
                "rollout_url": f"https://openreward.ai/rollout/{rollout.event_id}" if rollout else None,
            }
            result_path = log_dir / f"trial_{trial_id}_result.json"
            result_path.write_text(json.dumps(trial_result, indent=2))

        self._close_log(log_file)

        return AgentResult(
            success=True,
            reward=last_reward,
            finished=finished,
            turns_used=turns_used,
            input_tokens=total_input_tokens or None,
            output_tokens=total_output_tokens or None,
            duration_ms=duration_ms,
        )

    # --- Logging helpers ---

    def _log_jsonl(self, log_file: Any, event: dict) -> None:
        if log_file:
            log_file.write(json.dumps(event) + "\n")

    def _log_assistant(self, log_file: Any, rollout: Any, response: LLMResponse) -> None:
        # JSONL
        event: dict[str, Any] = {
            "type": "assistant",
            "text_content": response.text_content,
            "reasoning_content": response.reasoning_content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ],
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
        if response.cache_hit:
            event["cache_hit"] = True
        self._log_jsonl(log_file, event)

        # Rollout
        if rollout:
            if response.reasoning_content:
                rollout.log(ReasoningItem(content=response.reasoning_content))
            if response.text_content:
                rollout.log(AssistantMessage(content=response.text_content))

    def _log_tool_call(self, log_file: Any, rollout: Any, tc: Any) -> None:
        self._log_jsonl(log_file, {
            "type": "tool_call",
            "tool_call_id": tc.id,
            "tool_name": tc.name,
            "arguments": tc.arguments,
        })
        if rollout:
            rollout.log(ToolCall(
                name=tc.name,
                content=json.dumps(tc.arguments),
                call_id=tc.id,
            ))

    def _log_tool_result(
        self,
        log_file: Any,
        rollout: Any,
        call_id: str,
        output: str,
        reward: float | None,
        finished: bool,
        rollout_info: RolloutInfo | None = None,
    ) -> None:
        self._log_jsonl(log_file, {
            "type": "tool_result",
            "tool_call_id": call_id,
            "output": output,
            "reward": reward,
            "finished": finished,
        })
        if rollout:
            rollout.log(
                ToolResult(content=output, call_id=call_id),
                reward=reward,
                is_finished=finished,
                rollout_info=rollout_info,
            )

    def _log_compaction(
        self,
        log_file: Any,
        rollout: Any,
        cr: CompactionResult,
        compaction_count: int,
        proactive: bool,
    ) -> None:
        """Log a compaction event to both JSONL and rollout."""
        # JSONL — include the summary text so the trajectory is replayable
        self._log_jsonl(log_file, {
            "type": "compaction",
            "compaction_count": compaction_count,
            "method": cr.method,
            "success": cr.success,
            "original_message_count": cr.original_message_count,
            "new_message_count": len(cr.new_messages),
            "summary": cr.summary if cr.summary else None,
            "proactive": proactive,
        })

        # Rollout — log as SystemMessage + ToolCall("compact") + ToolResult
        if rollout:
            trigger = "proactive" if proactive else "reactive"
            compact_call_id = f"compact_{compaction_count}"

            # System message announcing compaction
            rollout.log(SystemMessage(
                content=(
                    f"Context compaction triggered ({trigger}). "
                    f"Method: {cr.method}. "
                    f"Messages: {cr.original_message_count} -> {len(cr.new_messages)}."
                ),
            ))

            # ToolCall representing the compaction operation
            rollout.log(ToolCall(
                name="compact",
                content=json.dumps({
                    "method": cr.method,
                    "trigger": trigger,
                    "compaction_count": compaction_count,
                    "original_message_count": cr.original_message_count,
                }),
                call_id=compact_call_id,
            ))

            # ToolResult with the summary or prune info
            if cr.success and cr.summary:
                rollout.log(ToolResult(
                    content=cr.summary,
                    call_id=compact_call_id,
                ))
            else:
                rollout.log(ToolResult(
                    content=(
                        f"Compaction {'succeeded' if cr.success else 'failed'}. "
                        f"Fell back to {cr.method}. "
                        f"Kept {len(cr.new_messages)} messages."
                    ),
                    call_id=compact_call_id,
                ))

    def _log_rollout_system_and_prompt(
        self,
        rollout: Any,
        system_prompt: str,
        user_prompt: str,
        task_index: int,
    ) -> None:
        if rollout:
            rollout.log(
                SystemMessage(content=system_prompt),
                rollout_info=RolloutInfo(task_index=task_index, harness="resum"),
            )
            rollout.log(UserMessage(content=user_prompt))

    def _log_summary(
        self,
        log_file: Any,
        ctx: TrialContext,
        reward: float | None,
        finished: bool,
        turns: int,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int,
    ) -> None:
        self._log_jsonl(log_file, {
            "type": "openreward_summary",
            "task_spec": ctx.task_spec,
            "env": ctx.env_name,
            "model": ctx.model,
            "final_reward": reward,
            "finished": finished,
            "usage": {
                "input_tokens": input_tokens or None,
                "output_tokens": output_tokens or None,
                "duration_ms": duration_ms,
            },
        })

    def _close_log(self, log_file: Any) -> None:
        if log_file:
            log_file.close()
