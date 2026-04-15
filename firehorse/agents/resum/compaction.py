"""Conversation compaction for the ReSum agent.

When the context window overflows, the agent compacts the conversation history
into a summary using the same LLM, then continues with a fresh message list.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from firehorse.agents.resum.providers.base import ProviderClient

MAX_COMPACTIONS = 3
MIN_MESSAGES_FOR_COMPACTION = 5
MAX_SUMMARY_TOKENS = 20000  # Claude Code p99.99 = 17,387 tokens; 20K provides headroom
PROACTIVE_COMPACT_THRESHOLD = 0.80  # 80% of context window
MICRO_COMPACT_PLACEHOLDER = "[Tool output cleared]"

COMPACTION_PROMPT = """\
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's \
explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions \
that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and \
ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to \
do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay \
special attention to the most recent messages and include full code snippets where applicable and include a \
summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to \
specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for \
understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary \
request, paying special attention to the most recent messages from both user and assistant. Include file \
names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you \
were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit \
requests, and the task you were working on immediately before this summary request. If your last task was \
concluded, then only list next steps if they are explicitly in line with the users request. Do not start on \
tangential requests or really old requests that were already completed without confirming with the user first.
   If there is a next step, include direct quotes from the most recent conversation showing exactly what \
task you were working on and where you left off. This should be verbatim to ensure there's no drift in \
task interpretation.

Please provide your summary based on the conversation so far, following this structure and ensuring \
precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to \
follow these instructions when creating the above summary. Examples of instructions include:

## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made \
and how you fixed them.

# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim."""


def simple_prune(messages: list[Any], keep_last_n: int = 10) -> list[Any]:
    """Fallback: keep the first message (system/prompt) and last N messages."""
    if len(messages) <= keep_last_n + 1:
        return messages
    return [messages[0]] + messages[-keep_last_n:]


def should_compact_proactively(
    last_input_tokens: int | None,
    context_window: int | None,
    threshold: float = PROACTIVE_COMPACT_THRESHOLD,
) -> bool:
    """Check if proactive compaction should be triggered based on token usage."""
    if last_input_tokens is None or context_window is None:
        return False
    return last_input_tokens >= context_window * threshold


def micro_compact(messages: list[Any], provider_name: str, protect_last_n: int = 5) -> tuple[list[Any], int]:
    """Replace old tool result content with a placeholder, preserving recent exchanges.

    This is the cheapest form of context recovery — no API call needed.
    Inspired by Claude Code's MicroCompact and OpenCode's tool result pruning.

    Returns (messages, cleared_count).
    """
    if provider_name in ("openai", "openrouter"):
        return _micro_compact_openai(messages, protect_last_n)
    elif provider_name == "anthropic":
        return _micro_compact_anthropic(messages, protect_last_n)
    elif provider_name == "google":
        return _micro_compact_google(messages, protect_last_n)
    return messages, 0


def _micro_compact_openai(messages: list[Any], protect_last_n: int) -> tuple[list[Any], int]:
    """Micro-compact for OpenAI/OpenRouter dict-based messages.

    Handles both Chat Completions format (role=tool) and Responses API
    format (type=function_call_output).
    """
    # Find all tool result message indices
    tool_indices = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        # Chat Completions format (OpenRouter)
        if (m.get("role") == "tool"
                and m.get("content") != MICRO_COMPACT_PLACEHOLDER):
            tool_indices.append(i)
        # Responses API format (OpenAI)
        elif (m.get("type") == "function_call_output"
                and m.get("output") != MICRO_COMPACT_PLACEHOLDER):
            tool_indices.append(i)

    if len(tool_indices) <= protect_last_n:
        return messages, 0

    to_clear = tool_indices[:-protect_last_n]
    for i in to_clear:
        if messages[i].get("type") == "function_call_output":
            messages[i] = {**messages[i], "output": MICRO_COMPACT_PLACEHOLDER}
        else:
            messages[i] = {**messages[i], "content": MICRO_COMPACT_PLACEHOLDER}
    return messages, len(to_clear)


def _micro_compact_anthropic(messages: list[Any], protect_last_n: int) -> tuple[list[Any], int]:
    """Micro-compact for Anthropic dict-based messages with tool_result blocks."""
    # Collect all (msg_index, block_index) for tool_result blocks
    tool_locs: list[tuple[int, int]] = []
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if (isinstance(block, dict) and block.get("type") == "tool_result"
                    and block.get("content") != MICRO_COMPACT_PLACEHOLDER):
                tool_locs.append((mi, bi))

    if len(tool_locs) <= protect_last_n:
        return messages, 0

    to_clear = tool_locs[:-protect_last_n]
    for mi, bi in to_clear:
        messages[mi]["content"][bi] = {
            **messages[mi]["content"][bi],
            "content": MICRO_COMPACT_PLACEHOLDER,
        }
    return messages, len(to_clear)


def _micro_compact_google(messages: list[Any], protect_last_n: int) -> tuple[list[Any], int]:
    """Micro-compact for Google Content objects with function_response parts."""
    from google.genai import types

    # Collect all (msg_index, part_index) for function_response parts
    fr_locs: list[tuple[int, int]] = []
    for mi, msg in enumerate(messages):
        if not hasattr(msg, "parts") or not msg.parts:
            continue
        for pi, part in enumerate(msg.parts):
            if (hasattr(part, "function_response") and part.function_response
                    and part.function_response.response
                    and part.function_response.response.get("result") != MICRO_COMPACT_PLACEHOLDER):
                fr_locs.append((mi, pi))

    if len(fr_locs) <= protect_last_n:
        return messages, 0

    to_clear = fr_locs[:-protect_last_n]
    for mi, pi in to_clear:
        part = messages[mi].parts[pi]
        new_part = types.Part.from_function_response(
            name=part.function_response.name,
            response={"result": MICRO_COMPACT_PLACEHOLDER},
        )
        parts = list(messages[mi].parts)
        parts[pi] = new_part
        messages[mi] = types.Content(role=messages[mi].role, parts=parts)
    return messages, len(to_clear)


@dataclass
class CompactionResult:
    """Result of a compaction attempt."""
    new_messages: list[Any]
    success: bool
    summary: str | None = None  # The generated summary text (None if pruned)
    original_message_count: int = 0
    method: str = "none"  # "summary", "prune", or "none"


async def compact_conversation(
    provider: ProviderClient,
    messages: list[Any],
    system_prompt: str,
    original_prompt: str,
    compaction_count: int,
) -> CompactionResult:
    """Attempt to compact conversation history via LLM summarization.

    Returns a CompactionResult with the new messages, success flag, and summary text.
    """
    original_count = len(messages)

    if compaction_count >= MAX_COMPACTIONS:
        print(f"[resum] Max compactions ({MAX_COMPACTIONS}) reached, pruning instead", file=sys.stderr)
        return CompactionResult(
            new_messages=simple_prune(messages),
            success=False,
            original_message_count=original_count,
            method="prune",
        )

    if len(messages) < MIN_MESSAGES_FOR_COMPACTION:
        print(f"[resum] Too few messages ({len(messages)}) to compact, pruning instead", file=sys.stderr)
        return CompactionResult(
            new_messages=simple_prune(messages),
            success=False,
            original_message_count=original_count,
            method="prune",
        )

    # Format messages as plain text
    conversation_text = provider.messages_to_text(messages)

    # Try compaction with progressively smaller summary tokens
    max_tokens = MAX_SUMMARY_TOKENS
    for attempt in range(3):
        try:
            print(
                f"[resum] Compaction attempt {attempt + 1}/3 "
                f"(messages={len(messages)}, max_summary_tokens={max_tokens})",
                file=sys.stderr,
            )
            summary = await provider.call_for_compaction(
                conversation_text=conversation_text,
                compaction_prompt=COMPACTION_PROMPT,
                max_tokens=max_tokens,
            )
            print(f"[resum] Compaction succeeded (summary={len(summary)} chars)", file=sys.stderr)
            new_messages = provider.rebuild_after_compaction(system_prompt, original_prompt, summary)
            return CompactionResult(
                new_messages=new_messages,
                success=True,
                summary=summary,
                original_message_count=original_count,
                method="summary",
            )

        except Exception as e:
            print(f"[resum] Compaction attempt {attempt + 1} failed: {e}", file=sys.stderr)
            max_tokens = max_tokens // 2
            if max_tokens < 1000:
                break

    print("[resum] All compaction attempts failed, falling back to pruning", file=sys.stderr)
    return CompactionResult(
        new_messages=simple_prune(messages),
        success=False,
        original_message_count=original_count,
        method="prune",
    )
