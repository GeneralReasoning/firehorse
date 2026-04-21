# ReSum Agent

ReSum is a multi-provider LLM agent for the firehorse evaluation harness. Unlike the `claude-code` agent (which spawns a CLI subprocess), ReSum implements the agentic tool-use loop directly via LLM API calls, supporting OpenAI, Anthropic, Google Gemini, and OpenRouter.

## Quick Start

```bash
# Install with provider dependencies
pip install -e ".[resum]"

# Run a task
firehorse --env YourOrg/YourEnv --agent resum --model openai/gpt-4o --max-tasks 1

# Different providers
firehorse --agent resum --model anthropic/claude-sonnet-4-6 ...
firehorse --agent resum --model google/gemini-2.5-pro ...
firehorse --agent resum --model openrouter/qwen/qwen3-coder ...
```

## Architecture

```
ReSumAgent
    |
    |-- ProviderClient (ABC)
    |       |-- OpenAIProvider
    |       |-- OpenRouterProvider (extends OpenAI with different base_url)
    |       |-- AnthropicProvider
    |       |-- GoogleProvider
    |
    |-- compaction.py (shared compaction logic)
```

The core loop is provider-agnostic. Each `ProviderClient` normalizes the differences between provider APIs (message formats, tool schemas, error codes) behind a common interface. The agent only interacts through `ProviderClient` methods like `call()`, `append_tool_result()`, and `format_tools()`.

### Core Loop

```
1. Parse provider from model string (e.g. "openai/gpt-4o" -> OpenAI + "gpt-4o")
2. Resolve API key (ctx.secrets -> environment variables)
3. Format tools + build initial messages
4. LOOP (up to max_turns):
   a. response = provider.call(messages, tools)
   b. If context_overflow -> compact conversation -> continue
   c. Accumulate tokens, append assistant message
   d. If no tool_calls -> nudge with "Continue working" user message
   e. For each tool_call:
      - result = session.call_tool(name, args)
      - Append result to messages
      - If finished -> break
5. Return AgentResult with reward, token usage, timing
```

## Model String Convention

The model string determines which provider to use:

| Prefix | Provider | Example |
|--------|----------|---------|
| `openai/` | OpenAI | `openai/gpt-4o-mini` |
| `anthropic/` | Anthropic | `anthropic/claude-sonnet-4-6` |
| `google/` | Google Gemini | `google/gemini-2.5-pro` |
| `openrouter/` | OpenRouter | `openrouter/qwen/qwen3-coder` |

If no prefix matches, falls back to checking `--provider-url`.

## API Key Resolution

Keys are resolved in priority order: `ctx.secrets` dict first, then environment variables.

| Provider | Secret Key | Env Var |
|----------|-----------|---------|
| openai | `openai_api_key` | `OPENAI_API_KEY` |
| anthropic | `anthropic_api_key` | `ANTHROPIC_API_KEY` |
| google | `google_api_key` | `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| openrouter | `openrouter_api_key` | `OPENROUTER_API_KEY` |

## Logging

ReSum produces two output files per trial when `output_dir` is set:

- **`trial_{task_id}.jsonl`** — One JSON line per event (prompt, assistant turns, tool calls, tool results, compactions, summary). Matches the schema used by the `claude-code` agent for downstream compatibility.
- **`trial_{task_id}_result.json`** — Final result with reward, token usage, timing, compaction count, and rollout URL.

If `ctx.logging` is enabled with a rollout client, events are also streamed to the OpenReward rollout API for live monitoring.

## Thinking / Reasoning

The `--effort` flag controls thinking/reasoning effort for the ReSum agent, using the same provider-specific mappings as the `react` agent:

```bash
# Anthropic with adaptive thinking
firehorse --env MyOrg/my-env --agent resum \
  --model anthropic/claude-sonnet-4-6 --effort high

# Google Gemini with thinking enabled
firehorse --env MyOrg/my-env --agent resum \
  --model google/gemini-3.1-flash-lite-preview --effort high
```

- **Anthropic**: Adaptive thinking with `effort` parameter (low/medium/high/max)
- **OpenAI**: `reasoning_effort` for o-series models only
- **Google Gemini 3.x**: `thinking_level` (low/medium/high)
- **Google Gemini 2.5**: `thinking_budget_tokens` (1024/5000/16000/24576)
- **OpenRouter**: Passes through to underlying provider

Note: Compaction calls (summarization) do not use thinking — only the main agent loop does.

## Conversation Compaction

Compaction is the mechanism that lets ReSum handle tasks that exceed a model's context window. ReSum uses a 3-layer strategy inspired by research into how Claude Code, Codex CLI, Aider, and other production agent frameworks manage context.

### How Other Frameworks Handle Compaction

| Framework | Trigger | Summary Budget | Max Compactions | Model |
|-----------|---------|---------------|-----------------|-------|
| **Claude Code** | Proactive at ~83.5% of context | 20K tokens | 3 failures circuit breaker | Same model, thinking disabled |
| **Codex CLI** | Proactive at 95% of context | 20K tokens | No limit (known issue) | Same model or server-side |
| **Aider** | Token count threshold | Half of max_tokens | Depth 3 recursion | Weak model first, fallback to main |
| **SWE-agent** | No compaction | N/A | N/A | N/A (exits on overflow) |
| **ReSum** | Proactive at 80% + reactive on overflow | 20K tokens | 3 max compactions | Same model |

### The 3-Layer Strategy

ReSum uses three progressively more aggressive layers to manage context:

#### Layer 1: Micro-Compaction (No API Call)

Replaces old tool result content with `[Tool output cleared]`, preserving the most recent 5 tool exchanges. This is the cheapest form of context recovery — no LLM call needed, conversation structure is preserved, and it often reclaims enough space to continue without full summarization. Inspired by Claude Code's MicroCompact and OpenCode's tool result pruning.

#### Layer 2: Full LLM Compaction

Calls the same LLM (no tools) with the compaction prompt to produce a structured summary. The summary covers:
- Primary request and intent
- Key technical concepts
- Files and code sections (with snippets)
- Errors encountered and fixes applied
- All user messages (critical for preserving intent)
- Pending tasks and current work
- Next steps

Messages are rebuilt as: `[system_prompt, original_task_prompt, summary]`.

#### Layer 3: Simple Prune (Fallback)

Keeps the first message (system prompt/initial task) and the last 10 messages. This is the last resort when compaction fails or is exhausted.

### When Compaction Triggers

Compaction triggers in two ways:

**Proactive (before overflow):** After each successful LLM call, if `input_tokens >= 80% of context_window`, ReSum proactively compacts. This prevents the degradation that occurs as models approach their context limit (the "lost in the middle" effect documented by Liu et al., 2023). Proactive compaction only fires when the model's context window size is known — either from the built-in lookup table or a user-provided override.

**Reactive (on overflow):** When a provider returns `LLMResponse(context_overflow=True)`, ReSum tries micro-compaction first, then full compaction. Each provider detects overflow differently:
- **OpenAI/OpenRouter**: `BadRequestError` with "context length" or "maximum"+"token"
- **Anthropic**: `BadRequestError` with "too long" / "context" / "token"
- **Google**: Exception messages containing "token", "context", or "too long"

### Flow Diagram

```
After each successful LLM call:
    input_tokens >= 80% of context_window?
        yes -> Layer 1 (micro-compact) -> if nothing cleared -> Layer 2 (full compact)
        no  -> continue normally

On context_overflow=True:
    Already micro-compacted?
        no  -> Layer 1 (micro-compact) -> retry LLM call
        yes -> Layer 2 (full compact):
                  compaction_count >= 3? -> Layer 3 (simple prune)
                  < 5 messages?          -> Layer 3 (simple prune)
                  Try summary (20K tokens)
                      success -> rebuild messages, continue
                      fail    -> retry at 10K tokens
                          fail -> retry at 5K tokens
                              fail -> Layer 3 (simple prune)
```

### Constants and Their Justification

#### `MAX_COMPACTIONS = 3`

Matches Claude Code (circuit breaker after 3 consecutive failures) and Aider (depth 3 recursion). Each compaction is lossy — nuances in tool outputs, exact error messages, and intermediate reasoning get reduced to bullet points. After 3 compactions, cumulative information loss means the agent is operating on a significantly degraded understanding of the task. Three compactions provides roughly 4x the effective context capacity (each cycle reclaims most of the window).

#### `MAX_SUMMARY_TOKENS = 20,000`

Aligned with Claude Code and Codex CLI, both of which use 20K. Claude Code's empirical data shows that the p99.99 summary length is 17,387 tokens — our previous value of 14,800 would have truncated ~1% of summaries. The additional 5,200 tokens of headroom is negligible relative to modern context windows (128K-2M) but meaningfully improves summary quality for complex tasks.

#### `PROACTIVE_COMPACT_THRESHOLD = 0.80` (80%)

Proactive compaction at 80% of the context window. Claude Code fires at ~83.5%, Codex CLI at 95%. We chose 80% as a slightly more conservative default — proactive compaction is cheaper than reactive (the compaction call itself has more room), and empirically models start degrading in quality before the hard limit. This can be adjusted per deployment by setting `context_window` on TrialContext to a smaller effective value.

#### `MIN_MESSAGES_FOR_COMPACTION = 5`

Matches Claude Code's minimum (5 text block messages). With fewer than 5 messages, the overhead of a compaction call isn't justified — the "summary" would be roughly the same length as the original. Also prevents an edge case where the first LLM call overflows (the task prompt itself is too large).

#### `micro_compact(protect_last_n=5)`

Preserves the 5 most recent tool results verbatim while clearing older ones. Claude Code protects the most recent ~40K tokens of tool outputs; our approach is count-based (5 exchanges) for simplicity across providers. Five exchanges typically covers the agent's immediate working context — the files it just read, the last few tool results it's reasoning about.

#### Progressive Retry with Token Halving

On compaction failure, retry up to 3 times halving `max_tokens`: 20K -> 10K -> 5K. Below 1K, retries stop (too lossy). This handles cases where the compaction call itself overflows — a shorter summary is better than no summary.

#### `simple_prune(keep_last_n=10)`

The last resort fallback. Keeps the first message (system/task prompt) and last 10 messages (~5 tool-call exchanges). Preserves recency at the cost of long-range context.

### Known Context Windows

Each provider includes a lookup table of known model context windows for proactive compaction. Users can override with `context_window` on TrialContext for unlisted models.

| Provider | Model | Context Window |
|----------|-------|---------------|
| Anthropic | claude-opus-4-6 | 1,000,000 |
| Anthropic | claude-sonnet-4-6 | 1,000,000 |
| OpenAI | gpt-5.4 | 1,000,000 |
| OpenAI | gpt-4.1 | 1,047,576 |
| OpenAI | gpt-4o | 128,000 |
| Google | gemini-2.5-pro | 2,000,000 |
| Google | gemini-2.5-flash | 1,048,576 |
| OpenRouter | deepseek/deepseek-v3.2 | 163,840 |
| OpenRouter | z-ai/glm-5.1 | 204,800 |
| OpenRouter | moonshot/kimi-k2.5 | 256,000 |
| OpenRouter | minimax/minimax-m2.7 | 205,000 |
| OpenRouter | qwen/qwen-3.5-397b-a17b | 1,000,000 |

## Differences from Other Agents

| Feature | ReSum | claude-code | react |
|---------|-------|-------------|-------|
| LLM interaction | Direct API calls | CLI subprocess | Direct API calls |
| Providers | OpenAI, Anthropic, Google, OpenRouter | Anthropic only | OpenAI, Anthropic, Google, OpenRouter |
| Compaction | 3-layer (micro + LLM + prune) | 3-layer (micro + session memory + full) | None |
| Proactive compaction | Yes (80% threshold) | Yes (~83.5% threshold) | No |
| Context window awareness | Built-in lookup + user override | Built-in | No |
| Retry with backoff | Yes (per-provider) | Handled by Claude CLI | No |
| Provider abstraction | `ProviderClient` ABC | N/A | Separate `_run_*` methods |
| JSONL logging | Yes | Yes | Yes |
| Rollout logging | Yes | Yes | Yes |

## File Layout

```
firehorse/agents/resum/
    __init__.py                    # Exports ReSumAgent
    agent.py                       # Core agent loop, logging, result building
    compaction.py                  # Compaction prompt, compact_conversation(), simple_prune()
    providers/
        __init__.py                # parse_provider(), resolve_api_key(), get_provider()
        base.py                    # ProviderClient ABC, LLMResponse, ToolCallInfo
        openai_provider.py         # OpenAI chat completions
        anthropic_provider.py      # Anthropic messages API
        google_provider.py         # Google Gemini
        openrouter_provider.py     # OpenRouter (OpenAI-compatible)
```

## Testing

```bash
# Unit tests (no API keys needed)
pytest tests/test_resum.py -v

# Integration tests (requires API keys in environment or .env)
pytest tests/test_resum_integration.py -v -s -m integration
```
