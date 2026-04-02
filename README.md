<p align="center">
  <img src="assets/firehorselogo.png" alt="Firehorse" width="400">
</p>

<p align="center">
  <a href="https://docs.openreward.ai/">
    <img src="https://img.shields.io/badge/docs-docs.openreward.ai-blue" alt="Docs">
  </a>
</p>

🔥🐴 Firehorse is a library of agent harnesses that can be used to run models against [OpenReward environments](https://openreward.ai/environments).

It automatically bridges popular harnesses such as Claude Code, Codex and ReAct with OpenReward environments, allowing you to quickly evaluate models without setting up environment infrastructure yourself. 

When running an evaluation, Firehorse manages concurrent trial execution and produces structured trajectory logs and aggregate results. It supports multiple LLM providers out of the box.

## Features

- **Multiple agent types** — Claude Code (MCP-based), ReAct, ReSum (with context compaction), and Codex
- **Multi-provider** — Anthropic, OpenAI, Google Gemini, OpenRouter, or any OpenAI-compatible endpoint
- **Concurrent execution** — run trials in parallel with configurable concurrency
- **Structured logging** — JSONL trajectories, per-trial results, and aggregate run summaries
- **OpenReward integration** — real-time rollout streaming to the OpenReward platform
- **Secret management** — automatic provider key detection and per-trial secret injection

## Installation

```bash
pip install firehorse
```

Install with agent-specific extras for ReAct or ReSum:

```bash
pip install "firehorse[react]"   # Anthropic, OpenAI, Google SDKs
pip install "firehorse[resum]"   # Same deps + context compaction support
```

## Quick Start

```bash
# Run Claude Code agent against an environment
firehorse \
  --env MyOrg/my-environment \
  --agent claude-code \
  --model anthropic/claude-sonnet-4-6

# Run ReAct agent with OpenAI
firehorse \
  --env MyOrg/my-environment \
  --agent react \
  --model openai/gpt-4o \
  --n-concurrent 4

# Run via OpenRouter
firehorse \
  --env MyOrg/my-environment \
  --agent claude-code \
  --model openrouter/deepseek/deepseek-v3

# Save results locally
firehorse \
  --env MyOrg/my-environment \
  --agent claude-code \
  --model anthropic/claude-sonnet-4-6 \
  --output-dir ./results
```

## Agent Types

### `claude-code` (default)

Spawns the [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI as a subprocess and bridges environment tools to it via MCP. Claude Code's built-in filesystem tools (Read, Edit, Bash, etc.) are disabled — the agent interacts with the environment exclusively through the MCP server. This gives the agent access to Claude Code's full capabilities (extended thinking, sub-agents, web search) while keeping it sandboxed to the environment's tool surface.

By default, environment tools that overlap with Claude Code built-ins (e.g. `bash`, `read`, `edit`) use Claude Code's own tool descriptions rather than the environment's. A system prompt mapping section tells the model which MCP tool names correspond to which built-ins (e.g. "use `mcp__openreward__bash` instead of `Bash`"). Pass `--use-env-descriptions` to use the environment's original descriptions instead.

**Providers:** Anthropic, OpenRouter

### `react`

A lightweight Reason-Act loop that calls LLM APIs directly. Each turn, the model receives the conversation history and available tools, produces a response with tool calls, and the harness executes them against the environment session. No subprocess, no MCP — just a direct API loop. Supports native tool-use formats for each provider (Anthropic Messages API, OpenAI Responses API, Google Gemini, OpenRouter Chat Completions).

**Providers:** Anthropic, OpenAI, Google, OpenRouter, custom

### `resum`

Extends the ReAct loop with automatic conversation compaction. When the context window fills up (80% threshold), the agent summarizes the conversation into a structured summary preserving file paths, code snippets, decisions, and pending tasks, then continues with a fresh context. Supports up to 3 compactions per trial and includes micro-compaction (clearing large tool outputs) as a first pass before full summarization. Designed for long-horizon tasks that would otherwise overflow the context window.

**Providers:** Anthropic, OpenAI, Google, OpenRouter, custom

### `codex`

Runs the [Codex CLI](https://github.com/openai/codex) with environment tools bridged via MCP. The built-in shell is sandboxed to read-only, and the agent uses environment-provided tools for all interactions. When the environment provides a `bash` tool, redundant filesystem tools (read, write, edit, grep, glob) are excluded from MCP by default. For non-OpenAI providers, a local auth-injecting proxy routes requests through OpenRouter.

**Providers:** OpenAI, OpenRouter, custom

## Environment Variables

```bash
# Required
OPENREWARD_API_KEY=<your-api-key>

# LLM provider (set the one you need)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
```

## CLI Reference

```
firehorse --env ENV --model MODEL [OPTIONS]

Required:
  --env              Environment name (e.g. MyOrg/my-environment)
  --model            Model identifier (e.g. anthropic/claude-sonnet-4-6)

Options:
  --agent            Agent type: claude-code, react, resum, codex (default: claude-code)
  --split            Task split to evaluate (default: test)
  --n-concurrent     Max parallel trials (default: 1)
  --max-tasks        Limit number of tasks to evaluate
  --max-turns        Max tool call turns per trial
  --run-name         Name for this evaluation run
  --effort           Thinking effort: low, medium, high, max (default: high)
  --provider-url     Custom API base URL for non-standard endpoints
  --output-dir       Directory for JSONL trajectory logs and results
  --secret KEY=VAL   Inject a session secret (repeatable)
  --disable-builtin-tools  Comma-separated list of tools to disable
  --use-env-descriptions   Use environment tool descriptions instead of built-in ones
  --use-all-filesystem-tools  Expose all filesystem tools via MCP (codex only)
  --no-logging       Disable OpenReward rollout streaming
```

## Output

When `--output-dir` is specified, firehorse writes:

```
output_dir/
├── run_result.json          # Aggregate results across all trials
├── trial_0.jsonl            # Full agent trajectory
├── trial_0_result.json      # Per-trial summary (reward, tokens, cost, duration)
├── trial_0_rewards.jsonl    # Reward signal at each tool call
└── ...
```

Each trial result includes:

| Field | Description |
|-------|-------------|
| `reward` | Final reward score from the environment |
| `finished` | Whether the environment signaled task completion |
| `turns_used` | Number of tool call turns |
| `input_tokens` | Total input tokens consumed |
| `output_tokens` | Total output tokens consumed |
| `cost_usd` | Estimated API cost |
| `duration_seconds` | Wall-clock time |

## Documentation

Full documentation is available at [docs.openreward.ai](https://docs.openreward.ai/).
