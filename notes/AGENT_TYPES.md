# Agent Types

### `claude-code`

Spawns the [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI as a subprocess and bridges environment tools to it via MCP. Claude Code's built-in filesystem tools (Read, Edit, Bash, etc.) are disabled — the agent interacts with the environment exclusively through the MCP server. This gives the agent access to Claude Code's full capabilities (extended thinking, sub-agents, web search) while keeping it sandboxed to the environment's tool surface.

By default, environment tools that overlap with Claude Code built-ins (e.g. `bash`, `read`, `edit`) use Claude Code's own tool descriptions rather than the environment's. A system prompt mapping section tells the model which MCP tool names correspond to which built-ins (e.g. "use `mcp__openreward__bash` instead of `Bash`"). Pass `--use-env-descriptions` to use the environment's original descriptions instead.

**Providers:** Anthropic, OpenRouter

> **Note:** Claude Code's built-in context compaction does not work for non-Anthropic models via OpenRouter. OpenRouter reports per-turn token usage as zero for many providers, which prevents Claude Code from detecting when the context window is full. For long-horizon tasks with non-Anthropic models, use `--agent resum` instead, which implements its own compaction.

### `codex`

Runs the [Codex CLI](https://github.com/openai/codex) with environment tools bridged via MCP. The subprocess runs with `--dangerously-bypass-approvals-and-sandbox` since all side-effects happen in the OpenReward environment via MCP. When the environment provides a `bash` tool, redundant filesystem tools (read, write, edit, grep, glob) are excluded from MCP by default.

**Providers:** OpenAI only

### `react`

A lightweight Reason-Act loop that calls LLM APIs directly. Each turn, the model receives the conversation history and available tools, produces a response with tool calls, and the harness executes them against the environment session. No subprocess, no MCP — just a direct API loop. Supports native tool-use formats for each provider (Anthropic Messages API, OpenAI Responses API, Google Gemini, OpenRouter Chat Completions).

**Providers:** Anthropic, OpenAI, Google, OpenRouter, custom

### `resum`

Extends the ReAct loop with automatic conversation compaction. When the context window fills up (80% threshold), the agent summarizes the conversation into a structured summary preserving file paths, code snippets, decisions, and pending tasks, then continues with a fresh context. Supports up to 3 compactions per trial and includes micro-compaction (clearing large tool outputs) as a first pass before full summarization. Designed for long-horizon tasks that would otherwise overflow the context window.

**Providers:** Anthropic, OpenAI, Google, OpenRouter, custom

### `gemini`

Runs the [Gemini CLI](https://github.com/google/gemini-cli) with environment tools bridged via MCP, analogous to the `codex` agent for OpenAI models.

**Providers:** Google only
