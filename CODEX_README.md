# Codex CLI Integration for Firehorse

Firehorse agent that wraps [OpenAI Codex CLI](https://github.com/openai/codex) (`codex exec`) to run OpenReward environment evaluations using OpenAI models.

## Prerequisites

1. **Install Codex CLI**:
   ```bash
   npm install -g @openai/codex
   ```

2. **Authenticate** (one of):
   ```bash
   # Option A: Login via ChatGPT account
   codex login

   # Option B: Set API key
   export OPENAI_API_KEY=sk-...
   ```

3. **Verify installation**:
   ```bash
   codex --version
   ```

## Usage

```bash
firehorse --agent codex --model openai/gpt-5-codex --env <namespace/environment>
```

### Examples

```bash
# Run with GPT-5 Codex (direct OpenAI)
firehorse --agent codex --model openai/gpt-5-codex --env GeneralReasoning/KellyBench

# Run via OpenRouter
export OPENROUTER_API_KEY=sk-or-...
firehorse --agent codex --model openrouter/openai/gpt-4.1 --env GeneralReasoning/KellyBench

# Run with GPT-4.1 and limit tasks
firehorse --agent codex --model openai/gpt-4.1 --env GeneralReasoning/portfolio --max-tasks 5

# Control reasoning effort
firehorse --agent codex --model openai/gpt-5-codex --effort high --env GeneralReasoning/KellyBench

# Expose all filesystem tools (not just bash)
firehorse --agent codex --model openai/gpt-5-codex --env MyEnv/thing --use-all-filesystem-tools

# With output logging
firehorse --agent codex --model openai/gpt-5-codex --env GeneralReasoning/KellyBench --output-dir ./results
```

## Supported Models

| Prefix | Example | Notes |
|--------|---------|-------|
| `openai/` | `openai/gpt-5-codex`, `openai/gpt-4.1` | Direct OpenAI access |
| `openrouter/` | `openrouter/openai/gpt-4.1` | Via model_providers config. Requires `OPENROUTER_API_KEY` |

## How It Works

The Codex agent follows the same subprocess + MCP bridge pattern as the Claude Code agent:

```
firehorse trial runner
    |
    v
CodexAgent.run()
    |-- creates temp working directory
    |-- builds MCP server config (passed via -c dotted-path flags)
    |-- filters env tools (only bash exposed by default)
    |-- (for openrouter/) configures model_provider with support_namespaces=false
    |-- launches: codex exec --json --sandbox read-only ...
    |
    v
codex exec subprocess
    |-- connects to API (OpenAI direct, or OpenRouter/custom via model_provider config)
    |-- connects to MCP server (firehorse.mcp bridge)
    |-- receives environment tools via MCP
    |-- executes agent loop (tool calls + reasoning)
    |-- emits JSONL events to stdout
    |
    v
MCP bridge (firehorse.mcp)
    |-- creates OpenReward session
    |-- exposes environment tools as MCP tools
    |-- tracks rewards and finished state
    |-- writes result.json on episode completion
```

### MCP Bridge

The same `firehorse.mcp` bridge module is reused from the Claude Code integration. It:

- Creates an OpenReward session for the task
- Exposes environment tools via MCP stdio transport
- Intercepts tool call results to track rewards
- Injects `[OR_REWARD:{...}]` tags for reward parsing
- Signals `[EPISODE COMPLETE]` when the environment finishes
- Writes `result.json` with final reward/finished/call-count data

### Tool Filtering

When the environment provides a `bash` tool, the other filesystem tools (`read`, `write`, `edit`, `grep`, `glob`) are **excluded from MCP by default**. The rationale: bash can do everything these tools do, and exposing fewer tools reduces confusion for the model.

| Environment provides | Exposed via MCP |
|---|---|
| `bash`, `read`, `write`, `edit`, `grep`, `glob`, `answer` | `bash`, `answer` (filesystem tools filtered) |
| `read`, `write`, `answer` (no `bash`) | `read`, `write`, `answer` (no filtering) |
| `bash`, `answer` | `bash`, `answer` (nothing to filter) |

Use `--use-all-filesystem-tools` to override this and expose all environment tools.

Planning tools (`todo_write`) are always excluded from MCP — Codex's built-in versions are preferred.

### Sandbox Policy

The built-in shell is **always sandboxed** with `--sandbox read-only`. This prevents the agent from using its built-in shell for writes or mutations — all environment interactions must go through MCP tools.

The system prompt explicitly instructs the agent to use the MCP `bash` tool instead of the built-in shell.

### System Prompt

The Codex agent uses the [upstream Codex system prompt](https://github.com/openai/codex/blob/main/codex-rs/core/prompt.md) as a base, with an appended section that:

1. Lists available MCP tools from the environment
2. Identifies the MCP `bash` tool as the primary tool (replacing the built-in shell)
3. Instructs the agent to stop on `[EPISODE COMPLETE]`

Since `codex exec` has no `--system-prompt` flag, the full prompt is prepended to the user/task prompt.

### Output Format

When `--output-dir` is specified, each trial produces:

```
output-dir/
  trial_<id>.jsonl           # Full JSONL event stream from Codex
  trial_<id>_rewards.jsonl   # Reward sidecar from MCP bridge
  trial_<id>_result.json     # Per-trial summary (reward, finished, usage)
```

The JSONL events follow Codex's native format:
```jsonl
{"model":"gpt-5-codex","approval":"never","sandbox":"...","provider":"openai",...}
{"prompt":"..."}
{"id":"0","msg":{"type":"task_started","model_context_window":272000}}
{"id":"0","msg":{"type":"mcp_tool_call_begin","call_id":"...","invocation":{...}}}
{"id":"0","msg":{"type":"mcp_tool_call_end","call_id":"...","result":{"Ok":{...}}}}
{"id":"0","msg":{"type":"agent_message","message":"..."}}
{"id":"0","msg":{"type":"token_count","info":{"total_token_usage":{...}}}}
```

## Differences from Claude Code Agent

| Feature | Claude Code | Codex |
|---------|------------|-------|
| MCP config | `--mcp-config <file.json>` | `-c` dotted-path flags |
| System prompt | `--append-system-prompt` flag | Prepended to user prompt |
| Reasoning effort | `--effort` flag | `-c model_reasoning_effort=...` |
| Built-in tool control | `--disallowed-tools` (always disables filesystem built-ins) | `--sandbox read-only` (restricts built-in shell) |
| Filesystem tool filtering | All env tools exposed via MCP | Only `bash` exposed by default (use `--use-all-filesystem-tools` for all) |
| Cost tracking | Yes (`total_cost_usd` in result event) | No (Codex doesn't report cost) |
| Token tracking | From `result` event | From `token_count` events |
| Subagent support | Yes (tracks `parent_tool_use_id`) | No |
| Provider support | Anthropic, OpenRouter, custom | OpenAI, OpenRouter, custom (via model_providers config) |

### OpenRouter / Custom Providers

For non-OpenAI providers, the agent configures a custom `model_provider` via
`-c model_providers.fh.*` flags. This sets the `base_url`, `env_key` (for
bearer auth), `wire_api`, and `support_namespaces=false` (to avoid the
`namespace` tool shape that OpenRouter's Responses API rejects). No local
proxy is needed.

## Known Limitations

- **No cost reporting**: Codex CLI doesn't emit cost data. The `cost_usd` field in results will be `None`.
- **No subagent tracking**: Unlike Claude Code, Codex doesn't expose subagent/parent relationships in its event stream.
- **Built-in shell still readable**: The `--sandbox read-only` mode blocks writes but the built-in shell can still read local files (e.g., `ls`, `cat`, `rg`). The system prompt instructs the agent to use MCP tools, but this isn't enforced for reads.
- **OpenRouter token tracking**: Token usage may not be reported when using OpenRouter.
