# Claude Code Agent in Firehorse

Firehorse uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code) as its agent runtime. Claude Code is launched as a subprocess for each trial, with environment tools proxied via MCP.

## Architecture

```
firehorse CLI
  └── orchestrator (concurrent trials)
        └── trial runner
              ├── OpenReward session (HTTP/SSE)
              └── Claude Code subprocess
                    ├── --print --verbose --output-format stream-json
                    ├── --mcp-config (MCP bridge → OpenReward session)
                    ├── --disallowed-tools (built-ins replaced by env)
                    ├── --append-system-prompt (MCP tool name mapping)
                    └── stdout → JSONL + OpenReward rollout logging
```

Each trial:
1. Opens an OpenReward session for the task
2. Fetches the environment's prompt and tools
3. Spawns Claude Code with the prompt, tools (via MCP), and a system prompt
4. Streams stdout events to JSONL files and OpenReward rollout API
5. Reads the reward/finished signal from the MCP bridge's result file

## Model Configuration

### Anthropic (direct)

Uses the Anthropic API directly. Requires `ANTHROPIC_API_KEY`.

```bash
firehorse --env GeneralReasoning/portfolio \
  --model anthropic/claude-sonnet-4-5 \
  --split test --max-tasks 10

# With extended thinking control
firehorse --env GeneralReasoning/portfolio \
  --model anthropic/claude-opus-4-6 \
  --effort max
```

The `anthropic/` prefix is stripped before passing to `claude --model`. Extended thinking (`--effort`) is only enabled for Anthropic models.

### OpenRouter

Routes through OpenRouter's Anthropic-compatible endpoint. Requires `OPENROUTER_API_KEY`.

Claude Code only speaks the **Anthropic Messages API**. It cannot talk to OpenAI or Google APIs directly. OpenRouter solves this by providing an Anthropic-compatible endpoint that translates requests to any model provider.

```bash
# Non-Anthropic model via OpenRouter
firehorse --env GeneralReasoning/portfolio \
  --model openrouter/z-ai/glm-5

# OpenAI model via OpenRouter
firehorse --env GeneralReasoning/portfolio \
  --model openrouter/openai/gpt-4.1

# Google model via OpenRouter
firehorse --env GeneralReasoning/portfolio \
  --model openrouter/google/gemini-2.5-pro

# Anthropic model via OpenRouter (extended thinking still works)
firehorse --env GeneralReasoning/portfolio \
  --model openrouter/anthropic/claude-opus-4-6 \
  --effort high
```

Under the hood, this sets:
- `ANTHROPIC_BASE_URL=https://openrouter.ai/api` — points Claude Code at OpenRouter
- `ANTHROPIC_AUTH_TOKEN=<OPENROUTER_API_KEY>` — authenticates with OpenRouter
- `ANTHROPIC_API_KEY=<OPENROUTER_API_KEY>` — overrides any existing Anthropic key

The model name after `openrouter/` is passed as-is to `claude --model` (e.g., `z-ai/glm-5`, `openai/gpt-4.1`). OpenRouter translates between the Anthropic Messages API format and the target model's native format.

`--effort` is automatically skipped for non-Anthropic models since extended thinking is an Anthropic-only feature. If the model name starts with `anthropic/` (even via OpenRouter), `--effort` is applied.

### Custom provider (--provider-url)

For any **Anthropic-compatible** proxy (LiteLLM, vLLM with Anthropic support, etc.). The endpoint must accept Anthropic Messages API format.

```bash
firehorse --env GeneralReasoning/portfolio \
  --model my-custom-model \
  --provider-url https://my-proxy.example.com/v1
```

Sets `ANTHROPIC_BASE_URL` to the provided URL. This does **not** work with native OpenAI or Google endpoints — use `openrouter/` for those.

## Tool Override Behavior

Filesystem built-in tools are **always disabled**, regardless of what the environment provides:

| Claude built-ins disabled | When |
|---|---|
| `Bash`, `Read`, `Write`, `Edit`, `Grep`, `Glob` | Always — filesystem access is only available through environment-provided sandbox tools |
| `NotebookEdit` | When the environment provides `notebookedit` |

If the environment provides filesystem tools (e.g. `bash`, `read`, `edit`), the agent uses those sandboxed versions. If the environment does **not** provide filesystem tools, the agent has no filesystem access at all. Non-filesystem built-ins (`WebSearch`, `WebFetch`, etc.) remain available.

Planning tools (`TodoWrite`, `Task`) always use Claude's built-in versions. The environment's `todo_write` tool is excluded from MCP to avoid conflicts.

Disabled built-ins are passed via `--disallowed-tools` to Claude Code.

### System Prompt Tool Mapping

Claude Code's default system prompt references built-in tool names (`Read`, `Edit`, etc.) that are disabled when replaced by MCP tools. To prevent the model from trying to use non-existent tools, firehorse appends a tool mapping section via `--append-system-prompt` that:

1. Lists all available MCP tools by their actual names (e.g. `mcp__openreward__read`, `mcp__openreward__submit_answer`)
2. Explicitly maps disabled built-ins to their MCP replacements (e.g. "use `mcp__openreward__read` instead of `Read`")
3. Clarifies that other built-in tools (`WebSearch`, `WebFetch`, `TodoWrite`, `Agent`, etc.) remain available

### Builtin Tool Descriptions

By default, MCP tool descriptions are overridden with rich descriptions extracted from Claude Code's source (see `builtin_descriptions.py`). These descriptions reference other tools using MCP names with "if available" language (e.g. "Use mcp__openreward__read if available") since not all environments provide every tool. The original Claude Code descriptions are preserved in `legacy_builtin_descriptions.py`.

Use `--use-env-descriptions` to use the environment's original tool descriptions instead.

## Thinking / Reasoning

Claude Code emits thinking blocks in its stream-json output. Firehorse captures these for both logging targets:

| Block type | Source | JSONL | Rollout |
|---|---|---|---|
| `thinking` | Anthropic models, some OpenRouter models | Raw event preserved | Logged as `ReasoningItem` |
| `redacted_thinking` | Anthropic (encrypted), OpenRouter (`openrouter.reasoning:<base64>`) | Raw event preserved | Decoded if OpenRouter, `[redacted thinking]` if Anthropic |
| `text` | All models | Raw event preserved | Logged as `AssistantMessage` |
| `tool_use` | All models | Raw event preserved | Logged as `ToolCall` |
| `tool_result` | All models | Raw event preserved | Logged as `ToolResult` with reward/finished extracted |

Interleaving order is preserved — events are processed sequentially as they stream from Claude Code's stdout.

## Output Files

For each run, firehorse writes to the output directory (default: auto-generated from run name):

```
output_dir/
├── run_result.json                      # Aggregate results for the run
├── trial_{task_id}.jsonl                # Full stream-json trajectory
├── trial_{task_id}_result.json          # Per-trial result summary
├── trial_{task_id}_rewards.jsonl        # Reward sidecar (one line per tool call)
└── trial_{task_id}_subagent_{id}.jsonl  # Subagent trajectories (if Task tool used)
```

### JSONL trajectory format

Each line is a JSON event from Claude Code's `--output-format stream-json`, plus two synthetic events:

- **`openreward_prompt`** (first line) — the system prompt and environment prompt passed via CLI flags (these don't appear in Claude Code's stream output)
- **`openreward_summary`** (last line) — task spec, bridge result (reward/finished/calls), and usage (cost/tokens/duration)

### Rewards sidecar

One JSON line per MCP tool call, tracking dense rewards:

```json
{"call_count": 1, "tool": "bash", "reward": 0.0, "finished": false, "total_reward": 0.0, "timestamp": 1775007137.1}
{"call_count": 2, "tool": "answer", "reward": 1.0, "finished": true, "total_reward": 1.0, "timestamp": 1775007308.9}
```

### Per-trial result.json

```json
{
  "task_id": "mdp_portfolio",
  "environment": "GeneralReasoning/portfolio",
  "model": "openrouter/z-ai/glm-5",
  "final_reward": 1.0,
  "finished": true,
  "tool_calls": 11,
  "duration_seconds": 207.6,
  "usage": {"cost_usd": 0.146, "input_tokens": 23099, "output_tokens": 4132},
  "rollout_url": "https://openreward.ai/rollout/a182b9ba-..."
}
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For `anthropic/` models | Anthropic API key |
| `OPENROUTER_API_KEY` | For `openrouter/` models | OpenRouter API key (covers OpenAI, Google, etc.) |
| `OPENREWARD_API_KEY` | Always | OpenReward platform API key |
| `OPENREWARD_URL` | No | Override OpenReward base URL (default: https://openreward.ai) |

Session secrets (e.g., API keys the environment needs) are auto-detected from env vars or passed explicitly:

```bash
firehorse --env MyEnv/thing \
  --model anthropic/claude-sonnet-4-5 \
  --secret openai_api_key=sk-... \
  --secret custom_key=value
```

## CLI Reference

```
firehorse --env <owner/name> --model <model> [options]

Required:
  --env              Environment (e.g., GeneralReasoning/portfolio)
  --model            Model identifier (see Model Configuration above)

Options:
  --agent            Agent type (default: claude-code)
  --n-concurrent     Max parallel trials (default: 1)
  --split            Task split (default: test)
  --max-tasks        Limit number of tasks
  --run-name         Custom run name
  --max-turns        Max tool call turns per trial
  --effort           Thinking effort: low|medium|high|max (default: high, Anthropic only)
  --provider-url     Custom API base URL for non-Anthropic models
  --disable-builtin-tools  Comma-separated Claude built-in tools to disable
  --no-logging       Disable OpenReward rollout logging
  --secret KEY=VAL   Session secret (repeatable)
  --use-env-descriptions  Use environment tool descriptions instead of Claude Code built-ins
  --output-dir       Output directory for trajectory logs
```
