<p align="center">
  <img src="assets/firehorselogo.png" alt="Firehorse" width="400">
</p>

<p align="center">
  <a href="https://docs.openreward.ai/">
    <img src="https://img.shields.io/badge/docs-openreward.ai-blue" alt="Docs">
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  </a>
</p>

🔥🐴 Firehorse is a library of agent harnesses for running models against [OpenReward environments](https://openreward.ai/environments).

It bridges popular harnesses (Claude Code, Codex, Gemini CLI, ReAct, ReSum) with OpenReward, letting you sample agentic trajectories without setting up environment infrastructure. Firehorse manages concurrent trial execution and produces structured trajectory logs and aggregate results.

## Quickstart

Install the `firehorse` library:

```bash
pip install firehorse-cli
```

Set up your environment variables - get an OpenReward key [here](https://openreward.ai/keys):

```bash
export OPENREWARD_API_KEY=your-openreward-key
export OPENROUTER_API_KEY=your-openrouter-key # or other env if using diff model provider
```

Ensure you have the harness CLI installed (in this case Claude Code) and then run:

```bash
# Run Claude Code agent against an environment
firehorse \
  --env Eigent/SETA \
  --agent claude-code \
  --model openrouter/moonshotai/kimi-k2.6
  --split train
  --output-dir ./kimi-seta
```

## Prerequisites

- **Python 3.10+**
- **OpenReward API key** — get one at [openreward.ai](https://openreward.ai/keys)
- **LLM provider API key** — Anthropic, OpenAI, Google, or OpenRouter

For specific agents:
- **`claude-code`** — requires [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed (tested with v2.1.88)
- **`codex`** — requires [Codex CLI](https://github.com/openai/codex) installed (tested with v0.121.0)
- **`gemini`** — requires [Gemini CLI](https://github.com/google/gemini-cli) installed (tested with v0.38.2)

## Agent Types

| Agent | Description | Providers |
|-------|-------------|-----------|
| `resum` (default) | ReAct loop with compaction when context fills up | Anthropic, OpenAI, Google, OpenRouter, custom |
| `claude-code` | Claude Code CLI with environment tools via MCP | Anthropic, OpenRouter |
| `codex` | Codex CLI with environment tools via MCP | OpenAI |
| `react` | Direct LLM API Reason-Act loop| Anthropic, OpenAI, Google, OpenRouter, custom |
| `gemini` | Gemini CLI with environment tools via MCP | Google |

## Thinking / Reasoning

The `--effort` flag controls how much thinking/reasoning the model does. It's supported by all agents. Omitting the flag (or passing `--effort none`) leaves the reasoning parameter unset so each provider uses its own default. The effort level maps to each provider's native thinking mechanism:

| Provider | Mechanism | low | medium | high | max |
|---|---|---|---|---|---|
| **Anthropic** | Adaptive thinking (`effort` param) | low | medium | high | max (Opus only) |
| **OpenAI** | `reasoning_effort` | low | medium | high | xhigh |
| **Google Gemini 3.x** | `thinking_level` | low | medium | high | high |
| **OpenRouter** | Passes through to underlying provider | — | — | — | — |

```bash
# High thinking (opt-in; default is no effort flag — provider picks its own default)
firehorse --env GeneralReasoning/CTF --model anthropic/claude-sonnet-4-6 --effort high

# Max thinking for deep reasoning tasks
firehorse --env Naman/R2E-Gym --agent codex --model openai/gpt-5.4 --split all --effort xhigh

# Low thinking for speed
firehorse --env collinear/YC-Bench --agent react --model google/gemini-3.1-flash-lite-preview --effort low
```

Each agent maps `--effort` to its provider's native parameter. Models that don't support thinking (e.g., GPT-4.1) ignore the flag.

## CLI Reference

```
firehorse --env ENV --model MODEL [OPTIONS]

Required:
  --env              Environment name (e.g. MyOrg/my-environment)
  --model            Model identifier (e.g. anthropic/claude-sonnet-4-6)

Options:
  --agent            Agent type: claude-code, codex, gemini, react, resum (default: resum)
  --variant          Environment variant (e.g. 'mathnocode' for GeneralReasoning/MATH) (default: none)
  --split            Task split to evaluate (default: test)
  --n-concurrent     Max parallel trials (default: 1)
  --max-tasks        Limit number of tasks to evaluate
  --max-turns        Max tool call turns per trial
  --run-name         Name for this evaluation run
  --effort           Thinking effort: none, low, medium, high, max, xhigh (default: none — use model default)
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

### JSONL Trajectory Format

All agents share the same bookend events in `trial_*.jsonl`:

```jsonl
{"type": "openreward_prompt", "system_prompt": "...", "environment_prompt": "..."}
... agent-specific events ...
{"type": "openreward_summary", "task_spec": {...}, "env": "...", "model": "...", "usage": {...}}
```

The events in between depend on whether the agent is API-based or CLI-based.

**API agents** (`react`, `resum`) produce normalized firehorse events:
- `assistant` — model response with text, reasoning, and tool calls
- `tool_call` — tool invocation with name, arguments, and call ID (resum only; react embeds in the assistant event)
- `tool_result` — tool output with explicit `reward` and `finished` fields

**CLI agents** (`claude-code`, `codex`, `gemini`) pass through the raw CLI stream format:
- **claude-code**: Claude's `stream-json` events (`assistant`/`user` with `message.content` blocks)
- **codex**: Codex's `--json` events (`item.started`/`item.completed` with nested `item.type`)
- **gemini**: Gemini's `stream-json` events (`message` deltas, `tool_use`, `tool_result`)

Reward signals are available in the `trial_*_rewards.jsonl` sidecar file and in OpenReward rollouts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Documentation

Full documentation is available at [docs.openreward.ai](https://docs.openreward.ai/).
