# ReAct Agent

A multi-provider ReAct (Reason-Act) agent for evaluating LLMs on OpenReward environments. Unlike the `claude-code` agent which shells out to the Claude CLI via an MCP bridge, the `react` agent calls LLM APIs directly and routes tool calls through the OpenReward session.

## Supported Providers

| Provider | Model format | API used | API key env var |
|----------|-------------|----------|-----------------|
| Anthropic | `anthropic/claude-sonnet-4-6` | Messages API | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-4o` | Responses API | `OPENAI_API_KEY` |
| Google | `google/gemini-2.5-flash` | Generate Content API | `GEMINI_API_KEY` |
| OpenRouter | `openrouter/deepseek/deepseek-v3.2` | Chat Completions API | `OPENROUTER_API_KEY` |

## Quick Start

```bash
pip install -e ".[react]"

# Anthropic
firehorse --env GeneralReasoning/counter --agent react \
  --model anthropic/claude-sonnet-4-6 --split train --max-tasks 1

# OpenAI
firehorse --env GeneralReasoning/counter --agent react \
  --model openai/gpt-4o --split train --max-tasks 1

# Google Gemini
firehorse --env GeneralReasoning/counter --agent react \
  --model google/gemini-2.5-flash --split train --max-tasks 1

# OpenRouter (any model)
firehorse --env GeneralReasoning/counter --agent react \
  --model openrouter/anthropic/claude-sonnet-4-6 --split train --max-tasks 1
```

## How It Works

The agent runs a simple loop for each trial:

```
1. Open an OpenReward session for the task
2. Get the prompt and tools from the session
3. Send the prompt + tools to the LLM
4. Parse tool calls from the LLM response
5. Execute each tool call via session.call_tool()
6. Append tool results to the conversation
7. If the environment signals finished=True, stop
8. Otherwise, go to step 3
```

The loop terminates when:
- The environment returns `finished=True` on a tool call
- The LLM responds without any tool calls (text-only response)
- `--max-turns` is reached

### Provider Differences

Each provider has slightly different APIs, but the core loop is the same. The key differences are:

- **Tool format**: The OpenReward SDK handles conversion via `session.list_tools(format="anthropic"|"openai"|"google"|"openrouter")`
- **System prompt delivery**: Anthropic uses `system=`, OpenAI uses `instructions=`, Google uses `system_instruction=`, OpenRouter uses a system message in the messages list
- **Tool call parsing**: Each provider returns tool calls in a different structure (Anthropic `tool_use` blocks, OpenAI `function_call` items, Google `function_call` parts, OpenRouter `tool_calls` on the message)
- **Tool result format**: Anthropic supports native image blocks in results; other providers receive text-only results with image placeholders

### Architecture

```
firehorse CLI
  -> orchestrator.py      # lists tasks, auto-detects secrets, runs trials concurrently
    -> trial.py            # opens session, builds TrialContext, calls agent.run()
      -> react.py          # the ReAct loop
        -> LLM API         # anthropic/openai/google SDK
        -> session.call_tool()  # OpenReward environment
```

The react agent does NOT use the MCP bridge (`firehorse.mcp`). It calls `session.call_tool()` directly, which is simpler and avoids the subprocess overhead of the `claude-code` agent.

## Thinking / Reasoning

The `--effort` flag enables and controls thinking/reasoning for providers that support it:

```bash
# Anthropic with adaptive thinking
firehorse --env MyOrg/my-env --agent react \
  --model anthropic/claude-sonnet-4-6 --effort high

# Google Gemini with thinking
firehorse --env MyOrg/my-env --agent react \
  --model google/gemini-3.1-flash-lite-preview --effort high

# OpenAI o-series with reasoning
firehorse --env MyOrg/my-env --agent react \
  --model openai/o3 --effort medium
```

How effort maps to each provider:

- **Anthropic**: Uses adaptive thinking (`thinking: {type: "adaptive"}` + `effort` parameter). All effort levels supported; `max` only works on Opus.
- **OpenAI**: Uses `reasoning_effort` for o-series models (o1, o3, o4). Non-o-series models (GPT-4.1, GPT-5.4) ignore the flag.
- **Google Gemini 3.x**: Uses `thinking_level` (low/medium/high). `max` maps to `high`.
- **Google Gemini 2.5**: Uses `thinking_budget_tokens` (low=1024, medium=5000, high=16000, max=24576).
- **OpenRouter**: Passes `reasoning_effort` through for models that support it.

## Secrets

**LLM API keys** are read from environment variables by the SDK clients automatically. Set the appropriate env var for your provider before running.

**Environment secrets** (e.g., API keys the environment needs to access external services) are handled separately by the orchestrator. They are auto-detected from env vars based on what the environment declares as required, or passed explicitly with `--secret KEY=VALUE`.

## Output

Each trial produces files in the output directory (defaults to a timestamped run name, or set with `--output-dir`):

```
output-dir/
  trial_0.jsonl              # Full agent trajectory (JSONL)
  trial_0_result.json        # Per-trial summary
  trial_1.jsonl
  trial_1_result.json
  run_result.json            # Aggregate run summary
```

### JSONL Trajectory Format

Each line is a JSON object with a `type` field:

```jsonl
{"type": "system", "content": "You are an agent..."}
{"type": "user", "content": "You are playing a counting game..."}
{"type": "assistant", "provider": "anthropic", "raw": {"role": "assistant", "content": [...]}}
{"type": "tool_result", "provider": "anthropic", "tool_use_id": "tu_001", "tool_name": "bash", "reward": null, "finished": false}
{"type": "tool_results", "provider": "anthropic", "raw": {"role": "user", "content": [...]}}
{"type": "openreward_summary", "task_spec": {...}, "reward": 1.0, "finished": true, ...}
```

Each `tool_result` entry includes `reward` and `finished` from the environment's response. For batched providers (Anthropic, Google), individual `tool_result` entries are written per tool call, followed by the batched `tool_results` entry containing the raw provider message.

### Per-Trial Result JSON

```json
{
  "task_id": 0,
  "task_spec": {"count": 31, "goal": -36},
  "environment": "GeneralReasoning/counter",
  "agent": "react",
  "model": "anthropic/claude-sonnet-4-6",
  "split": "train",
  "final_reward": 1.0,
  "finished": true,
  "tool_calls": 2,
  "duration_seconds": 8.2,
  "usage": {"input_tokens": 1550, "output_tokens": 137},
  "rollout_url": "https://openreward.ai/rollout/..."
}
```

### Rollout Logging

When logging is enabled (default), each trial is recorded as an OpenReward rollout viewable at `https://openreward.ai/rollout/<id>`. The agent uses provider-specific rollout log methods (`log_anthropic_message`, `log_openai_response`, etc.) so the rollout viewer can render the conversation correctly. Disable with `--no-logging`.

## CLI Options

```
--env ENV               Environment name (e.g. GeneralReasoning/counter)
--agent react           Use the react agent
--model MODEL           Provider/model (e.g. anthropic/claude-sonnet-4-6)
--split SPLIT           Dataset split (default: test)
--max-tasks N           Limit number of tasks
--max-turns N           Max tool calls per trial
--n-concurrent N        Parallel trials (default: 1)
--output-dir DIR        Where to write logs
--secret KEY=VALUE      Pass secrets to the environment (repeatable)
--provider-url URL      Override the LLM API base URL
--no-logging            Disable OpenReward rollout logging
```

## Tests

```bash
pytest tests/test_react.py -v
```

32 tests covering provider parsing, tool output formatting, per-provider core loops (mocked), max_turns enforcement, error handling, JSONL logging (including per-message reward/finished fields), and rollout logging.
