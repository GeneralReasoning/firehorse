"""Resume a previously-killed firehorse trial from its on-disk JSONL.

Today's flow:
    firehorse resume <results-dir> [--max-turns N] [--max-tasks 1]

The results dir is the one firehorse already creates per trial:

    results/<stamp>_<env-slug>_c<C>_t<T>/
        run.log
        run_result.json
        trial_<task_id>.jsonl
        trial_<task_id>_rewards.jsonl
        trial_<task_id>_result.json

Everything needed to resume is already in `trial_*.jsonl`:

  * line 1 — `openreward_prompt` with the env's system+environment prompt.
  * line 2 — `thread.started` with codex's `thread_id` (= codex session id).
  * every successful `item.completed` of type `mcp_tool_call` with
    `arguments` and `result.content[*].text` containing the `OR_REWARD`
    marker.

The resume process:
  1. Parse the JSONL for: env name, task_spec, model, effort, thread_id,
     and the ordered sequence of successful mcp_tool_calls.
  2. Start a fresh `firehorse run`, except:
       - the MCP bridge is told to **replay** the tool-call sequence
         (via `OPENREWARD_REPLAY_PATH=<json-file>`) before answering
         tools/list, so the new OR session arrives at the same env
         state as the dead one.
       - codex is launched with `exec resume <thread_id>` so it
         already knows the prior conversation.

This module owns the parsing + manifest building. The replay-during-init
logic lives in firehorse/mcp/bridge.py; the codex resume flag is wired
in firehorse/agents/codex.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ResumeState:
    """Everything pulled out of an existing trial JSONL needed to resume."""
    results_dir: Path
    trial_jsonl: Path
    env_name: str
    task_spec: dict
    task_id: str
    split: str
    model: str
    effort: Optional[str]
    thread_id: Optional[str]
    tool_calls: list[dict]  # [{tool, arguments, reward_seen, status}]
    total_reward_seen: float
    last_completed_index: int


_REWARD_MARKER = "[OR_REWARD:"


def _find_trial_jsonl(results_dir: Path) -> Path:
    """Find the single `trial_*.jsonl` that isn't a sidecar (_rewards / _result)."""
    candidates = [
        p for p in results_dir.glob("trial_*.jsonl")
        if not p.name.endswith("_rewards.jsonl") and "_result" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No trial_*.jsonl in {results_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple trial JSONLs in {results_dir} — resume supports single-trial dirs only"
        )
    return candidates[0]


def _read_run_result(results_dir: Path) -> dict:
    """Best-effort read of run_result.json for env/model/split fallback."""
    p = results_dir / "run_result.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_reward_marker(text: str) -> Optional[dict]:
    """`OR_REWARD:{"r": 0.0, "f": false}` is appended to most tool results."""
    if _REWARD_MARKER not in text:
        return None
    try:
        start = text.index(_REWARD_MARKER) + len(_REWARD_MARKER)
        end = text.index("]", start)
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def parse_results_dir(results_dir: Path) -> ResumeState:
    """Walk the trial JSONL once and pull out everything needed to resume."""
    trial_path = _find_trial_jsonl(results_dir)
    run_result = _read_run_result(results_dir)

    env_name: str = run_result.get("environment") or ""
    model: str = run_result.get("model") or ""
    split: str = run_result.get("split") or ""
    task_spec: dict = {}
    task_id: str = ""
    effort: Optional[str] = None
    thread_id: Optional[str] = None
    tool_calls: list[dict] = []
    total_reward = 0.0
    last_completed_index = -1

    with trial_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "openreward_prompt":
                # System prompt already carries env name in its text; the
                # most reliable source is run_result.json (we already
                # read it above). No-op here.
                continue
            if t == "openreward_summary":
                ts = ev.get("task_spec") or {}
                if ts:
                    task_spec = ts
                    task_id = str(ts.get("id") or ts.get("index") or "")
                if ev.get("env") and not env_name:
                    env_name = ev["env"]
                if ev.get("model") and not model:
                    model = ev["model"]
                continue
            if t == "thread.started":
                tid = ev.get("thread_id")
                if tid:
                    thread_id = tid
                continue
            if t == "item.completed":
                item = ev.get("item") or {}
                if item.get("type") != "mcp_tool_call":
                    continue
                if item.get("status") != "completed":
                    continue
                tool = item.get("tool") or ""
                args = item.get("arguments") or {}
                result = item.get("result") or {}
                # Pull reward marker from any text block in result.content
                reward_obj: Optional[dict] = None
                contents = (result.get("content") or []) if isinstance(result, dict) else []
                for block in contents:
                    if not isinstance(block, dict):
                        continue
                    txt = block.get("text") or ""
                    rew = _parse_reward_marker(txt)
                    if rew is not None:
                        reward_obj = rew
                        break
                r = float(reward_obj["r"]) if reward_obj and "r" in reward_obj else 0.0
                f_flag = bool(reward_obj["f"]) if reward_obj and "f" in reward_obj else False
                total_reward += r
                last_completed_index = len(tool_calls)
                tool_calls.append({
                    "tool": tool,
                    "arguments": args,
                    "reward_seen": r,
                    "finished_seen": f_flag,
                })

    if not env_name:
        raise RuntimeError(
            f"Could not determine env name from {results_dir} (run_result.json + "
            f"trial JSONL openreward_summary both missing it)"
        )

    return ResumeState(
        results_dir=results_dir.resolve(),
        trial_jsonl=trial_path.resolve(),
        env_name=env_name,
        task_spec=task_spec,
        task_id=task_id,
        split=split or "test",
        model=model,
        effort=effort,
        thread_id=thread_id,
        tool_calls=tool_calls,
        total_reward_seen=total_reward,
        last_completed_index=last_completed_index,
    )


def write_replay_manifest(state: ResumeState, dest: Path) -> Path:
    """Write the ordered tool-call sequence MCP bridge will replay during
    initialize(). Just the (tool, arguments) pairs — no rewards (those
    will be re-emitted by the live env)."""
    payload = {
        "env_name": state.env_name,
        "task_spec": state.task_spec,
        "task_id": state.task_id,
        "split": state.split,
        "thread_id": state.thread_id,
        "tool_calls": [
            {"tool": tc["tool"], "arguments": tc["arguments"]}
            for tc in state.tool_calls
        ],
        "expected_total_reward": state.total_reward_seen,
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def summarize(state: ResumeState) -> str:
    """One-line summary for the operator."""
    return (
        f"env={state.env_name} task_id={state.task_id} split={state.split} "
        f"model={state.model} thread_id={state.thread_id or '?'} "
        f"replay_calls={len(state.tool_calls)} prior_reward={state.total_reward_seen:.2f}"
    )


async def replay_against_fresh_session(
    state: ResumeState,
    secrets: Optional[dict] = None,
    print_every: int = 1,
) -> dict:
    """Open a fresh OR session, replay every recorded tool call directly
    via the SDK, and return a summary dict.

    This does NOT spawn an MCP bridge and does NOT involve any LLM —
    it's the "rebuild env state" half of resume. Useful as a diagnostic
    on its own (`firehorse replay <dir>`).
    """
    from openreward.client import AsyncOpenReward
    from openreward.api.environments.types import Task

    # Parse env_name → namespace + server_name (matches bridge.py logic)
    namespace: Optional[str] = None
    server_name = state.env_name
    if "/" in state.env_name:
        namespace, server_name = state.env_name.split("/", 1)
    task_env_name = server_name

    client = AsyncOpenReward()
    env = client.environments.get(state.env_name)
    task = Task(
        server_name=server_name,
        environment_name=task_env_name,
        task_spec=state.task_spec,
        namespace=namespace,
    )

    print(
        f"[replay] opening fresh session for {state.env_name} (this can "
        f"take 30-90s on a cold OR backend)...",
        file=sys.stderr, flush=True,
    )

    summary: dict = {
        "env_name": state.env_name,
        "task_id": state.task_id,
        "total_calls": len(state.tool_calls),
        "ok": 0,
        "fail": 0,
        "total_reward_seen": 0.0,
        "errors": [],
    }
    _REWARD = "[OR_REWARD:"

    async with env.session(task, secrets) as session:
        # Only env-side tools can be replayed against OR. Codex built-ins
        # (bash, multi_edit, ls, ...) are agent-local; sending them to
        # OR would record bogus "unknown tool" entries on the rollout.
        or_tools = await session.list_tools()
        or_tool_names = {t.name for t in or_tools}
        summary["skipped"] = 0
        print(f"[replay] session open. Replaying {len(state.tool_calls)} "
              f"tool calls ({len(or_tool_names)} OR tools available)...",
              file=sys.stderr, flush=True)
        for i, tc in enumerate(state.tool_calls):
            tool = tc["tool"]
            args = tc.get("arguments") or {}
            if tool not in or_tool_names:
                summary["skipped"] += 1
                print(
                    f"[replay] {i+1:>4}/{len(state.tool_calls)} "
                    f"tool={tool:<24s} SKIP (agent-side, not an OR tool)",
                    file=sys.stderr, flush=True,
                )
                continue
            try:
                result = await session.call_tool(tool, args)
                summary["ok"] += 1
                # Pull reward marker from result text if present
                blocks = getattr(result, "content", None) or []
                for b in blocks:
                    txt = getattr(b, "text", None) or ""
                    if _REWARD in txt:
                        try:
                            chunk = txt[txt.index(_REWARD) + len(_REWARD):]
                            chunk = chunk[: chunk.index("]")]
                            obj = json.loads(chunk)
                            summary["total_reward_seen"] += float(obj.get("r", 0))
                        except Exception:
                            pass
                if print_every == 1 or (i + 1) % print_every == 0 or (i + 1) == len(state.tool_calls):
                    print(
                        f"[replay] {i+1:>4}/{len(state.tool_calls)} "
                        f"tool={tool:<24s} ok  total_reward={summary['total_reward_seen']:.1f}",
                        file=sys.stderr, flush=True,
                    )
            except Exception as e:
                summary["fail"] += 1
                msg = f"call {i+1} tool={tool} raised {type(e).__name__}: {e}"
                summary["errors"].append(msg)
                print(f"[replay] {msg}", file=sys.stderr, flush=True)
    return summary
