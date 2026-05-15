"""Firehorse CLI — agent evaluation harness for OpenReward environments."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from firehorse.config import RunConfig
from firehorse.orchestrator import run_evaluation


async def command_run(
    env: str,
    agent: str,
    model: str,
    variant: str | None = None,
    n_concurrent: int = 1,
    split: str = "test",
    max_tasks: int | None = None,
    skip_tasks: int = 0,
    plan_mode: bool = False,
    run_name: str | None = None,
    max_turns: int | None = None,
    provider_url: str | None = None,
    disable_builtin_tools: str | None = None,
    secrets: dict[str, str] | None = None,
    output_dir: str | None = None,
    effort: str | None = None,
    logging: bool = True,
    use_env_descriptions: bool = False,
    use_all_filesystem_tools: bool = False,
    toolset: str | None = None,
) -> int:
    disabled = []
    if disable_builtin_tools:
        disabled = [t.strip() for t in disable_builtin_tools.split(",") if t.strip()]

    config = RunConfig(
        env=env,
        agent=agent,
        model=model,
        variant=variant,
        n_concurrent=n_concurrent,
        split=split,
        max_tasks=max_tasks,
        skip_tasks=skip_tasks,
        plan_mode=plan_mode,
        run_name=run_name,
        max_turns=max_turns,
        provider_url=provider_url,
        disable_builtin_tools=disabled,
        secrets=secrets or {},
        output_dir=output_dir,
        effort=effort,
        logging=logging,
        use_builtin_descriptions=not use_env_descriptions,
        use_all_filesystem_tools=use_all_filesystem_tools,
        toolset=toolset,
    )

    summary = await run_evaluation(config)
    return 0 if summary.failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firehorse",
        description="Agent evaluation harness for OpenReward environments",
    )
    parser.add_argument("--env", required=True, help="Environment name (e.g. GeneralReasoning/portfolio)")
    parser.add_argument("--agent", default="resum", help="Agent type (default: resum)")
    parser.add_argument("--model", required=True, help="Model identifier (e.g. anthropic/claude-opus-4-6)")
    parser.add_argument("--variant", default=None, help="Environment variant (e.g. 'mathnocode' for GeneralReasoning/MATH)")
    parser.add_argument("--n-concurrent", type=int, default=1, help="Max parallel trials (default: 1)")
    parser.add_argument("--split", default="test", help="Which split to evaluate (default: test)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--skip-tasks", type=int, default=0, help="Skip first N tasks (default: 0)")
    parser.add_argument("--run-name", default=None, help="Name for this run")
    parser.add_argument("--max-turns", type=int, default=None, help="Max tool call turns per trial")
    parser.add_argument(
        "--effort", default=None, choices=["none", "low", "medium", "high", "max", "xhigh"],
        help="Reasoning/thinking effort passed to the model (default: none — use model's own default)",
    )
    parser.add_argument(
        "--provider-url", default=None,
        help="Custom API base URL for non-Anthropic models (e.g. https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--disable-builtin-tools", default=None,
        help="Comma-separated list of Claude built-in tools to disable",
    )
    parser.add_argument(
        "--plan-mode", action="store_true", default=False,
        help="Enable Claude Code plan mode (claude-code agent only)",
    )
    parser.add_argument(
        "--no-logging", action="store_true", default=False,
        help="Disable OpenReward rollout logging (default: logging enabled)",
    )
    parser.add_argument(
        "--secret", action="append", default=[], metavar="KEY=VALUE",
        help="Session secret (can be repeated, e.g. --secret openai_api_key=sk-...)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write per-trial JSONL trajectory logs",
    )
    parser.add_argument(
        "--use-env-descriptions", action="store_true", default=False,
        help="Use environment tool descriptions instead of Claude Code built-in descriptions (default: use Claude Code descriptions)",
    )
    parser.add_argument(
        "--use-all-filesystem-tools", action="store_true", default=False,
        help="Codex: expose all filesystem tools via MCP (default: only bash, others filtered)",
    )
    parser.add_argument(
        "--toolset", default=None,
        help="Override the SDK toolset (default: auto-pick by agent). Pass an "
             "empty string to skip the toolset entirely — required for envs "
             "without a declared sandbox (e.g. RJT1990/TheTraitors).",
    )
    return parser


def _resume(argv: list[str]) -> int:
    """`firehorse resume <results-dir>` — replay a killed trial.

    Parses the existing trial JSONL to recover (env, task_spec, model,
    codex thread_id, prior tool-call sequence), writes a small manifest
    to a temp file, and re-invokes `command_run` with two env vars set:

      OPENREWARD_REPLAY_PATH      → the manifest (mcp/bridge.py replays).
      FIREHORSE_RESUME_THREAD_ID  → codex thread id (codex.py uses
                                    `codex exec resume <id>` to re-attach
                                    to the prior conversation).
    """
    import argparse, tempfile
    from firehorse.resume import parse_results_dir, summarize, write_replay_manifest

    p = argparse.ArgumentParser(
        prog="firehorse resume",
        description="Resume a killed firehorse trial from its results dir.",
    )
    p.add_argument("results_dir", help="A results/<stamp>_<env>_… directory")
    p.add_argument("--agent", default="codex",
                   help="Agent to use for the resume (default: codex; only codex supports resume today)")
    p.add_argument("--model", default=None,
                   help="Override the model (default: same as the dead run)")
    p.add_argument("--effort", default="max",
                   choices=["none", "low", "medium", "high", "max", "xhigh"])
    p.add_argument("--max-turns", type=int, default=None,
                   help="Optional cap for the resumed run")
    p.add_argument("--secret", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--toolset", default=None,
                   help='Override toolset (pass "" to skip the SDK wrapper)')
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"[resume] not a directory: {results_dir}", file=sys.stderr)
        return 2

    state = parse_results_dir(results_dir)
    print(f"[resume] {summarize(state)}", file=sys.stderr)

    # Recover the original rollout id so we can replay its messages into
    # the new rollout (so the openreward.ai view of the resumed run
    # mirrors the dead one). All firehorse agents print
    # `[<agent>] Rollout: https://openreward.ai/rollout/<id>` to stderr
    # right after creating their rollout, so run.log has it.
    from firehorse.rollout_replay import extract_rollout_id_from_run_log
    orig_rollout_id = extract_rollout_id_from_run_log(results_dir)
    if orig_rollout_id:
        print(f"[resume] original rollout id: {orig_rollout_id}", file=sys.stderr)
    else:
        print(
            "[resume] could not find original rollout id in run.log — "
            "resumed rollout will start with a fresh message history "
            "(env state will still be rebuilt).",
            file=sys.stderr,
        )

    # Note: we no longer use codex's `exec resume <thread_id>`. The agent
    # starts as a fresh codex process and discovers env state via the
    # env's own tools (view_current_squad, etc.). The thread_id from the
    # JSONL is informational only.
    if state.thread_id is None:
        print("[resume] (no codex thread_id in JSONL — agent will start fresh "
              "and rediscover state via env tools.)", file=sys.stderr)

    # Manifest goes alongside the existing trial files so it's not lost
    # when temp dirs cycle. Re-creatable from the JSONL, so safe to overwrite.
    manifest_path = results_dir / "resume_manifest.json"
    write_replay_manifest(state, manifest_path)
    print(f"[resume] wrote {manifest_path}", file=sys.stderr)

    # Drop a new output dir so we don't trample the prior trial's logs.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env_slug = state.env_name.replace("/", "_")
    new_out = results_dir.parent / f"{stamp}_{env_slug}_RESUMED"
    new_out.mkdir(parents=True, exist_ok=True)

    # Stamp the resumed run with provenance so downstream tooling (compare
    # scripts, dashboards) can join it back to the original trial.
    from collections import Counter
    _tool_hist = Counter(tc["tool"] for tc in state.tool_calls)
    resume_meta = {
        "kind": "firehorse.resume",
        "resumed_at": datetime.now().isoformat(timespec="seconds"),
        "cloned_from": {
            "results_dir": str(results_dir),
            "results_dir_name": results_dir.name,
            "trial_jsonl": str(state.trial_jsonl),
            "thread_id": state.thread_id,
            "rollout_id": orig_rollout_id,
            "rollout_url": (
                f"https://openreward.ai/rollout/{orig_rollout_id}"
                if orig_rollout_id else None
            ),
            "replay_calls": len(state.tool_calls),
            "replay_calls_by_tool": dict(_tool_hist.most_common()),
            "last_completed_index": state.last_completed_index,
            "prior_reward_seen": state.total_reward_seen,
        },
        "env": state.env_name,
        "task_id": state.task_id,
        "split": state.split,
        "model": args.model or state.model,
        "effort": args.effort,
        "agent": args.agent,
        "max_turns": args.max_turns,
        "replay_manifest": str(manifest_path),
        "replay_progress_file": str(new_out / "replay_progress.jsonl"),
    }
    (new_out / "resume_meta.json").write_text(
        json.dumps(resume_meta, indent=2), encoding="utf-8"
    )
    print(
        f"[resume] cloned from {results_dir.name} "
        f"(thread={state.thread_id or '?'}, "
        f"{len(state.tool_calls)} prior calls, "
        f"prior_reward={state.total_reward_seen:.2f}) → {new_out.name}",
        file=sys.stderr,
    )

    secrets = {}
    for s in (args.secret or []):
        if "=" not in s:
            print(f"Invalid --secret format: {s!r}", file=sys.stderr)
            return 1
        k, v = s.split("=", 1)
        secrets[k] = v

    # Only env var we need: OPENREWARD_REPLAY_PATH. The MCP bridge picks
    # this up during initialize() and fast-replays every prior tool call
    # against the fresh OR session, so by the time codex queries tools/list
    # the env is back in its pre-crash state. Codex itself starts fresh.
    # The codex agent (codex.py) auto-bumps its MCP startup_timeout when
    # this var is set — replay can take minutes for long rollouts.
    os.environ["OPENREWARD_REPLAY_PATH"] = str(manifest_path)
    # Pass the original rollout id to the agent harness so it can replay
    # the dead session's messages into the new rollout right after
    # creating it (firehorse/rollout_replay.py:maybe_replay_into).
    if orig_rollout_id:
        os.environ["OPENREWARD_REPLAY_ROLLOUT_ID"] = orig_rollout_id
    # Bridge mirrors every replay step to this file; we tail it below
    # so the user sees per-step progress in real time. Codex 0.129
    # buffers MCP subprocess stderr too aggressively for the bridge's
    # own prints to surface live.
    progress_path = new_out / "replay_progress.jsonl"
    os.environ["OPENREWARD_REPLAY_PROGRESS_FILE"] = str(progress_path)
    print(
        f"[resume] replay manifest has {len(state.tool_calls)} prior tool calls. "
        f"Tailing live progress from {progress_path.name}.",
        file=sys.stderr,
    )

    # Background thread: tail the bridge's replay_progress.jsonl and print
    # `[REPLAY i/M tool=X]` per line. Exits on the "complete" event.
    #
    # The bridge (a subprocess of codex) writes the file; we read it. On
    # Windows, a long-lived file handle in process A doesn't reliably see
    # new data appended by process B even with line-buffered writes —
    # readline() returns EOF and stays there. Workaround: poll file size
    # via stat(), and on every cycle reopen → seek(last_pos) → read.
    import threading, time as _time
    _stop_tail = threading.Event()

    def _emit(ev: dict) -> bool:
        """Print one progress event. Returns True if this was 'complete'."""
        et = ev.get("event")
        if et == "begin":
            print(
                f"[REPLAY] begin: {ev.get('total')} prior tool calls "
                f"to replay against fresh OR session",
                file=sys.stderr, flush=True,
            )
        elif et == "call":
            if ev.get("skipped"):
                marker = f"SKIP ({ev.get('reason','agent-side')})"
            else:
                marker = "ok" if ev.get("ok") else "FAIL"
            print(
                f"[REPLAY {ev.get('i'):>4}/{ev.get('total')}] "
                f"tool={ev.get('tool')} {marker}",
                file=sys.stderr, flush=True,
            )
        elif et == "complete":
            print(
                f"[REPLAY] complete: ok={ev.get('ok')} "
                f"fail={ev.get('fail')} skipped={ev.get('skipped',0)} "
                f"— handing off to agent",
                file=sys.stderr, flush=True,
            )
            return True
        elif et == "msg-replay-begin":
            print(
                f"[ROLLOUT-REPLAY] pushing {ev.get('total')} prior messages "
                f"from rollout {ev.get('source_rollout')}",
                file=sys.stderr, flush=True,
            )
        elif et == "msg-replay-complete":
            print(
                f"[ROLLOUT-REPLAY] complete: logged={ev.get('logged')} "
                f"skipped={ev.get('skipped')} unknown={ev.get('unknown')}",
                file=sys.stderr, flush=True,
            )
        return False

    def _tail_progress() -> None:
        last_pos = 0
        leftover = ""  # carry partial last-line between reads
        seen_complete = False
        # Up to ~20 min total wait time (cold OR backend + long replay).
        idle_deadline = _time.time() + 1200
        while not _stop_tail.is_set():
            try:
                if not progress_path.exists():
                    if _time.time() > idle_deadline:
                        return
                    _time.sleep(0.5)
                    continue
                size = progress_path.stat().st_size
                if size > last_pos:
                    # Reopen fresh each cycle to bust Windows cross-process
                    # file-cache visibility issues.
                    with progress_path.open("r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        chunk = f.read()
                        last_pos = f.tell()
                    data = leftover + chunk
                    lines = data.split("\n")
                    leftover = lines[-1]  # last (possibly partial) line
                    for ln in lines[:-1]:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            ev = json.loads(ln)
                        except Exception:
                            continue
                        if _emit(ev):
                            seen_complete = True
                    if seen_complete:
                        return
                    # Reset idle deadline whenever we see progress.
                    idle_deadline = _time.time() + 1200
                _time.sleep(0.25)
            except Exception as _e:
                print(
                    f"[REPLAY] tail error: {type(_e).__name__}: {_e}",
                    file=sys.stderr, flush=True,
                )
                _time.sleep(0.5)

    _tail_thread = threading.Thread(target=_tail_progress, daemon=True)
    _tail_thread.start()

    try:
        exit_code = asyncio.run(command_run(
            env=state.env_name,
            agent=args.agent,
            model=args.model or state.model,
            split=state.split,
            max_tasks=1,
            max_turns=args.max_turns,
            secrets=secrets,
            output_dir=str(new_out),
            effort=None if args.effort in (None, "none") else args.effort,
            toolset=args.toolset,
        ))
    finally:
        os.environ.pop("OPENREWARD_REPLAY_PATH", None)
        os.environ.pop("OPENREWARD_REPLAY_PROGRESS_FILE", None)
        os.environ.pop("OPENREWARD_REPLAY_ROLLOUT_ID", None)
        _stop_tail.set()

    return exit_code


def _replay(argv: list[str]) -> int:
    """`firehorse replay <results-dir>` — REPLAY ONLY, no LLM.

    Opens a fresh OR session, calls every recorded tool from the trial
    JSONL in order, and prints progress + final cumulative reward.
    Useful to verify that a dead trial's state can be reconstructed
    before committing to a full `firehorse resume`.
    """
    import argparse
    from firehorse.resume import (
        parse_results_dir, summarize, replay_against_fresh_session
    )

    p = argparse.ArgumentParser(
        prog="firehorse replay",
        description="Replay a trial's tool-call sequence against a fresh OR session (no LLM).",
    )
    p.add_argument("results_dir", help="A results/<stamp>_<env>_… directory")
    p.add_argument("--secret", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--print-every", type=int, default=1,
                   help="Print progress every N calls (default 1 = every call)")
    args = p.parse_args(argv)

    rd = Path(args.results_dir).resolve()
    if not rd.is_dir():
        print(f"[replay] not a directory: {rd}", file=sys.stderr)
        return 2

    state = parse_results_dir(rd)
    print(f"[replay] {summarize(state)}", file=sys.stderr)

    secrets: dict = {}
    for s in (args.secret or []):
        if "=" not in s:
            print(f"Invalid --secret format: {s!r}", file=sys.stderr)
            return 1
        k, v = s.split("=", 1)
        secrets[k] = v

    summary = asyncio.run(replay_against_fresh_session(
        state, secrets=secrets or None, print_every=args.print_every,
    ))

    print("==================== REPLAY SUMMARY ====================", file=sys.stderr)
    print(f"  env:              {summary['env_name']}", file=sys.stderr)
    print(f"  task_id:          {summary['task_id']}", file=sys.stderr)
    print(f"  tool calls total: {summary['total_calls']}", file=sys.stderr)
    print(f"  succeeded:        {summary['ok']}", file=sys.stderr)
    print(f"  failed:           {summary['fail']}", file=sys.stderr)
    print(f"  cumulative reward seen during replay: {summary['total_reward_seen']:.2f}", file=sys.stderr)
    if summary["errors"]:
        print(f"  first error: {summary['errors'][0]}", file=sys.stderr)
    return 0 if summary["fail"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    # Subcommands: `firehorse resume <dir>` / `firehorse replay <dir>` — peel off
    # before argparse on the main flag set runs (it doesn't know them).
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "resume":
        return _resume(argv[1:])
    if argv and argv[0] == "replay":
        return _replay(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    # Reject env names without <namespace>/<env> early — otherwise the
    # OR session-server returns a confusing nested aiohttp stack trace
    # ending in `400 X-Deployment must be in the format ...`.
    env_parts = args.env.split("/")
    if len(env_parts) != 2 or not env_parts[0] or not env_parts[1]:
        suggestion = (
            args.env if "/" in args.env else f"GeneralReasoning/{args.env}"
        )
        print(
            f"firehorse: error: --env must be in the format "
            f"<namespace>/<environment>, got {args.env!r}\n"
            f"           e.g. --env {suggestion}",
            file=sys.stderr,
        )
        return 2

    secrets = {}
    for s in (args.secret or []):
        if "=" not in s:
            print(f"Invalid --secret format: {s!r} (expected KEY=VALUE)", file=sys.stderr)
            return 1
        k, v = s.split("=", 1)
        secrets[k] = v

    try:
        exit_code = asyncio.run(command_run(
            env=args.env,
            agent=args.agent,
            model=args.model,
            variant=args.variant,
            n_concurrent=args.n_concurrent,
            split=args.split,
            max_tasks=args.max_tasks,
            skip_tasks=args.skip_tasks,
            plan_mode=args.plan_mode,
            run_name=args.run_name,
            max_turns=args.max_turns,
            provider_url=args.provider_url,
            disable_builtin_tools=args.disable_builtin_tools,
            secrets=secrets,
            output_dir=args.output_dir,
            effort=None if args.effort in (None, "none") else args.effort,
            logging=not args.no_logging,
            use_env_descriptions=args.use_env_descriptions,
            use_all_filesystem_tools=args.use_all_filesystem_tools,
            toolset=args.toolset,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
