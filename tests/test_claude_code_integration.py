"""End-to-end integration test for the claude-code agent.

Runs the real `firehorse` CLI with `--max-tasks 1` against a cheap OpenReward
environment. Requires the `claude` CLI on PATH plus live ANTHROPIC_API_KEY and
OPENREWARD_API_KEY.

Catches the "silent subprocess exit after init" failure mode: when claude CLI
initializes, reports MCP connected, then exits without producing any
`assistant` events (e.g. due to a stale API key), the trajectory contains only
openreward_prompt + system/init + openreward_summary.

Run with:
    pytest tests/test_claude_code_integration.py -v -s

Tests are marked with @pytest.mark.integration and skipped if prerequisites
are missing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

integration = pytest.mark.integration


def _has_key(name: str) -> bool:
    return bool(os.environ.get(name))


skip_no_anthropic = pytest.mark.skipif(
    not _has_key("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
skip_no_openreward = pytest.mark.skipif(
    not _has_key("OPENREWARD_API_KEY"),
    reason="OPENREWARD_API_KEY not set",
)
skip_no_claude_cli = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not on PATH (install: npm install -g @anthropic-ai/claude-code)",
)
skip_no_firehorse_cli = pytest.mark.skipif(
    shutil.which("firehorse") is None,
    reason="firehorse CLI not on PATH (install this repo with `pip install -e .`)",
)


def _parse_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@integration
@skip_no_anthropic
@skip_no_openreward
@skip_no_claude_cli
@skip_no_firehorse_cli
def test_claude_code_end_to_end_produces_healthy_trajectory(tmp_path):
    """Full CLI invocation. Verifies the trajectory records real assistant
    activity, not just init + summary (the silent-exit failure mode).

    Uses GeneralReasoning/MATH (no Docker) + claude-haiku-4-5 + effort=low to
    keep the run <60s and cost pennies.
    """
    cmd = [
        "firehorse",
        "--env", "GeneralReasoning/MATH",
        "--agent", "claude-code",
        "--model", "anthropic/claude-haiku-4-5-20251001",
        "--max-tasks", "1",
        "--max-turns", "10",
        "--effort", "low",
        "--run-name", "integration-claude-code",
        "--output-dir", str(tmp_path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=240,
        cwd=tmp_path,
    )
    # Surface both streams on any anomaly to aid debugging in CI logs.
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
    assert proc.returncode == 0, f"firehorse exited {proc.returncode}"
    assert "[claude-code] Rollout:" in proc.stderr, (
        "missing rollout URL log — did rollout creation fail silently?"
    )

    # Find the single trial trajectory (exclude *_rewards.jsonl sidecar).
    jsonls = [
        p for p in tmp_path.glob("trial_*.jsonl")
        if not p.name.endswith("_rewards.jsonl")
    ]
    assert len(jsonls) == 1, f"expected one trial trajectory, got {jsonls}"
    events = _parse_jsonl(jsonls[0])
    assert events, "trajectory file is empty"

    types = [e.get("type") for e in events]
    subtypes = [(e.get("type"), e.get("subtype")) for e in events]

    assert types[0] == "openreward_prompt", f"first event should be prompt, got {types[0]}"
    assert ("system", "init") in subtypes, f"missing system/init. Types: {types}"
    assert types[-1] == "openreward_summary", f"last event should be summary, got {types[-1]}"

    # THE REGRESSION CHECK: silent-exit runs only produce prompt+init+summary.
    # A healthy run must produce at least one assistant event.
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    assert assistant_events, (
        f"claude produced no assistant events after init — silent subprocess exit. "
        f"Observed types: {types}\n"
        f"STDERR tail:\n{proc.stderr[-2000:]}"
    )

    # And at least one terminal `result` event with cost/token data.
    result_events = [e for e in events if e.get("type") == "result"]
    assert result_events, f"missing terminal result event. Types: {types}"

    # result.json sidecar should report no error and a live rollout URL.
    result_jsons = list(tmp_path.glob("trial_*_result.json"))
    assert result_jsons, "no trial_*_result.json found"
    data = json.loads(result_jsons[0].read_text())
    assert data.get("error") is None, f"trial error surfaced: {data['error']}"
    assert data.get("rollout_url", "").startswith("https://openreward.ai/rollout/"), (
        f"rollout_url missing or malformed: {data.get('rollout_url')!r}"
    )
