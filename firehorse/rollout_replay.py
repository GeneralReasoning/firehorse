"""Replay an original OR rollout's messages into a new rollout.

Used by `firehorse resume` to make the resumed rollout visually mirror
the dead one: the agent's prior assistant / tool_call / tool_result
messages are pushed into the new rollout via the SDK before the fresh
agent starts emitting, so on openreward.ai the resumed rollout's first
~N messages are identical to the original's.

This is harness-agnostic: every firehorse agent (codex, claude_code,
gemini, react, resum) creates a rollout the same way
(`ctx.rollout_client.rollout.create(...)`) and then calls
`main_rollout.log(...)` per event. We tap into that single chokepoint.

The bridge-side `session.call_tool(...)` replay (firehorse/mcp/bridge.py)
is a separate concern: it rebuilds env state inside OR's session so the
agent can keep going. Rollout-message replay (this module) is purely
for visibility on the openreward.ai rollout view.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

from openreward.api.rollouts.serializers.base import (
    AssistantMessage,
    ReasoningItem,
    SystemMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


_ROLLOUT_URL_RE = re.compile(
    r"Rollout:\s*https?://[^\s]+?/rollout/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def extract_rollout_id_from_run_log(results_dir: Path) -> Optional[str]:
    """Scan run.log for `[<agent>] Rollout: https://openreward.ai/rollout/<id>`.

    Every firehorse agent prints this line in the same format right after
    creating its main rollout. We use that to recover the original
    rollout id from a dead trial's results dir.
    """
    log = results_dir / "run.log"
    if not log.exists():
        return None
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _ROLLOUT_URL_RE.search(text)
    return m.group(1) if m else None


def fetch_rollout_messages(rollout_id: str, api_key: str) -> list[dict]:
    """GET /v1/rollouts/<id> on api.openreward.ai and return its messages.

    Uses urllib so we don't add an httpx dependency in agent paths that
    don't otherwise need it.
    """
    base = os.environ.get("OPENREWARD_API_URL")
    if not base:
        base = "https://api.openreward.ai"
    url = f"{base.rstrip('/')}/v1/rollouts/{rollout_id}"
    req = urllib.request.Request(
        url,
        headers={"X-Api-Key": api_key, "User-Agent": "firehorse-resume"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    msgs = data.get("messages") or []
    msgs.sort(key=lambda m: m.get("index", 0))
    return msgs


def _msg_to_upload(m: dict) -> Optional[Any]:
    """Map an OR rollout message dict to the SDK's UploadType variant.

    Returns None for types we don't handle (so the caller can skip them).
    `content` arrives as a string for messages, JSON-string for tool_call
    arguments, and string for tool_result. We pass through verbatim.
    """
    t = m.get("type")
    content = m.get("content")
    # OR API sometimes returns content as a list/dict — coerce to string.
    if not isinstance(content, str) and content is not None:
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)
    if content is None:
        content = ""

    if t == "system_message":
        return SystemMessage(content=content)
    if t == "user_message":
        return UserMessage(content=content)
    if t == "assistant_message":
        return AssistantMessage(content=content)
    if t == "reasoning":
        return ReasoningItem(
            content=content or None,
            content_reference=m.get("contentReference"),
            summary=m.get("summary"),
        )
    if t == "tool_call":
        return ToolCall(
            name=m.get("name") or "unknown",
            content=content if content.strip() else "{}",
            call_id=m.get("callId") or f"call_{m.get('index', '?')}",
        )
    if t == "tool_result":
        return ToolResult(
            content=content,
            call_id=m.get("callId") or f"call_{m.get('index', '?')}",
        )
    return None


def replay_messages_into(
    main_rollout: Any,
    rollout_id: str,
    api_key: str,
    *,
    skip_system_and_user: bool = True,
    progress_fh: Any = None,
) -> dict:
    """Fetch <rollout_id>'s messages and push them into `main_rollout`.

    The agent has already logged its own SystemMessage + UserMessage by
    the time we run (every harness does that right after rollout.create),
    so by default we skip orig[system_message] + orig[user_message] to
    avoid duplicates. The env prompts are byte-identical between runs,
    so the visual match starts at index 0 anyway.

    Returns a small summary dict so the caller can log + write metadata.
    """
    summary = {"fetched": 0, "logged": 0, "skipped": 0, "unknown": 0}
    try:
        messages = fetch_rollout_messages(rollout_id, api_key)
    except Exception as e:
        print(
            f"[rollout-replay] could not fetch orig rollout {rollout_id}: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr, flush=True,
        )
        summary["error"] = f"{type(e).__name__}: {e}"
        return summary

    summary["fetched"] = len(messages)
    print(
        f"[rollout-replay] fetched {len(messages)} prior messages from "
        f"rollout {rollout_id}; pushing into the new rollout so the "
        f"resumed view mirrors the original.",
        file=sys.stderr, flush=True,
    )
    if progress_fh is not None:
        try:
            progress_fh.write(json.dumps({
                "event": "msg-replay-begin",
                "total": len(messages),
                "source_rollout": rollout_id,
            }) + "\n")
        except Exception:
            pass

    saw_system = False
    saw_user = False
    for i, m in enumerate(messages):
        t = m.get("type")
        if skip_system_and_user:
            # Skip only the FIRST system_message and FIRST user_message —
            # if the original had multiple, those are part of the
            # agent's working history and should be preserved.
            if t == "system_message" and not saw_system:
                saw_system = True
                summary["skipped"] += 1
                continue
            if t == "user_message" and not saw_user:
                saw_user = True
                summary["skipped"] += 1
                continue
        upload = _msg_to_upload(m)
        if upload is None:
            summary["unknown"] += 1
            print(
                f"[rollout-replay] {i+1:>4}/{len(messages)} type={t!r} — "
                f"no SDK mapping, skipping",
                file=sys.stderr, flush=True,
            )
            continue
        try:
            main_rollout.log(upload)
            summary["logged"] += 1
        except Exception as e:
            summary["unknown"] += 1
            print(
                f"[rollout-replay] {i+1:>4}/{len(messages)} type={t!r} "
                f"raised {type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
            continue
        if progress_fh is not None and (i + 1) % 25 == 0:
            try:
                progress_fh.write(json.dumps({
                    "event": "msg-replay-progress",
                    "i": i + 1,
                    "total": len(messages),
                    "logged": summary["logged"],
                }) + "\n")
            except Exception:
                pass

    print(
        f"[rollout-replay] DONE — logged={summary['logged']} "
        f"skipped={summary['skipped']} unknown={summary['unknown']} "
        f"of {summary['fetched']} fetched.",
        file=sys.stderr, flush=True,
    )
    if progress_fh is not None:
        try:
            progress_fh.write(json.dumps({
                "event": "msg-replay-complete",
                **summary,
            }) + "\n")
        except Exception:
            pass
    return summary


def resume_metadata() -> dict:
    """Resume-specific metadata fields to merge into rollout.create(metadata=...).

    Empty dict when not in resume mode, so call sites can always splat
    `**resume_metadata()` unconditionally.
    """
    rid = os.environ.get("OPENREWARD_REPLAY_ROLLOUT_ID")
    if not rid:
        return {}
    return {
        "resumed_from_rollout_id": rid,
        "resumed_from_rollout_url": f"https://openreward.ai/rollout/{rid}",
        "resume_kind": "firehorse",
    }


def maybe_replay_into(main_rollout: Any) -> Optional[dict]:
    """Idiom for agent harnesses: call this right after rollout.create.

    No-ops unless `OPENREWARD_REPLAY_ROLLOUT_ID` is set (firehorse CLI
    sets this in `firehorse resume`). Picks up the OR API key from
    `OPENREWARD_API_KEY` in the agent process env.
    """
    rid = os.environ.get("OPENREWARD_REPLAY_ROLLOUT_ID")
    if not rid or main_rollout is None:
        return None
    api_key = os.environ.get("OPENREWARD_API_KEY")
    if not api_key:
        print(
            "[rollout-replay] OPENREWARD_REPLAY_ROLLOUT_ID is set but "
            "OPENREWARD_API_KEY is missing — cannot fetch original "
            "rollout; new rollout will start fresh.",
            file=sys.stderr, flush=True,
        )
        return None
    progress_path = os.environ.get("OPENREWARD_REPLAY_PROGRESS_FILE")
    progress_fh = None
    if progress_path:
        try:
            # Append, since the bridge's tool-call replay already opened+wrote+closed.
            progress_fh = open(progress_path, "a", buffering=1)
        except Exception:
            progress_fh = None
    try:
        return replay_messages_into(
            main_rollout, rid, api_key, progress_fh=progress_fh,
        )
    finally:
        if progress_fh is not None:
            try:
                progress_fh.close()
            except Exception:
                pass
