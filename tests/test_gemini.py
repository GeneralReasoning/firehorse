"""Tests for firehorse.agents.gemini helper functions."""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from firehorse.agents.gemini import (
    _resolve_model_gemini,
    _compute_gemini_excluded_tools,
    _build_gemini_mcp_prompt,
    _log_gemini_event_to_rollout,
)


# ---------------------------------------------------------------------------
# _resolve_model_gemini
# ---------------------------------------------------------------------------

class TestResolveModelGemini:
    def test_google_prefix_strips(self):
        name = _resolve_model_gemini("google/gemini-2.5-flash")
        assert name == "gemini-2.5-flash"

    def test_google_prefix_pro(self):
        name = _resolve_model_gemini("google/gemini-2.5-pro")
        assert name == "gemini-2.5-pro"

    def test_no_prefix_raises(self):
        with pytest.raises(ValueError, match="requires 'google/' prefix"):
            _resolve_model_gemini("gemini-2.5-flash")

    def test_wrong_prefix_raises(self):
        with pytest.raises(ValueError, match="requires 'google/' prefix"):
            _resolve_model_gemini("anthropic/claude-sonnet-4-5")


# ---------------------------------------------------------------------------
# _compute_gemini_excluded_tools
# ---------------------------------------------------------------------------

class TestComputeGeminiExcludedTools:
    def test_bash_present_excludes_filesystem_tools(self):
        result = _compute_gemini_excluded_tools(
            ["bash", "read", "write", "edit", "grep", "glob", "answer"],
        )
        assert sorted(result) == ["edit", "glob", "grep", "read", "write"]

    def test_no_bash_excludes_nothing(self):
        result = _compute_gemini_excluded_tools(
            ["read", "write", "answer"],
        )
        assert result == []

    def test_use_all_filesystem_tools_keeps_everything(self):
        result = _compute_gemini_excluded_tools(
            ["bash", "read", "write", "edit", "grep", "glob"],
            use_all_filesystem_tools=True,
        )
        assert result == []

    def test_always_use_builtin_excluded(self):
        result = _compute_gemini_excluded_tools(
            ["bash", "todo_write", "answer"],
        )
        assert "todo_write" in result

    def test_empty_tools(self):
        result = _compute_gemini_excluded_tools([])
        assert result == []

    def test_bash_case_insensitive(self):
        result = _compute_gemini_excluded_tools(["Bash", "Read", "Write"])
        assert sorted(r.lower() for r in result) == ["read", "write"]

    def test_bash_only_no_filesystem_tools(self):
        """When env only provides bash and no other filesystem tools, nothing extra excluded."""
        result = _compute_gemini_excluded_tools(["bash", "answer", "submit"])
        assert result == []


# ---------------------------------------------------------------------------
# _build_gemini_mcp_prompt
# ---------------------------------------------------------------------------

class TestBuildGeminiMcpPrompt:
    def test_bash_prompt_mentions_mcp_bash(self):
        result = _build_gemini_mcp_prompt(["bash", "answer"], excluded=[])
        assert "bash" in result.lower()
        assert "built-in shell" in result.lower()
        assert "EPISODE COMPLETE" in result

    def test_excluded_tools_not_listed(self):
        result = _build_gemini_mcp_prompt(
            ["bash", "read", "write", "answer"],
            excluded=["read", "write"],
        )
        assert "answer" in result
        assert "`read`" not in result
        assert "`write`" not in result

    def test_empty_available_returns_empty(self):
        result = _build_gemini_mcp_prompt(["bash"], excluded=["bash"])
        assert result == ""

    def test_no_bash_lists_other_tools(self):
        result = _build_gemini_mcp_prompt(["answer", "submit"], excluded=[])
        assert "answer" in result
        assert "submit" in result
        assert "Primary tool" not in result

    def test_mentions_not_using_builtin(self):
        result = _build_gemini_mcp_prompt(["bash"], excluded=[])
        assert "built-in" in result.lower()


# ---------------------------------------------------------------------------
# _log_gemini_event_to_rollout
# ---------------------------------------------------------------------------

class _FakeRollout:
    def __init__(self):
        self.logged: list = []

    def log(self, msg, **kwargs):
        self.logged.append((msg, kwargs))


class TestLogGeminiEventToRollout:
    def test_assistant_message_accumulated(self):
        rollout = _FakeRollout()
        accumulated = []
        event = {"type": "message", "role": "assistant", "content": "Hello", "delta": True}
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        # Not logged yet — accumulated
        assert len(rollout.logged) == 0
        assert accumulated == ["Hello"]

    def test_tool_use_flushes_accumulated_text(self):
        rollout = _FakeRollout()
        accumulated = ["Hello ", "world"]
        event = {
            "type": "tool_use",
            "tool_name": "bash",
            "tool_id": "bash_123_0",
            "parameters": {"command": "ls"},
        }
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        # Should have flushed AssistantMessage + logged ToolCall
        assert len(rollout.logged) == 2
        assert rollout.logged[0][0].content == "Hello world"
        assert rollout.logged[1][0].name == "bash"
        assert rollout.logged[1][0].call_id == "bash_123_0"
        assert accumulated == []

    def test_tool_use_no_accumulated(self):
        rollout = _FakeRollout()
        accumulated = []
        event = {
            "type": "tool_use",
            "tool_name": "read",
            "tool_id": "read_456_0",
            "parameters": {"path": "/tmp/test.py"},
        }
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][0].name == "read"
        assert json.loads(rollout.logged[0][0].content) == {"path": "/tmp/test.py"}

    def test_tool_result_success(self):
        rollout = _FakeRollout()
        accumulated = []
        event = {
            "type": "tool_result",
            "tool_id": "bash_123_0",
            "status": "success",
        }
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][0].call_id == "bash_123_0"

    def test_tool_result_with_reward(self):
        rollout = _FakeRollout()
        accumulated = []
        event = {
            "type": "tool_result",
            "tool_id": "submit_789_0",
            "status": "success",
            "content": 'Done [OR_REWARD:{"r":1.0,"f":true}]',
        }
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][1]["reward"] == 1.0
        assert rollout.logged[0][1]["is_finished"] is True
        # OR_REWARD marker should be stripped from content
        assert "OR_REWARD" not in rollout.logged[0][0].content

    def test_init_event_ignored(self):
        rollout = _FakeRollout()
        accumulated = []
        event = {"type": "init", "session_id": "abc", "model": "gemini-2.5-flash"}
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        assert len(rollout.logged) == 0

    def test_result_event_ignored(self):
        rollout = _FakeRollout()
        accumulated = []
        event = {"type": "result", "status": "success", "stats": {"input_tokens": 100}}
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        assert len(rollout.logged) == 0

    def test_user_message_ignored(self):
        rollout = _FakeRollout()
        accumulated = []
        event = {"type": "message", "role": "user", "content": "prompt text"}
        _log_gemini_event_to_rollout(event, rollout, accumulated)
        assert len(rollout.logged) == 0
        assert accumulated == []

    def test_multiple_deltas_accumulated(self):
        rollout = _FakeRollout()
        accumulated = []
        # Simulate multiple delta messages
        _log_gemini_event_to_rollout(
            {"type": "message", "role": "assistant", "content": "I'll ", "delta": True},
            rollout, accumulated,
        )
        _log_gemini_event_to_rollout(
            {"type": "message", "role": "assistant", "content": "help you.", "delta": True},
            rollout, accumulated,
        )
        assert accumulated == ["I'll ", "help you."]
        assert len(rollout.logged) == 0

        # Flush via tool_use
        _log_gemini_event_to_rollout(
            {"type": "tool_use", "tool_name": "bash", "tool_id": "t1", "parameters": {}},
            rollout, accumulated,
        )
        assert rollout.logged[0][0].content == "I'll help you."
