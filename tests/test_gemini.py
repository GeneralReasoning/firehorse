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
    _build_gemini_settings,
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
    def test_excludes_generic_names(self):
        result = _compute_gemini_excluded_tools(
            ["bash", "read", "write", "edit", "grep", "glob", "run_shell_command", "read_file"],
        )
        assert sorted(result) == ["bash", "edit", "glob", "grep", "read", "write"]

    def test_keeps_gemini_native_names(self):
        result = _compute_gemini_excluded_tools(
            ["bash", "run_shell_command", "read_file", "write_file", "replace", "grep_search"],
        )
        excluded = set(r.lower() for r in result)
        assert "run_shell_command" not in excluded
        assert "read_file" not in excluded
        assert "replace" not in excluded

    def test_use_all_filesystem_tools_keeps_everything(self):
        result = _compute_gemini_excluded_tools(
            ["bash", "read", "write", "edit"],
            use_all_filesystem_tools=True,
        )
        assert result == []

    def test_todo_write_excluded(self):
        result = _compute_gemini_excluded_tools(
            ["bash", "todo_write", "write_todos"],
        )
        assert "todo_write" in result
        assert "write_todos" not in result

    def test_empty_tools(self):
        result = _compute_gemini_excluded_tools([])
        assert result == []


# ---------------------------------------------------------------------------
# _build_gemini_settings
# ---------------------------------------------------------------------------

class TestBuildGeminiSettings:
    def test_no_tool_restrictions(self):
        """Settings should not restrict tools — Gemini CLI auto-namespaces MCP tools."""
        settings = _build_gemini_settings({"OPENREWARD_API_KEY": "test"})
        assert "tools" not in settings

    def test_mcp_server_configured(self):
        mcp_env = {"OPENREWARD_API_KEY": "key", "OPENREWARD_ENV_NAME": "Foo/Bar"}
        settings = _build_gemini_settings(mcp_env)
        assert "mcpServers" in settings
        assert "openreward" in settings["mcpServers"]
        srv = settings["mcpServers"]["openreward"]
        assert srv["args"] == ["-m", "firehorse.mcp"]
        assert srv["env"] == mcp_env

    def test_max_turns_included(self):
        settings = _build_gemini_settings({}, max_turns=50)
        assert settings["max_turns"] == 50

    def test_max_turns_capped_at_100(self):
        settings = _build_gemini_settings({}, max_turns=200)
        assert settings["max_turns"] == 100

    def test_no_max_turns_omitted(self):
        settings = _build_gemini_settings({}, max_turns=None)
        assert "max_turns" not in settings

    def test_effort_sets_thinking_budget(self):
        settings = _build_gemini_settings({}, effort="high")
        assert settings["thinkingBudget"] == 16000

    def test_effort_low(self):
        settings = _build_gemini_settings({}, effort="low")
        assert settings["thinkingBudget"] == 1024

    def test_effort_max(self):
        settings = _build_gemini_settings({}, effort="max")
        assert settings["thinkingBudget"] == 24576

    def test_no_effort_omits_thinking_budget(self):
        settings = _build_gemini_settings({}, effort=None)
        assert "thinkingBudget" not in settings


# ---------------------------------------------------------------------------
# _build_gemini_mcp_prompt
# ---------------------------------------------------------------------------

class TestBuildGeminiMcpPrompt:
    def test_includes_episode_complete(self):
        result = _build_gemini_mcp_prompt(["bash", "answer"], excluded=[])
        assert "EPISODE COMPLETE" in result

    def test_empty_available_returns_empty(self):
        result = _build_gemini_mcp_prompt(["bash"], excluded=["bash"])
        assert result == ""

    def test_lists_shell_as_primary(self):
        """Prompt should list run_shell_command as primary with openreward_ prefix."""
        result = _build_gemini_mcp_prompt(["run_shell_command", "read_file"], excluded=[])
        assert "`openreward_run_shell_command`" in result

    def test_lists_additional_tools_with_prefix(self):
        """Prompt should list additional MCP tools with openreward_ prefix."""
        result = _build_gemini_mcp_prompt(["run_shell_command", "read_file", "submit_answer"], excluded=[])
        assert "`openreward_read_file`" in result
        assert "`openreward_submit_answer`" in result


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
