"""Tests for firehorse.agents.codex helper functions."""
from __future__ import annotations

import base64
import json
import os
from unittest import mock

import pytest

from firehorse.agents.codex import (
    _resolve_model_codex,
    _compute_codex_excluded_tools,
    _build_codex_mcp_prompt,
    _extract_mcp_text,
    _log_codex_event_to_rollout,
)


# ---------------------------------------------------------------------------
# _resolve_model_codex
# ---------------------------------------------------------------------------

class TestResolveModelCodex:
    def test_openai_prefix_strips(self):
        name, provider_config = _resolve_model_codex("openai/gpt-5-codex", None)
        assert name == "gpt-5-codex"
        assert provider_config is None

    @mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"})
    def test_openrouter_returns_provider_config(self):
        name, provider_config = _resolve_model_codex("openrouter/openai/gpt-4.1", None)
        assert name == "openai/gpt-4.1"
        assert provider_config == {
            "name": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "env_key": "OPENROUTER_API_KEY",
            "wire_api": "responses",
            "support_namespaces": False,
        }

    def test_openrouter_missing_key_raises(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                _resolve_model_codex("openrouter/openai/gpt-4.1", None)

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_provider_url_returns_provider_config(self):
        name, provider_config = _resolve_model_codex("custom-model", "https://my-proxy.com/v1")
        assert name == "custom-model"
        assert provider_config == {
            "name": "Custom",
            "base_url": "https://my-proxy.com/v1",
            "env_key": "OPENAI_API_KEY",
            "wire_api": "responses",
            "support_namespaces": False,
        }

    def test_no_prefix_no_provider_raises(self):
        with pytest.raises(ValueError, match="requires 'openai/' prefix"):
            _resolve_model_codex("gpt-5-codex", None)


# ---------------------------------------------------------------------------
# _compute_codex_excluded_tools
# ---------------------------------------------------------------------------

class TestComputeCodexExcludedTools:
    def test_bash_present_excludes_filesystem_tools(self):
        result = _compute_codex_excluded_tools(
            ["bash", "read", "write", "edit", "grep", "glob", "answer"],
        )
        assert sorted(result) == ["edit", "glob", "grep", "read", "write"]

    def test_no_bash_excludes_nothing(self):
        result = _compute_codex_excluded_tools(
            ["read", "write", "answer"],
        )
        assert result == []

    def test_use_all_filesystem_tools_keeps_everything(self):
        result = _compute_codex_excluded_tools(
            ["bash", "read", "write", "edit", "grep", "glob"],
            use_all_filesystem_tools=True,
        )
        assert result == []

    def test_always_use_builtin_excluded(self):
        result = _compute_codex_excluded_tools(
            ["bash", "todo_write", "answer"],
        )
        assert "todo_write" in result

    def test_empty_tools(self):
        result = _compute_codex_excluded_tools([])
        assert result == []

    def test_bash_case_insensitive(self):
        result = _compute_codex_excluded_tools(["Bash", "Read", "Write"])
        assert sorted(r.lower() for r in result) == ["read", "write"]

    def test_bash_only_no_filesystem_tools(self):
        """When env only provides bash and no other filesystem tools, nothing extra excluded."""
        result = _compute_codex_excluded_tools(["bash", "answer", "submit"])
        assert result == []


# ---------------------------------------------------------------------------
# _build_codex_mcp_prompt
# ---------------------------------------------------------------------------

class TestBuildCodexMcpPrompt:
    def test_bash_prompt_mentions_mcp_bash(self):
        result = _build_codex_mcp_prompt(["bash", "answer"], excluded=[])
        assert "bash" in result.lower()
        assert "built-in shell" in result.lower()
        assert "EPISODE COMPLETE" in result

    def test_excluded_tools_not_listed(self):
        result = _build_codex_mcp_prompt(
            ["bash", "read", "write", "answer"],
            excluded=["read", "write"],
        )
        assert "answer" in result
        assert "`read`" not in result
        assert "`write`" not in result

    def test_empty_available_returns_empty(self):
        result = _build_codex_mcp_prompt(["bash"], excluded=["bash"])
        assert result == ""

    def test_no_bash_lists_other_tools(self):
        result = _build_codex_mcp_prompt(["answer", "submit"], excluded=[])
        assert "answer" in result
        assert "submit" in result
        assert "Primary tool" not in result  # no bash = no primary tool section


# ---------------------------------------------------------------------------
# _extract_mcp_text
# ---------------------------------------------------------------------------

class TestExtractMcpText:
    def test_structured_content(self):
        data = {"content": [{"type": "text", "text": "hello"}]}
        assert _extract_mcp_text(data) == "hello"

    def test_string_passthrough(self):
        assert _extract_mcp_text("plain text") == "plain text"

    def test_dict_without_content(self):
        data = {"result": "ok"}
        assert _extract_mcp_text(data) == json.dumps(data)


# ---------------------------------------------------------------------------
# _log_codex_event_to_rollout
# ---------------------------------------------------------------------------

class _FakeRollout:
    def __init__(self):
        self.logged: list = []

    def log(self, msg, **kwargs):
        self.logged.append((msg, kwargs))


class TestLogCodexEventToRollout:
    # --- Legacy v0.39 (nested msg) format ---

    def test_agent_message_logged(self):
        rollout = _FakeRollout()
        event = {"msg": {"type": "agent_message", "message": "Hello"}}
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][0].content == "Hello"

    def test_mcp_tool_call_begin(self):
        rollout = _FakeRollout()
        event = {
            "msg": {
                "type": "mcp_tool_call_begin",
                "call_id": "c1",
                "invocation": {"tool": "bash", "arguments": {"command": "ls"}},
            }
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][0].name == "bash"

    def test_mcp_tool_call_end_extracts_reward(self):
        rollout = _FakeRollout()
        event = {
            "msg": {
                "type": "mcp_tool_call_end",
                "call_id": "c1",
                "result": {
                    "Ok": {
                        "content": [{"type": "text", "text": 'Done [OR_REWARD:{"r":1.0,"f":true}]'}],
                        "isError": False,
                    }
                },
            }
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        item, kwargs = rollout.logged[0]
        assert kwargs["reward"] == 1.0
        assert kwargs["is_finished"] is True
        assert "OR_REWARD" not in item.content
        assert item.content == "Done"

    def test_flat_event_format(self):
        """v0.118 flat event format (no nested msg)."""
        rollout = _FakeRollout()
        event = {"type": "agent_message", "message": "Flat format"}
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][0].content == "Flat format"

    def test_exec_command_begin(self):
        rollout = _FakeRollout()
        event = {
            "msg": {
                "type": "exec_command_begin",
                "call_id": "c2",
                "command": ["ls", "-la"],
                "cwd": "/tmp",
            }
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][0].name == "shell"

    # --- Current CLI: item-based format ---

    def test_item_completed_agent_message(self):
        """item.completed with agent_message logs AssistantMessage."""
        rollout = _FakeRollout()
        event = {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "Working on it"},
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        assert rollout.logged[0][0].content == "Working on it"

    def test_item_started_mcp_tool_call(self):
        """item.started with mcp_tool_call logs ToolCall."""
        rollout = _FakeRollout()
        event = {
            "type": "item.started",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "openreward",
                "tool": "bash",
                "arguments": {"command": "ls"},
                "result": None,
                "status": "in_progress",
            },
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        tc = rollout.logged[0][0]
        assert tc.name == "bash"
        assert tc.call_id == "item_1"
        assert json.loads(tc.content) == {"command": "ls"}

    def test_item_completed_mcp_tool_call_with_result(self):
        """item.completed with mcp_tool_call and result logs ToolResult."""
        rollout = _FakeRollout()
        event = {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "openreward",
                "tool": "bash",
                "arguments": {"command": "ls"},
                "result": {
                    "content": [{"type": "text", "text": "file.txt\n"}],
                    "structured_content": None,
                },
                "status": "completed",
            },
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        tr, kwargs = rollout.logged[0]
        assert tr.content == "file.txt\n"
        assert tr.call_id == "item_1"
        assert kwargs["reward"] is None
        assert kwargs["is_finished"] is False

    def test_item_completed_mcp_tool_call_extracts_reward(self):
        """item.completed mcp_tool_call with OR_REWARD tag extracts reward."""
        rollout = _FakeRollout()
        event = {
            "type": "item.completed",
            "item": {
                "id": "item_5",
                "type": "mcp_tool_call",
                "tool": "answer",
                "arguments": {"answer": "42"},
                "result": {
                    "content": [
                        {"type": "text", "text": "Task complete"},
                        {"type": "text", "text": ""},
                        {"type": "text", "text": '\n\n[EPISODE COMPLETE] [OR_REWARD:{"r":1.0,"f":true}]'},
                    ],
                },
                "status": "completed",
            },
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        tr, kwargs = rollout.logged[0]
        assert kwargs["reward"] == 1.0
        assert kwargs["is_finished"] is True
        assert "OR_REWARD" not in tr.content

    def test_turn_completed_ignored(self):
        """turn.completed events should not log anything to rollout."""
        rollout = _FakeRollout()
        event = {
            "type": "turn.completed",
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 0

    def test_thread_started_ignored(self):
        """thread.started events should not log anything to rollout."""
        rollout = _FakeRollout()
        event = {"type": "thread.started", "thread_id": "abc-123"}
        _log_codex_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 0
