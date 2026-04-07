"""Tests for firehorse.agents.codex helper functions.

TODO: Ideally add unit tests here to for CodexAgent.run() with a mocked subprocess
"""
from __future__ import annotations

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
        name, env, proxy = _resolve_model_codex("openai/gpt-5-codex", None)
        assert name == "gpt-5-codex"
        assert env == {}
        assert proxy is None

    @mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key"})
    def test_openrouter_returns_proxy_target(self):
        name, env, proxy = _resolve_model_codex("openrouter/openai/gpt-4.1", None)
        assert name == "openai/gpt-4.1"
        assert env["_PROXY_API_KEY"] == "or-test-key"
        assert proxy == "https://openrouter.ai/api/v1"

    def test_openrouter_missing_key_raises(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                _resolve_model_codex("openrouter/openai/gpt-4.1", None)

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_provider_url_uses_proxy(self):
        name, env, proxy = _resolve_model_codex("custom-model", "https://my-proxy.com/v1")
        assert name == "custom-model"
        assert env["_PROXY_API_KEY"] == "sk-test"
        assert proxy == "https://my-proxy.com/v1"

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
        _, kwargs = rollout.logged[0]
        assert kwargs["reward"] == 1.0
        assert kwargs["is_finished"] is True

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
