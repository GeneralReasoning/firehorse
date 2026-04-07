"""Tests for firehorse.agents.claude_code helper functions.

TODO: Is it possible to add good unit tests for ClaudeCodeAgent.run()
"""
from __future__ import annotations

import base64
import json
import os
from unittest import mock

import pytest

from firehorse.agents.claude_code import (
    _resolve_model, _compute_disallowed_builtins, _log_event_to_rollout,
    _build_tool_mapping_prompt, _build_submission_reminder, _sanitize_prompt,
    _SUBPROCESS_LINE_LIMIT,
)


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:
    def test_anthropic_prefix_strips_and_is_anthropic(self):
        name, env, is_anthropic = _resolve_model("anthropic/claude-sonnet-4-5", None)
        assert name == "claude-sonnet-4-5"
        assert env == {}
        assert is_anthropic is True

    def test_anthropic_prefix_with_nested_slash(self):
        name, env, is_anthropic = _resolve_model("anthropic/claude-opus-4-6", None)
        assert name == "claude-opus-4-6"
        assert is_anthropic is True

    @mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key-123"})
    def test_openrouter_non_anthropic_model(self):
        name, env, is_anthropic = _resolve_model("openrouter/z-ai/glm-5", None)
        assert name == "z-ai/glm-5"
        assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "or-test-key-123"
        assert env["ANTHROPIC_API_KEY"] == "or-test-key-123"
        assert is_anthropic is False

    @mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test-key-123"})
    def test_openrouter_anthropic_model_is_anthropic(self):
        name, env, is_anthropic = _resolve_model("openrouter/anthropic/claude-opus-4-6", None)
        assert name == "anthropic/claude-opus-4-6"
        assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert is_anthropic is True

    @mock.patch.dict(os.environ, {}, clear=False)
    def test_openrouter_missing_api_key_raises(self):
        # Ensure OPENROUTER_API_KEY is not set
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                _resolve_model("openrouter/z-ai/glm-5", None)

    def test_provider_url_sets_anthropic_base_url(self):
        name, env, is_anthropic = _resolve_model(
            "some-model", "https://my-proxy.example.com/v1"
        )
        assert name == "some-model"
        assert env == {"ANTHROPIC_BASE_URL": "https://my-proxy.example.com/v1"}
        assert is_anthropic is False

    def test_no_prefix_no_provider_url_raises(self):
        with pytest.raises(ValueError, match="needs a provider prefix"):
            _resolve_model("z-ai/glm-5", None)

    def test_error_suggests_openrouter_for_openai(self):
        with pytest.raises(ValueError, match="openrouter/openai/gpt-4.1"):
            _resolve_model("openai/gpt-4.1", None)

    def test_error_suggests_openrouter_for_google(self):
        with pytest.raises(ValueError, match="openrouter/google/gemini-2.5-pro"):
            _resolve_model("google/gemini-2.5-pro", None)


# ---------------------------------------------------------------------------
# _compute_disallowed_builtins
# ---------------------------------------------------------------------------

class TestComputeDisallowedBuiltins:
    """Filesystem built-ins are always disabled, regardless of what the env provides."""

    FILESYSTEM = ["Bash", "Edit", "Glob", "Grep", "Read", "Write"]

    def test_always_disables_filesystem_builtins_with_bash(self):
        result = _compute_disallowed_builtins(["bash", "answer", "ls"])
        assert sorted(result) == self.FILESYSTEM

    def test_always_disables_filesystem_builtins_without_bash(self):
        result = _compute_disallowed_builtins(["answer", "ls", "custom_tool"])
        assert sorted(result) == self.FILESYSTEM

    def test_always_disables_filesystem_builtins_empty(self):
        result = _compute_disallowed_builtins([])
        assert sorted(result) == self.FILESYSTEM

    def test_bash_case_insensitive(self):
        result = _compute_disallowed_builtins(["Bash"])
        assert sorted(result) == self.FILESYSTEM

    def test_notebookedit_added_on_top_of_filesystem(self):
        result = _compute_disallowed_builtins(["notebookedit", "answer"])
        assert sorted(result) == sorted(self.FILESYSTEM + ["NotebookEdit"])

    def test_env_filesystem_tools_dont_duplicate(self):
        """Env providing read/write/etc doesn't add duplicates."""
        result = _compute_disallowed_builtins(["read", "write", "grep"])
        assert sorted(result) == self.FILESYSTEM


# ---------------------------------------------------------------------------
# _build_tool_mapping_prompt
# ---------------------------------------------------------------------------

class TestBuildToolMappingPrompt:
    def test_generates_mapping_for_known_tools(self):
        result = _build_tool_mapping_prompt(["bash", "read", "edit"])
        assert "mcp__openreward__bash" in result
        assert "mcp__openreward__read" in result
        assert "mcp__openreward__edit" in result
        assert "instead of `Bash`" in result
        assert "instead of `Read`" in result
        assert "instead of `Edit`" in result

    def test_includes_non_mapped_tools(self):
        result = _build_tool_mapping_prompt(["bash", "submit_answer", "custom_tool"])
        assert "mcp__openreward__bash" in result
        assert "mcp__openreward__submit_answer" in result
        assert "mcp__openreward__custom_tool" in result
        assert "Additional environment tools:" in result

    def test_excludes_always_use_builtin(self):
        result = _build_tool_mapping_prompt(["bash", "todo_write", "todowrite"])
        assert "todo_write" not in result
        assert "todowrite" not in result

    def test_empty_tools_returns_empty(self):
        result = _build_tool_mapping_prompt([])
        assert result == ""

    def test_custom_server_name(self):
        result = _build_tool_mapping_prompt(["bash"], mcp_server_name="myserver")
        assert "mcp__myserver__bash" in result
        assert "openreward" not in result

    def test_other_builtins_mentioned_as_available(self):
        result = _build_tool_mapping_prompt(["bash", "read"])
        assert "Other built-in tools" in result
        assert "remain available" in result

    def test_toolsearch_warning_present(self):
        result = _build_tool_mapping_prompt(["bash", "read"])
        assert "Do NOT use ToolSearch" in result
        assert "cannot find MCP tools" in result

    def test_toolsearch_warning_absent_when_empty(self):
        result = _build_tool_mapping_prompt([])
        assert "ToolSearch" not in result


# ---------------------------------------------------------------------------
# _log_event_to_rollout
# ---------------------------------------------------------------------------

class _FakeRollout:
    """Captures rollout.log() calls for assertions."""
    def __init__(self):
        self.logged: list = []

    def log(self, msg, **kwargs):
        self.logged.append((msg, kwargs))


class TestLogEventToRollout:
    def test_thinking_block_logged_as_reasoning(self):
        rollout = _FakeRollout()
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": "Let me think...", "summary": ""}]},
        }
        _log_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        item = rollout.logged[0][0]
        assert item.content == "Let me think..."

    def test_redacted_thinking_openrouter_decoded(self):
        reasoning = json.dumps({"text": "decoded reasoning", "type": "reasoning.text"})
        b64 = base64.b64encode(reasoning.encode()).decode()
        rollout = _FakeRollout()
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "redacted_thinking", "data": f"openrouter.reasoning:{b64}"}]},
        }
        _log_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        item = rollout.logged[0][0]
        assert item.content == "decoded reasoning"

    def test_redacted_thinking_anthropic_fallback(self):
        rollout = _FakeRollout()
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "redacted_thinking", "data": "encrypted-blob-here"}]},
        }
        _log_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        item = rollout.logged[0][0]
        assert item.content == "[redacted thinking]"

    def test_text_block_logged_as_assistant_message(self):
        rollout = _FakeRollout()
        event = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello world"}]},
        }
        _log_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        item = rollout.logged[0][0]
        assert item.content == "Hello world"

    def test_tool_use_logged_as_tool_call(self):
        rollout = _FakeRollout()
        event = {
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use",
                "name": "bash",
                "input": {"command": "ls"},
                "id": "tc_123",
            }]},
        }
        _log_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        item = rollout.logged[0][0]
        assert item.name == "bash"
        assert item.call_id == "tc_123"

    def test_interleaved_blocks_preserve_order(self):
        """Thinking, text, and tool_use in one message stay in order."""
        reasoning = json.dumps({"text": "step 1", "type": "reasoning.text"})
        b64 = base64.b64encode(reasoning.encode()).decode()
        rollout = _FakeRollout()
        event = {
            "type": "assistant",
            "message": {"content": [
                {"type": "thinking", "thinking": "step 0", "summary": ""},
                {"type": "redacted_thinking", "data": f"openrouter.reasoning:{b64}"},
                {"type": "text", "text": "Here's what I'll do"},
                {"type": "tool_use", "name": "bash", "input": {}, "id": "tc_1"},
            ]},
        }
        _log_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 4
        types = [type(item[0]).__name__ for item in rollout.logged]
        assert types == ["ReasoningItem", "ReasoningItem", "AssistantMessage", "ToolCall"]

    def test_tool_result_extracts_reward(self):
        rollout = _FakeRollout()
        event = {
            "type": "user",
            "message": {"content": [{
                "type": "tool_result",
                "content": 'Result ok [OR_REWARD:{"r":1.0,"f":true}]',
                "tool_use_id": "tc_123",
            }]},
        }
        _log_event_to_rollout(event, rollout)
        assert len(rollout.logged) == 1
        _, kwargs = rollout.logged[0]
        assert kwargs["reward"] == 1.0
        assert kwargs["is_finished"] is True


# ---------------------------------------------------------------------------
# _build_submission_reminder
# ---------------------------------------------------------------------------

class TestBuildSubmissionReminder:
    def test_detects_submit_answer_tool(self):
        result = _build_submission_reminder(["bash", "create_file", "submit_answer", "view"])
        assert "mcp__openreward__submit_answer" in result
        assert "MUST call" in result

    def test_detects_answer_tool(self):
        result = _build_submission_reminder(["bash", "answer", "read"])
        assert "mcp__openreward__answer" in result

    def test_detects_submit_tool(self):
        result = _build_submission_reminder(["bash", "submit", "view"])
        assert "mcp__openreward__submit" in result

    def test_no_submission_tool_returns_empty(self):
        result = _build_submission_reminder(["bash", "create_file", "str_replace", "view"])
        assert result == ""

    def test_custom_server_name(self):
        result = _build_submission_reminder(["submit_answer"], mcp_server_name="myserver")
        assert "mcp__myserver__submit_answer" in result
        assert "openreward" not in result

    def test_multiple_submission_tools(self):
        result = _build_submission_reminder(["submit_answer", "answer"])
        assert "mcp__openreward__submit_answer" in result
        assert "mcp__openreward__answer" in result

    def test_case_insensitive_detection(self):
        result = _build_submission_reminder(["Submit_Answer"])
        assert "mcp__openreward__Submit_Answer" in result
        assert "MUST call" in result

    def test_empty_tools_returns_empty(self):
        result = _build_submission_reminder([])
        assert result == ""


# ---------------------------------------------------------------------------
# _sanitize_prompt
# ---------------------------------------------------------------------------

class TestSanitizePrompt:
    def test_leading_dash_gets_space_prepended(self):
        assert _sanitize_prompt("- You are given...") == " - You are given..."

    def test_no_leading_dash_unchanged(self):
        assert _sanitize_prompt("You are given...") == "You are given..."

    def test_empty_string_unchanged(self):
        assert _sanitize_prompt("") == ""

    def test_leading_double_dash(self):
        assert _sanitize_prompt("--flag something") == " --flag something"

    def test_leading_space_already_present(self):
        assert _sanitize_prompt(" - list item") == " - list item"


# ---------------------------------------------------------------------------
# _SUBPROCESS_LINE_LIMIT
# ---------------------------------------------------------------------------

class TestSubprocessLineLimit:
    def test_value_is_500kb(self):
        assert _SUBPROCESS_LINE_LIMIT == 500 * 1024

    def test_exceeds_asyncio_default(self):
        assert _SUBPROCESS_LINE_LIMIT > 2**16
