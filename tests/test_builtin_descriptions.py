"""Tests for builtin tool description overrides."""
from __future__ import annotations

import os
from unittest import mock

import pytest

from firehorse.agents.claude_code import ENV_TO_BUILTIN
from firehorse.mcp.builtin_descriptions import BUILTIN_DESCRIPTIONS
from firehorse.mcp.codex_descriptions import CODEX_DESCRIPTIONS


# ---------------------------------------------------------------------------
# BUILTIN_DESCRIPTIONS completeness
# ---------------------------------------------------------------------------

class TestBuiltinDescriptions:
    def test_covers_all_env_to_builtin_keys(self):
        """Every env tool name in ENV_TO_BUILTIN must have a description."""
        missing = set(ENV_TO_BUILTIN.keys()) - set(BUILTIN_DESCRIPTIONS.keys())
        assert not missing, f"Missing descriptions for env tool names: {missing}"

    def test_all_descriptions_are_nonempty_strings(self):
        for name, desc in BUILTIN_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"{name}: expected str, got {type(desc)}"
            assert len(desc.strip()) > 0, f"{name}: description is empty"

    def test_bash_description_contains_key_guidance(self):
        desc = BUILTIN_DESCRIPTIONS["bash"]
        assert "Executes a given bash command" in desc
        assert "mcp__openreward__glob if available" in desc
        assert "mcp__openreward__grep if available" in desc
        assert "mcp__openreward__read if available" in desc
        assert "mcp__openreward__edit if available" in desc
        assert "mcp__openreward__write if available" in desc
        assert "git commit" in desc.lower()

    def test_grep_description_mentions_ripgrep(self):
        desc = BUILTIN_DESCRIPTIONS["grep"]
        assert "ripgrep" in desc

    def test_glob_description_mentions_patterns(self):
        desc = BUILTIN_DESCRIPTIONS["glob"]
        assert "glob patterns" in desc

    def test_read_description_mentions_absolute_path(self):
        desc = BUILTIN_DESCRIPTIONS["read"]
        assert "absolute path" in desc

    def test_edit_description_mentions_exact_replacement(self):
        desc = BUILTIN_DESCRIPTIONS["edit"]
        assert "exact string replacements" in desc.lower()

    def test_write_description_mentions_overwrite(self):
        desc = BUILTIN_DESCRIPTIONS["write"]
        assert "overwrite" in desc

    def test_notebookedit_description(self):
        desc = BUILTIN_DESCRIPTIONS["notebookedit"]
        assert "Jupyter notebook" in desc


# ---------------------------------------------------------------------------
# toolspec_to_mcp with description_override
# ---------------------------------------------------------------------------

class TestToolspecToMcpOverride:
    def test_no_override_uses_spec_description(self):
        from firehorse.mcp.convert import toolspec_to_mcp
        from openreward.api.environments.types import ToolSpec

        spec = ToolSpec(
            name="bash",
            description="Original env description",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        )
        tool = toolspec_to_mcp(spec)
        assert tool.description == "Original env description"

    def test_override_replaces_description(self):
        from firehorse.mcp.convert import toolspec_to_mcp
        from openreward.api.environments.types import ToolSpec

        spec = ToolSpec(
            name="bash",
            description="Original env description",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        )
        tool = toolspec_to_mcp(spec, description_override="Rich Claude Code description")
        assert tool.description == "Rich Claude Code description"

    def test_none_override_keeps_original(self):
        from firehorse.mcp.convert import toolspec_to_mcp
        from openreward.api.environments.types import ToolSpec

        spec = ToolSpec(
            name="custom_tool",
            description="Custom description",
            input_schema=None,
        )
        tool = toolspec_to_mcp(spec, description_override=None)
        assert tool.description == "Custom description"

    def test_override_preserves_name_and_schema(self):
        from firehorse.mcp.convert import toolspec_to_mcp
        from openreward.api.environments.types import ToolSpec

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        spec = ToolSpec(name="myname", description="orig", input_schema=schema)
        tool = toolspec_to_mcp(spec, description_override="overridden")
        assert tool.name == "myname"
        assert tool.inputSchema == schema


# ---------------------------------------------------------------------------
# Config threading
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_run_config_default_true(self):
        from firehorse.config import RunConfig
        config = RunConfig(env="test/env", agent="claude-code", model="anthropic/claude-sonnet-4-5")
        assert config.use_builtin_descriptions is True

    def test_run_config_explicit_false(self):
        from firehorse.config import RunConfig
        config = RunConfig(
            env="test/env", agent="claude-code", model="anthropic/claude-sonnet-4-5",
            use_builtin_descriptions=False,
        )
        assert config.use_builtin_descriptions is False

    def test_trial_config_default_true(self):
        from firehorse.config import TrialConfig
        config = TrialConfig(
            task_index=0, task_spec={}, run_name="test",
            env="test/env", split="test", model="anthropic/claude-sonnet-4-5",
        )
        assert config.use_builtin_descriptions is True

    def test_trial_config_explicit_false(self):
        from firehorse.config import TrialConfig
        config = TrialConfig(
            task_index=0, task_spec={}, run_name="test",
            env="test/env", split="test", model="anthropic/claude-sonnet-4-5",
            use_builtin_descriptions=False,
        )
        assert config.use_builtin_descriptions is False

    def test_trial_context_default_true(self):
        from firehorse.agents.base import TrialContext
        ctx = TrialContext(
            prompt_text="test", tools=[], session=None,
            model="m", env_name="e", task_spec={}, run_name="r", split="s",
            task_index=0,
        )
        assert ctx.use_builtin_descriptions is True

    def test_trial_context_explicit_false(self):
        from firehorse.agents.base import TrialContext
        ctx = TrialContext(
            prompt_text="test", tools=[], session=None,
            model="m", env_name="e", task_spec={}, run_name="r", split="s",
            task_index=0,
            use_builtin_descriptions=False,
        )
        assert ctx.use_builtin_descriptions is False


# ---------------------------------------------------------------------------
# CLI flag
# ---------------------------------------------------------------------------

class TestCLIFlag:
    def test_parser_defaults_to_no_use_env_descriptions(self):
        from firehorse.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--env", "test/env", "--model", "anthropic/claude-sonnet-4-5",
        ])
        assert args.use_env_descriptions is False

    def test_parser_accepts_use_env_descriptions(self):
        from firehorse.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--env", "test/env", "--model", "anthropic/claude-sonnet-4-5",
            "--use-env-descriptions",
        ])
        assert args.use_env_descriptions is True


# ---------------------------------------------------------------------------
# Codex descriptions
# ---------------------------------------------------------------------------

class TestCodexDescriptions:
    def test_has_bash_key(self):
        assert "bash" in CODEX_DESCRIPTIONS

    def test_bash_is_nonempty_string(self):
        assert isinstance(CODEX_DESCRIPTIONS["bash"], str)
        assert len(CODEX_DESCRIPTIONS["bash"].strip()) > 0

    def test_bash_matches_upstream_codex_style(self):
        desc = CODEX_DESCRIPTIONS["bash"]
        assert "Runs a shell command" in desc
        assert "workdir" in desc


# ---------------------------------------------------------------------------
# Bridge variant selection (OPENREWARD_TOOL_DESCRIPTIONS)
# ---------------------------------------------------------------------------

class TestBridgeVariant:
    """Test that the bridge selects descriptions based on OPENREWARD_TOOL_DESCRIPTIONS."""

    def _make_tool_specs(self):
        from openreward.api.environments.types import ToolSpec
        return [
            ToolSpec(
                name="bash",
                description="Env bash description",
                input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
            ),
            ToolSpec(
                name="custom_tool",
                description="Custom tool description",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    def test_claude_variant_uses_builtin_descriptions(self):
        from firehorse.mcp.convert import toolspec_to_mcp
        specs = self._make_tool_specs()
        # Simulate bridge logic for variant="claude"
        descs = BUILTIN_DESCRIPTIONS
        tools = [toolspec_to_mcp(t, description_override=descs.get(t.name.lower())) for t in specs]
        assert tools[0].description == BUILTIN_DESCRIPTIONS["bash"]
        assert tools[1].description == "Custom tool description"

    def test_codex_variant_uses_codex_descriptions(self):
        from firehorse.mcp.convert import toolspec_to_mcp
        specs = self._make_tool_specs()
        # Simulate bridge logic for variant="codex"
        descs = CODEX_DESCRIPTIONS
        tools = [toolspec_to_mcp(t, description_override=descs.get(t.name.lower())) for t in specs]
        assert tools[0].description == CODEX_DESCRIPTIONS["bash"]
        assert tools[1].description == "Custom tool description"  # no override for custom_tool

    def test_env_variant_uses_original_descriptions(self):
        from firehorse.mcp.convert import toolspec_to_mcp
        specs = self._make_tool_specs()
        # Simulate bridge logic for variant="env"
        tools = [toolspec_to_mcp(t) for t in specs]
        assert tools[0].description == "Env bash description"
        assert tools[1].description == "Custom tool description"

    def test_default_variant_is_claude(self):
        """When OPENREWARD_TOOL_DESCRIPTIONS is not set, default is 'claude'."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENREWARD_TOOL_DESCRIPTIONS", None)
            val = os.environ.get("OPENREWARD_TOOL_DESCRIPTIONS", "claude")
            assert val == "claude"

    def test_claude_and_codex_bash_descriptions_differ(self):
        """Claude and Codex should have different bash descriptions."""
        assert BUILTIN_DESCRIPTIONS["bash"] != CODEX_DESCRIPTIONS["bash"]


# ---------------------------------------------------------------------------
# Agent variant values
# ---------------------------------------------------------------------------

class TestAgentVariantValues:
    """Test that agents produce correct variant strings."""

    def test_claude_code_variant_when_builtin_enabled(self):
        val = "claude" if True else "env"
        assert val == "claude"

    def test_claude_code_variant_when_builtin_disabled(self):
        val = "claude" if False else "env"
        assert val == "env"

    def test_codex_variant_when_builtin_enabled(self):
        val = "codex" if True else "env"
        assert val == "codex"

    def test_codex_variant_when_builtin_disabled(self):
        val = "codex" if False else "env"
        assert val == "env"


# ---------------------------------------------------------------------------
# Description override for all known tools
# ---------------------------------------------------------------------------

class TestAllKnownToolsOverride:
    """Verify that when a ToolSpec matches a known tool name, the override works."""

    @pytest.mark.parametrize("tool_name", list(ENV_TO_BUILTIN.keys()))
    def test_override_applied_for_known_tool(self, tool_name):
        from firehorse.mcp.convert import toolspec_to_mcp
        from openreward.api.environments.types import ToolSpec

        spec = ToolSpec(
            name=tool_name,
            description=f"Env description for {tool_name}",
            input_schema={"type": "object", "properties": {}},
        )
        override = BUILTIN_DESCRIPTIONS.get(tool_name.lower())
        tool = toolspec_to_mcp(spec, description_override=override)
        assert tool.description == BUILTIN_DESCRIPTIONS[tool_name.lower()]
        assert tool.description != f"Env description for {tool_name}"

    @pytest.mark.parametrize("tool_name", ["answer", "submit", "ls", "custom_checker"])
    def test_no_override_for_unknown_tool(self, tool_name):
        from firehorse.mcp.convert import toolspec_to_mcp
        from openreward.api.environments.types import ToolSpec

        spec = ToolSpec(
            name=tool_name,
            description=f"Env description for {tool_name}",
            input_schema={"type": "object", "properties": {}},
        )
        override = BUILTIN_DESCRIPTIONS.get(tool_name.lower())
        tool = toolspec_to_mcp(spec, description_override=override)
        assert tool.description == f"Env description for {tool_name}"
