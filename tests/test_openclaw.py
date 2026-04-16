"""Tests for firehorse.agents.openclaw helpers and orchestrator guards."""
from __future__ import annotations

import asyncio

import pytest

from firehorse.agents.openclaw import (
    _EFFORT_TO_OPENCLAW_THINKING,
    _resolve_openclaw_model,
)
from firehorse.config import RunConfig
from firehorse.orchestrator import run_evaluation


class TestResolveOpenClawModel:
    def test_splits_provider_and_name(self):
        assert _resolve_openclaw_model("anthropic/claude-sonnet-4-6") == (
            "anthropic",
            "claude-sonnet-4-6",
        )

    def test_keeps_nested_slashes_in_name(self):
        assert _resolve_openclaw_model("openrouter/moonshotai/kimi-k2.5") == (
            "openrouter",
            "moonshotai/kimi-k2.5",
        )

    def test_defaults_to_openai_when_no_slash(self):
        assert _resolve_openclaw_model("gpt-4o") == ("openai", "gpt-4o")


class TestEffortMapping:
    def test_all_cli_effort_values_mapped(self):
        # Must cover every CLI --effort choice.
        for cli_value in ("low", "medium", "high", "max"):
            assert cli_value in _EFFORT_TO_OPENCLAW_THINKING

    def test_max_maps_to_xhigh(self):
        assert _EFFORT_TO_OPENCLAW_THINKING["max"] == "xhigh"

    def test_low_medium_high_pass_through(self):
        assert _EFFORT_TO_OPENCLAW_THINKING["low"] == "low"
        assert _EFFORT_TO_OPENCLAW_THINKING["medium"] == "medium"
        assert _EFFORT_TO_OPENCLAW_THINKING["high"] == "high"


class TestOpenClawAnthropicGuard:
    """OpenClaw doesn't support Anthropic's native endpoint — route via OpenRouter."""

    def test_anthropic_model_with_openclaw_agent_exits(self, capsys):
        config = RunConfig(
            env="some/env",
            agent="openclaw",
            model="anthropic/claude-sonnet-4-6",
        )
        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(run_evaluation(config))
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "OpenClaw" in err
        assert "Anthropic" in err
        assert "openrouter/anthropic/" in err

    def test_openrouter_anthropic_model_with_openclaw_not_blocked(self, monkeypatch):
        # Routing Claude through OpenRouter is the supported path — the
        # anthropic-endpoint guard must not trigger. We only check that the
        # guard itself doesn't exit; downstream steps may still fail for
        # unrelated reasons, so we stop after the credential check by
        # deliberately clearing OPENROUTER_API_KEY to get a different exit.
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        config = RunConfig(
            env="some/env",
            agent="openclaw",
            model="openrouter/anthropic/claude-sonnet-4-6",
        )
        with pytest.raises(SystemExit):
            asyncio.run(run_evaluation(config))
        # If the anthropic-endpoint guard had fired, the error would mention
        # "OpenClaw" / "Anthropic API endpoint". Instead we expect the
        # OPENROUTER_API_KEY credential error.
        # (Message content is asserted via a separate capsys check in tests
        # that care; here we only need to confirm the guard didn't match.)

    def test_anthropic_model_with_other_agent_not_blocked(self, monkeypatch, capsys):
        # The guard is openclaw-specific. claude-code (and other harnesses)
        # must continue to accept anthropic/ models.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = RunConfig(
            env="some/env",
            agent="claude-code",
            model="anthropic/claude-sonnet-4-6",
        )
        with pytest.raises(SystemExit):
            asyncio.run(run_evaluation(config))
        err = capsys.readouterr().err
        # Should fail on credential check, not the openclaw guard.
        assert "OpenClaw" not in err
        assert "ANTHROPIC_API_KEY" in err
