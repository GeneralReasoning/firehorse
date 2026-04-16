"""Integration tests for the CodexAgent that call a real Codex CLI + provider.

These tests require:
  - ``codex`` CLI installed (``bun add -g @openai/codex@0.120`` or similar)
  - ``OPENREWARD_API_KEY`` — to list tasks / log rollouts
  - ``OPENROUTER_API_KEY`` — used by the openrouter/ model path

Run with:
    pytest tests/test_codex_integration.py -v -s -m integration

Most tests are marked ``@pytest.mark.integration`` and skipped when the
corresponding binary or API key is missing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


integration = pytest.mark.integration


def _has_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var))


def _codex_version() -> tuple[int, int, int] | None:
    if shutil.which("codex") is None:
        return None
    try:
        out = subprocess.run(
            ["codex", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    from firehorse.agents.codex import _parse_codex_version
    return _parse_codex_version(out)


skip_no_codex = pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="codex CLI not installed",
)
skip_no_openrouter = pytest.mark.skipif(
    not _has_key("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
skip_no_openreward = pytest.mark.skipif(
    not _has_key("OPENREWARD_API_KEY"),
    reason="OPENREWARD_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Version-check unit tests (always run — don't need codex installed)
# ---------------------------------------------------------------------------

class TestCodexVersionGate:
    def test_parse_version_from_codex_cli_output(self):
        from firehorse.agents.codex import _parse_codex_version
        assert _parse_codex_version("codex-cli 0.120.0") == (0, 120, 0)
        assert _parse_codex_version("codex 0.65.0") == (0, 65, 0)

    def test_parse_version_garbage_returns_none(self):
        from firehorse.agents.codex import _parse_codex_version
        assert _parse_codex_version("codex") is None
        assert _parse_codex_version("") is None

    def test_warn_on_unsupported_version(self, capsys):
        from firehorse.agents.codex import _warn_if_unsupported_codex_version
        _warn_if_unsupported_codex_version("codex-cli 0.121.0")
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "0.121.0" in err
        assert "@openai/codex@0.120" in err

    def test_no_warning_on_supported_version(self, capsys):
        from firehorse.agents.codex import _warn_if_unsupported_codex_version
        _warn_if_unsupported_codex_version("codex-cli 0.120.0")
        assert "WARNING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Command-line wiring tests (no network)
# ---------------------------------------------------------------------------

class TestCodexCliWiring:
    """Verify the exact -c flags we send to `codex exec`.

    Regression guard: these flags are the fix for the bug where Codex v0.120+
    talking to OpenRouter needs a custom model_provider (so env_key / bearer
    auth is injected) and --dangerously-bypass-approvals-and-sandbox (so
    exec-mode MCP calls aren't auto-cancelled for lack of an approver).
    """

    @pytest.mark.asyncio
    async def test_openrouter_emits_model_provider_and_bypass_flags(
        self, monkeypatch, tmp_path,
    ):
        import asyncio
        from firehorse.agents import codex as codex_mod

        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        monkeypatch.setenv("OPENREWARD_API_KEY", "")

        captured: dict = {}

        async def fake_exec(*args, **kw):
            captured["cmd"] = list(args)
            captured["env"] = kw.get("env", {})

            class P:
                returncode = 0
                stdout = asyncio.StreamReader()
                stderr = asyncio.StreamReader()
                async def wait(self):
                    return 0
                def kill(self):
                    pass

            p = P()
            p.stdout.feed_eof()
            p.stderr.feed_eof()
            return p

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        from firehorse.agents.base import TrialContext

        class _Sess:
            class task:
                server_name = "MATH"
                environment_name = "math"
                namespace = "GeneralReasoning"

        ctx = TrialContext(
            prompt_text="hi",
            tools=[],
            session=_Sess(),
            model="openrouter/z-ai/glm-5.1",
            env_name="GeneralReasoning/MATH",
            task_spec={"id": "t1"},
            run_name="test",
            split="train",
            variant="math",
            task_index=0,
            effort="high",
            output_dir=str(tmp_path),
            logging=False,
        )

        agent = codex_mod.CodexAgent()
        await agent.run(ctx)

        cmd = captured["cmd"]
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd

        cfg_values = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-c"]
        joined = " \n".join(cfg_values)
        assert 'model_providers.fh.base_url="https://openrouter.ai/api/v1"' in joined
        assert 'model_providers.fh.env_key="OPENROUTER_API_KEY"' in joined
        assert 'model_providers.fh.support_namespaces=false' in joined
        assert 'model_provider="fh"' in joined

    @pytest.mark.asyncio
    async def test_openai_model_omits_model_provider(
        self, monkeypatch, tmp_path,
    ):
        import asyncio
        from firehorse.agents import codex as codex_mod
        from firehorse.agents.base import TrialContext

        captured: dict = {}

        async def fake_exec(*args, **kw):
            captured["cmd"] = list(args)

            class P:
                returncode = 0
                stdout = asyncio.StreamReader()
                stderr = asyncio.StreamReader()
                async def wait(self):
                    return 0
                def kill(self):
                    pass

            p = P()
            p.stdout.feed_eof()
            p.stderr.feed_eof()
            return p

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        class _Sess:
            class task:
                server_name = "MATH"
                environment_name = "math"
                namespace = "GeneralReasoning"

        ctx = TrialContext(
            prompt_text="hi",
            tools=[],
            session=_Sess(),
            model="openai/gpt-5-codex",
            env_name="GeneralReasoning/MATH",
            task_spec={"id": "t1"},
            run_name="test",
            split="train",
            task_index=0,
            effort="high",
            output_dir=str(tmp_path),
            logging=False,
        )

        agent = codex_mod.CodexAgent()
        await agent.run(ctx)

        cfg_values = [captured["cmd"][i + 1] for i, a in enumerate(captured["cmd"]) if a == "-c"]
        joined = " \n".join(cfg_values)
        assert "model_providers.fh" not in joined
        assert 'model_provider="fh"' not in joined
        # sanity: the native OpenAI model name is passed through stripped
        assert "--model" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("--model") + 1] == "gpt-5-codex"


# ---------------------------------------------------------------------------
# Real end-to-end test — the regression the user hit
# ---------------------------------------------------------------------------

@integration
@skip_no_codex
@skip_no_openrouter
@skip_no_openreward
class TestCodexOpenRouterEndToEnd:
    """Run the exact command from the user's bug report against one real task.

    This is the regression guard for:
      firehorse --env GeneralReasoning/MATH --agent codex \\
                --model openrouter/z-ai/glm-5.1 \\
                --split train --variant math
    """

    def test_single_task_completes_with_reward(self):
        version = _codex_version()
        if version is not None and version[:2] > (0, 120):
            pytest.skip(
                f"codex-cli {version[0]}.{version[1]}.x is known-broken with "
                "openrouter (namespace tool shape rejected). Pin to 0.120."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "firehorse.cli",
                    "--env", "GeneralReasoning/MATH",
                    "--agent", "codex",
                    "--model", "openrouter/z-ai/glm-5.1",
                    "--n-concurrent", "1",
                    "--output-dir", tmpdir,
                    "--split", "train",
                    "--variant", "math",
                    "--max-tasks", "1",
                    "--no-logging",
                ],
                capture_output=True, text=True, timeout=300,
            )

            # Surface logs on failure so pytest -v shows why.
            if proc.returncode != 0:
                print("STDOUT:", proc.stdout)
                print("STDERR:", proc.stderr)

            result_path = Path(tmpdir) / "trial_0_result.json"
            assert result_path.exists(), f"trial_0_result.json not written\nstderr:\n{proc.stderr}"

            result = json.loads(result_path.read_text())
            assert result["agent"] == "codex"
            assert result["model"] == "openrouter/z-ai/glm-5.1"
            assert result["finished"] is True, f"task didn't finish: {result}"
            assert result["final_reward"] == 1.0, f"expected reward 1.0, got {result}"
            assert (result.get("tool_calls") or 0) >= 1, "no MCP tool calls recorded"
