"""Integration tests for the ReSumAgent that call real provider APIs.

These tests require actual API keys set as environment variables:
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - GOOGLE_API_KEY or GEMINI_API_KEY
  - OPENROUTER_API_KEY

Run with:
    pytest tests/test_resum_integration.py -v -s

Tests are marked with @pytest.mark.integration and skipped if the
corresponding API key is not available.
"""
from __future__ import annotations

import os
import tempfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from firehorse.agents.base import TrialContext
from firehorse.agents.resum import ReSumAgent

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

integration = pytest.mark.integration


def _has_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var))


skip_no_openai = pytest.mark.skipif(
    not _has_key("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
skip_no_anthropic = pytest.mark.skipif(
    not _has_key("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
skip_no_google = pytest.mark.skipif(
    not (_has_key("GOOGLE_API_KEY") or _has_key("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY/GEMINI_API_KEY not set",
)
skip_no_openrouter = pytest.mark.skipif(
    not _has_key("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Fake environment
# ---------------------------------------------------------------------------

@dataclass
class FakeTextBlock:
    text: str
    detail: Any = None
    type: str = "text"


@dataclass
class FakeToolSpec:
    name: str
    description: str
    input_schema: dict | None = None


@dataclass
class FakeToolOutput:
    blocks: list = field(default_factory=list)
    metadata: Any = None
    reward: float | None = None
    finished: bool = False


class SimpleCalculatorSession:
    """A mock session that provides a simple calculator tool.

    The tool accepts an 'expression' string and returns eval(expression).
    After the first correct call, it marks finished=True with reward=1.0.
    """

    def __init__(self):
        self._tools = [
            FakeToolSpec(
                name="calculate",
                description="Evaluate a simple math expression. Returns the numeric result.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A simple math expression like '2 + 3' or '10 * 5'",
                        },
                    },
                    "required": ["expression"],
                },
            ),
        ]
        self.call_tool_calls: list[tuple[str, dict]] = []

    async def list_tools(self, format=None):
        return self._tools

    async def call_tool(self, name: str, input: dict) -> FakeToolOutput:
        self.call_tool_calls.append((name, input))
        if name == "calculate":
            expr = input.get("expression", "")
            try:
                result = eval(expr)  # safe for simple math
                return FakeToolOutput(
                    blocks=[FakeTextBlock(text=str(result))],
                    reward=1.0,
                    finished=True,
                )
            except Exception as e:
                return FakeToolOutput(
                    blocks=[FakeTextBlock(text=f"Error: {e}")],
                    reward=0.0,
                    finished=False,
                )
        return FakeToolOutput(
            blocks=[FakeTextBlock(text=f"Unknown tool: {name}")],
            finished=False,
        )


class MultiStepSession:
    """A mock session with two tools: read_file and submit_answer.

    read_file returns a fixed answer, submit_answer checks if it matches.
    Tests the agent's ability to chain tool calls.
    """

    def __init__(self):
        self._secret_answer = "42"
        self._tools = [
            FakeToolSpec(
                name="read_file",
                description="Read the contents of a file. The file contains the answer to submit.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read",
                        },
                    },
                    "required": ["path"],
                },
            ),
            FakeToolSpec(
                name="submit_answer",
                description="Submit your final answer. Must match the content of the file.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to submit",
                        },
                    },
                    "required": ["answer"],
                },
            ),
        ]
        self.call_tool_calls: list[tuple[str, dict]] = []

    async def list_tools(self, format=None):
        return self._tools

    async def call_tool(self, name: str, input: dict) -> FakeToolOutput:
        self.call_tool_calls.append((name, input))
        if name == "read_file":
            return FakeToolOutput(
                blocks=[FakeTextBlock(text=f"The answer is: {self._secret_answer}")],
                reward=None,
                finished=False,
            )
        elif name == "submit_answer":
            answer = input.get("answer", "")
            if answer.strip() == self._secret_answer:
                return FakeToolOutput(
                    blocks=[FakeTextBlock(text="Correct!")],
                    reward=1.0,
                    finished=True,
                )
            else:
                return FakeToolOutput(
                    blocks=[FakeTextBlock(text=f"Wrong answer: {answer}")],
                    reward=0.0,
                    finished=True,
                )
        return FakeToolOutput(blocks=[FakeTextBlock(text="Unknown tool")])


def _make_ctx(model: str, session: Any, **overrides) -> TrialContext:
    secrets = {}
    if "openai" in model:
        secrets["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    elif "anthropic" in model:
        secrets["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    elif "google" in model:
        secrets["google_api_key"] = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    elif "openrouter" in model:
        secrets["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY", "")

    # Populate tools from the session (mirroring what trial.py does)
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        tools = session._tools
    else:
        tools = asyncio.run(session.list_tools())

    defaults = dict(
        prompt_text="You have a calculator tool. Please calculate: 2 + 3",
        tools=tools,
        session=session,
        model=model,
        env_name="test/calculator",
        task_spec={"id": "integration_test"},
        run_name="integration_test",
        split="test",
        max_turns=10,
        provider_url=None,
        disable_builtin_tools=[],
        secrets=secrets,
        output_dir=None,
        effort="high",
        logging=False,
        rollout_client=None,
    )
    defaults.update(overrides)
    return TrialContext(**defaults)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@integration
class TestOpenAIIntegration:
    @skip_no_openai
    @pytest.mark.asyncio
    async def test_simple_calculator(self):
        """OpenAI provider can call a tool and finish."""
        session = SimpleCalculatorSession()
        ctx = _make_ctx("openai/gpt-4o-mini", session)

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used >= 1
        assert result.input_tokens is not None and result.input_tokens > 0
        assert len(session.call_tool_calls) >= 1
        assert session.call_tool_calls[0][0] == "calculate"

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_multi_step(self):
        """OpenAI provider chains read_file then submit_answer."""
        session = MultiStepSession()
        ctx = _make_ctx(
            "openai/gpt-4o-mini",
            session,
            prompt_text="Read the file at /data/answer.txt, then submit the answer you find.",
        )

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        tool_names = [name for name, _ in session.call_tool_calls]
        assert "read_file" in tool_names
        assert "submit_answer" in tool_names

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_jsonl_logging(self):
        """Verify JSONL output is written during integration test."""
        session = SimpleCalculatorSession()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _make_ctx("openai/gpt-4o-mini", session, output_dir=tmpdir)

            agent = ReSumAgent()
            result = await agent.run(ctx)

            jsonl_path = Path(tmpdir) / "trial_integration_test.jsonl"
            assert jsonl_path.exists()

            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            types = [e["type"] for e in events]
            assert "openreward_prompt" in types
            assert "assistant" in types
            assert "tool_call" in types
            assert "openreward_summary" in types

            result_path = Path(tmpdir) / "trial_integration_test_result.json"
            assert result_path.exists()


@integration
class TestAnthropicIntegration:
    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_simple_calculator(self):
        """Anthropic provider can call a tool and finish."""
        session = SimpleCalculatorSession()
        ctx = _make_ctx("anthropic/claude-haiku-4-5-20251001", session)

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used >= 1
        assert len(session.call_tool_calls) >= 1

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_multi_step(self):
        """Anthropic provider chains read_file then submit_answer."""
        session = MultiStepSession()
        ctx = _make_ctx(
            "anthropic/claude-haiku-4-5-20251001",
            session,
            prompt_text="Read the file at /data/answer.txt, then submit the answer you find.",
        )

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        tool_names = [name for name, _ in session.call_tool_calls]
        assert "read_file" in tool_names
        assert "submit_answer" in tool_names


@integration
class TestGoogleIntegration:
    @skip_no_google
    @pytest.mark.asyncio
    async def test_simple_calculator(self):
        """Google Gemini provider can call a tool and finish."""
        session = SimpleCalculatorSession()
        ctx = _make_ctx("google/gemini-2.0-flash", session)

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used >= 1

    @skip_no_google
    @pytest.mark.asyncio
    async def test_multi_step(self):
        """Google Gemini provider chains read_file then submit_answer."""
        session = MultiStepSession()
        ctx = _make_ctx(
            "google/gemini-2.0-flash",
            session,
            prompt_text="Read the file at /data/answer.txt, then submit the answer you find.",
        )

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True


@integration
class TestOpenRouterIntegration:
    @skip_no_openrouter
    @pytest.mark.asyncio
    async def test_simple_calculator(self):
        """OpenRouter provider can call a tool and finish."""
        session = SimpleCalculatorSession()
        ctx = _make_ctx("openrouter/openai/gpt-4o-mini", session)

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used >= 1

    @skip_no_openrouter
    @pytest.mark.asyncio
    async def test_multi_step(self):
        """OpenRouter provider chains read_file then submit_answer."""
        session = MultiStepSession()
        ctx = _make_ctx(
            "openrouter/openai/gpt-4o-mini",
            session,
            prompt_text="Read the file at /data/answer.txt, then submit the answer you find.",
        )

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True


# ---------------------------------------------------------------------------
# Cross-provider consistency
# ---------------------------------------------------------------------------

@integration
class TestCrossProvider:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        pytest.param("openai/gpt-4o-mini", marks=skip_no_openai),
        pytest.param("anthropic/claude-haiku-4-5-20251001", marks=skip_no_anthropic),
        pytest.param("google/gemini-2.0-flash", marks=skip_no_google),
        pytest.param("openrouter/openai/gpt-4o-mini", marks=skip_no_openrouter),
    ])
    async def test_calculator_all_providers(self, model: str):
        """All providers should be able to solve a simple calculator task."""
        session = SimpleCalculatorSession()

        # Build secrets based on provider
        secrets = {}
        if model.startswith("openai/"):
            secrets["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
        elif model.startswith("anthropic/"):
            secrets["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        elif model.startswith("google/"):
            secrets["google_api_key"] = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        elif model.startswith("openrouter/"):
            secrets["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY", "")

        ctx = TrialContext(
            prompt_text="You have a calculator tool. Please calculate: 7 * 8",
            tools=session._tools,
            session=session,
            model=model,
            env_name="test/calculator",
            task_spec={"id": f"cross_provider_{model.split('/')[0]}"},
            run_name="cross_provider_test",
            split="test",
            max_turns=10,
            secrets=secrets,
            logging=False,
        )

        agent = ReSumAgent()
        result = await agent.run(ctx)

        assert result.success is True, f"Provider {model} failed: {result.error}"
        assert result.finished is True, f"Provider {model} did not finish"
        assert result.reward == 1.0, f"Provider {model} got reward {result.reward}"
        assert result.turns_used >= 1
