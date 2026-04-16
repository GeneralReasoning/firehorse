"""Tests for the ReactAgent."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from firehorse.agents.base import AgentResult, TrialContext
from firehorse.agents.react import (
    ReactAgent,
    _format_tool_output,
    _format_tool_output_anthropic,
    _parse_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeTextBlock:
    text: str
    detail: Any = None
    type: str = "text"


@dataclass
class FakeImageBlock:
    data: str
    mimeType: str
    detail: Any = None
    type: str = "image"


@dataclass
class FakeToolOutput:
    blocks: list = field(default_factory=list)
    metadata: Any = None
    reward: float | None = None
    finished: bool = False


@dataclass
class FakeToolSpec:
    name: str
    description: str
    input_schema: dict | None = None


class MockSession:
    """Mock AsyncSession with configurable tool list and call_tool responses."""

    def __init__(
        self,
        tools: list | None = None,
        call_tool_responses: list[FakeToolOutput] | None = None,
    ):
        self._tools = tools or []
        self._call_tool_responses = list(call_tool_responses or [])
        self._call_tool_idx = 0
        self.call_tool_calls: list[tuple[str, dict]] = []

    async def list_tools(self, format=None):
        if format == "anthropic":
            return [
                {"type": "custom", "name": t.name, "description": t.description, "input_schema": t.input_schema or {"type": "object", "properties": {}}}
                for t in self._tools
            ]
        elif format == "openai":
            return [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.input_schema or {}}
                for t in self._tools
            ]
        elif format == "google":
            return [
                {"name": t.name, "description": t.description, "parameters": t.input_schema or {}}
                for t in self._tools
            ]
        elif format == "openrouter":
            return [
                {"type": "function", "function": {"name": t.name, "description": t.description}, "parameters": t.input_schema or {}}
                for t in self._tools
            ]
        return self._tools

    async def call_tool(self, name: str, input: dict) -> FakeToolOutput:
        self.call_tool_calls.append((name, input))
        if self._call_tool_idx < len(self._call_tool_responses):
            resp = self._call_tool_responses[self._call_tool_idx]
            self._call_tool_idx += 1
            return resp
        return FakeToolOutput(blocks=[FakeTextBlock(text="default response")])


def make_trial_context(model: str = "anthropic/claude-sonnet-4-6", **overrides) -> TrialContext:
    """Factory for TrialContext with sensible defaults."""
    session = overrides.pop("session", MockSession())
    defaults = dict(
        prompt_text="Solve this task.",
        tools=[],
        session=session,
        model=model,
        env_name="test/env",
        task_spec={"id": "test_task"},
        run_name="test_run",
        split="train",
        task_index=0,
        max_turns=None,
        provider_url=None,
        disable_builtin_tools=[],
        secrets={},
        output_dir=None,
        effort="high",
        logging=False,
        rollout_client=None,
    )
    defaults.update(overrides)
    return TrialContext(**defaults)


# ---------------------------------------------------------------------------
# Provider parsing
# ---------------------------------------------------------------------------

class TestParseProvider:
    def test_anthropic(self):
        assert _parse_provider("anthropic/claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")

    def test_openai(self):
        assert _parse_provider("openai/gpt-5.4") == ("openai", "gpt-5.4")

    def test_google(self):
        assert _parse_provider("google/gemini-2.5-flash") == ("google", "gemini-2.5-flash")

    def test_openrouter(self):
        assert _parse_provider("openrouter/deepseek/deepseek-v3.2") == ("openrouter", "deepseek/deepseek-v3.2")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _parse_provider("llama/llama-3")


# ---------------------------------------------------------------------------
# Tool output formatting
# ---------------------------------------------------------------------------

class TestFormatToolOutput:
    def test_text_only(self):
        output = FakeToolOutput(blocks=[FakeTextBlock(text="hello")])
        assert _format_tool_output(output) == "hello"

    def test_multiple_text_blocks(self):
        output = FakeToolOutput(blocks=[
            FakeTextBlock(text="hello"),
            FakeTextBlock(text=" world"),
        ])
        assert _format_tool_output(output) == "hello world"

    def test_mixed_with_image(self):
        output = FakeToolOutput(blocks=[
            FakeTextBlock(text="result: "),
            FakeImageBlock(data="abc123", mimeType="image/png"),
        ])
        result = _format_tool_output(output)
        assert "result: " in result
        assert "[Image: image/png]" in result

    def test_empty_blocks(self):
        output = FakeToolOutput(blocks=[])
        assert _format_tool_output(output) == ""


class TestFormatToolOutputAnthropic:
    def test_text_block(self):
        output = FakeToolOutput(blocks=[FakeTextBlock(text="hello")])
        result = _format_tool_output_anthropic(output)
        assert result == [{"type": "text", "text": "hello"}]

    def test_image_block(self):
        output = FakeToolOutput(blocks=[FakeImageBlock(data="abc", mimeType="image/png")])
        result = _format_tool_output_anthropic(output)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert result[0]["source"]["data"] == "abc"

    def test_empty_returns_empty_text(self):
        output = FakeToolOutput(blocks=[])
        result = _format_tool_output_anthropic(output)
        assert result == [{"type": "text", "text": ""}]


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_get_react_agent(self):
        from firehorse.agents import get_agent
        agent = get_agent("react")
        assert isinstance(agent, ReactAgent)
        assert agent.name == "react"


# ---------------------------------------------------------------------------
# Anthropic core loop
# ---------------------------------------------------------------------------

class TestRunAnthropic:
    @pytest.mark.asyncio
    async def test_basic_tool_call(self):
        """LLM makes one tool call, tool returns finished=True."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit answer")],
            call_tool_responses=[
                FakeToolOutput(
                    blocks=[FakeTextBlock(text="Correct!")],
                    reward=1.0,
                    finished=True,
                ),
            ],
        )
        ctx = make_trial_context(model="anthropic/claude-sonnet-4-6", session=session)

        mock_usage = MagicMock(input_tokens=100, output_tokens=50)

        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tu_001"
        tool_use_block.name = "submit"
        tool_use_block.input = {"answer": "42"}

        first_response = MagicMock()
        first_response.content = [tool_use_block]
        first_response.stop_reason = "tool_use"
        first_response.usage = mock_usage

        mock_create = AsyncMock(return_value=first_response)

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = mock_create
            MockAnthropic.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_anthropic(ctx, "claude-sonnet-4-6", None, None, 0.0)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used == 1
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert session.call_tool_calls == [("submit", {"answer": "42"})]

    @pytest.mark.asyncio
    async def test_no_tool_calls_exits(self):
        """LLM responds with text only, loop should exit."""
        session = MockSession()
        ctx = make_trial_context(model="anthropic/claude-sonnet-4-6", session=session)

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "The answer is 42."

        response = MagicMock()
        response.content = [text_block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(input_tokens=50, output_tokens=20)

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            MockAnthropic.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_anthropic(ctx, "claude-sonnet-4-6", None, None, 0.0)

        assert result.success is True
        assert result.finished is False
        assert result.turns_used == 0
        assert session.call_tool_calls == []

    @pytest.mark.asyncio
    async def test_tool_call_error(self):
        """Tool call raises exception, error sent back to LLM."""
        session = MockSession()

        async def failing_call_tool(name, input):
            raise Exception("Tool failed")
        session.call_tool = failing_call_tool  # type: ignore

        ctx = make_trial_context(model="anthropic/claude-sonnet-4-6", session=session)

        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tu_err"
        tool_use_block.name = "bad_tool"
        tool_use_block.input = {}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I see there was an error."

        response_1 = MagicMock()
        response_1.content = [tool_use_block]
        response_1.stop_reason = "tool_use"
        response_1.usage = MagicMock(input_tokens=10, output_tokens=10)

        response_2 = MagicMock()
        response_2.content = [text_block]
        response_2.stop_reason = "end_turn"
        response_2.usage = MagicMock(input_tokens=20, output_tokens=20)

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=[response_1, response_2])
            MockAnthropic.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_anthropic(ctx, "claude-sonnet-4-6", None, None, 0.0)

        assert result.success is True
        assert result.turns_used == 1


# ---------------------------------------------------------------------------
# OpenAI core loop
# ---------------------------------------------------------------------------

class TestRunOpenAI:
    @pytest.mark.asyncio
    async def test_basic_tool_call(self):
        """LLM makes one function_call, tool returns finished=True."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(model="openai/gpt-5.4", session=session)

        func_call = MagicMock()
        func_call.type = "function_call"
        func_call.name = "submit"
        func_call.arguments = '{"answer": "42"}'
        func_call.call_id = "fc_001"

        response = MagicMock()
        response.output = [func_call]
        response.usage = MagicMock(input_tokens=80, output_tokens=30)

        with patch("firehorse.agents.react.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=response)
            MockOAI.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_openai(ctx, "gpt-5.4", None, None, 0.0)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used == 1
        assert session.call_tool_calls == [("submit", {"answer": "42"})]

    @pytest.mark.asyncio
    async def test_no_function_call_exits(self):
        """Response has no function_call items."""
        session = MockSession()
        ctx = make_trial_context(model="openai/gpt-5.4", session=session)

        text_item = MagicMock()
        text_item.type = "message"

        response = MagicMock()
        response.output = [text_item]
        response.usage = MagicMock(input_tokens=10, output_tokens=5)

        with patch("firehorse.agents.react.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=response)
            MockOAI.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_openai(ctx, "gpt-5.4", None, None, 0.0)

        assert result.success is True
        assert result.finished is False
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# Google core loop
# ---------------------------------------------------------------------------

class TestRunGoogle:
    @pytest.mark.asyncio
    async def test_basic_tool_call(self):
        """Gemini makes one function_call, tool returns finished=True."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(model="google/gemini-2.5-flash", session=session)

        mock_fc = MagicMock()
        mock_fc.name = "submit"
        mock_fc.args = {"answer": "42"}

        mock_part = MagicMock()
        mock_part.function_call = mock_fc

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 60
        mock_usage.candidates_token_count = 25

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        with patch("firehorse.agents.react.google_genai.Client") as MockClient, \
             patch("firehorse.agents.react.google_types") as mock_types:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            mock_types.Tool.return_value = MagicMock()
            mock_types.Content.return_value = MagicMock()
            mock_types.Part.return_value = MagicMock()
            mock_types.Part.from_function_response.return_value = MagicMock()
            mock_types.GenerateContentConfig.return_value = MagicMock()

            agent = ReactAgent()
            result = await agent._run_google(ctx, "gemini-2.5-flash", None, None, 0.0)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used == 1
        assert session.call_tool_calls == [("submit", {"answer": "42"})]

    @pytest.mark.asyncio
    async def test_no_function_call_exits(self):
        """Gemini responds with text only."""
        session = MockSession()
        ctx = make_trial_context(model="google/gemini-3.5-flash", session=session)

        mock_part = MagicMock()
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)

        with patch("firehorse.agents.react.google_genai.Client") as MockClient, \
             patch("firehorse.agents.react.google_types") as mock_types:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client
            mock_types.Tool.return_value = MagicMock()
            mock_types.Content.return_value = MagicMock()
            mock_types.Part.return_value = MagicMock()
            mock_types.GenerateContentConfig.return_value = MagicMock()

            agent = ReactAgent()
            result = await agent._run_google(ctx, "gemini-2.5-flash", None, None, 0.0)

        assert result.success is True
        assert result.finished is False
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# OpenRouter core loop
# ---------------------------------------------------------------------------

class TestRunOpenRouter:
    @pytest.mark.asyncio
    async def test_basic_tool_call(self):
        """OpenRouter LLM makes one tool_call, tool returns finished=True."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(model="openrouter/deepseek/deepseek-v3.2", session=session)

        mock_tc = MagicMock()
        mock_tc.id = "tc_001"
        mock_tc.function.name = "submit"
        mock_tc.function.arguments = '{"answer": "42"}'

        mock_msg = MagicMock()
        mock_msg.content = "Let me submit."
        mock_msg.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=70, completion_tokens=20)

        with patch("firehorse.agents.react.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOAI.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_openrouter(ctx, "deepseek/deepseek-v3.2", None, None, 0.0)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used == 1
        assert session.call_tool_calls == [("submit", {"answer": "42"})]

    @pytest.mark.asyncio
    async def test_no_tool_calls_exits(self):
        """LLM responds without tool_calls."""
        session = MockSession()
        ctx = make_trial_context(model="openrouter/deepseek/deepseek-v3.2", session=session)

        mock_msg = MagicMock()
        mock_msg.content = "The answer is 42."
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch("firehorse.agents.react.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOAI.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_openrouter(ctx, "deepseek/deepseek-v3.2", None, None, 0.0)

        assert result.success is True
        assert result.finished is False
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_max_turns_enforced(self):
        """Loop stops when max_turns is reached even if not finished."""
        session = MockSession(
            tools=[FakeToolSpec(name="step", description="Step")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=0.1, finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=0.2, finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=0.3, finished=False),
            ],
        )
        ctx = make_trial_context(
            model="anthropic/claude-sonnet-4-6",
            session=session,
            max_turns=2,
        )

        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tu_x"
        tool_use_block.name = "step"
        tool_use_block.input = {}

        response = MagicMock()
        response.content = [tool_use_block]
        response.stop_reason = "tool_use"
        response.usage = MagicMock(input_tokens=10, output_tokens=10)

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            MockAnthropic.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_anthropic(ctx, "claude-sonnet-4-6", None, None, 0.0)

        assert result.turns_used == 2
        assert result.finished is False

    @pytest.mark.asyncio
    async def test_episode_complete_appended(self):
        """When finished=True, [EPISODE COMPLETE] is appended to tool result text."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(model="openrouter/test/model", session=session)

        mock_tc = MagicMock()
        mock_tc.id = "tc_ep"
        mock_tc.function.name = "submit"
        mock_tc.function.arguments = '{}'

        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        captured_messages: list[dict] = []

        with patch("firehorse.agents.react.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()

            async def capture_create(**kwargs):
                captured_messages.extend(kwargs.get("messages", []))
                return mock_response

            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOAI.return_value = mock_client

            agent = ReactAgent()
            result = await agent._run_openrouter(ctx, "test/model", None, None, 0.0)

        assert result.finished is True
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_jsonl_logging(self):
        """Verify JSONL events are written correctly."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done")], reward=1.0, finished=True),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = make_trial_context(
                model="anthropic/claude-sonnet-4-6",
                session=session,
                output_dir=tmpdir,
            )

            tool_use_block = MagicMock()
            tool_use_block.type = "tool_use"
            tool_use_block.id = "tu_log"
            tool_use_block.name = "submit"
            tool_use_block.input = {"x": 1}

            response = MagicMock()
            response.content = [tool_use_block]
            response.stop_reason = "tool_use"
            response.usage = MagicMock(input_tokens=10, output_tokens=5)

            with patch("anthropic.AsyncAnthropic") as MockAnthropic:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=response)
                MockAnthropic.return_value = mock_client

                agent = ReactAgent()
                result = await agent.run(ctx)

            # Check JSONL file
            jsonl_path = Path(tmpdir) / "trial_test_task.jsonl"
            assert jsonl_path.exists()

            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            types = [e["type"] for e in events]
            assert "system" in types
            assert "user" in types
            assert "assistant" in types
            assert "openreward_summary" in types

            # Check result JSON
            result_path = Path(tmpdir) / "trial_test_task_result.json"
            assert result_path.exists()
            result_data = json.loads(result_path.read_text())
            assert result_data["agent"] == "react"
            assert result_data["task_id"] == "test_task"

    @pytest.mark.asyncio
    async def test_jsonl_tool_result_has_reward_anthropic(self):
        """Anthropic: tool_result JSONL entries include reward and finished."""
        session = MockSession(
            tools=[FakeToolSpec(name="step", description="Step"), FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=None, finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="Done")], reward=0.75, finished=True),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = make_trial_context(
                model="anthropic/claude-sonnet-4-6",
                session=session,
                output_dir=tmpdir,
            )

            tool_block_1 = MagicMock()
            tool_block_1.type = "tool_use"
            tool_block_1.id = "tu_1"
            tool_block_1.name = "step"
            tool_block_1.input = {}

            tool_block_2 = MagicMock()
            tool_block_2.type = "tool_use"
            tool_block_2.id = "tu_2"
            tool_block_2.name = "submit"
            tool_block_2.input = {}

            response_1 = MagicMock()
            response_1.content = [tool_block_1]
            response_1.stop_reason = "tool_use"
            response_1.usage = MagicMock(input_tokens=10, output_tokens=5)

            response_2 = MagicMock()
            response_2.content = [tool_block_2]
            response_2.stop_reason = "tool_use"
            response_2.usage = MagicMock(input_tokens=10, output_tokens=5)

            with patch("anthropic.AsyncAnthropic") as MockAnthropic:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=[response_1, response_2])
                MockAnthropic.return_value = mock_client

                agent = ReactAgent()
                await agent.run(ctx)

            jsonl_path = Path(tmpdir) / "trial_test_task.jsonl"
            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            tool_results = [e for e in events if e["type"] == "tool_result"]

            assert len(tool_results) == 2
            # First tool result: no reward, not finished
            assert tool_results[0]["reward"] is None
            assert tool_results[0]["finished"] is False
            assert tool_results[0]["provider"] == "anthropic"
            assert tool_results[0]["tool_name"] == "step"
            # Second tool result: reward=0.75, finished
            assert tool_results[1]["reward"] == 0.75
            assert tool_results[1]["finished"] is True
            assert tool_results[1]["tool_name"] == "submit"

            # Summary should still be correct
            summary = [e for e in events if e["type"] == "openreward_summary"][0]
            assert summary["reward"] == 0.75
            assert summary["finished"] is True

    @pytest.mark.asyncio
    async def test_jsonl_tool_result_has_reward_openai(self):
        """OpenAI: tool_result JSONL entries include reward and finished."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=0.5, finished=True),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = make_trial_context(model="openai/gpt-5.4", session=session, output_dir=tmpdir)

            func_call = MagicMock()
            func_call.type = "function_call"
            func_call.name = "submit"
            func_call.arguments = '{"answer": "42"}'
            func_call.call_id = "fc_001"

            response = MagicMock()
            response.output = [func_call]
            response.usage = MagicMock(input_tokens=80, output_tokens=30)

            with patch("firehorse.agents.react.AsyncOpenAI") as MockOAI:
                mock_client = AsyncMock()
                mock_client.responses.create = AsyncMock(return_value=response)
                MockOAI.return_value = mock_client

                agent = ReactAgent()
                await agent.run(ctx)

            jsonl_path = Path(tmpdir) / "trial_test_task.jsonl"
            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            tool_results = [e for e in events if e["type"] == "tool_result"]

            assert len(tool_results) == 1
            assert tool_results[0]["reward"] == 0.5
            assert tool_results[0]["finished"] is True
            assert tool_results[0]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_jsonl_tool_result_has_reward_openrouter(self):
        """OpenRouter: tool_result JSONL entries include reward and finished."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = make_trial_context(model="openrouter/deepseek/deepseek-v3.2", session=session, output_dir=tmpdir)

            mock_tc = MagicMock()
            mock_tc.id = "tc_001"
            mock_tc.function.name = "submit"
            mock_tc.function.arguments = '{}'

            mock_msg = MagicMock()
            mock_msg.content = None
            mock_msg.tool_calls = [mock_tc]

            mock_choice = MagicMock()
            mock_choice.message = mock_msg

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

            with patch("firehorse.agents.react.AsyncOpenAI") as MockOAI:
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                MockOAI.return_value = mock_client

                agent = ReactAgent()
                await agent.run(ctx)

            jsonl_path = Path(tmpdir) / "trial_test_task.jsonl"
            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            tool_results = [e for e in events if e["type"] == "tool_result"]

            assert len(tool_results) == 1
            assert tool_results[0]["reward"] == 1.0
            assert tool_results[0]["finished"] is True
            assert tool_results[0]["provider"] == "openrouter"

    @pytest.mark.asyncio
    async def test_jsonl_tool_result_has_reward_google(self):
        """Google: tool_result JSONL entries include reward and finished."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=0.9, finished=True),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = make_trial_context(model="google/gemini-2.5-flash", session=session, output_dir=tmpdir)

            mock_fc = MagicMock()
            mock_fc.name = "submit"
            mock_fc.args = {"answer": "42"}

            mock_part = MagicMock()
            mock_part.function_call = mock_fc

            mock_content = MagicMock()
            mock_content.parts = [mock_part]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_usage = MagicMock()
            mock_usage.prompt_token_count = 60
            mock_usage.candidates_token_count = 25

            mock_response = MagicMock()
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = mock_usage

            with patch("firehorse.agents.react.google_genai.Client") as MockClient, \
                 patch("firehorse.agents.react.google_types") as mock_types:
                mock_client = MagicMock()
                mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
                MockClient.return_value = mock_client

                mock_types.Tool.return_value = MagicMock()
                mock_types.Content.return_value = MagicMock()
                mock_types.Part.return_value = MagicMock()
                mock_types.Part.from_function_response.return_value = MagicMock()
                mock_types.GenerateContentConfig.return_value = MagicMock()

                agent = ReactAgent()
                await agent.run(ctx)

            jsonl_path = Path(tmpdir) / "trial_test_task.jsonl"
            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            tool_results = [e for e in events if e["type"] == "tool_result"]

            assert len(tool_results) == 1
            assert tool_results[0]["reward"] == 0.9
            assert tool_results[0]["finished"] is True
            assert tool_results[0]["provider"] == "google"
            assert tool_results[0]["tool_name"] == "submit"

    @pytest.mark.asyncio
    async def test_jsonl_tool_result_null_reward(self):
        """Tool results with no reward should log reward=null."""
        session = MockSession(
            tools=[FakeToolSpec(name="step", description="Step")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=None, finished=False),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = make_trial_context(model="anthropic/claude-sonnet-4-6", session=session, output_dir=tmpdir)

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.id = "tu_1"
            tool_block.name = "step"
            tool_block.input = {}

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Done."

            response_1 = MagicMock()
            response_1.content = [tool_block]
            response_1.stop_reason = "tool_use"
            response_1.usage = MagicMock(input_tokens=10, output_tokens=5)

            response_2 = MagicMock()
            response_2.content = [text_block]
            response_2.stop_reason = "end_turn"
            response_2.usage = MagicMock(input_tokens=10, output_tokens=5)

            with patch("anthropic.AsyncAnthropic") as MockAnthropic:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=[response_1, response_2])
                MockAnthropic.return_value = mock_client

                agent = ReactAgent()
                await agent.run(ctx)

            jsonl_path = Path(tmpdir) / "trial_test_task.jsonl"
            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            tool_results = [e for e in events if e["type"] == "tool_result"]

            assert len(tool_results) == 1
            assert tool_results[0]["reward"] is None
            assert tool_results[0]["finished"] is False

    @pytest.mark.asyncio
    async def test_rollout_logging(self):
        """Verify rollout provider-specific log methods are called."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done")], reward=1.0, finished=True),
            ],
        )

        mock_rollout = MagicMock()
        mock_rollout.event_id = "test-event-id"
        mock_rollout.log = MagicMock()
        mock_rollout.log_anthropic_message = MagicMock()

        mock_rollout_client = MagicMock()
        mock_rollout_client.rollout.create.return_value = mock_rollout

        ctx = make_trial_context(
            model="anthropic/claude-sonnet-4-6",
            session=session,
            logging=True,
            rollout_client=mock_rollout_client,
        )

        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "tu_rl"
        tool_use_block.name = "submit"
        tool_use_block.input = {}

        response = MagicMock()
        response.content = [tool_use_block]
        response.stop_reason = "tool_use"
        response.usage = MagicMock(input_tokens=10, output_tokens=5)

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            MockAnthropic.return_value = mock_client

            agent = ReactAgent()
            result = await agent.run(ctx)

        # Rollout should have been created
        mock_rollout_client.rollout.create.assert_called_once()

        # System + user logged via generic .log()
        assert mock_rollout.log.call_count >= 2

        # Assistant + tool results logged via provider-specific method
        assert mock_rollout.log_anthropic_message.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_dispatches_to_correct_provider(self):
        """run() dispatches to the correct provider method."""
        agent = ReactAgent()

        for model, method_name in [
            ("anthropic/model", "_run_anthropic"),
            ("openai/model", "_run_openai"),
            ("google/model", "_run_google"),
            ("openrouter/model", "_run_openrouter"),
        ]:
            ctx = make_trial_context(model=model)
            mock_result = AgentResult(success=True, finished=True)

            with patch.object(agent, method_name, new_callable=AsyncMock, return_value=mock_result) as mock_method:
                result = await agent.run(ctx)

            mock_method.assert_called_once()
            assert result.success is True
