"""Tests for the ReSumAgent."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from firehorse.agents.base import AgentResult, TrialContext
from firehorse.agents.resum import ReSumAgent
from firehorse.agents.resum.providers import parse_provider, resolve_api_key
from firehorse.agents.resum.providers.base import LLMResponse, ProviderClient, ToolCallInfo
from firehorse.agents.resum.compaction import (
    CompactionResult,
    compact_conversation,
    micro_compact,
    should_compact_proactively,
    simple_prune,
    COMPACTION_PROMPT,
    MAX_COMPACTIONS,
    MICRO_COMPACT_PLACEHOLDER,
    PROACTIVE_COMPACT_THRESHOLD,
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
        return self._tools

    async def call_tool(self, name: str, input: dict) -> FakeToolOutput:
        self.call_tool_calls.append((name, input))
        if self._call_tool_idx < len(self._call_tool_responses):
            resp = self._call_tool_responses[self._call_tool_idx]
            self._call_tool_idx += 1
            return resp
        return FakeToolOutput(blocks=[FakeTextBlock(text="default response")])


class MockProvider(ProviderClient):
    """Mock provider that returns preconfigured LLMResponses."""

    def __init__(self, responses: list[LLMResponse], context_window_size: int | None = None):
        self._responses = list(responses)
        self._call_idx = 0
        self.call_count = 0
        self.messages_history: list[list[Any]] = []
        self._context_window_size = context_window_size

    @property
    def context_window(self) -> int | None:
        return self._context_window_size

    def format_tools(self, tools):
        return [{"name": t.name} for t in tools]

    def build_initial_messages(self, system_prompt, user_prompt):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def call(self, messages, tools, max_tokens=16384, effort=None):
        self.call_count += 1
        self.messages_history.append(list(messages))
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        return LLMResponse(raw_message=None, text_content="I'm done.")

    def append_assistant(self, messages, response):
        messages.append({"role": "assistant", "response": response})

    def append_tool_result(self, messages, call_id, tool_name, output):
        messages.append({"role": "tool", "call_id": call_id, "content": output})

    def append_user_message(self, messages, content):
        messages.append({"role": "user", "content": content})

    def messages_to_text(self, messages):
        return "\n".join(str(m) for m in messages)

    def rebuild_after_compaction(self, system_prompt, original_prompt, summary):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_prompt},
            {"role": "user", "content": f"[SUMMARY] {summary}"},
        ]

    async def call_for_compaction(self, conversation_text, compaction_prompt, max_tokens):
        return "Compacted summary of the conversation."


def make_trial_context(model: str = "openai/gpt-4o", **overrides) -> TrialContext:
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
        secrets={"openai_api_key": "test-key"},
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
        assert parse_provider("anthropic/claude-opus-4-6", None) == ("anthropic", "claude-opus-4-6")

    def test_openai(self):
        assert parse_provider("openai/gpt-4o", None) == ("openai", "gpt-4o")

    def test_google(self):
        assert parse_provider("google/gemini-2.5-pro", None) == ("google", "gemini-2.5-pro")

    def test_openrouter(self):
        assert parse_provider("openrouter/qwen/qwen3-coder", None) == ("openrouter", "qwen/qwen3-coder")

    def test_unknown_with_provider_url(self):
        assert parse_provider("my-model", "https://api.example.com") == ("openai", "my-model")

    def test_unknown_with_openrouter_url(self):
        assert parse_provider("my-model", "https://openrouter.ai/api/v1") == ("openrouter", "my-model")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot determine provider"):
            parse_provider("unknown-model", None)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_from_secrets(self):
        assert resolve_api_key("openai", {"openai_api_key": "sk-secret"}) == "sk-secret"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key")
        assert resolve_api_key("anthropic", {}) == "sk-env-key"

    def test_openrouter_falls_back_to_openai(self):
        assert resolve_api_key("openrouter", {"openai_api_key": "sk-oai"}) == "sk-oai"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key found"):
            resolve_api_key("openai", {})


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_get_resum_agent(self):
        from firehorse.agents import get_agent
        agent = get_agent("resum")
        assert isinstance(agent, ReSumAgent)
        assert agent.name == "resum"


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

class TestCompaction:
    def test_simple_prune_short(self):
        messages = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
        result = simple_prune(messages, keep_last_n=10)
        assert len(result) == 3

    def test_simple_prune_long(self):
        messages = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg-{i}"} for i in range(20)
        ]
        result = simple_prune(messages, keep_last_n=5)
        assert len(result) == 6  # first + last 5
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "msg-19"

    @pytest.mark.asyncio
    async def test_compact_max_reached(self):
        provider = MockProvider([])
        messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        cr = await compact_conversation(
            provider, messages, "system", "prompt", compaction_count=MAX_COMPACTIONS,
        )
        assert not cr.success
        assert cr.method == "prune"
        assert cr.summary is None
        assert len(cr.new_messages) <= len(messages)

    @pytest.mark.asyncio
    async def test_compact_too_few_messages(self):
        provider = MockProvider([])
        messages = [{"role": "user", "content": "hello"}]
        cr = await compact_conversation(
            provider, messages, "system", "prompt", compaction_count=0,
        )
        assert not cr.success
        assert cr.method == "prune"

    @pytest.mark.asyncio
    async def test_compact_success(self):
        provider = MockProvider([])
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
        cr = await compact_conversation(
            provider, messages, "system", "original prompt", compaction_count=0,
        )
        assert cr.success
        assert cr.method == "summary"
        assert cr.summary is not None
        assert "Compacted summary" in cr.summary
        assert cr.original_message_count == 10
        assert len(cr.new_messages) == 3  # system + original + summary
        assert cr.new_messages[0]["content"] == "system"
        assert cr.new_messages[1]["content"] == "original prompt"
        assert "[SUMMARY]" in cr.new_messages[2]["content"]

    @pytest.mark.asyncio
    async def test_compact_failure_falls_back(self):
        provider = MockProvider([])

        async def failing_compact(*args, **kwargs):
            raise Exception("LLM error")
        provider.call_for_compaction = failing_compact  # type: ignore

        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
        cr = await compact_conversation(
            provider, messages, "system", "prompt", compaction_count=0,
        )
        assert not cr.success
        assert cr.method == "prune"
        assert cr.summary is None
        # Should have fallen back to pruning
        assert len(cr.new_messages) <= len(messages)

    def test_compaction_prompt_exists(self):
        assert len(COMPACTION_PROMPT) > 100
        assert "summary" in COMPACTION_PROMPT.lower()


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

class TestReSumAgentLoop:
    @pytest.mark.asyncio
    async def test_basic_tool_call_finished(self):
        """Agent makes one tool call, tool returns finished=True."""
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
        ctx = make_trial_context(session=session)

        mock_provider = MockProvider([
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_1", name="submit", arguments={"answer": "42"})],
                text_content=None,
                input_tokens=100,
                output_tokens=50,
            ),
        ])

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used == 1
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert session.call_tool_calls == [("submit", {"answer": "42"})]

    @pytest.mark.asyncio
    async def test_no_tool_calls_nudge(self):
        """Agent responds with text only, then nudged, then finishes."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(session=session, max_turns=10)

        mock_provider = MockProvider([
            # First response: text only (no tools)
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[],
                text_content="Let me think about this...",
                input_tokens=50,
                output_tokens=20,
            ),
            # Second response: tool call after nudge
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_1", name="submit", arguments={})],
                text_content=None,
                input_tokens=60,
                output_tokens=30,
            ),
        ])

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.turns_used == 1
        assert result.input_tokens == 110
        assert result.output_tokens == 50
        # Verify nudge message was appended
        assert mock_provider.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_call_error_continues(self):
        """Tool call raises exception, agent continues."""
        session = MockSession(tools=[FakeToolSpec(name="bad_tool", description="Fails")])

        async def failing_call_tool(name, input):
            raise Exception("Tool exploded")
        session.call_tool = failing_call_tool  # type: ignore

        ctx = make_trial_context(session=session, max_turns=5)

        mock_provider = MockProvider([
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_err", name="bad_tool", arguments={})],
                input_tokens=10,
                output_tokens=10,
            ),
            # After error, agent responds with text (no more tools)
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[],
                text_content="I see there was an error.",
                input_tokens=20,
                output_tokens=20,
            ),
            # Then it does nothing
            LLMResponse(raw_message=None, text_content="Done thinking."),
        ])

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.turns_used == 1  # One tool call attempted

    @pytest.mark.asyncio
    async def test_context_overflow_triggers_compaction(self):
        """Context overflow triggers compaction and continues."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(session=session, max_turns=10)

        mock_provider = MockProvider([
            # First call overflows
            LLMResponse(raw_message=None, context_overflow=True),
            # After compaction, tool call succeeds
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_1", name="submit", arguments={})],
                input_tokens=50,
                output_tokens=20,
            ),
        ])

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert mock_provider.call_count == 2

    @pytest.mark.asyncio
    async def test_max_turns_enforced(self):
        """Loop stops when max_turns is reached."""
        session = MockSession(
            tools=[FakeToolSpec(name="step", description="Step")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=0.1, finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=0.2, finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], reward=0.3, finished=False),
            ],
        )
        ctx = make_trial_context(session=session, max_turns=2)

        # Return tool calls indefinitely
        responses = [
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id=f"tc_{i}", name="step", arguments={})],
                input_tokens=10,
                output_tokens=10,
            )
            for i in range(10)
        ]
        mock_provider = MockProvider(responses)

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is False
        # max_turns=2, so only 2 steps of the loop but each has 1 tool call
        assert result.turns_used == 2

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_per_step(self):
        """Agent makes multiple tool calls in a single response."""
        session = MockSession(
            tools=[
                FakeToolSpec(name="read", description="Read"),
                FakeToolSpec(name="submit", description="Submit"),
            ],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="file content")], reward=None, finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(session=session)

        mock_provider = MockProvider([
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[
                    ToolCallInfo(id="tc_1", name="read", arguments={"path": "/tmp/x"}),
                    ToolCallInfo(id="tc_2", name="submit", arguments={"answer": "42"}),
                ],
                input_tokens=100,
                output_tokens=50,
            ),
        ])

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.reward == 1.0
        assert result.turns_used == 2  # Two tool calls
        assert session.call_tool_calls == [
            ("read", {"path": "/tmp/x"}),
            ("submit", {"answer": "42"}),
        ]


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

class TestReSumLogging:
    @pytest.mark.asyncio
    async def test_jsonl_output(self):
        """Verify JSONL events and result JSON are written."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = make_trial_context(session=session, output_dir=tmpdir)

            mock_provider = MockProvider([
                LLMResponse(
                    raw_message={"role": "assistant"},
                    tool_calls=[ToolCallInfo(id="tc_1", name="submit", arguments={"x": 1})],
                    input_tokens=100,
                    output_tokens=50,
                ),
            ])

            with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
                 patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
                 patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
                agent = ReSumAgent()
                result = await agent.run(ctx)

            # Check JSONL file
            jsonl_path = Path(tmpdir) / "trial_test_task.jsonl"
            assert jsonl_path.exists()

            events = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            types = [e["type"] for e in events]
            assert "openreward_prompt" in types
            assert "assistant" in types
            assert "tool_call" in types
            assert "tool_result" in types
            assert "openreward_summary" in types

            # Check result JSON
            result_path = Path(tmpdir) / "trial_test_task_result.json"
            assert result_path.exists()
            result_data = json.loads(result_path.read_text())
            assert result_data["agent"] == "resum"
            assert result_data["task_id"] == "test_task"
            assert result_data["finished"] is True
            assert result_data["final_reward"] == 1.0

    @pytest.mark.asyncio
    async def test_rollout_logging(self):
        """Verify rollout log methods are called."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )

        mock_rollout = MagicMock()
        mock_rollout.event_id = "test-event-id"
        mock_rollout.log = MagicMock()

        mock_rollout_client = MagicMock()
        mock_rollout_client.rollout.create.return_value = mock_rollout

        ctx = make_trial_context(
            session=session,
            logging=True,
            rollout_client=mock_rollout_client,
        )

        mock_provider = MockProvider([
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_1", name="submit", arguments={})],
                text_content="Let me submit.",
                input_tokens=10,
                output_tokens=5,
            ),
        ])

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        mock_rollout_client.rollout.create.assert_called_once()
        # System + User + AssistantMessage + ToolCall + ToolResult = at least 5 calls
        assert mock_rollout.log.call_count >= 4


# ---------------------------------------------------------------------------
# Provider-specific unit tests
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_basic_call(self):
        """OpenAI provider makes a Responses API call."""
        # Mock a function_call output item
        mock_fc = MagicMock()
        mock_fc.type = "function_call"
        mock_fc.call_id = "tc_001"
        mock_fc.name = "submit"
        mock_fc.arguments = '{"answer": "42"}'

        # Mock a message output item with text
        mock_text_block = MagicMock()
        mock_text_block.type = "output_text"
        mock_text_block.text = "Let me submit."

        mock_msg_item = MagicMock()
        mock_msg_item.type = "message"
        mock_msg_item.content = [mock_text_block]

        mock_response = MagicMock()
        mock_response.output = [mock_msg_item, mock_fc]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch("openai.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=mock_response)
            MockOAI.return_value = mock_client

            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            provider._client = mock_client

            result = await provider.call(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "submit"
        assert result.tool_calls[0].arguments == {"answer": "42"}
        assert result.text_content == "Let me submit."
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert not result.context_overflow

    @pytest.mark.asyncio
    async def test_context_overflow(self):
        """OpenAI provider detects context overflow."""
        import openai as openai_mod

        with patch("openai.AsyncOpenAI") as MockOAI:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                side_effect=openai_mod.BadRequestError(
                    message="This model's maximum context length is 128000 tokens",
                    response=MagicMock(status_code=400),
                    body=None,
                )
            )
            MockOAI.return_value = mock_client

            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            provider._client = mock_client

            result = await provider.call(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
            )

        assert result.context_overflow is True

    def test_message_management(self):
        """Test append/rebuild operations (Responses API format)."""
        with patch("openai.AsyncOpenAI"):
            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")

        # build_initial_messages: system prompt stored internally, only user message returned
        messages = provider.build_initial_messages("system prompt", "user prompt")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert provider._system_prompt == "system prompt"

        provider.append_user_message(messages, "continue")
        assert len(messages) == 2
        assert messages[1]["content"] == "continue"

        # rebuild_after_compaction: system prompt stored internally
        rebuilt = provider.rebuild_after_compaction("sys", "prompt", "summary text")
        assert len(rebuilt) == 2
        assert rebuilt[0]["content"] == "prompt"
        assert "summary text" in rebuilt[1]["content"]

    def test_messages_to_text(self):
        """Test message formatting for compaction (Responses API format)."""
        with patch("openai.AsyncOpenAI"):
            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"type": "function_call", "call_id": "tc1", "name": "bash", "arguments": '{"cmd": "ls"}'},
            {"type": "function_call_output", "call_id": "tc1", "output": "result data"},
        ]
        text = provider.messages_to_text(messages)
        assert "USER: Hello" in text
        assert "ASSISTANT: Hi there" in text
        assert "Tool Call" in text
        assert "TOOL OUTPUT" in text


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_basic_call(self):
        """Anthropic provider makes a messages API call."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me help."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_001"
        tool_block.name = "submit"
        tool_block.input = {"answer": "42"}

        mock_response = MagicMock()
        mock_response.content = [text_block, tool_block]
        mock_response.usage = MagicMock(input_tokens=80, output_tokens=40)

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            MockAnthropic.return_value = mock_client

            from firehorse.agents.resum.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(model="claude-opus-4-6", api_key="test-key")
            provider._client = mock_client

            result = await provider.call(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
            )

        assert result.text_content == "Let me help."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "submit"
        assert result.input_tokens == 80

    def test_tool_result_grouping(self):
        """Anthropic groups consecutive tool results into one user message."""
        with patch("anthropic.AsyncAnthropic"):
            from firehorse.agents.resum.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(model="claude-opus-4-6", api_key="test-key")

        messages: list[dict] = []
        provider.append_tool_result(messages, "tc1", "tool_a", "result1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1

        # Second tool result should be appended to same message
        provider.append_tool_result(messages, "tc2", "tool_b", "result2")
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2


class TestOpenRouterProvider:
    def test_inherits_openai(self):
        with patch("openai.AsyncOpenAI") as MockOAI:
            MockOAI.return_value = AsyncMock()
            from firehorse.agents.resum.providers.openrouter_provider import OpenRouterProvider
            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenRouterProvider(model="qwen/qwen3", api_key="test-key")
            assert isinstance(provider, OpenAIProvider)


# ---------------------------------------------------------------------------
# Tool output extraction
# ---------------------------------------------------------------------------

class TestToolOutputExtraction:
    def test_text_only(self):
        from firehorse.agents.resum.agent import _extract_tool_output_text
        output = FakeToolOutput(blocks=[FakeTextBlock(text="hello")])
        assert _extract_tool_output_text(output) == "hello"

    def test_mixed(self):
        from firehorse.agents.resum.agent import _extract_tool_output_text
        output = FakeToolOutput(blocks=[
            FakeTextBlock(text="result: "),
            FakeImageBlock(data="abc", mimeType="image/png"),
        ])
        text = _extract_tool_output_text(output)
        assert "result:" in text
        assert "[Image: image/png]" in text

    def test_empty(self):
        from firehorse.agents.resum.agent import _extract_tool_output_text
        output = FakeToolOutput(blocks=[])
        assert _extract_tool_output_text(output) == ""


# ---------------------------------------------------------------------------
# Micro-compaction
# ---------------------------------------------------------------------------

class TestMicroCompact:
    def test_openai_format(self):
        """Micro-compact clears old tool results in OpenAI message format."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "response": "..."},
            {"role": "tool", "call_id": "tc1", "content": "long result 1"},
            {"role": "assistant", "response": "..."},
            {"role": "tool", "call_id": "tc2", "content": "long result 2"},
            {"role": "assistant", "response": "..."},
            {"role": "tool", "call_id": "tc3", "content": "long result 3"},
            {"role": "assistant", "response": "..."},
            {"role": "tool", "call_id": "tc4", "content": "long result 4"},
            {"role": "assistant", "response": "..."},
            {"role": "tool", "call_id": "tc5", "content": "long result 5"},
            {"role": "assistant", "response": "..."},
            {"role": "tool", "call_id": "tc6", "content": "long result 6"},
            {"role": "assistant", "response": "..."},
            {"role": "tool", "call_id": "tc7", "content": "recent result 7"},
        ]
        result, cleared = micro_compact(messages, "openai", protect_last_n=3)

        assert cleared == 4  # tc1-tc4 cleared, tc5-tc7 protected
        # Check cleared messages have placeholder
        assert result[3]["content"] == MICRO_COMPACT_PLACEHOLDER  # tc1
        assert result[5]["content"] == MICRO_COMPACT_PLACEHOLDER  # tc2
        assert result[7]["content"] == MICRO_COMPACT_PLACEHOLDER  # tc3
        assert result[9]["content"] == MICRO_COMPACT_PLACEHOLDER  # tc4
        # Check protected messages are untouched
        assert result[11]["content"] == "long result 5"
        assert result[13]["content"] == "long result 6"
        assert result[15]["content"] == "recent result 7"

    def test_openai_nothing_to_clear(self):
        """No clearing when all tool results are within protection window."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "call_id": "tc1", "content": "result 1"},
            {"role": "tool", "call_id": "tc2", "content": "result 2"},
        ]
        result, cleared = micro_compact(messages, "openai", protect_last_n=5)
        assert cleared == 0

    def test_openai_already_cleared(self):
        """Don't re-clear already-cleared messages."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "call_id": "tc1", "content": MICRO_COMPACT_PLACEHOLDER},
            {"role": "tool", "call_id": "tc2", "content": "result 2"},
        ]
        result, cleared = micro_compact(messages, "openai", protect_last_n=1)
        assert cleared == 0  # tc1 already cleared, tc2 is protected

    def test_openrouter_uses_openai_format(self):
        """OpenRouter uses the same format as OpenAI."""
        messages = [
            {"role": "tool", "call_id": "tc1", "content": "old result"},
            {"role": "tool", "call_id": "tc2", "content": "new result"},
        ]
        result, cleared = micro_compact(messages, "openrouter", protect_last_n=1)
        assert cleared == 1
        assert result[0]["content"] == MICRO_COMPACT_PLACEHOLDER

    def test_anthropic_format(self):
        """Micro-compact clears old tool results in Anthropic message format."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "tool_a", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "old result 1"},
            ]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu2", "name": "tool_b", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu2", "content": "old result 2"},
            ]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu3", "name": "tool_c", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu3", "content": "recent result"},
            ]},
        ]
        result, cleared = micro_compact(messages, "anthropic", protect_last_n=1)

        assert cleared == 2
        # First two tool_results cleared
        assert result[2]["content"][0]["content"] == MICRO_COMPACT_PLACEHOLDER
        assert result[4]["content"][0]["content"] == MICRO_COMPACT_PLACEHOLDER
        # Last one protected
        assert result[6]["content"][0]["content"] == "recent result"

    def test_unknown_provider_noop(self):
        """Unknown provider returns unchanged messages."""
        messages = [{"role": "tool", "content": "data"}]
        result, cleared = micro_compact(messages, "unknown_provider")
        assert cleared == 0


# ---------------------------------------------------------------------------
# Proactive compaction threshold
# ---------------------------------------------------------------------------

class TestProactiveCompaction:
    def test_should_compact_at_threshold(self):
        assert should_compact_proactively(80000, 100000, threshold=0.80) is True

    def test_should_not_compact_below_threshold(self):
        assert should_compact_proactively(70000, 100000, threshold=0.80) is False

    def test_should_not_compact_unknown_context(self):
        assert should_compact_proactively(80000, None) is False

    def test_should_not_compact_unknown_tokens(self):
        assert should_compact_proactively(None, 100000) is False

    def test_exact_threshold(self):
        assert should_compact_proactively(80000, 100000, threshold=0.80) is True

    def test_default_threshold(self):
        assert should_compact_proactively(160001, 200000) is True
        assert should_compact_proactively(159999, 200000) is False


# ---------------------------------------------------------------------------
# Context window lookup
# ---------------------------------------------------------------------------

class TestContextWindowLookup:
    def test_openai_known_model(self):
        with patch("openai.AsyncOpenAI"):
            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            assert provider.context_window == 128000

    def test_openai_unknown_model(self):
        with patch("openai.AsyncOpenAI"):
            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model="gpt-99", api_key="test-key")
            assert provider.context_window is None

    def test_openai_user_override(self):
        with patch("openai.AsyncOpenAI"):
            from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model="gpt-4o", api_key="test-key", context_window=500000)
            assert provider.context_window == 500000  # Override beats lookup

    def test_anthropic_known_model(self):
        with patch("anthropic.AsyncAnthropic"):
            from firehorse.agents.resum.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(model="claude-opus-4-6", api_key="test-key")
            assert provider.context_window == 1000000

    def test_openrouter_known_model(self):
        with patch("openai.AsyncOpenAI"):
            from firehorse.agents.resum.providers.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(model="deepseek/deepseek-v3.2", api_key="test-key")
            assert provider.context_window == 163840

    def test_openrouter_user_override(self):
        with patch("openai.AsyncOpenAI"):
            from firehorse.agents.resum.providers.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(model="custom/model", api_key="test-key", context_window=300000)
            assert provider.context_window == 300000


# ---------------------------------------------------------------------------
# Agent loop: micro-compaction and proactive compaction
# ---------------------------------------------------------------------------

class TestReSumCompactionIntegration:
    @pytest.mark.asyncio
    async def test_micro_compact_before_full_compact(self):
        """Agent tries micro-compaction first on overflow, then full compaction."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(session=session, max_turns=10)

        mock_provider = MockProvider([
            # First call overflows — micro-compaction will find nothing since
            # MockProvider uses minimal messages, so it goes straight to full compaction
            LLMResponse(raw_message=None, context_overflow=True),
            # After compaction, succeeds
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_1", name="submit", arguments={})],
                input_tokens=50,
                output_tokens=20,
            ),
        ])

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True

    @pytest.mark.asyncio
    async def test_micro_compact_with_tool_results(self):
        """Agent micro-compacts old tool results on overflow."""
        session = MockSession(
            tools=[FakeToolSpec(name="step", description="Step")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(session=session, max_turns=20)

        responses = []
        for i in range(6):
            responses.append(
                LLMResponse(
                    raw_message={"role": "assistant"},
                    tool_calls=[ToolCallInfo(id=f"tc_{i}", name="step", arguments={})],
                    input_tokens=50,
                    output_tokens=20,
                )
            )
        # After 6 tool calls, overflow
        responses.append(LLMResponse(raw_message=None, context_overflow=True))
        # After micro-compaction, succeed with final tool call
        responses.append(
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_final", name="step", arguments={})],
                input_tokens=50,
                output_tokens=20,
            )
        )

        mock_provider = MockProvider(responses)

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True

    @pytest.mark.asyncio
    async def test_proactive_compaction_triggers(self):
        """Agent proactively compacts when input_tokens exceeds 80% of context window."""
        session = MockSession(
            tools=[FakeToolSpec(name="step", description="Step")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="ok")], finished=False),
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(session=session, max_turns=10)

        mock_provider = MockProvider(
            [
                # First call: input tokens at 85% of 128K (gpt-4o) — should trigger proactive compaction
                LLMResponse(
                    raw_message={"role": "assistant"},
                    tool_calls=[ToolCallInfo(id="tc_1", name="step", arguments={})],
                    input_tokens=108800,  # 85% of 128000
                    output_tokens=20,
                ),
                # After proactive compaction, finish
                LLMResponse(
                    raw_message={"role": "assistant"},
                    tool_calls=[ToolCallInfo(id="tc_2", name="step", arguments={})],
                    input_tokens=5000,
                    output_tokens=20,
                ),
            ],
            context_window_size=128000,
        )

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert result.turns_used >= 2

    @pytest.mark.asyncio
    async def test_proactive_compaction_skipped_when_no_context_window(self):
        """Proactive compaction is skipped when context_window is unknown."""
        session = MockSession(
            tools=[FakeToolSpec(name="submit", description="Submit")],
            call_tool_responses=[
                FakeToolOutput(blocks=[FakeTextBlock(text="Done!")], reward=1.0, finished=True),
            ],
        )
        ctx = make_trial_context(session=session, max_turns=10)

        mock_provider = MockProvider([
            LLMResponse(
                raw_message={"role": "assistant"},
                tool_calls=[ToolCallInfo(id="tc_1", name="submit", arguments={})],
                input_tokens=999999,  # Very high, but no context_window to compare against
                output_tokens=20,
            ),
        ])
        # context_window defaults to None on MockProvider (no override)

        with patch("firehorse.agents.resum.agent.get_provider", return_value=mock_provider), \
             patch("firehorse.agents.resum.agent.parse_provider", return_value=("openai", "gpt-4o")), \
             patch("firehorse.agents.resum.agent.resolve_api_key", return_value="test-key"):
            agent = ReSumAgent()
            result = await agent.run(ctx)

        assert result.success is True
        assert result.finished is True
        assert mock_provider.call_count == 1  # No extra compaction call
