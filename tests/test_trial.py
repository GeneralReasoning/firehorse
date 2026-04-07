"""Tests for firehorse.trial retry logic."""
from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from firehorse.agents.base import AgentResult
from firehorse.trial import _is_mcp_failure, _MCP_RETRY_MAX, _MCP_RETRY_BASE_DELAY, run_trial
from firehorse.config import TrialConfig


# ---------------------------------------------------------------------------
# _is_mcp_failure
# ---------------------------------------------------------------------------

class TestIsMcpFailure:
    def test_mcp_failure_detected(self):
        result = AgentResult(
            success=False,
            error="MCP server 'openreward' failed to connect — no environment tools available",
        )
        assert _is_mcp_failure(result) is True

    def test_non_mcp_error_not_detected(self):
        result = AgentResult(
            success=False,
            error="No result file produced. Exit code: 1. stdout_lines: 0",
        )
        assert _is_mcp_failure(result) is False

    def test_success_result_not_detected(self):
        result = AgentResult(success=True)
        assert _is_mcp_failure(result) is False

    def test_none_error_not_detected(self):
        result = AgentResult(success=False, error=None)
        assert _is_mcp_failure(result) is False


# ---------------------------------------------------------------------------
# _MCP_RETRY constants
# ---------------------------------------------------------------------------

class TestMcpRetryConstants:
    def test_max_retries(self):
        assert _MCP_RETRY_MAX == 8 # fixed the test

    def test_base_delay(self):
        assert _MCP_RETRY_BASE_DELAY == 2.0


# ---------------------------------------------------------------------------
# run_trial retry behaviour
# ---------------------------------------------------------------------------

def _make_trial_config(**overrides) -> TrialConfig:
    defaults = dict(
        task_index=0,
        task_spec={"id": "test-task"},
        run_name="test-run",
        env="test/env",
        split="test",
        model="anthropic/claude-sonnet-4-5",
        logging=False,
    )
    defaults.update(overrides)
    return TrialConfig(**defaults)


def _make_mock_env():
    """Build a mock AsyncEnvironment whose session yields mock prompt/tools."""
    mock_block = mock.MagicMock()
    mock_block.text = "Do the task."

    mock_session = mock.AsyncMock()
    mock_session.get_prompt = mock.AsyncMock(return_value=[mock_block])
    mock_session.list_tools = mock.AsyncMock(return_value=[])
    mock_session.task = mock.MagicMock(
        server_name="srv", environment_name="env", namespace=None,
    )

    # Make the session usable as an async context manager
    mock_session.__aenter__ = mock.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = mock.AsyncMock(return_value=False)

    mock_env = mock.MagicMock()
    mock_env.session = mock.MagicMock(return_value=mock_session)
    return mock_env


MCP_FAILURE = AgentResult(
    success=False,
    error="MCP server 'openreward' failed to connect — no environment tools available",
)
SUCCESS_RESULT = AgentResult(success=True, reward=1.0, finished=True)
OTHER_FAILURE = AgentResult(success=False, error="Some other error")


class TestRunTrialRetry:
    @pytest.mark.asyncio
    async def test_retries_on_mcp_failure_then_succeeds(self):
        env = _make_mock_env()
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(side_effect=[MCP_FAILURE, SUCCESS_RESULT])
        config = _make_trial_config()
        task = mock.MagicMock(task_spec={"id": "test-task"})

        with mock.patch("firehorse.trial.asyncio.sleep", new_callable=mock.AsyncMock):
            result = await run_trial(env, task, agent, config)

        assert result.success is True
        assert result.reward == 1.0
        assert agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_mcp_failure(self):
        env = _make_mock_env()
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(return_value=OTHER_FAILURE)
        config = _make_trial_config()
        task = mock.MagicMock(task_spec={"id": "test-task"})

        result = await run_trial(env, task, agent, config)

        assert result.success is False
        assert agent.run.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        env = _make_mock_env()
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(return_value=MCP_FAILURE)
        config = _make_trial_config()
        task = mock.MagicMock(task_spec={"id": "test-task"})

        with mock.patch("firehorse.trial.asyncio.sleep", new_callable=mock.AsyncMock):
            result = await run_trial(env, task, agent, config)

        assert result.success is False
        assert "MCP" in result.error
        assert agent.run.call_count == _MCP_RETRY_MAX

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        env = _make_mock_env()
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(return_value=SUCCESS_RESULT)
        config = _make_trial_config()
        task = mock.MagicMock(task_spec={"id": "test-task"})

        result = await run_trial(env, task, agent, config)

        assert result.success is True
        assert agent.run.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_uses_exponential_backoff(self):
        env = _make_mock_env()
        agent = mock.AsyncMock()
        agent.run = mock.AsyncMock(side_effect=[MCP_FAILURE, MCP_FAILURE, SUCCESS_RESULT])
        config = _make_trial_config()
        task = mock.MagicMock(task_spec={"id": "test-task"})

        with mock.patch("firehorse.trial.asyncio.sleep", new_callable=mock.AsyncMock) as mock_sleep, \
             mock.patch("firehorse.trial.random.uniform", return_value=0.0): # the tests will add jitter, if we want them to pass we should stop that
            result = await run_trial(env, task, agent, config)

        assert result.success is True
        assert agent.run.call_count == 3
        # First retry: 2s, second retry: 4s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)
