from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openreward.api.environments.types import ToolSpec


@dataclass
class ToolCallInfo:
    """Normalized tool call extracted from an LLM response."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    # Provider-native message object for appending to the messages list
    raw_message: Any
    # Parsed tool calls
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    # Assistant text content (may be None if only tool calls)
    text_content: str | None = None
    # Reasoning / thinking content (if extended thinking is enabled)
    reasoning_content: str | None = None
    # Token usage
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Context overflow signal — the agent loop handles compaction
    context_overflow: bool = False


class ProviderClient(ABC):
    """Abstract interface for LLM provider clients.

    Each provider implementation manages messages in its own native format.
    The agent loop interacts only through this interface.
    """

    @abstractmethod
    def format_tools(self, tools: list[ToolSpec]) -> Any:
        """Convert OpenReward ToolSpecs to provider-native tool format."""

    @abstractmethod
    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> list[Any]:
        """Create the initial message list with system + user prompt."""

    @abstractmethod
    async def call(
        self,
        messages: list[Any],
        tools: Any,
        max_tokens: int = 16384,
        effort: str | None = None,
    ) -> LLMResponse:
        """Call the LLM and return a normalized response.

        On context overflow, return LLMResponse(context_overflow=True)
        instead of raising, so the agent loop can handle compaction.

        effort: Thinking/reasoning effort level ("low", "medium", "high", "max").
        Provider implementations map this to their native thinking parameters.
        """

    @abstractmethod
    def append_assistant(self, messages: list[Any], response: LLMResponse) -> None:
        """Append the assistant's response to the message list."""

    @abstractmethod
    def append_tool_result(
        self,
        messages: list[Any],
        call_id: str,
        tool_name: str,
        output: str,
    ) -> None:
        """Append a tool result to the message list."""

    @abstractmethod
    def append_user_message(self, messages: list[Any], content: str) -> None:
        """Append a user message to the message list."""

    @abstractmethod
    def messages_to_text(self, messages: list[Any]) -> str:
        """Flatten messages to plain text for compaction summarization."""

    @abstractmethod
    def rebuild_after_compaction(
        self,
        system_prompt: str,
        original_prompt: str,
        summary: str,
    ) -> list[Any]:
        """Rebuild the message list after compaction with system + original prompt + summary."""

    @property
    def context_window(self) -> int | None:
        """Return the model's context window size in tokens, if known.

        Used for proactive compaction. Returns None if unknown (disabling
        proactive compaction). Providers override with a lookup dict of
        known model sizes. User can override via constructor parameter.
        """
        return None

    @abstractmethod
    async def call_for_compaction(
        self,
        conversation_text: str,
        compaction_prompt: str,
        max_tokens: int,
    ) -> str:
        """Call the LLM (no tools) to generate a compaction summary."""
