from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openreward.api.environments.client import AsyncSession
    from openreward.api.environments.types import ToolSpec


@dataclass
class TrialContext:
    prompt_text: str
    tools: list[ToolSpec]
    session: AsyncSession
    model: str
    env_name: str
    task_spec: dict
    run_name: str
    split: str
    variant: str | None = None
    task_index: int | None = None
    max_turns: int | None = None
    provider_url: str | None = None
    disable_builtin_tools: list[str] = field(default_factory=list)
    secrets: dict[str, str] = field(default_factory=dict)
    output_dir: str | None = None
    effort: str = "high"
    logging: bool = True
    use_builtin_descriptions: bool = True
    use_all_filesystem_tools: bool = False  # Codex: expose all filesystem tools via MCP, not just bash
    plan_mode: bool = False
    toolset_name: str | None = None  # Built-in toolset name (e.g. "claude-code", "codex")
    rollout_client: Any = None  # AsyncOpenReward, for creating rollouts
    context_window: int | None = None  # Override model's default context window size


@dataclass
class AgentResult:
    success: bool
    reward: float | None = None
    finished: bool = False
    turns_used: int = 0
    error: str | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None


class BaseAgent(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def setup(self) -> None:
        """One-time setup (e.g. check that the agent binary is installed)."""

    @abstractmethod
    async def run(self, ctx: TrialContext) -> AgentResult:
        """Execute the agent on a single task. Returns the result."""
