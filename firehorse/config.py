# TODO: remove this comment and run isort

from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RunConfig:
    env: str
    agent: Literal["claude-code", "react", "resum", "codex"]
    model: str
    n_concurrent: int = 1
    split: str = "test"
    max_tasks: int | None = None
    skip_tasks: int = 0
    plan_mode: bool = False
    run_name: str | None = None
    max_turns: int | None = None
    provider_url: str | None = None
    disable_builtin_tools: list[str] = field(default_factory=list)
    secrets: dict[str, str] = field(default_factory=dict)
    output_dir: str | None = None
    effort: Literal["low", "medium", "high", "max"] = "high"
    logging: bool = True
    use_builtin_descriptions: bool = True
    use_all_filesystem_tools: bool = False

    def __post_init__(self) -> None:
        if self.n_concurrent < 1:
            raise ValueError(f"n_concurrent must be >= 1, got {self.n_concurrent}")
        if self.max_tasks is not None and self.max_tasks < 1:
            raise ValueError(f"max_tasks must be >= 1, got {self.max_tasks}")
        if self.skip_tasks < 0:
            raise ValueError(f"skip_tasks must be >= 0, got {self.skip_tasks}")
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        if self.effort not in ("low", "medium", "high", "max"):
            raise ValueError(f"effort must be low/medium/high/max, got {self.effort!r}")
        if self.agent not in ("claude-code", "react", "resum", "codex"):
            raise ValueError(f"agent must be claude-code/react/resum/codex, got {self.agent!r}")

    def effective_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.env.replace('/', '-')}-{self.agent}-{ts}"


@dataclass(frozen=True)
class TrialConfig:
    task_index: int
    task_spec: dict
    run_name: str
    env: str
    split: str
    model: str
    max_turns: int | None = None
    provider_url: str | None = None
    disable_builtin_tools: list[str] = field(default_factory=list)
    secrets: dict[str, str] = field(default_factory=dict)
    output_dir: str | None = None
    effort: str = "high"
    logging: bool = True
    use_builtin_descriptions: bool = True
    use_all_filesystem_tools: bool = False
    plan_mode: bool = False
