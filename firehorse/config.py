from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RunConfig:
    env: str
    agent: str
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
    effort: str = "high"
    logging: bool = True
    use_builtin_descriptions: bool = True
    use_all_filesystem_tools: bool = False

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
