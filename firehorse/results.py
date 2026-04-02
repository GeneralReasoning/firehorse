from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrialResult:
    task_index: int
    task_spec: dict
    success: bool
    reward: float | None = None
    finished: bool = False
    turns_used: int = 0
    error: str | None = None
    duration_seconds: float = 0.0
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict:
        return {
            "task_index": self.task_index,
            "task_id": self.task_spec.get("id", self.task_index),
            "task_spec": self.task_spec,
            "success": self.success,
            "final_reward": self.reward,
            "finished": self.finished,
            "tool_calls": self.turns_used,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 1),
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class RunSummary:
    run_name: str
    environment: str
    agent: str
    model: str
    split: str
    total_tasks: int
    completed: int
    failed: int
    avg_reward: float | None
    std_reward: float | None = None
    total_cost_usd: float | None = None
    results: list[TrialResult] = field(default_factory=list)

    @classmethod
    def from_results(
        cls,
        results: list[TrialResult | BaseException],
        run_name: str,
        environment: str,
        agent: str,
        model: str,
        split: str,
    ) -> RunSummary:
        trial_results: list[TrialResult] = []
        errors = 0
        rewards: list[float] = []
        total_cost = 0.0

        for r in results:
            if isinstance(r, BaseException):
                errors += 1
                trial_results.append(TrialResult(
                    task_index=-1,
                    task_spec={},
                    success=False,
                    error=str(r),
                ))
            else:
                trial_results.append(r)
                if not r.success:
                    errors += 1
                if r.reward is not None:
                    rewards.append(r.reward)
                if r.cost_usd is not None:
                    total_cost += r.cost_usd

        avg = sum(rewards) / len(rewards) if rewards else None
        std = None
        if avg is not None and len(rewards) > 1:
            std = math.sqrt(sum((x - avg) ** 2 for x in rewards) / (len(rewards) - 1))

        return cls(
            run_name=run_name,
            environment=environment,
            agent=agent,
            model=model,
            split=split,
            total_tasks=len(results),
            completed=len(results) - errors,
            failed=errors,
            avg_reward=avg,
            std_reward=std,
            total_cost_usd=total_cost if total_cost > 0 else None,
            results=trial_results,
        )

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "environment": self.environment,
            "agent": self.agent,
            "model": self.model,
            "split": self.split,
            "n_tasks": self.total_tasks,
            "n_completed": self.completed,
            "n_failed": self.failed,
            "avg_reward": round(self.avg_reward, 4) if self.avg_reward is not None else None,
            "std_reward": round(self.std_reward, 4) if self.std_reward is not None else None,
            "total_cost_usd": round(self.total_cost_usd, 4) if self.total_cost_usd is not None else None,
            "trials": [r.to_dict() for r in self.results],
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def print_report(self, file=sys.stderr) -> None:
        print("\n" + "=" * 60, file=file)
        print(f"Run: {self.run_name}", file=file)
        print(f"Environment: {self.environment} | Agent: {self.agent} | Model: {self.model}", file=file)
        print(f"Split: {self.split}", file=file)
        print("-" * 60, file=file)
        print(f"Total tasks: {self.total_tasks}", file=file)
        print(f"Completed:   {self.completed}", file=file)
        print(f"Failed:      {self.failed}", file=file)
        if self.avg_reward is not None:
            std_str = f" +/- {self.std_reward:.4f}" if self.std_reward is not None else ""
            print(f"Avg reward:  {self.avg_reward:.4f}{std_str}", file=file)
        else:
            print("Avg reward:  N/A", file=file)
        if self.total_cost_usd is not None:
            print(f"Total cost:  ${self.total_cost_usd:.2f}", file=file)
        print("=" * 60, file=file)
