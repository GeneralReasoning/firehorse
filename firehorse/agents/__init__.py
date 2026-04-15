from __future__ import annotations

from firehorse.agents.base import BaseAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def _register_defaults():
    from firehorse.agents.claude_code import ClaudeCodeAgent
    from firehorse.agents.codex import CodexAgent
    from firehorse.agents.hermes import HermesAgent
    from firehorse.agents.openclaw import OpenClawAgent
    from firehorse.agents.react import ReactAgent
    from firehorse.agents.resum import ReSumAgent
    AGENT_REGISTRY["claude-code"] = ClaudeCodeAgent
    AGENT_REGISTRY["codex"] = CodexAgent
    AGENT_REGISTRY["openclaw"] = OpenClawAgent
    AGENT_REGISTRY["hermes"] = HermesAgent
    AGENT_REGISTRY["react"] = ReactAgent
    AGENT_REGISTRY["resum"] = ReSumAgent


def get_agent(name: str) -> BaseAgent:
    if not AGENT_REGISTRY:
        _register_defaults()
    cls = AGENT_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(AGENT_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown agent: {name!r}. Available: {available}")
    return cls()
