from __future__ import annotations

import os
from typing import TYPE_CHECKING

from firehorse.agents.resum.providers.base import ProviderClient

if TYPE_CHECKING:
    pass


def parse_provider(model: str, provider_url: str | None) -> tuple[str, str]:
    """Parse a model string into (provider_name, model_id).

    Examples:
        "anthropic/claude-opus-4-6" -> ("anthropic", "claude-opus-4-6")
        "openai/gpt-4o"            -> ("openai", "gpt-4o")
        "openrouter/qwen/qwen3"    -> ("openrouter", "qwen/qwen3")
        "google/gemini-2.5-pro"    -> ("google", "gemini-2.5-pro")
    """
    for prefix in ("anthropic", "openai", "openrouter", "google"):
        if model.startswith(f"{prefix}/"):
            return prefix, model[len(prefix) + 1:]
    if provider_url:
        if "openrouter" in provider_url:
            return "openrouter", model
        return "openai", model
    raise ValueError(
        f"Cannot determine provider for model {model!r}. "
        f"Use a prefix (anthropic/, openai/, openrouter/, google/) or pass --provider-url."
    )


def resolve_api_key(provider: str, secrets: dict[str, str]) -> str:
    """Resolve the API key for a provider from secrets or environment variables."""
    key_map: dict[str, list[tuple[str, str]]] = {
        "openai": [("openai_api_key", "OPENAI_API_KEY")],
        "anthropic": [("anthropic_api_key", "ANTHROPIC_API_KEY")],
        "google": [
            ("google_api_key", "GOOGLE_API_KEY"),
            ("gemini_api_key", "GEMINI_API_KEY"),
        ],
        "openrouter": [
            ("openrouter_api_key", "OPENROUTER_API_KEY"),
            ("openai_api_key", "OPENAI_API_KEY"),
        ],
    }
    candidates = key_map.get(provider, [])
    for secret_key, env_var in candidates:
        if secret_key in secrets:
            return secrets[secret_key]
        val = os.environ.get(env_var, "")
        if val:
            return val
    raise ValueError(
        f"No API key found for provider {provider!r}. "
        f"Set it via --secret or environment variable."
    )


def get_provider(
    provider_name: str,
    model_id: str,
    api_key: str,
    provider_url: str | None = None,
    context_window: int | None = None,
) -> ProviderClient:
    """Instantiate the appropriate ProviderClient for the given provider."""
    if provider_name == "openai":
        from firehorse.agents.resum.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model_id, api_key=api_key, base_url=provider_url, context_window=context_window)

    elif provider_name == "openrouter":
        from firehorse.agents.resum.providers.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(model=model_id, api_key=api_key, context_window=context_window)

    elif provider_name == "anthropic":
        from firehorse.agents.resum.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model_id, api_key=api_key, context_window=context_window)

    elif provider_name == "google":
        from firehorse.agents.resum.providers.google_provider import GoogleProvider
        return GoogleProvider(model=model_id, api_key=api_key, context_window=context_window)

    else:
        raise ValueError(f"Unknown provider: {provider_name!r}")
