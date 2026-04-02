from __future__ import annotations

from firehorse.agents.resum.providers.openai_provider import OpenAIProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Context windows for models commonly accessed via OpenRouter.
# Model IDs here are the OpenRouter format (already stripped of "openrouter/" prefix).
OPENROUTER_CONTEXT_WINDOWS = {
    # DeepSeek
    "deepseek/deepseek-v3.2": 163840,
    "deepseek/deepseek-v3": 128000,
    "deepseek/deepseek-r1": 128000,
    # Zhipu / GLM
    "z-ai/glm-5.1": 204800,
    "z-ai/glm-5": 200000,
    # Moonshot / Kimi
    "moonshot/kimi-k2.5": 256000,
    "moonshot/kimi-k2-thinking": 256000,
    "moonshot/kimi-k2-thinking-turbo": 256000,
    # MiniMax
    "minimax/minimax-m2.7": 205000,
    "minimax/minimax-m2.5": 196608,
    # Alibaba / Qwen
    "qwen/qwen-3.5-397b-a17b": 1000000,
    "qwen/qwen3-max-2026-01-23": 262144,
    # OpenAI via OpenRouter
    "openai/gpt-5.4": 1000000,
    "openai/gpt-5.4-mini": 400000,
    "openai/gpt-4.1": 1047576,
    "openai/gpt-4.1-mini": 1047576,
    "openai/gpt-4o": 128000,
    "openai/gpt-4o-mini": 128000,
    # Google via OpenRouter
    "google/gemini-2.5-pro": 2000000,
    "google/gemini-2.5-flash": 1048576,
    "google/gemini-2.0-flash": 1048576,
    # Anthropic via OpenRouter
    "anthropic/claude-opus-4-6": 1000000,
    "anthropic/claude-sonnet-4-6": 1000000,
    "anthropic/claude-haiku-4-5-20251001": 200000,
}


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider — OpenAI-compatible with a different base URL."""

    def __init__(self, model: str, api_key: str, context_window: int | None = None):
        super().__init__(model=model, api_key=api_key, base_url=OPENROUTER_BASE_URL, context_window=context_window)

    @property
    def context_window(self) -> int | None:
        if self._context_window_override is not None:
            return self._context_window_override
        return OPENROUTER_CONTEXT_WINDOWS.get(self.model)
