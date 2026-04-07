"""Shared provider utilities (context window lookup, etc.)."""
from __future__ import annotations

import asyncio
import json
import os

# Default context window when OpenRouter lookup fails.
FALLBACK_CONTEXT_WINDOW = 128_000


async def get_openrouter_context_window(model_name: str) -> int | None:
    """Fetch context window size from OpenRouter's model metadata.

    Queries the /api/v1/models endpoint and finds the matching model.
    Returns context_length in tokens, or None on failure.
    """
    import ssl
    import urllib.request

    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = None

        req = urllib.request.Request("https://openrouter.ai/api/v1/models")
        if or_key:
            req.add_header("Authorization", f"Bearer {or_key}")

        def _fetch():
            with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
                return json.loads(resp.read())

        data = await asyncio.get_event_loop().run_in_executor(None, _fetch)
        for model in data.get("data", []):
            if model.get("id") == model_name:
                return model.get("context_length")
    except Exception as e:
        # TODO: log this properly. Silent fallback to FALLBACK_CONTEXT_WINDOW (128K)
        # can cause compaction to trigger too early or too late for models with
        # different actual context sizes.
        import sys
        print(f"[providers] Failed to fetch context window for {model_name}: {e}", file=sys.stderr)
    return None
