"""Entry point for the MCP bridge: python -m firehorse.mcp"""
import asyncio
import builtins
import os
import signal
import sys

# The MCP bridge talks to its client (codex / claude-code) over
# stdin/stdout using JSON-RPC. Any non-JSON byte written to stdout
# corrupts the stream — the client errors with "expected value at line
# 1 column 1" and tears down the transport.
#
# The openreward SDK writes to stdout in two ways:
#   1. A raw `print(...)` in api/environments/client.py when a session
#      starts ("Environment session started (sid: ...)").
#   2. structlog loggers (log_utils.py, rollouts/rollout.py) configured
#      to write to sys.stdout.
#
# We address both before anything imports the SDK:
#   * Suppress non-error structlog output via OPENREWARD_LOG_LEVEL.
#   * Replace builtins.print so that any bare `print(...)` from
#     anywhere in the bridge process (SDK or otherwise) goes to stderr
#     unless the caller explicitly passed file=. The MCP server uses
#     sys.stdout.write / sys.stdout.buffer directly, not print, so this
#     does not affect JSON-RPC framing.
os.environ.setdefault("OPENREWARD_LOG_LEVEL", "ERROR")

_real_print = builtins.print


def _safe_print(*args, **kwargs):
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    return _real_print(*args, **kwargs)


builtins.print = _safe_print

from firehorse.mcp.bridge import OpenRewardBridge


async def main():
    bridge = OpenRewardBridge()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(bridge.shutdown()))
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    try:
        await bridge.run()
    except Exception as e:
        print(f"[openreward-bridge] Fatal error: {e}", file=sys.stderr)
        bridge._write_result_file()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
