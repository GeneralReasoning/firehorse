"""Entry point for the MCP bridge: python -m firehorse.mcp"""
import asyncio
import signal
import sys

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
