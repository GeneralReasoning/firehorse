"""Lightweight async reverse proxy that injects Bearer auth for OpenRouter.

Codex CLI v0.118+ uses the Responses API with WebSocket streaming, falling back
to HTTP SSE. When `openai_base_url` points to a custom endpoint, the HTTP
fallback does not send the Authorization header. This proxy sits on localhost,
rejects WebSocket upgrades (so Codex falls back to HTTP), injects the Bearer
token, and streams the SSE response back.

Uses aiohttp ClientSession for upstream but raw asyncio server to ensure
WebSocket upgrade requests are rejected instantly (not absorbed by a framework).
"""
from __future__ import annotations

import asyncio
import ssl
import sys
from typing import Any

import aiohttp


class OpenRouterProxy:
    def __init__(self, api_key: str, target_base: str = "https://openrouter.ai/api/v1"):
        self._api_key = api_key
        self._target_base = target_base.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._server: asyncio.Server | None = None
        self.port: int = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        self._session = aiohttp.ClientSession(
            connector=conn,
            timeout=aiohttp.ClientTimeout(total=300, sock_read=300),
        )
        self._server = await asyncio.start_server(
            self._handle_connection, "127.0.0.1", 0,
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection from Codex."""
        peer = writer.get_extra_info("peername")
        try:
            await self._process_request(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError) as e:
            print(f"[codex-proxy] conn closed ({peer}): {e}", file=sys.stderr)
        except asyncio.TimeoutError:
            print(f"[codex-proxy] timeout ({peer})", file=sys.stderr)
        except Exception as e:
            print(f"[codex-proxy] conn error ({peer}): {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _process_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        # Read request line
        request_line = await asyncio.wait_for(reader.readline(), timeout=10)
        if not request_line:
            return
        parts = request_line.decode("utf-8", errors="replace").strip().split(" ", 2)
        if len(parts) < 2:
            return
        method, path = parts[0], parts[1]

        # Read headers
        headers: dict[str, str] = {}
        content_length = -1
        is_websocket = False
        is_chunked = False
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" in decoded:
                key, val = decoded.split(":", 1)
                key = key.strip()
                val = val.strip()
                headers[key] = val
                kl = key.lower()
                if kl == "content-length":
                    content_length = int(val)
                if kl == "transfer-encoding" and "chunked" in val.lower():
                    is_chunked = True
                if kl == "upgrade" and val.lower() == "websocket":
                    is_websocket = True

        # Reject WebSocket upgrades immediately
        if is_websocket:
            writer.write(
                b"HTTP/1.1 404 Not Found\r\n"
                b"Content-Length: 0\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            await writer.drain()
            return

        # Read body
        body = b""
        if is_chunked:
            # Read chunked transfer encoding
            while True:
                size_line = await asyncio.wait_for(reader.readline(), timeout=30)
                chunk_size = int(size_line.strip(), 16)
                if chunk_size == 0:
                    await reader.readline()  # trailing \r\n
                    break
                chunk = await asyncio.wait_for(reader.readexactly(chunk_size), timeout=30)
                body += chunk
                await reader.readline()  # trailing \r\n after chunk
        elif content_length > 0:
            body = await asyncio.wait_for(reader.readexactly(content_length), timeout=30)

        # Decompress body if Content-Encoding is set (Codex sends zstd).
        # Header keys are stored as-sent (e.g. "content-encoding" lowercase from Codex).
        content_encoding = ""
        for k, v in headers.items():
            if k.lower() == "content-encoding":
                content_encoding = v.lower()
                break
        if content_encoding == "zstd" and body:
            import zstandard
            dctx = zstandard.ZstdDecompressor()
            # Use streaming decompressor since frames may lack content size
            body = dctx.decompress(body, max_output_size=16 * 1024 * 1024)

        target_url = f"{self._target_base}{path}"
        print(f"[codex-proxy] {method} {path} body={len(body)}b", file=sys.stderr)

        # Build forwarded headers (strip content-encoding since we decompressed)
        fwd_headers: dict[str, str] = {}
        for k, v in headers.items():
            kl = k.lower()
            if kl in ("host", "transfer-encoding", "connection", "upgrade", "content-encoding", "content-length"):
                continue
            fwd_headers[k] = v
        fwd_headers["Authorization"] = f"Bearer {self._api_key}"
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        assert self._session is not None

        try:
            async with self._session.request(
                method=method,
                url=target_url,
                headers=fwd_headers,
                data=body if body else None,
            ) as upstream:
                ct = upstream.headers.get("Content-Type", "")
                is_sse = "text/event-stream" in ct
                print(f"[codex-proxy] <- {upstream.status} sse={is_sse}", file=sys.stderr)

                # Build response line and headers
                status_line = f"HTTP/1.1 {upstream.status} {upstream.reason}\r\n"
                writer.write(status_line.encode())

                for k, v in upstream.headers.items():
                    kl = k.lower()
                    if kl in ("transfer-encoding", "content-encoding"):
                        continue
                    writer.write(f"{k}: {v}\r\n".encode())
                # Use chunked transfer for streaming
                writer.write(b"Transfer-Encoding: chunked\r\n")
                writer.write(b"\r\n")
                await writer.drain()

                # Stream response body
                async for chunk in upstream.content.iter_any():
                    if not chunk:
                        continue
                    # Chunked encoding
                    writer.write(f"{len(chunk):x}\r\n".encode())
                    writer.write(chunk)
                    writer.write(b"\r\n")
                    await writer.drain()

                # End chunked transfer
                writer.write(b"0\r\n\r\n")
                await writer.drain()

        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as e:
            print(f"[codex-proxy] upstream error: {e}", file=sys.stderr)
            error_body = str(e).encode()
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n")
            writer.write(f"Content-Length: {len(error_body)}\r\n".encode())
            writer.write(b"\r\n")
            writer.write(error_body)
            await writer.drain()
