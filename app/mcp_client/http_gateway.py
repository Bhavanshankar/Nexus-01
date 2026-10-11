"""Production MCP gateway: FastMCP clients over Streamable HTTP to four separate server processes.

Endpoints default to the loopback ports the servers bind to and can be overridden with
``NEXUS_BANKING_URL``, ``NEXUS_INSURANCE_URL``, ``NEXUS_WEALTH_URL`` and
``NEXUS_CONCIERGE_OPS_URL``. The orchestrator's request ID is propagated to the servers in the
``X-Nexus-Request-Id`` header so server log lines carry the same correlation ID.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Awaitable, TypeVar

from app.logging_config import current_request_id, get_logger, trace
from app.mcp_client.gateway import MCPGateway, MCPPrimitiveError, MCPUnavailable

logger = get_logger(__name__)
T = TypeVar("T")

DEFAULT_URLS = {
    "banking": ("NEXUS_BANKING_URL", "http://127.0.0.1:8101/mcp"),
    "insurance": ("NEXUS_INSURANCE_URL", "http://127.0.0.1:8102/mcp"),
    "wealth": ("NEXUS_WEALTH_URL", "http://127.0.0.1:8103/mcp"),
    "concierge_ops": ("NEXUS_CONCIERGE_OPS_URL", "http://127.0.0.1:8104/mcp"),
}


@trace(logger)
def server_urls() -> dict[str, str]:
    return {server: os.getenv(env, default) for server, (env, default) in DEFAULT_URLS.items()}


class HttpMCPGateway(MCPGateway):
    @trace(logger)
    def __init__(self, urls: dict[str, str] | None = None, *, timeout_s: float = 15.0) -> None:
        super().__init__()
        self.urls = urls or server_urls()
        self.timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="nexus-mcp-client", daemon=True)
        self._thread.start()

    @trace(logger)
    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # -- helpers -------------------------------------------------------------------------------

    @trace(logger)
    def _client(self, server: str):
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        headers = {}
        request_id = current_request_id.get()
        if request_id:
            headers["X-Nexus-Request-Id"] = request_id
        transport = StreamableHttpTransport(self.urls[server], headers=headers)
        return Client(transport, timeout=self.timeout_s, init_timeout=self.timeout_s)

    @trace(logger)
    def _run(self, server: str, primitive: str, name: str, coro: Awaitable[T]) -> T:
        from mcp.shared.exceptions import McpError

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=self.timeout_s * 2)
        except McpError as exc:
            # The server answered with a protocol-level error (e.g. prompt argument rejected).
            raise MCPPrimitiveError(server, primitive, name, type(exc).__name__) from None
        except Exception:
            future.cancel()
            # Connection refused, reset, timeout, or the process went away mid-call.
            raise MCPUnavailable(server, primitive, name) from None

    # -- transport -----------------------------------------------------------------------------

    @trace(logger)
    def _call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        client = self._client(server)  # created in the calling thread: captures the request ID header

        async def op():
            async with client:
                result = await client.call_tool(tool, arguments, raise_on_error=False)
                return result.structured_content, result.is_error

        content, is_error = self._run(server, "tool", tool, op())
        if not isinstance(content, dict) or "ok" not in content:
            raise MCPPrimitiveError(server, "tool", tool, "malformed_envelope" if not is_error else "tool_error")
        return content

    @trace(logger)
    def _read_resource(self, server: str, uri: str) -> dict[str, Any]:
        client = self._client(server)

        async def op():
            async with client:
                return await client.read_resource(uri)

        contents = self._run(server, "resource", uri.split("://", 1)[0], op())
        try:
            return json.loads(contents[0].text)
        except (IndexError, AttributeError, json.JSONDecodeError):
            raise MCPPrimitiveError(server, "resource", uri.split("://", 1)[0], "malformed_resource") from None

    @trace(logger)
    def _get_prompt(self, server: str, name: str, arguments: dict[str, str]) -> str:
        client = self._client(server)

        async def op():
            async with client:
                return await client.get_prompt(name, arguments)

        result = self._run(server, "prompt", name, op())
        try:
            return "\n".join(m.content.text for m in result.messages)
        except AttributeError:
            raise MCPPrimitiveError(server, "prompt", name, "malformed_prompt") from None

    # -- diagnostics ---------------------------------------------------------------------------

    @trace(logger)
    def list_primitives(self) -> dict[str, dict[str, list[str]]]:
        """List each server's tools, resource templates and prompts (used by health checks)."""

        async def op(client):
            async with client:
                tools = [t.name for t in await client.list_tools()]
                templates = [t.uriTemplate for t in await client.list_resource_templates()]
                prompts = [p.name for p in await client.list_prompts()]
                return {"tools": tools, "resources": templates, "prompts": prompts}

        return {server: self._run(server, "list", "primitives", op(self._client(server))) for server in self.urls}
