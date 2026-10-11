"""Demo/test MCP gateways with the production interface. NOT used by production wiring.

* ``InProcessMCPGateway`` calls the servers' business functions directly against a chosen SQLite
  file, with JSON round-tripping so results look exactly like transport results. It lets
  ``run_local_demo.py`` run in seconds without any MCP client or server process (PDF section 8).
* ``FastMCPMemoryGateway`` speaks the real MCP protocol through FastMCP's in-memory transport, so
  tests exercise genuine registration, schema validation and serialisation without sockets.

Both support fault injection (``offline`` servers, or a server going offline after N calls) for the
"one MCP server offline mid-request" stress test.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.logging_config import get_logger, trace
from app.mcp_client.gateway import MCPGateway, MCPPrimitiveError, MCPUnavailable

logger = get_logger(__name__)

_RESOURCE_PATTERNS = {
    "customer": (re.compile(r"^customer://([^/]+)/profile$"), "concierge_ops", "customer_profile_resource"),
    "account": (re.compile(r"^account://([^/]+)/summary$"), "banking", "account_summary_resource"),
    "policy": (re.compile(r"^policy://([^/]+)/summary$"), "insurance", "policy_summary_resource"),
    "knowledgebase": (re.compile(r"^knowledgebase://([^/]+)/excerpt$"), "concierge_ops", "knowledgebase_excerpt_resource"),
}


@trace(logger)
def _server_modules() -> dict[str, Any]:
    from mcp_servers import banking_server, concierge_ops_server, insurance_server, wealth_server

    return {
        "banking": banking_server,
        "insurance": insurance_server,
        "wealth": wealth_server,
        "concierge_ops": concierge_ops_server,
    }


class _FaultInjection:
    @trace(logger)
    def __init__(self, offline: set[str] | None, offline_after: dict[str, int] | None) -> None:
        self.offline = set(offline or ())
        self.offline_after = dict(offline_after or {})
        self.counts: dict[str, int] = {}

    @trace(logger)
    def check(self, server: str, primitive: str, name: str) -> None:
        self.counts[server] = self.counts.get(server, 0) + 1
        limit = self.offline_after.get(server)
        if limit is not None and self.counts[server] > limit:
            self.offline.add(server)
        if server in self.offline:
            raise MCPUnavailable(server, primitive, name)


class InProcessMCPGateway(MCPGateway):
    @trace(logger)
    def __init__(
        self,
        db_path: str | Path,
        *,
        offline: set[str] | None = None,
        offline_after: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.db_path = str(db_path)
        self.faults = _FaultInjection(offline, offline_after)
        self._modules = _server_modules()

    @trace(logger)
    def take_offline(self, server: str) -> None:
        self.faults.offline.add(server)

    @contextmanager
    def _database(self) -> Iterator[None]:
        previous = os.environ.get("NEXUS_DB_PATH")
        os.environ["NEXUS_DB_PATH"] = self.db_path
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("NEXUS_DB_PATH", None)
            else:
                os.environ["NEXUS_DB_PATH"] = previous

    @staticmethod
    @trace(logger)
    def _round_trip(value: Any) -> Any:
        return json.loads(json.dumps(value))

    @trace(logger)
    def _call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.faults.check(server, "tool", tool)
        function = getattr(self._modules[server], tool)
        with self._database():
            try:
                return self._round_trip(function(**arguments))
            except TypeError:
                raise MCPPrimitiveError(server, "tool", tool, "invalid_arguments") from None

    @trace(logger)
    def _read_resource(self, server: str, uri: str) -> dict[str, Any]:
        scheme = uri.split("://", 1)[0]
        self.faults.check(server, "resource", scheme)
        pattern, _, function_name = _RESOURCE_PATTERNS[scheme]
        match = pattern.fullmatch(uri)
        if not match:
            raise MCPPrimitiveError(server, "resource", scheme, "unknown_resource")
        with self._database():
            return self._round_trip(getattr(self._modules[server], function_name)(match.group(1)))

    @trace(logger)
    def _get_prompt(self, server: str, name: str, arguments: dict[str, str]) -> str:
        self.faults.check(server, "prompt", name)
        with self._database():
            try:
                return getattr(self._modules[server], name)(**arguments)
            except (ValueError, TypeError):
                raise MCPPrimitiveError(server, "prompt", name, "prompt_rejected_arguments") from None


class FastMCPMemoryGateway(MCPGateway):
    """Real MCP protocol (FastMCP in-memory transport) against the four registered servers."""

    @trace(logger)
    def __init__(self, db_path: str | Path, *, offline: set[str] | None = None, offline_after: dict[str, int] | None = None) -> None:
        super().__init__()
        self.db_path = str(db_path)
        self.faults = _FaultInjection(offline, offline_after)
        self._modules = _server_modules()

    @trace(logger)
    def _run(self, server: str, primitive: str, name: str, op) -> Any:
        from fastmcp import Client
        from mcp.shared.exceptions import McpError

        self.faults.check(server, primitive, name)

        async def runner():
            async with Client(self._modules[server].mcp) as client:
                return await op(client)

        previous = os.environ.get("NEXUS_DB_PATH")
        os.environ["NEXUS_DB_PATH"] = self.db_path
        try:
            return asyncio.run(runner())
        except McpError:
            raise MCPPrimitiveError(server, primitive, name, "mcp_error") from None
        finally:
            if previous is None:
                os.environ.pop("NEXUS_DB_PATH", None)
            else:
                os.environ["NEXUS_DB_PATH"] = previous

    @trace(logger)
    def _call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async def op(client):
            return (await client.call_tool(tool, arguments, raise_on_error=False)).structured_content

        return self._run(server, "tool", tool, op)

    @trace(logger)
    def _read_resource(self, server: str, uri: str) -> dict[str, Any]:
        async def op(client):
            return json.loads((await client.read_resource(uri))[0].text)

        return self._run(server, "resource", uri.split("://", 1)[0], op)

    @trace(logger)
    def _get_prompt(self, server: str, name: str, arguments: dict[str, str]) -> str:
        async def op(client):
            result = await client.get_prompt(name, arguments)
            return "\n".join(m.content.text for m in result.messages)

        return self._run(server, "prompt", name, op)
