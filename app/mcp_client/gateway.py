"""MCP gateway interface used by the orchestrator, retrieval and agents.

All backend reads and writes go through a gateway. Implementations:

* ``HttpMCPGateway`` (production): real FastMCP clients over Streamable HTTP to the four
  independently running server processes.
* ``InProcessMCPGateway`` / ``FastMCPMemoryGateway`` (``app.mcp_client.in_process``): demo and
  test adapters with the same interface. Production wiring never uses them.

The gateway - not any model - records each call (``calls``), so the pipeline trace proves which
MCP primitives were really invoked. It also enforces a client-side allowlist: a tool can only be
called on the server that owns it.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from app.logging_config import get_logger, trace
from app.models.response import MCPCallRecord

logger = get_logger(__name__)

TOOL_SERVER: dict[str, str] = {
    "get_account_summary": "banking",
    "get_transaction_history": "banking",
    "get_linked_accounts": "banking",
    "get_policy_details": "insurance",
    "check_coverage_clause": "insurance",
    "get_claim_status": "insurance",
    "get_portfolio_summary": "wealth",
    "get_investment_product_details": "wealth",
    "check_product_suitability": "wealth",
    "get_customer_kyc_status": "concierge_ops",
    "run_escalation_check": "concierge_ops",
    "send_customer_notification": "concierge_ops",
    "write_audit_log": "concierge_ops",
}
RESOURCE_SERVER: dict[str, str] = {
    "customer": "concierge_ops",
    "account": "banking",
    "policy": "insurance",
    "knowledgebase": "concierge_ops",
}
PROMPT_SERVER: dict[str, str] = {
    "request_classification_prompt": "concierge_ops",
    "response_drafting_prompt": "concierge_ops",
}
SERVERS = ("banking", "insurance", "wealth", "concierge_ops")


class MCPUnavailable(RuntimeError):
    """The server could not be reached or did not answer (offline, refused, timed out)."""

    def __init__(self, server: str, primitive: str, name: str) -> None:
        super().__init__(f"MCP server '{server}' unavailable for {primitive} '{name}'")
        self.server = server
        self.primitive = primitive
        self.name = name


class MCPPrimitiveError(RuntimeError):
    """The server answered but the primitive failed outside the tool's error envelope
    (e.g. a Prompt rejected its arguments, or a malformed response)."""

    def __init__(self, server: str, primitive: str, name: str, reason: str) -> None:
        super().__init__(f"MCP {primitive} '{name}' on '{server}' failed: {reason}")
        self.server = server
        self.primitive = primitive
        self.name = name
        self.reason = reason


class MCPGateway(ABC):
    """Synchronous facade over MCP Tools, Resources and Prompts."""

    @trace(logger)
    def __init__(self) -> None:
        self.calls: list[MCPCallRecord] = []
        # Optional hook invoked before every call: (server, primitive, name) -> None.
        # Tests use it to stop a server process mid-request.
        self.before_call: Callable[[str, str, str], None] | None = None

    # -- public API --------------------------------------------------------------------------

    @trace(logger)
    def call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call ``tool`` on its owning server; returns the tool's ``{ok, data|error}`` envelope."""
        server = TOOL_SERVER.get(tool)
        if server is None:
            raise ValueError(f"Unknown MCP tool: {tool}")
        envelope = self._invoke(server, "tool", tool, lambda: self._call_tool(server, tool, arguments))
        return envelope

    @trace(logger)
    def read_resource(self, uri: str) -> dict[str, Any]:
        scheme = uri.split("://", 1)[0]
        server = RESOURCE_SERVER.get(scheme)
        if server is None:
            raise ValueError(f"Unknown MCP resource scheme: {scheme}")
        return self._invoke(server, "resource", f"{scheme}://", lambda: self._read_resource(server, uri))

    @trace(logger)
    def get_prompt(self, name: str, arguments: dict[str, str]) -> str:
        server = PROMPT_SERVER.get(name)
        if server is None:
            raise ValueError(f"Unknown MCP prompt: {name}")
        return self._invoke(server, "prompt", name, lambda: self._get_prompt(server, name, arguments))

    @trace(logger)
    def reset_calls(self) -> None:
        self.calls = []

    # -- recording ---------------------------------------------------------------------------

    @trace(logger)
    def _invoke(self, server: str, primitive: str, name: str, operation: Callable[[], Any]) -> Any:
        if self.before_call is not None:
            self.before_call(server, primitive, name)
        started = time.perf_counter()
        try:
            result = operation()
        except MCPUnavailable:
            self._record(server, primitive, name, False, "service_unavailable", started)
            raise
        except MCPPrimitiveError:
            self._record(server, primitive, name, False, "primitive_error", started)
            raise
        ok = bool(result.get("ok")) if isinstance(result, dict) else isinstance(result, str)
        code = None
        if isinstance(result, dict) and not ok:
            code = (result.get("error") or {}).get("code", "unknown_error")
        self._record(server, primitive, name, ok, code, started)
        return result

    @trace(logger)
    def _record(self, server: str, primitive: str, name: str, ok: bool, code: str | None, started: float) -> None:
        self.calls.append(MCPCallRecord(
            server=server, primitive=primitive, name=name, ok=ok, error_code=code,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        ))

    # -- transport -----------------------------------------------------------------------------

    @abstractmethod
    def _call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def _read_resource(self, server: str, uri: str) -> dict[str, Any]: ...

    @abstractmethod
    def _get_prompt(self, server: str, name: str, arguments: dict[str, str]) -> str: ...
