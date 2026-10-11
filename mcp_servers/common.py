"""Shared protocol, validation, and least-privilege database helpers for the Nexus MCP servers.

Validation, masking and minimisation come from :mod:`app.guardrails`; table scopes come from
:mod:`app.db_access`. Nothing here is duplicated per server.

Caller-scope trust boundary
---------------------------
``customer`` and ``authorized_agent`` are always accepted. ``internal_service`` exposes extra
fields, and the servers have no caller authentication, so it is refused with
``authorization_error`` unless the operator starts the server with ``NEXUS_TRUST_INTERNAL_SCOPE=1``.
That flag is the documented trust boundary for a loopback-only demo; the orchestrator never sets
or needs it.
"""

import functools
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from app import guardrails
from app.db_access import DOMAIN_SCOPES, get_scoped_connection
from app.logging_config import current_request_id, get_logger, trace

logger = get_logger(__name__)

SERVER_TABLES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    name: (scope["read_tables"], scope["write_tables"]) for name, scope in DOMAIN_SCOPES.items()
}
REQUEST_ID_HEADER = "x-nexus-request-id"
_REQUEST_ID_RE = re.compile(r"^REQ-[A-Z0-9-]{6,40}$")
ERROR_CODES = {
    "validation_error", "not_found", "authorization_error", "database_error",
    "service_unavailable", "internal_error",
}


@trace(logger)
def new_request_id() -> str:
    """Return a server-generated correlation identifier."""
    return f"REQ-{uuid.uuid4().hex[:12].upper()}"


@trace(logger)
def correlation_id() -> str:
    """The caller's propagated ``X-Nexus-Request-Id`` when valid, otherwise a new identifier."""
    propagated = current_request_id.get()
    if propagated and _REQUEST_ID_RE.fullmatch(propagated):
        return propagated
    return new_request_id()


@trace(logger)
def _header_request_id() -> str | None:
    try:
        from fastmcp.server.dependencies import get_http_headers

        value = get_http_headers().get(REQUEST_ID_HEADER)
    except Exception:  # not inside an HTTP request (in-process call or tests)
        return None
    return value if isinstance(value, str) and _REQUEST_ID_RE.fullmatch(value) else None


def with_request_context(func: Callable[..., Any]) -> Callable[..., Any]:
    """Bind the propagated request ID to the logging context for the duration of a primitive call.

    Applied outside ``@trace`` so ENTER/EXIT/FAILED lines carry the orchestrator's request ID.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        header = _header_request_id()
        if header is None:
            return func(*args, **kwargs)
        token = current_request_id.set(header)
        try:
            return func(*args, **kwargs)
        finally:
            current_request_id.reset(token)

    return wrapper


@trace(logger)
def error(code: str, message: str, *, field: str | None = None, retryable: bool = False) -> dict[str, Any]:
    """Build the contract's safe, typed error envelope."""
    if code not in ERROR_CODES:
        code = "internal_error"
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "request_id": correlation_id(),
            "field": field,
        },
    }


@trace(logger)
def success(data: dict[str, Any]) -> dict[str, Any]:
    """Wrap a successful tool result in the shared response structure."""
    return {"ok": True, "data": data}


@trace(logger)
def id_error(kind: str, value: Any, field: str | None = None) -> dict[str, Any] | None:
    """Return a validation envelope when ``value`` is not an exact ``kind`` identifier."""
    if guardrails.is_valid_identifier(kind, value):
        return None
    name = field or kind
    return error("validation_error", f"{name} must match the {guardrails.ID_FORMATS[kind][1]} format", field=name)


@trace(logger)
def scope_error(caller_scope: Any) -> dict[str, Any] | None:
    """Validate ``caller_scope`` and enforce the internal_service trust boundary."""
    if caller_scope not in guardrails.CALLER_SCOPES:
        return error(
            "validation_error",
            f"caller_scope must be one of {', '.join(guardrails.CALLER_SCOPES)}",
            field="caller_scope",
        )
    if caller_scope == "internal_service" and os.getenv("NEXUS_TRUST_INTERNAL_SCOPE") != "1":
        return error(
            "authorization_error",
            "internal_service scope is not trusted by this server.",
            field="caller_scope",
        )
    return None


@trace(logger, redact_result=True)
def text_error(value: Any, field: str, maximum: int, *, source: str = "tool_argument") -> tuple[str | None, dict[str, Any] | None]:
    """Sanitise a free-text argument; return ``(clean_text, None)`` or ``(None, error_envelope)``."""
    try:
        return guardrails.sanitize_free_text(value, field=field, max_length=maximum, source=source), None
    except guardrails.PromptInjectionDetected:
        return None, error("validation_error", f"{field} contains disallowed instruction-like content.", field=field)
    except guardrails.InputValidationError as exc:
        return None, error("validation_error", str(exc), field=field)


@trace(logger)
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@trace(logger)
def iso_utc(value: str | None) -> str | None:
    """Render a stored ``YYYY-MM-DD HH:MM:SS`` (UTC) timestamp as ISO 8601 with ``Z``."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@trace(logger)
def money(amount_minor: int, currency: str = "USD") -> dict[str, Any]:
    return {"amount_minor": int(amount_minor), "currency": currency}


@contextmanager
def connection(
    server: str,
    *,
    read_tables: set[str],
    write_tables: set[str] | None = None,
    db_path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection scoped to this server and this operation's tables."""
    if server not in SERVER_TABLES:
        raise ValueError("Unknown MCP server")
    allowed_reads, allowed_writes = SERVER_TABLES[server]
    write_tables = set(write_tables or ())
    if not set(read_tables) <= allowed_reads or not write_tables <= allowed_writes:
        raise PermissionError("Requested database scope exceeds MCP server ownership")
    conn = get_scoped_connection(read_tables=read_tables, write_tables=write_tables, db_path=db_path)
    try:
        yield conn
    finally:
        conn.close()


@trace(logger)
def tool_failure(exc: Exception) -> dict[str, Any]:
    """Translate internal failures without exposing SQL, paths or exception text."""
    if isinstance(exc, FileNotFoundError):
        return error("service_unavailable", "The data service is unavailable.", retryable=True)
    if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
        return error("service_unavailable", "The data service is busy; retry shortly.", retryable=True)
    if isinstance(exc, (sqlite3.Error, PermissionError)):
        return error("database_error", "The data operation could not be completed.", retryable=True)
    return error("internal_error", "The operation could not be completed.")


@trace(logger, redact=lambda a: {**a, "conn": "<connection>"})
def audit_insert(
    conn: sqlite3.Connection,
    *,
    action_type: str,
    performed_by: str,
    customer_id: str | None,
    outcome: str,
    detail: str,
    timestamp: str | None = None,
) -> int:
    """Insert the mandatory audit record using an already scoped connection."""
    cursor = conn.execute(
        "INSERT INTO audit_log (timestamp, action_type, performed_by, customer_id, outcome, detail) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp or iso_now(), action_type, performed_by, customer_id, outcome, detail),
    )
    return int(cursor.lastrowid)


@trace(logger)
def parse_coverage_clauses(raw: str) -> list[dict[str, str]]:
    """Normalise the stored clause JSON into ``[{clause_id, title, text}]``."""
    parsed = json.loads(raw)
    clauses: list[dict[str, str]] = []
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            clauses.append({
                "clause_id": str(key),
                "title": str(key).replace("_", " ").title(),
                "text": str(value),
            })
    elif isinstance(parsed, list):
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            clauses.append({
                "clause_id": str(item.get("clause_id", index + 1)),
                "title": str(item.get("title", "Policy clause")),
                "text": str(item.get("text", "")),
            })
    else:
        raise ValueError("Unsupported coverage clause structure")
    return clauses


@trace(logger)
def register_tool(mcp: Any, func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a traced business function as an MCP tool under its own name.

    The module keeps the plain traced function (directly testable); FastMCP receives a
    request-context wrapper whose ``__wrapped__`` chain exposes the real signature and docstring.
    """
    mcp.tool(name=func.__name__)(with_request_context(func))
    return func


@trace(logger)
def register_resource(mcp: Any, uri: str, func: Callable[..., Any], *, mime_type: str = "application/json") -> Callable[..., Any]:
    mcp.resource(uri, name=func.__name__, mime_type=mime_type)(with_request_context(func))
    return func


@trace(logger)
def register_prompt(mcp: Any, func: Callable[..., Any]) -> Callable[..., Any]:
    mcp.prompt(name=func.__name__)(with_request_context(func))
    return func


@trace(logger)
def serve(mcp: Any, port_env: str, default_port: int) -> None:
    """Run one server as an independent process: Streamable HTTP on loopback (default), or stdio when
    ``NEXUS_MCP_TRANSPORT=stdio`` so an MCP host such as Claude Desktop can launch it directly. Logs
    always go to stderr or ``NEXUS_LOG_FILE``, never stdout, so the stdio protocol stream stays clean."""
    if os.getenv("NEXUS_MCP_TRANSPORT", "http").lower() == "stdio":
        mcp.run(transport="stdio", show_banner=False)
        return
    mcp.run(
        transport="http",
        host=os.getenv("NEXUS_MCP_HOST", "127.0.0.1"),
        port=int(os.getenv(port_env, str(default_port))),
    )
