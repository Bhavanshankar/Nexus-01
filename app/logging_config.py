"""Structured JSON logging and the ``@trace(logger)`` decorator used across Nexus.

Every traced call emits:

* ``ENTER`` (DEBUG): fully qualified function name, ``call_id``, redacted bound arguments.
* ``EXIT`` (DEBUG): same ``call_id``, ``duration_ms``, bounded redacted return preview.
* ``FAILED`` (ERROR): same ``call_id``, ``duration_ms``, exception type, redacted message and
  redacted full traceback. The original exception is always re-raised.

Lines are single JSON objects. ``LOG_LEVEL`` defaults to ``DEBUG``. Output goes to stderr, or
to ``NEXUS_LOG_FILE`` when that variable is set, so stdout stays free for MCP/demo output.

Redaction is delegated to :func:`app.guardrails.redact_for_logging` (imported lazily to avoid a
circular import). The redactor itself and this module's internals are deliberately not traced:
tracing the redactor would recurse into itself on every log line.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

PREVIEW_LIMIT = int(os.getenv("NEXUS_LOG_PREVIEW_LIMIT", "500"))

# Correlates log lines from one customer request across intake, agents and MCP calls.
current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nexus_request_id", default=None
)


def log_level() -> int:
    return getattr(logging, os.getenv("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG)


class JsonFormatter(logging.Formatter):
    """Render every record as one JSON object; dict messages are merged into the object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
        }
        request_id = current_request_id.get()
        if request_id:
            payload["request_id"] = request_id
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["event"] = "LOG"
            payload["message"] = _redact_text(record.getMessage())
        return json.dumps(payload, default=str, ensure_ascii=False)


_HANDLER: logging.Handler | None = None


def _shared_handler() -> logging.Handler:
    global _HANDLER
    if _HANDLER is None:
        log_file = os.getenv("NEXUS_LOG_FILE")
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            handler: logging.Handler = logging.FileHandler(log_file, encoding="utf-8")
        else:
            handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        _HANDLER = handler
    return _HANDLER


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes structured JSON lines at ``LOG_LEVEL`` (default DEBUG)."""
    logger = logging.getLogger(_module_name(name))
    logger.setLevel(log_level())
    if not any(getattr(h, "_nexus", False) for h in logger.handlers):
        handler = _shared_handler()
        handler._nexus = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def reset_logging_for_tests() -> None:
    """Drop the cached handler so a test can redirect output (e.g. ``NEXUS_LOG_FILE``)."""
    global _HANDLER
    if _HANDLER is not None:
        for name in list(logging.root.manager.loggerDict):
            existing = logging.getLogger(name)
            if _HANDLER in existing.handlers:
                existing.removeHandler(_HANDLER)
        _HANDLER.close()
    _HANDLER = None


def _redactor() -> Callable[[Any], Any]:
    from app.guardrails import redact_for_logging

    return redact_for_logging


def _redact_text(text: str) -> str:
    from app.guardrails import redact_text

    return redact_text(text)


def _module_name(name: str) -> str:
    """Real dotted module name, also when the module runs as ``__main__`` (``python -m pkg.mod``)."""
    if name != "__main__":
        return name
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and spec.name:
        return spec.name
    path = getattr(main, "__file__", None)
    return os.path.splitext(os.path.basename(path))[0] if path else name


def _qualified_name(func: Callable[..., Any]) -> str:
    return f"{_module_name(func.__module__)}.{func.__qualname__}"


def _bound_arguments(func: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
    except (TypeError, ValueError):
        arguments = {"args": list(args), "kwargs": dict(kwargs)}
    for receiver in ("self", "cls"):
        if receiver in arguments:
            owner = arguments[receiver]
            arguments[receiver] = f"<{owner.__name__ if isinstance(owner, type) else type(owner).__name__}>"
    return arguments


def _preview(value: Any, limit: int) -> str:
    try:
        text = json.dumps(_redactor()(value), default=str, ensure_ascii=False)
    except Exception:  # a preview must never break the traced call
        text = f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _shape(value: Any) -> str:
    size = len(value) if hasattr(value, "__len__") else None
    return f"<{type(value).__name__}{f': {size} chars' if isinstance(value, str) else (f': {size} items' if size is not None else '')}>"


def _safe_arguments(arguments: dict[str, Any], redact: Callable[[dict], Any] | None) -> Any:
    try:
        if redact is not None:
            arguments = redact(arguments)
        return _redactor()(arguments)
    except Exception:
        return {"arguments": "<redaction failed; omitted>"}


def _failure_record(name: str, call_id: str, started: float, exc: BaseException) -> dict[str, Any]:
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "event": "FAILED",
        "function": name,
        "call_id": call_id,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "exception_message": _redact_text(str(exc)),
        "traceback": _redact_text(rendered),
    }


def trace(
    logger: logging.Logger,
    redact: Callable[[dict[str, Any]], Any] | None = None,
    preview_limit: int = PREVIEW_LIMIT,
    redact_result: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a sync or async function with ENTER/EXIT/FAILED JSON tracing.

    ``redact`` optionally pre-processes bound arguments; the shared PII redactor always runs
    afterwards, so a custom redactor can only remove more data, never expose more.
    ``redact_result`` marks functions whose return value is content (sanitised customer or retrieved
    text): the EXIT preview then records only its type and size.
    The wrapper keeps ``__wrapped__`` so FastMCP and ``inspect.signature`` see the real signature.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = _qualified_name(func)

        def enter(args: tuple, kwargs: dict) -> tuple[str, float]:
            call_id = uuid.uuid4().hex
            logger.debug({
                "event": "ENTER",
                "function": name,
                "call_id": call_id,
                "arguments": _safe_arguments(_bound_arguments(func, args, kwargs), redact),
            })
            return call_id, time.perf_counter()

        def exit_(call_id: str, started: float, result: Any) -> None:
            logger.debug({
                "event": "EXIT",
                "function": name,
                "call_id": call_id,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "return_preview": _shape(result) if redact_result else _preview(result, preview_limit),
            })

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                call_id, started = enter(args, kwargs)
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    logger.error(_failure_record(name, call_id, started, exc))
                    raise
                exit_(call_id, started, result)
                return result

            async_wrapper.__nexus_traced__ = True  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_id, started = enter(args, kwargs)
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                logger.error(_failure_record(name, call_id, started, exc))
                raise
            exit_(call_id, started, result)
            return result

        wrapper.__nexus_traced__ = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


def is_traced(func: Any) -> bool:
    """True when ``func`` (or the function it wraps) carries the trace decorator."""
    seen = 0
    while func is not None and seen < 10:
        if getattr(func, "__nexus_traced__", False):
            return True
        func = getattr(func, "__wrapped__", None) or getattr(func, "fn", None)
        seen += 1
    return False
