import asyncio
import io
import json
import logging

import pytest
from pydantic import BaseModel

from app import logging_config
from app.logging_config import JsonFormatter, current_request_id, trace


@pytest.fixture
def capture():
    stream = io.StringIO()
    logger = logging.getLogger("tests.trace_capture")
    logger.handlers = []
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    def lines():
        return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]

    return logger, lines


def test_log_level_defaults_to_debug(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert logging_config.log_level() == logging.DEBUG
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert logging_config.log_level() == logging.WARNING


def test_enter_exit_are_json_with_matching_call_id_and_duration(capture):
    logger, lines = capture

    @trace(logger)
    def add(a, b=2):
        return a + b

    assert add(1) == 3
    enter, exit_ = lines()
    assert enter["event"] == "ENTER" and exit_["event"] == "EXIT"
    assert enter["level"] == "DEBUG" and exit_["level"] == "DEBUG"
    assert enter["call_id"] == exit_["call_id"] and len(enter["call_id"]) == 32
    assert enter["function"] == exit_["function"] == f"{__name__}.test_enter_exit_are_json_with_matching_call_id_and_duration.<locals>.add"
    assert enter["arguments"] == {"a": 1, "b": 2}  # bound arguments with defaults applied
    assert isinstance(exit_["duration_ms"], float) and exit_["return_preview"] == "3"


def test_failed_line_has_type_traceback_and_reraises(capture):
    logger, lines = capture

    @trace(logger)
    def boom(customer_id):
        raise ValueError(f"lookup failed for {customer_id} email a.b@x.com")

    with pytest.raises(ValueError):
        boom("CUS-20077")
    enter, failed = lines()
    assert failed["event"] == "FAILED" and failed["level"] == "ERROR"
    assert failed["call_id"] == enter["call_id"]
    assert failed["exception_type"] == "builtins.ValueError"
    assert "Traceback (most recent call last)" in failed["traceback"] and "in boom" in failed["traceback"]
    rendered = json.dumps([enter, failed])
    assert "CUS-20077" not in rendered and "a.b@x.com" not in rendered
    assert "****0077" in failed["exception_message"]


def test_pii_never_reaches_arguments_or_previews(capture):
    logger, lines = capture

    class Customer(BaseModel):
        customer_id: str
        email: str
        phone: str
        message: str

    @trace(logger)
    def handle(customer: Customer, account_id: str, note: str, value: str):
        return {"policy_id": "POL-IN-30091", "summary": "Customer said they lost their job", "nested": [{"phone": "+1-415-555-2001"}]}

    handle(
        Customer(customer_id="CUS-20077", email="arjun.mehta@meridian.com", phone="+1-415-555-2001", message="I lost my job"),
        "ACC-20077",
        "call +1-415-555-2001 about ACC-30142",
        "private free text",
    )
    rendered = json.dumps(lines())
    for raw in ("CUS-20077", "ACC-20077", "ACC-30142", "POL-IN-30091", "arjun.mehta@meridian.com", "+1-415-555-2001", "lost my job", "lost their job", "private free text"):
        assert raw not in rendered, raw
    assert "****0077" in rendered and "<redacted:" in rendered


def test_custom_redactor_can_only_remove_more(capture):
    logger, lines = capture

    @trace(logger, redact=lambda args: {**args, "token": "<omitted>"})
    def call(token, customer_id):
        return None

    call("secret-token", "CUS-20077")
    enter = lines()[0]
    assert enter["arguments"]["token"] == "<omitted>"
    assert enter["arguments"]["customer_id"] == "****0077"


def test_redact_result_hides_returned_content(capture):
    logger, lines = capture

    @trace(logger, redact_result=True)
    def clean(text):
        return "I can't afford it"

    clean("x")
    assert lines()[1]["return_preview"] == "<str: 17 chars>"


def test_async_functions_are_traced(capture):
    logger, lines = capture

    @trace(logger)
    async def fetch(n):
        await asyncio.sleep(0)
        return n * 2

    assert asyncio.run(fetch(4)) == 8
    enter, exit_ = lines()
    assert enter["call_id"] == exit_["call_id"] and exit_["return_preview"] == "8"


def test_request_id_context_is_attached_and_wrapped_signature_kept(capture):
    logger, lines = capture

    @trace(logger)
    def tool(account_id: str, caller_scope: str) -> dict:
        """doc"""
        return {}

    token = current_request_id.set("REQ-ABCDEF123456")
    try:
        tool("ACC-20077", "customer")
    finally:
        current_request_id.reset(token)
    assert {line["request_id"] for line in lines()} == {"REQ-ABCDEF123456"}
    import inspect

    assert list(inspect.signature(tool).parameters) == ["account_id", "caller_scope"]
    assert tool.__doc__ == "doc" and logging_config.is_traced(tool)


def test_preview_is_bounded(capture):
    logger, lines = capture

    @trace(logger, preview_limit=50)
    def big():
        return {f"k{i}": i for i in range(100)}

    big()
    preview = lines()[1]["return_preview"]
    assert len(preview) <= 50 + len("...<truncated>") and preview.endswith("...<truncated>")
