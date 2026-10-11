"""Deterministic intake gate: the first step of every request.

Runs before any model or MCP call. Rejects malformed requests with a typed ``IntakeError`` so an
invalid intake can never silently proceed to retrieval, drafting or a domain tool call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.guardrails import (
    InputValidationError,
    PromptInjectionDetected,
    find_identifiers,
    is_valid_identifier,
    mask_identifiers_in_text,
    sanitize_free_text,
)
from app.logging_config import get_logger, trace
from app.models.intake import CustomerRequest, IntakeError, IntakeErrorCode, ReferencedObjects

logger = get_logger(__name__)
MAX_MESSAGE_LENGTH = 2000


class IntakeRejected(Exception):
    """Raised by the intake layer; carries a typed, customer-safe ``IntakeError``."""

    def __init__(self, error: IntakeError) -> None:
        super().__init__(f"{error.code.value}: {error.message}")
        self.error = error


@dataclass(frozen=True)
class SanitizedInbound:
    customer_id: str
    message: str  # sanitised text (identifiers intact; never logged)
    model_message: str  # sanitised text with identifiers masked - what models see
    references: ReferencedObjects
    channel: str


@trace(logger)
def _reject(code: IntakeErrorCode, message: str, field: str | None = None) -> IntakeRejected:
    return IntakeRejected(IntakeError(code=code, message=message, field=field))


@trace(logger)
def validate_inbound(raw: CustomerRequest | Mapping[str, Any]) -> SanitizedInbound:
    """Validate structure, sanitise the message and reconcile identifiers from text and metadata."""
    try:
        request = raw if isinstance(raw, CustomerRequest) else CustomerRequest.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(p) for p in first.get("loc", ())) or None
        if first.get("type") == "extra_forbidden":
            raise _reject(IntakeErrorCode.INVALID_REQUEST, "The request contains fields that callers may not set.", location) from None
        if location and location.startswith("references"):
            raise _reject(IntakeErrorCode.MALFORMED_REFERENCE, "A referenced identifier has an invalid format.", location) from None
        raise _reject(IntakeErrorCode.INVALID_REQUEST, "The request is not well formed.", location) from None

    try:
        message = sanitize_free_text(request.message, field="message", max_length=MAX_MESSAGE_LENGTH, source="customer")
    except PromptInjectionDetected:
        raise _reject(IntakeErrorCode.PROMPT_INJECTION, "The message contains instruction-like content that cannot be processed.", "message") from None
    except InputValidationError as exc:
        raise _reject(IntakeErrorCode.INVALID_TEXT, str(exc), "message") from None

    in_text = find_identifiers(message)
    text_customers = in_text.pop("customer_id", [])
    supplied = request.customer_id.strip().upper() if isinstance(request.customer_id, str) and request.customer_id.strip() else None
    if supplied is not None and not is_valid_identifier("customer_id", supplied):
        raise _reject(IntakeErrorCode.MALFORMED_CUSTOMER_ID, "customer_id must use the CUS-XXXXX format.", "customer_id")
    if supplied is None:
        if len(text_customers) == 1:
            supplied = text_customers[0]
        elif len(text_customers) > 1:
            raise _reject(IntakeErrorCode.CONFLICTING_CUSTOMER_ID, "The request names more than one customer.", "customer_id")
        else:
            raise _reject(IntakeErrorCode.MISSING_CUSTOMER_ID, "A customer ID is required before the request can be processed.", "customer_id")
    elif text_customers and set(text_customers) != {supplied}:
        raise _reject(IntakeErrorCode.CONFLICTING_CUSTOMER_ID, "The customer ID in the message does not match the authenticated customer.", "customer_id")

    references = request.references.model_dump()
    for kind, values in in_text.items():
        if len(values) > 1:
            raise _reject(IntakeErrorCode.CONFLICTING_REFERENCE, f"The message names more than one {kind}.", kind)
        if references.get(kind) and references[kind] != values[0]:
            raise _reject(IntakeErrorCode.CONFLICTING_REFERENCE, f"The {kind} in the message does not match the request metadata.", kind)
        references[kind] = values[0]

    return SanitizedInbound(
        customer_id=supplied,
        message=message,
        model_message=mask_identifiers_in_text(message),
        references=ReferencedObjects.model_validate(references),
        channel=request.channel,
    )
