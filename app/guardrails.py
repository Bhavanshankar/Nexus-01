"""Shared Nexus guardrails.

One implementation used by the MCP servers, Resources, Prompts, retrieval, agents and the
orchestrator:

* exact identifier formats and typed format errors;
* ``sanitize_free_text`` - type/length validation, HTML stripping, Unicode/whitespace
  normalisation and deterministic regex prompt-injection detection;
* ``mask_identifier`` / ``mask_identifiers_in_text`` - last-four masking of account, policy and
  portfolio (and claim/customer) numbers;
* ``minimize_account_fields`` / ``minimize_portfolio_fields`` - caller-scope field filtering;
* ``redact_for_logging`` / ``redact_text`` - PII-safe structured logging;
* ``filter_final_output`` - the deterministic gate a draft must pass before it can reach a customer.

Block vs. escalate (final output gate)
--------------------------------------
``block``: the text itself is unsafe to show anyone outside the bank - unmasked identifiers,
email/phone numbers, an injection signature, or a claim that an action was completed when no
write tool performed it. The draft is withheld and replaced with a fixed safe notice.

``escalate``: the text may be harmless, but it is not proven - validation failed, claims are
unsupported, evidence citations are missing, or numbers in the draft are not in the evidence.
A human reviews it; the customer gets a fixed holding notice.
"""

from __future__ import annotations

import dataclasses
import html
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.logging_config import get_logger, trace

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Identifier formats (PDF section 4)
# ---------------------------------------------------------------------------

ID_FORMATS: dict[str, tuple[str, str]] = {
    "customer_id": (r"CUS-\d{5}", "CUS-XXXXX"),
    "account_id": (r"ACC-\d{5}", "ACC-XXXXX"),
    "policy_id": (r"POL-[A-Z]{2}-\d{5}", "POL-XX-XXXXX"),
    "portfolio_id": (r"PORT-\d{5}", "PORT-XXXXX"),
    "product_id": (r"PROD-[A-Z]{2}-\d{2}", "PROD-XX-XX"),
    "claim_id": (r"CLM-\d{5}", "CLM-XXXXX"),
}
_ID_FULL = {kind: re.compile(rf"^{pattern}$") for kind, (pattern, _) in ID_FORMATS.items()}

# Identifiers that must never appear unmasked in customer-facing text or logs.
SENSITIVE_ID_KINDS = ("customer_id", "account_id", "policy_id", "portfolio_id", "claim_id")
_SENSITIVE_ID_IN_TEXT = re.compile(
    r"\b(?:CUS-\d{5}|ACC-\d{5}|POL-[A-Z]{2}-\d{5}|PORT-\d{5}|CLM-\d{5})\b", re.IGNORECASE
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
# Phone shapes: +1-415-555-2001, (415) 555-2001, 415.555.2001, or 10-15 bare digits.
# ISO dates (2026-10-05) and timestamps deliberately do not match.
_PHONE = re.compile(
    r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])"
    r"|(?<![\w.*])\+?\d{10,15}(?![\w.])"
)
_MASKED_TOKEN = re.compile(r"\*{4}[A-Za-z0-9]{1,4}")


class GuardrailError(ValueError):
    """Base class for typed guardrail rejections. ``code`` is safe to return to callers."""

    code = "guardrail_violation"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class IdentifierFormatError(GuardrailError):
    code = "invalid_identifier"


class InputValidationError(GuardrailError):
    code = "invalid_input"


class PromptInjectionDetected(GuardrailError):
    """Untrusted text matched a deterministic injection signature."""

    code = "prompt_injection_detected"

    def __init__(self, signatures: list[str], *, field: str | None = None, source: str = "untrusted") -> None:
        super().__init__(f"{source} text matched injection signature(s): {', '.join(signatures)}", field=field)
        self.signatures = signatures
        self.source = source


class UnsupportedScopeError(GuardrailError):
    code = "unsupported_scope"


@trace(logger)
def is_valid_identifier(kind: str, value: Any) -> bool:
    if kind not in _ID_FULL:
        raise KeyError(f"Unknown identifier kind: {kind}")
    return isinstance(value, str) and _ID_FULL[kind].fullmatch(value) is not None


@trace(logger)
def validate_identifier(kind: str, value: Any, *, field: str | None = None) -> str:
    """Return ``value`` if it matches the exact format for ``kind``; raise a typed error otherwise."""
    if not is_valid_identifier(kind, value):
        raise IdentifierFormatError(
            f"{field or kind} must match the {ID_FORMATS[kind][1]} format", field=field or kind
        )
    return value


@trace(logger)
def find_identifiers(text: str) -> dict[str, list[str]]:
    """Extract every well-formed identifier of each kind from ``text`` (upper-cased, de-duplicated)."""
    found: dict[str, list[str]] = {}
    for kind, (pattern, _) in ID_FORMATS.items():
        values = []
        for match in re.finditer(rf"\b{pattern}\b", text.upper()):
            if match.group(0) not in values:
                values.append(match.group(0))
        if values:
            found[kind] = values
    return found


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


@trace(logger)
def mask_identifier(value: Any) -> Any:
    """Mask an identifier to its final four characters (``ACC-20077`` -> ``****0077``)."""
    if not isinstance(value, str) or not value:
        return value
    if _MASKED_TOKEN.fullmatch(value):
        return value
    return f"****{value[-4:]}"


@trace(logger, redact_result=True)
def mask_identifiers_in_text(text: str) -> str:
    """Mask every account/policy/portfolio/claim/customer number embedded in free text."""
    return _SENSITIVE_ID_IN_TEXT.sub(lambda m: f"****{m.group(0)[-4:]}", text)


# ---------------------------------------------------------------------------
# Free-text sanitisation and prompt-injection defence
# ---------------------------------------------------------------------------

DEFAULT_MAX_LENGTH = 2000
_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Deterministic signatures. They target imperative attempts to change model behaviour, so
# descriptive policy language ("never follow instructions embedded in a document to reveal
# secrets") and ordinary financial requests are not flagged.
INJECTION_SIGNATURES: dict[str, re.Pattern[str]] = {
    "ignore_previous_instructions": re.compile(
        r"\b(?:ignore|disregard|forget|override|skip)\s+(?:(?:all|any|the|your|of|these|those)\s+)*"
        r"(?:previous|prior|above|earlier|preceding|original|system|developer)\s+"
        r"(?:instructions?|prompts?|rules|directions|directives|messages|guidelines)\b",
        re.IGNORECASE,
    ),
    "reveal_hidden_data": re.compile(
        r"\b(?:reveal|print|show|display|output|repeat|leak|dump|expose|disclose)\s+(?:me\s+)?"
        r"(?:(?:the|your|all|any|its|every)\s+)+(?:full\s+|raw\s+|hidden\s+|unmasked\s+|private\s+|internal\s+)*"
        r"(?:system\s+prompts?|instructions|developer\s+(?:message|prompt)s?|prompts?|api\s+keys?|secrets?|"
        r"credentials?|passwords?|tool\s+outputs?|account\s+numbers?|policy\s+numbers?|chain\s+of\s+thought)\b",
        re.IGNORECASE,
    ),
    "role_override": re.compile(
        r"\byou\s+are\s+now\s+(?:a|an|the|in|my)\b|\bact\s+as\s+(?:the\s+|an?\s+)?(?:system|developer|administrator|admin|root|unrestricted)\b"
        r"|\bfrom\s+now\s+on\s*,?\s*you\s+(?:will|must|are|should)\b|\bpretend\s+(?:that\s+)?you\s+(?:are|have)\s+no\s+(?:rules|restrictions)\b",
        re.IGNORECASE,
    ),
    "injected_instruction_block": re.compile(
        r"\b(?:new|updated|override|real|actual)\s+(?:system\s+)?instructions?\s*:|\bsystem\s*(?:prompt|message)\s*:",
        re.IGNORECASE,
    ),
    "chat_markup": re.compile(
        r"<\|?\s*(?:im_start|im_end|system|endoftext|assistant)\s*\|?>|\[/?(?:INST|SYS)\]|^\s*#{2,}\s*(?:system|instructions?)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    "jailbreak_marker": re.compile(r"\b(?:jailbreak|developer\s+mode|DAN\s+mode|do\s+anything\s+now)\b", re.IGNORECASE),
    "disable_guardrails": re.compile(
        r"\b(?:disable|turn\s+off|deactivate|bypass|ignore)\s+(?:(?:the|all|your|any)\s+)*"
        r"(?:safety\s+(?:rules|checks|filters?)|guardrails?|content\s+filters?|output\s+filters?|security\s+(?:checks?|controls?))\b",
        re.IGNORECASE,
    ),
    "tool_invocation_request": re.compile(
        r"\b(?:call|invoke|execute|run|trigger)\s+(?:the\s+)?(?:send_customer_notification|write_audit_log|run_escalation_check)\b",
        re.IGNORECASE,
    ),
}


@trace(logger)
def detect_injection(text: str) -> list[str]:
    """Return the names of every injection signature found in ``text`` (empty when clean)."""
    if not isinstance(text, str):
        return []
    normalized = _normalize(text, keep_newlines=True)
    return [name for name, pattern in INJECTION_SIGNATURES.items() if pattern.search(normalized)]


def _normalize(text: str, *, keep_newlines: bool) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


@trace(logger, redact_result=True)
def strip_html(text: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _HTML_COMMENT.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return html.unescape(text)


@trace(logger, redact_result=True)
def sanitize_free_text(
    value: Any,
    *,
    field: str = "text",
    max_length: int = DEFAULT_MAX_LENGTH,
    source: str = "customer",
    allow_empty: bool = False,
    keep_newlines: bool = False,
) -> str:
    """Validate and clean untrusted text; raise a typed error instead of silently passing bad input.

    Applied to customer free text, every retrieved KB chunk and MCP text field before a model sees
    it, outbound notification bodies, and free-text Tool/Resource/Prompt arguments.

    Raises ``InputValidationError`` (wrong type, empty, too long) or ``PromptInjectionDetected``.
    """
    if not isinstance(value, str):
        raise InputValidationError(f"{field} must be a string", field=field)
    # HTML is decoded and stripped twice so entity-encoded tags (&lt;script&gt;) are also removed.
    cleaned = strip_html(strip_html(value))
    cleaned = _normalize(cleaned, keep_newlines=keep_newlines)
    cleaned = re.sub(r"[ 	]+([,.;:!?])", r"", cleaned)
    if not cleaned and not allow_empty:
        raise InputValidationError(f"{field} must not be empty", field=field)
    if len(cleaned) > max_length:
        raise InputValidationError(f"{field} exceeds the {max_length}-character limit", field=field)
    signatures = [name for name, pattern in INJECTION_SIGNATURES.items() if pattern.search(cleaned)]
    if signatures:
        raise PromptInjectionDetected(signatures, field=field, source=source)
    return cleaned


# ---------------------------------------------------------------------------
# Caller-scoped field minimisation (PDF section 4)
# ---------------------------------------------------------------------------

CALLER_SCOPES = ("customer", "authorized_agent", "internal_service")

# Fields each scope may see. Identifiers are always masked regardless of scope.
#   customer          - the authenticated customer viewing their own record.
#   authorized_agent  - the Nexus concierge acting for a verified customer: needs ownership
#                       (customer_id) and eligibility state, not balances.
#   internal_service  - back-office reconciliation; only honoured when the server's documented
#                       trust boundary allows it (see mcp_servers.common.scope_allowed).
ACCOUNT_SCOPE_FIELDS: dict[str, tuple[str, ...]] = {
    "customer": ("account_id", "account_type", "status", "auto_debit_allowed", "currency", "balance"),
    "authorized_agent": ("account_id", "customer_id", "account_type", "status", "auto_debit_allowed", "currency"),
    "internal_service": (
        "account_id", "customer_id", "account_type", "status", "auto_debit_allowed", "currency", "balance", "opened_at",
    ),
}
PORTFOLIO_SCOPE_FIELDS: dict[str, tuple[str, ...]] = {
    "customer": ("portfolio_id", "risk_profile", "status", "total_value"),
    "authorized_agent": ("portfolio_id", "customer_id", "risk_profile", "status"),
    "internal_service": ("portfolio_id", "customer_id", "risk_profile", "status", "total_value", "opened_at"),
}
TRANSACTION_SCOPE_FIELDS: dict[str, tuple[str, ...]] = {
    "customer": ("amount", "direction", "description", "occurred_at"),
    "authorized_agent": ("amount", "direction", "description", "occurred_at"),
    "internal_service": ("transaction_id", "amount", "direction", "description", "occurred_at"),
}
_MASKED_FIELDS = {"account_id", "policy_id", "portfolio_id", "claim_id"}


@trace(logger)
def validate_caller_scope(caller_scope: Any) -> str:
    if caller_scope not in CALLER_SCOPES:
        raise UnsupportedScopeError(
            f"caller_scope must be one of {', '.join(CALLER_SCOPES)}", field="caller_scope"
        )
    return caller_scope


@trace(logger)
def _minimize(record: Mapping[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in allowed:
        if name in record:
            value = record[name]
            output[name] = mask_identifier(value) if name in _MASKED_FIELDS else value
    return output


@trace(logger)
def minimize_account_fields(record: Mapping[str, Any], caller_scope: str) -> dict[str, Any]:
    """Return only the account fields ``caller_scope`` needs, with the account number masked."""
    return _minimize(record, ACCOUNT_SCOPE_FIELDS[validate_caller_scope(caller_scope)])


@trace(logger)
def minimize_portfolio_fields(record: Mapping[str, Any], caller_scope: str) -> dict[str, Any]:
    """Return only the portfolio fields ``caller_scope`` needs, with the portfolio number masked."""
    return _minimize(record, PORTFOLIO_SCOPE_FIELDS[validate_caller_scope(caller_scope)])


@trace(logger)
def minimize_transaction_fields(record: Mapping[str, Any], caller_scope: str) -> dict[str, Any]:
    minimized = _minimize(record, TRANSACTION_SCOPE_FIELDS[validate_caller_scope(caller_scope)])
    if "description" in minimized and isinstance(minimized["description"], str):
        minimized["description"] = minimize_free_text_pii(minimized["description"])[:300]
    return minimized


@trace(logger, redact_result=True)
def minimize_free_text_pii(text: str) -> str:
    """Mask identifiers and replace e-mail addresses / phone numbers inside stored free text."""
    text = mask_identifiers_in_text(text)
    text = _EMAIL.sub("[email]", text)
    return _PHONE.sub("[phone]", text)


# ---------------------------------------------------------------------------
# PII-safe logging
# ---------------------------------------------------------------------------

# Values under these keys are masked to the last four characters.
ID_LOG_KEYS = {
    "customer_id", "account_id", "policy_id", "portfolio_id", "claim_id",
    "funding_account_id", "linked_account_id",
}
# Values under these keys are personal data or free-form customer/retrieved content: they are
# replaced with a length marker so logs never carry the content under any key.
CONTENT_LOG_KEYS = {
    "phone", "email", "full_name", "customer_name", "first_name", "name_on_account",
    "message", "user_message", "customer_message", "customer_stated_intent", "raw_message",
    "description", "summary", "prior_interaction_summary", "recent_interaction_summary",
    "detail", "body", "content", "text", "excerpt", "chunk_text", "prompt", "draft",
    "retrieved_context_summary", "classified_intent", "payment_terms", "notes",
    "value", "query", "queries", "question", "statement", "evidence_texts", "grounding_summary",
    "model_message", "reasons_text", "kb_value", "mcp_value", "payload", "raw",
}
MAX_LOG_DEPTH = 6
MAX_LOG_ITEMS = 25
MAX_LOG_STRING = 300
# Any longer string is treated as content (customer text, retrieved text, prompts) and logged only
# as a length marker, whatever key it sits under. Short strings are codes, statuses and masked IDs.
MAX_INLINE_STRING = 64
LONG_STRING_SAFE_KEYS = {"function", "db_path", "path", "source_path", "url", "uri", "chunk_id", "model", "name"}


_PYDANTIC_INPUT = re.compile(r"""input_value=(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S+)""")


def redact_text(text: str) -> str:
    """Pattern-mask identifiers, e-mail addresses and phone numbers inside an arbitrary string, and
    drop the ``input_value=...`` echoes Pydantic puts in validation error messages."""
    if not isinstance(text, str):
        return text
    text = _PYDANTIC_INPUT.sub("input_value=<redacted>", text)
    text = _SENSITIVE_ID_IN_TEXT.sub(lambda m: f"****{m.group(0)[-4:]}", text)
    text = _EMAIL.sub("[email]", text)
    return _PHONE.sub("[phone]", text)


def redact_for_logging(value: Any, _depth: int = 0, _key: str | None = None) -> Any:
    """Recursively make ``value`` safe for a structured log line.

    Handles mappings, sequences, Pydantic models, dataclasses, exceptions and arbitrary objects.
    ID-shaped keys are masked, content keys are replaced with a length marker, every other string
    is pattern-masked and truncated, and containers are bounded in depth and size. Supports the
    data-minimisation principle: logs prove *that* a call happened and with which shape, without
    retaining the customer's data. (Not traced: it is the tracer's own redactor.)
    """
    key = (_key or "").lower()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if key in CONTENT_LOG_KEYS:
            return f"<redacted:{len(value)} chars>"
        if key in ID_LOG_KEYS:
            return mask_identifier.__wrapped__(value)  # type: ignore[attr-defined]
        text = redact_text(value)
        if len(text) > MAX_INLINE_STRING and key not in LONG_STRING_SAFE_KEYS:
            return f"<redacted:{len(value)} chars>"
        return text if len(text) <= MAX_LOG_STRING else text[:MAX_LOG_STRING] + "...<truncated>"
    if _depth >= MAX_LOG_DEPTH:
        return f"<{type(value).__name__}: depth limit>"
    if key in CONTENT_LOG_KEYS and isinstance(value, (Mapping, list, tuple)):
        return f"<redacted:{type(value).__name__}>"
    if isinstance(value, BaseException):
        return {"exception_type": type(value).__name__, "message": redact_text(str(value))[:MAX_LOG_STRING]}
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return redact_for_logging(value.model_dump(mode="json"), _depth + 1, _key)
        except Exception:
            return f"<{type(value).__name__}>"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        data = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        return redact_for_logging(data, _depth + 1, _key)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for index, (k, v) in enumerate(value.items()):
            if index >= MAX_LOG_ITEMS:
                output["<truncated>"] = f"{len(value) - MAX_LOG_ITEMS} more keys"
                break
            output[str(k)] = redact_for_logging(v, _depth + 1, str(k))
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        redacted = [redact_for_logging(v, _depth + 1, _key) for v in items[:MAX_LOG_ITEMS]]
        if len(items) > MAX_LOG_ITEMS:
            redacted.append(f"<{len(items) - MAX_LOG_ITEMS} more items>")
        return redacted
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return f"<{type(value).__name__}>"


# ---------------------------------------------------------------------------
# Final output filter
# ---------------------------------------------------------------------------

# A draft must never claim that a write happened: the tool inventory has no payment-setup tool.
COMPLETED_ACTION_CLAIMS = re.compile(
    r"\b(?:i|we)\s*(?:'ve|have|has)?\s*(?:now\s+)?(?:set\s+up|activated|enabled|enrolled|registered|scheduled|processed|"
    r"cancelled|canceled|changed|updated)\b"
    r"|\b(?:auto-?debit|payment|direct\s+debit|mandate)\s+(?:has\s+been|is\s+now|was)\s+(?:set\s+up|activated|enabled|scheduled|processed)\b"
    r"|\byou\s+are\s+now\s+enrolled\b",
    re.IGNORECASE,
)
GUARANTEE_LANGUAGE = re.compile(r"\b(?:guaranteed?|risk-free|cannot\s+lose|will\s+definitely)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w*])\$?\d[\d,]*(?:\.\d+)?%?")


@trace(logger)
def find_unmasked_identifiers(text: str) -> list[str]:
    """Return masked renderings of each unmasked sensitive identifier in ``text``."""
    return sorted({f"****{m.group(0)[-4:]}" for m in _SENSITIVE_ID_IN_TEXT.finditer(text)})


@trace(logger)
def ungrounded_numbers(text: str, evidence_texts: Iterable[str]) -> list[str]:
    """Numbers/amounts/percentages in ``text`` that never appear in any evidence text.

    Masked identifiers (``****0077``) and ordinals ``1.``-style list markers are ignored.
    """
    corpus = " ".join(evidence_texts)
    # Digits of masked identifiers count as evidence, so "ending 0077" matches "****0077".
    corpus_values = {_numeric(n) for n in _NUMBER.findall(corpus) + [m.group(0)[4:] for m in _MASKED_TOKEN.finditer(corpus)]}
    missing = []
    for raw in _NUMBER.findall(_MASKED_TOKEN.sub(" ", text)):
        value = _numeric(raw)
        if value is None or value in corpus_values:
            continue
        missing.append(raw)
    return sorted(set(missing))


@trace(logger)
def _numeric(raw: str) -> float | None:
    """'$1,250.00' -> 1250.0 so equivalent renderings of one amount compare equal."""
    cleaned = raw.strip("$%").replace(",", "").rstrip(".")
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


@dataclass
class OutputDecision:
    action: str  # "send" | "block" | "escalate"
    reasons: list[str] = field(default_factory=list)

    @property
    def sendable(self) -> bool:
        return self.action == "send"


@trace(logger)
def filter_final_output(
    message: str,
    *,
    cited_evidence_ids: Iterable[str],
    known_evidence_ids: Iterable[str],
    evidence_texts: Iterable[str],
    validation_passed: bool,
    unsupported_claims: Iterable[str] = (),
    customer_specific: bool = True,
) -> OutputDecision:
    """Deterministic last gate between a validated draft and the customer.

    A model saying it is grounded is not evidence; this gate checks the text itself.
    """
    block: list[str] = []
    escalate: list[str] = []
    if not isinstance(message, str) or not message.strip():
        return OutputDecision("block", ["empty_message"])
    if find_unmasked_identifiers(message):
        block.append("unmasked_identifier")
    if _EMAIL.search(message) or _PHONE.search(_MASKED_TOKEN.sub(" ", message)):
        block.append("contact_details_in_output")
    if detect_injection(message):
        block.append("injection_signature_in_output")
    if COMPLETED_ACTION_CLAIMS.search(message):
        block.append("claims_completed_action_without_write_tool")
    if GUARANTEE_LANGUAGE.search(message):
        escalate.append("guarantee_language")

    cited = set(cited_evidence_ids)
    known = set(known_evidence_ids)
    if customer_specific and not cited:
        escalate.append("no_evidence_cited")
    if cited - known:
        escalate.append("cites_unknown_evidence")
    if ungrounded_numbers(message, list(evidence_texts)):
        escalate.append("numbers_not_in_evidence")
    if not validation_passed:
        escalate.append("validation_failed")
    if list(unsupported_claims):
        escalate.append("unsupported_claims")

    if block:
        return OutputDecision("block", block + escalate)
    if escalate:
        return OutputDecision("escalate", escalate)
    return OutputDecision("send", [])
