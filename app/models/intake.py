"""Structured intake contracts (PDF section 3.2).

``CustomerRequest`` is what a channel submits: the free-text message plus trusted request metadata
(the authenticated customer ID and any object IDs the customer selected in the app). It cannot
carry a request ID or an escalation category - those are produced by Nexus.

``ClassificationProposal`` is the model's raw extraction. It is never trusted directly: the
Classifier merges it with deterministic signals and validates the result into ``IntakeRequest``.
"""

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.guardrails import ID_FORMATS, is_valid_identifier


class ProductLine(str, Enum):
    BANKING = "banking"
    INSURANCE = "insurance"
    WEALTH = "wealth"
    CROSS_PRODUCT = "cross_product"


class RequestType(str, Enum):
    BALANCE_INQUIRY = "balance_inquiry"
    TRANSACTION_HISTORY = "transaction_history"
    LINKED_ACCOUNTS = "linked_accounts"

    AUTO_DEBIT_SETUP = "auto_debit_setup"
    PAYMENT_TERMS = "payment_terms"
    COVERAGE_QUESTION = "coverage_question"
    CLAIM_STATUS = "claim_status"

    PORTFOLIO_QUERY = "portfolio_query"
    INVESTMENT_PRODUCT_QUERY = "investment_product_query"
    SUITABILITY_CHECK = "suitability_check"

    COMPLAINT = "complaint"
    OTHER = "other"


class Urgency(str, Enum):
    ROUTINE = "routine"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationCategory(str, Enum):
    NONE = "none"
    FINANCIAL_HARDSHIP = "financial_hardship"
    SAFEGUARDING_CONCERN = "safeguarding_concern"
    COMPLIANCE_OVERRIDE_REQUEST = "compliance_override_request"


HARD_CATEGORIES = frozenset({
    EscalationCategory.FINANCIAL_HARDSHIP,
    EscalationCategory.SAFEGUARDING_CONCERN,
    EscalationCategory.COMPLIANCE_OVERRIDE_REQUEST,
})


def _check_id(kind: str, value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().upper()
    if not is_valid_identifier(kind, value):
        raise ValueError(f"{kind} must use the {ID_FORMATS[kind][1]} format")
    return value


class ReferencedObjects(BaseModel):
    """Object identifiers the request concerns; each must match its exact format."""

    model_config = ConfigDict(extra="forbid")

    account_id: str | None = None
    policy_id: str | None = None
    portfolio_id: str | None = None
    product_id: str | None = None
    claim_id: str | None = None

    @field_validator("account_id", "policy_id", "portfolio_id", "product_id", "claim_id", mode="before")
    @classmethod
    def _validate(cls, value, info):
        return _check_id(info.field_name, value)

    def present(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v}


class CustomerRequest(BaseModel):
    """Inbound request as submitted by a channel. ``extra='forbid'`` rejects any attempt to supply
    a request ID, escalation category or other pipeline-owned field."""

    model_config = ConfigDict(extra="forbid")

    message: str
    customer_id: str | None = None
    references: ReferencedObjects = Field(default_factory=ReferencedObjects)
    channel: str = Field(default="chat", pattern=r"^(chat|phone|email|sms|branch)$")


class ClassificationProposal(BaseModel):
    """Untrusted model output for the Classifier role."""

    model_config = ConfigDict(extra="ignore")

    product_line: ProductLine
    request_type: RequestType
    urgency: Urgency
    customer_stated_intent: str = Field(min_length=1, max_length=500)
    escalation_category: EscalationCategory = EscalationCategory.NONE


class IntakeRequest(BaseModel):
    """
    Structured representation of a customer's inbound request.

    This is the contract produced by the Classifier Agent and validated
    before any retrieval or MCP operation is allowed.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        default_factory=lambda: f"REQ-{uuid4().hex[:12].upper()}",
        pattern=r"^REQ-[A-Z0-9]{12}$",
    )

    customer_id: str

    product_line: ProductLine
    request_type: RequestType
    urgency: Urgency

    customer_stated_intent: str = Field(
        min_length=1,
        max_length=2000,
    )

    escalation_category: EscalationCategory = EscalationCategory.NONE

    referenced_ids: ReferencedObjects = Field(default_factory=ReferencedObjects)

    # Deterministic signal names that set or confirmed escalation_category (no customer text).
    escalation_signals: list[str] = Field(default_factory=list)

    customer_tier: str = "standard"

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, value: str) -> str:
        if not is_valid_identifier("customer_id", value):
            raise ValueError("customer_id must use CUS-XXXXX format")
        return value

    @field_validator("customer_stated_intent")
    @classmethod
    def validate_intent(cls, value: str) -> str:
        from app.guardrails import mask_identifiers_in_text, sanitize_free_text

        return sanitize_free_text(mask_identifiers_in_text(value), field="customer_stated_intent", max_length=2000)

    @model_validator(mode="after")
    def _cross_product_consistency(self):
        if self.escalation_category in HARD_CATEGORIES and not self.escalation_signals:
            self.escalation_signals = [f"model:{self.escalation_category.value}"]
        return self

    @property
    def is_hard_category(self) -> bool:
        return self.escalation_category in HARD_CATEGORIES


class IntakeErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    MISSING_CUSTOMER_ID = "missing_customer_id"
    MALFORMED_CUSTOMER_ID = "malformed_customer_id"
    CONFLICTING_CUSTOMER_ID = "conflicting_customer_id"
    MALFORMED_REFERENCE = "malformed_reference"
    CONFLICTING_REFERENCE = "conflicting_reference"
    UNKNOWN_CUSTOMER = "unknown_customer"
    PROMPT_INJECTION = "prompt_injection"
    INVALID_TEXT = "invalid_text"
    INCOMPLETE_EXTRACTION = "incomplete_extraction"


class IntakeError(BaseModel):
    """Typed intake rejection. ``message`` is safe to show; it never echoes customer text."""

    model_config = ConfigDict(extra="forbid")

    code: IntakeErrorCode
    message: str
    field: str | None = None
