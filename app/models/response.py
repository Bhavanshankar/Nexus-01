from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .escalation import EscalationResult
from .validation import ValidationResult


class ResponseStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    ESCALATED = "escalated"
    BLOCKED = "blocked"
    FAILED = "failed"
    REJECTED = "rejected"


class AssertedFact(BaseModel):
    """A customer-specific fact the draft states, tied to an MCP fact ID for checking."""

    model_config = ConfigDict(extra="ignore")

    fact_id: str
    value: str | int | float | bool | None = None


class DraftProposal(BaseModel):
    """Untrusted model output for the Drafting role."""

    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1, max_length=5000)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    asserted_facts: list[AssertedFact] = Field(default_factory=list)
    proposed_next_step: str | None = None


class DraftResponse(BaseModel):
    """
    Customer-facing response proposed by the Drafting Agent.

    The draft must be grounded in GroundedContext.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str

    message: str = Field(
        min_length=1,
        max_length=5000,
    )

    evidence_ids: list[str] = Field(
        default_factory=list
    )

    asserted_facts: list[AssertedFact] = Field(default_factory=list)

    proposed_action: str | None = None

    holding_reply: bool = False

    status: ResponseStatus = ResponseStatus.DRAFT


class PipelineErrorCode(str, Enum):
    INVALID_INTAKE = "invalid_intake"
    PROMPT_INJECTION = "prompt_injection"
    UNKNOWN_CUSTOMER = "unknown_customer"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MCP_TOOL_ERROR = "mcp_tool_error"
    OWNERSHIP_MISMATCH = "ownership_mismatch"
    RETRIEVAL_INCOMPLETE = "retrieval_incomplete"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    ESCALATION_UNAVAILABLE = "escalation_unavailable"
    INVARIANT_VIOLATION = "invariant_violation"
    INTERNAL_ERROR = "internal_error"


class PipelineError(BaseModel):
    """Typed failure surfaced through the pipeline instead of a crash or a guessed answer."""

    model_config = ConfigDict(extra="forbid")

    code: PipelineErrorCode
    stage: str
    message: str
    server: str | None = None
    tool: str | None = None
    detail_code: str | None = None
    retryable: bool = False


class StepRecord(BaseModel):
    """One observable pipeline step (no customer text, identifiers masked)."""

    model_config = ConfigDict(extra="forbid")

    step: str
    role: str
    status: str
    duration_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class MCPCallRecord(BaseModel):
    """Evidence that an MCP primitive was actually invoked (recorded by the gateway, not the model)."""

    model_config = ConfigDict(extra="forbid")

    server: str
    primitive: str  # tool | resource | prompt
    name: str
    ok: bool
    error_code: str | None = None
    duration_ms: float = 0.0


class FinalResponse(BaseModel):
    """
    Final response produced by the Nexus orchestration pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    customer_id: str | None  # masked

    message: str

    status: ResponseStatus

    validation: ValidationResult | None = None

    escalation: EscalationResult | None = None

    error: PipelineError | None = None

    can_be_sent_to_customer: bool

    next_action: str | None = None

    evidence_ids: list[str] = Field(default_factory=list)

    conflicts: list[str] = Field(default_factory=list)

    blocked_sources: list[str] = Field(default_factory=list)

    output_gate: dict[str, Any] = Field(default_factory=dict)

    notification: dict[str, Any] | None = None

    audit: dict[str, Any] | None = None

    steps: list[StepRecord] = Field(default_factory=list)

    mcp_calls: list[MCPCallRecord] = Field(default_factory=list)
