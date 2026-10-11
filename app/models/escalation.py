from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .intake import EscalationCategory


class EscalationReason(str, Enum):
    NONE = "none"
    LOW_CONFIDENCE = "low_confidence"
    HARD_CATEGORY = "hard_category"
    VALIDATION_FAILURE = "validation_failure"
    MCP_FAILURE = "mcp_failure"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    OUTPUT_BLOCKED = "output_blocked"


class EscalationResult(BaseModel):
    """
    Deterministic escalation decision returned after validation.

    Hard escalation categories must override confidence. ``server_reason`` is exactly what
    concierge_ops.run_escalation_check returned; ``reason`` may be stricter when the orchestrator's
    own gate (validation failure, blocked output) also requires a human.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    customer_id: str  # masked

    should_escalate: bool

    category: EscalationCategory

    reason: EscalationReason
    server_reason: str

    draft_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    hard_escalation_triggered: bool = False

    decision_source: str = "concierge_ops.run_escalation_check"
    decision_id: str | None = None
    confidence_threshold: float | None = None

    explanation: str

    human_action_required: bool = False
