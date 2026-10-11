from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: ValidationSeverity

    message: str

    evidence_reference: str | None = None

    blocks_response: bool = False


class ValidationReview(BaseModel):
    """Untrusted model review (tone, compliance, factual support). Never used alone: the
    Validation Agent combines it with deterministic checks and takes the lower confidence."""

    model_config = ConfigDict(extra="ignore")

    factually_supported: bool
    tone_appropriate: bool = True
    compliance_ok: bool = True
    # Specific compliance questions; any True is a blocking concern.
    recommends_financial_product: bool = False
    guarantees_outcome: bool = False
    claims_action_performed: bool = False
    acknowledges_conflicts: bool = True
    unsupported_claims: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """
    Result produced by the Validation Agent.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str

    passed: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    issues: list[ValidationIssue] = Field(
        default_factory=list
    )

    unsupported_claims: list[str] = Field(
        default_factory=list
    )

    pii_detected: bool = False

    conflicts_detected: bool = False

    conflicts_acknowledged: bool = True

    grounding_valid: bool = True

    compliance_valid: bool = True

    tone_valid: bool = True

    deterministic_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    checks: dict[str, bool] = Field(default_factory=dict)
