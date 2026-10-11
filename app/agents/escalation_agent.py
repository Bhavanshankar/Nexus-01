"""Escalation role: obtain and verify the deterministic decision from Concierge Ops MCP.

The decision comes from the real ``run_escalation_check`` tool call made through the gateway (which
records it); no model participates, so a model cannot claim it "called the tool". The returned
decision is then verified: it must echo the request, and a hard category must escalate. A tool error
or unreachable server raises ``EscalationUnavailable`` - it never becomes a successful decision.
"""
from __future__ import annotations

from app.guardrails import mask_identifier
from app.logging_config import get_logger, trace
from app.mcp_client.gateway import MCPGateway, MCPPrimitiveError, MCPUnavailable
from app.models.escalation import EscalationReason, EscalationResult
from app.models.intake import EscalationCategory, IntakeRequest

logger = get_logger(__name__)


class EscalationUnavailable(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(f"escalation decision unavailable: {code}")
        self.code = code


class EscalationInvariantViolation(RuntimeError):
    """The decision tool's answer contradicts the hard-escalation rule or the request."""


class EscalationAgent:
    """Role 5 of 5: confidence-based and hard category-based escalation via MCP."""

    @trace(logger)
    def __init__(self, gateway: MCPGateway) -> None:
        self.gateway = gateway

    @trace(logger)
    def decide(self, intake: IntakeRequest, draft_confidence: float, *, validation_passed: bool = True) -> EscalationResult:
        category = intake.escalation_category.value
        confidence = round(max(0.0, min(1.0, float(draft_confidence))), 3)
        try:
            envelope = self.gateway.call_tool("run_escalation_check", {
                "customer_id": intake.customer_id,
                "request_category": category,
                "draft_confidence": confidence,
            })
        except (MCPUnavailable, MCPPrimitiveError):
            raise EscalationUnavailable("service_unavailable") from None
        if not envelope.get("ok"):
            raise EscalationUnavailable(str((envelope.get("error") or {}).get("code", "tool_error")))

        data = envelope["data"]
        if data.get("customer_id") != intake.customer_id or data.get("request_category") != category:
            raise EscalationInvariantViolation("decision does not match the request")
        if intake.is_hard_category and (data.get("should_escalate") is not True or data.get("hard_escalation_triggered") is not True):
            raise EscalationInvariantViolation("hard category returned without escalation")

        server_reason = str(data.get("reason", "none"))
        should_escalate = bool(data["should_escalate"])
        reason = EscalationReason(server_reason) if server_reason in {r.value for r in EscalationReason} else EscalationReason.MANUAL_REVIEW_REQUIRED
        if not should_escalate and not validation_passed:
            should_escalate, reason = True, EscalationReason.VALIDATION_FAILURE
        explanation = {
            EscalationReason.HARD_CATEGORY: f"'{category}' always requires a human, regardless of confidence.",
            EscalationReason.LOW_CONFIDENCE: f"Confidence {confidence} is below the server threshold {data.get('confidence_threshold')}.",
            EscalationReason.VALIDATION_FAILURE: "The draft failed validation, so it cannot be sent automatically.",
            EscalationReason.NONE: "No hard category and confidence meets the threshold.",
        }.get(reason, "Manual review required.")
        return EscalationResult(
            request_id=intake.request_id,
            customer_id=mask_identifier(intake.customer_id),
            should_escalate=should_escalate,
            category=EscalationCategory(category),
            reason=reason,
            server_reason=server_reason,
            draft_confidence=confidence,
            hard_escalation_triggered=bool(data.get("hard_escalation_triggered")),
            decision_id=data.get("decision_id"),
            confidence_threshold=data.get("confidence_threshold"),
            explanation=explanation,
            human_action_required=should_escalate,
        )
