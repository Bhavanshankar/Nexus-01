"""Nexus orchestrator: one customer request through intake -> retrieval -> five agents -> MCP.

Callable interface::

    orchestrator = build_production_orchestrator()        # HTTP MCP clients + CrewAI + Chroma
    final = orchestrator.handle({"message": "...", "customer_id": "CUS-20077",
                                 "references": {"account_id": "ACC-20077", "policy_id": "POL-IN-30091"}})

Runnable process (an MCP *client* of the four servers)::

    python -m nexus_orchestrator serve            # HTTP API on 127.0.0.1:8100 (POST /v1/requests, GET /health)
    python -m nexus_orchestrator ask --customer-id CUS-20077 --account-id ACC-20077 --policy-id POL-IN-30091 "message"
    python -m nexus_orchestrator check            # list each server's Tools / Resources / Prompts

Call path (every step is recorded in ``FinalResponse.steps``; every MCP call in ``mcp_calls``):

1. intake gate (deterministic)      - sanitise, validate IDs; reject with zero tool calls
2. identity check (MCP Resource)    - customer:// profile; unknown -> reject with zero tool calls
3. ClassifierAgent                  - request_classification_prompt (MCP) + model + hard signals
   KYC check (first MCP tool call)   - get_customer_kyc_status, only after intake is validated
4. RetrieverAgent                   - KB search + live MCP facts + ownership + reconciliation
5. DraftingAgent                    - response_drafting_prompt (MCP) + model
6. ValidationAgent                  - deterministic checks + model review -> confidence
7. EscalationAgent                  - run_escalation_check (MCP), verified
8. final output gate (deterministic)- guardrails.filter_final_output
9. delivery + audit (MCP writes)    - send_customer_notification (only after the gate), write_audit_log

This module never opens SQLite and never imports MCP server code: all backend data arrives through
an ``MCPGateway``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from app.agents import ClassifierAgent, DraftingAgent, EscalationAgent, RetrieverAgent, ValidationAgent
from app.agents._shared import AgentStepError
from app.agents.escalation_agent import EscalationInvariantViolation, EscalationUnavailable
from app.agents.llm import LLMBackend
from app.guardrails import filter_final_output, mask_identifier
from app.intake.gate import IntakeRejected, validate_inbound
from app.logging_config import current_request_id, get_logger, trace
from app.mcp_client.gateway import MCPGateway, MCPPrimitiveError, MCPUnavailable
from app.models.escalation import EscalationReason, EscalationResult
from app.models.intake import CustomerRequest, IntakeError, IntakeErrorCode, IntakeRequest
from app.models.response import (
    DraftResponse,
    FinalResponse,
    PipelineError,
    PipelineErrorCode,
    ResponseStatus,
    StepRecord,
)
from app.models.retrieval import GroundedContext
from app.models.validation import ValidationResult
from app.retrieval.hybrid import HybridRetriever, OwnershipMismatch

logger = get_logger("nexus_orchestrator")

HOLDING_MESSAGE = (
    "Thank you{name}. Your request needs a Meridian specialist, who will review it personally and contact you. "
    "Nothing has been changed on your accounts or policies."
)
BLOCKED_MESSAGE = (
    "Thank you{name}. We couldn't complete an automated answer for this request, so a Meridian specialist "
    "will review it and contact you."
)
UNAVAILABLE_MESSAGE = (
    "Sorry{name} - we couldn't verify your {what} records right now, so we can't answer this safely. "
    "Please try again shortly; a specialist can also help."
)
SERVER_LABELS = {"banking": "banking", "insurance": "insurance", "wealth": "investment", "concierge_ops": "customer", "knowledge_base": "policy guidance"}


@trace(logger)
def new_request_id() -> str:
    """Server-generated request identifier (never supplied by the caller)."""
    return f"REQ-{uuid.uuid4().hex[:12].upper()}"


class _Run:
    """Per-request working state: step records and the pieces of the final response."""

    @trace(logger)
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.steps: list[StepRecord] = []
        self.customer_id: str | None = None
        self.first_name: str | None = None
        self.kyc_status: str | None = None
        self.channel = "chat"

    @trace(logger)
    def step(self, step: str, role: str, started: float, status: str, **detail: Any) -> None:
        self.steps.append(StepRecord(
            step=step, role=role, status=status,
            duration_ms=round((time.perf_counter() - started) * 1000, 3), detail=detail,
        ))

    @trace(logger)
    def greeting(self) -> str:
        return f", {self.first_name}" if self.first_name else ""


class NexusOrchestrator:
    """Explicit controller over the five roles with injected model, retrieval and MCP interfaces."""

    @trace(logger)
    def __init__(
        self,
        *,
        gateway: MCPGateway,
        llm: LLMBackend,
        kb_search: Any,
        test_injections: Iterable[str] = (),
        deliver_notifications: bool = True,
        audit: bool = True,
    ) -> None:
        self.gateway = gateway
        self.llm = llm
        self.classifier = ClassifierAgent(llm, gateway)
        self.retriever = RetrieverAgent(HybridRetriever(kb_search, gateway, test_injections=test_injections))
        self.drafter = DraftingAgent(llm, gateway)
        self.validator = ValidationAgent(llm)
        self.escalator = EscalationAgent(gateway)
        self.deliver_notifications = deliver_notifications
        self.audit_enabled = audit
        self._lock = threading.Lock()  # one request at a time per gateway (call records are per request)

    # ------------------------------------------------------------------------------- entry point

    @trace(logger)
    def handle(self, request: CustomerRequest | Mapping[str, Any]) -> FinalResponse:
        """Process one request end to end. Never raises: every failure becomes a typed response."""
        with self._lock:
            run = _Run(new_request_id())
            token = current_request_id.set(run.request_id)
            self.gateway.reset_calls()
            try:
                return self._pipeline(run, request)
            except Exception:  # last resort: the trace decorator has logged the redacted traceback
                return self._final(run, ResponseStatus.FAILED, BLOCKED_MESSAGE.format(name=run.greeting()), error=PipelineError(
                    code=PipelineErrorCode.INTERNAL_ERROR, stage="orchestrator", message="The request could not be completed."))
            finally:
                current_request_id.reset(token)

    # --------------------------------------------------------------------------------- pipeline

    @trace(logger)
    def _pipeline(self, run: _Run, request: CustomerRequest | Mapping[str, Any]) -> FinalResponse:
        # 1. Intake gate - deterministic, no tool calls.
        started = time.perf_counter()
        try:
            inbound = validate_inbound(request)
        except IntakeRejected as rejected:
            run.step("intake_gate", "deterministic", started, "rejected", code=rejected.error.code.value)
            return self._rejected(run, rejected.error)
        run.customer_id, run.channel = inbound.customer_id, inbound.channel
        run.step("intake_gate", "deterministic", started, "passed", references=sorted(inbound.references.present()))

        # 2. Identity check - one read of the customer:// Resource (not a tool call). Intake must be
        #    validated before any tool runs (PDF 3.2), so an unknown customer is rejected here with
        #    zero tool calls; the profile also supplies tier/first name/prior interaction to the prompt.
        started = time.perf_counter()
        try:
            profile_envelope = self.gateway.read_resource(f"customer://{inbound.customer_id}/profile")
        except (MCPUnavailable, MCPPrimitiveError):
            run.step("identity_check", "mcp_resource", started, "failed", code="service_unavailable")
            return self._unavailable(run, "concierge_ops", "customer://{customer_id}/profile", "identity_check", "service_unavailable", None)
        if not profile_envelope.get("ok"):
            code = (profile_envelope.get("error") or {}).get("code")
            run.step("identity_check", "mcp_resource", started, "rejected", code=code)
            if code == "not_found":
                return self._rejected(run, IntakeError(code=IntakeErrorCode.UNKNOWN_CUSTOMER, message="We could not find a customer with that ID.", field="customer_id"))
            return self._unavailable(run, "concierge_ops", "customer://{customer_id}/profile", "identity_check", code or "resource_error", None)
        profile = profile_envelope.get("data") or {}
        run.first_name = profile.get("first_name")
        run.step("identity_check", "mcp_resource", started, "passed", tier=profile.get("customer_tier"))

        # 3. Classifier.
        started = time.perf_counter()
        try:
            intake = self.classifier.classify(inbound, request_id=run.request_id, profile=profile)
        except IntakeRejected as rejected:
            run.step("classify", "ClassifierAgent", started, "rejected", code=rejected.error.code.value)
            return self._rejected(run, rejected.error)
        except (MCPUnavailable, MCPPrimitiveError):
            run.step("classify", "ClassifierAgent", started, "failed", code="service_unavailable")
            return self._unavailable(run, "concierge_ops", "request_classification_prompt", "classify", "service_unavailable", None)
        run.step("classify", "ClassifierAgent", started, "ok", product_line=intake.product_line.value,
                 request_type=intake.request_type.value, urgency=intake.urgency.value,
                 escalation_category=intake.escalation_category.value, signals=intake.escalation_signals)

        # 3b. KYC status - first tool call, made only after the intake schema is validated. The KYC
        #     rule itself is enforced server-side by send_customer_notification.
        started = time.perf_counter()
        try:
            kyc = self.gateway.call_tool("get_customer_kyc_status", {"customer_id": intake.customer_id})
        except (MCPUnavailable, MCPPrimitiveError):
            run.step("kyc_check", "mcp", started, "failed", code="service_unavailable")
            return self._unavailable(run, "concierge_ops", "get_customer_kyc_status", "kyc_check", "service_unavailable", intake)
        if not kyc.get("ok"):
            code = (kyc.get("error") or {}).get("code") or "tool_error"
            run.step("kyc_check", "mcp", started, "failed", code=code)
            return self._unavailable(run, "concierge_ops", "get_customer_kyc_status", "kyc_check", code, intake)
        run.kyc_status = kyc["data"]["kyc_status"]
        run.step("kyc_check", "mcp", started, "ok", kyc_status=run.kyc_status)

        # 4. Retriever.
        started = time.perf_counter()
        try:
            context = self.retriever.retrieve(intake)
        except OwnershipMismatch as mismatch:
            run.step("retrieve", "RetrieverAgent", started, "failed", code="ownership_mismatch", record=mismatch.key)
            error = PipelineError(code=PipelineErrorCode.OWNERSHIP_MISMATCH, stage="retrieve", server=mismatch.server, tool=mismatch.tool,
                                  message="A referenced record does not belong to this customer.")
            escalation = self._try_escalate(run, intake, 0.0)
            return self._final(run, ResponseStatus.FAILED, BLOCKED_MESSAGE.format(name=run.greeting()), intake=intake,
                               error=error, escalation=escalation)
        run.step("retrieve", "RetrieverAgent", started, "complete" if context.retrieval_complete else "incomplete",
                 kb_status=context.kb_quality.status if context.kb_quality else None,
                 evidence=[e.evidence_id for e in context.evidence], conflicts=[c.conflict_id for c in context.conflicts],
                 blocked_sources=[b.source_reference for b in context.blocked_sources],
                 service_errors=[f"{e.server}.{e.tool}:{e.code}" for e in context.service_errors],
                 missing=context.missing_requirements)
        if not context.retrieval_complete:
            return self._incomplete(run, intake, context)

        # 5. Drafting.
        started = time.perf_counter()
        try:
            draft = self.drafter.draft(intake, context, customer_name=run.first_name)
        except (AgentStepError, MCPUnavailable, MCPPrimitiveError) as exc:
            run.step("draft", "DraftingAgent", started, "failed", code=getattr(exc, "code", "service_unavailable"))
            escalation = self._try_escalate(run, intake, 0.0)
            return self._final(run, ResponseStatus.FAILED, BLOCKED_MESSAGE.format(name=run.greeting()), intake=intake, context=context,
                               escalation=escalation, error=PipelineError(code=PipelineErrorCode.MODEL_OUTPUT_INVALID, stage="draft",
                                                                          message="A safe draft could not be produced."))
        run.step("draft", "DraftingAgent", started, "ok", holding_reply=draft.holding_reply, cited=draft.evidence_ids,
                 asserted_facts=[a.fact_id for a in draft.asserted_facts])

        # 6. Validation.
        started = time.perf_counter()
        validation = self.validator.validate(draft, context, intake)
        run.step("validate", "ValidationAgent", started, "passed" if validation.passed else "failed", confidence=validation.confidence,
                 deterministic_confidence=validation.deterministic_confidence, model_confidence=validation.model_confidence,
                 issues=[i.code for i in validation.issues], conflicts_seen=len(context.conflicts))

        # 7. Escalation - the deterministic MCP decision, verified.
        started = time.perf_counter()
        try:
            escalation = self.escalator.decide(intake, validation.confidence, validation_passed=validation.passed)
        except EscalationUnavailable as exc:
            run.step("escalate", "EscalationAgent", started, "failed", code=exc.code)
            return self._final(run, ResponseStatus.FAILED, UNAVAILABLE_MESSAGE.format(name=run.greeting(), what="customer"), intake=intake,
                               context=context, validation=validation, error=PipelineError(
                                   code=PipelineErrorCode.ESCALATION_UNAVAILABLE, stage="escalate", server="concierge_ops",
                                   tool="run_escalation_check", detail_code=exc.code, retryable=True,
                                   message="The escalation decision service is unavailable; nothing was sent."))
        except EscalationInvariantViolation:
            run.step("escalate", "EscalationAgent", started, "failed", code="invariant_violation")
            return self._final(run, ResponseStatus.ESCALATED, HOLDING_MESSAGE.format(name=run.greeting()), intake=intake, context=context,
                               validation=validation, error=PipelineError(code=PipelineErrorCode.INVARIANT_VIOLATION, stage="escalate",
                                                                          message="The escalation decision failed verification; routed to a human."))
        run.step("escalate", "EscalationAgent", started, "escalate" if escalation.should_escalate else "no_escalation",
                 reason=escalation.reason.value, server_reason=escalation.server_reason, decision_id=escalation.decision_id,
                 hard=escalation.hard_escalation_triggered)

        # 8. Final output gate.
        started = time.perf_counter()
        if escalation.should_escalate:
            run.step("output_gate", "deterministic", started, "skipped_escalated")
            return self._final(run, ResponseStatus.ESCALATED, HOLDING_MESSAGE.format(name=run.greeting()), intake=intake, context=context,
                               validation=validation, escalation=escalation, draft=draft)
        decision = filter_final_output(
            draft.message,
            cited_evidence_ids=draft.evidence_ids,
            known_evidence_ids=context.evidence_ids(),
            evidence_texts=[e.content for e in context.evidence],
            validation_passed=validation.passed,
            unsupported_claims=validation.unsupported_claims,
            customer_specific=context.customer_specific,
        )
        run.step("output_gate", "deterministic", started, decision.action, reasons=decision.reasons)
        gate = {"action": decision.action, "reasons": decision.reasons}
        if decision.action != "send":
            escalation = escalation.model_copy(update={
                "should_escalate": True, "human_action_required": True,
                "reason": EscalationReason.OUTPUT_BLOCKED if decision.action == "block" else EscalationReason.VALIDATION_FAILURE,
                "explanation": f"Final output gate: {', '.join(decision.reasons)}.",
            })
            status = ResponseStatus.BLOCKED if decision.action == "block" else ResponseStatus.ESCALATED
            message = (BLOCKED_MESSAGE if decision.action == "block" else HOLDING_MESSAGE).format(name=run.greeting())
            return self._final(run, status, message, intake=intake, context=context, validation=validation,
                               escalation=escalation, draft=draft, gate=gate)

        # 9. Delivery (only after the gate) - the answer is customer-ready.
        notification = self._deliver(run, draft.message) if self.deliver_notifications else None
        if notification and not notification.get("ok") and notification.get("code") == "authorization_error":
            escalation = escalation.model_copy(update={
                "should_escalate": True, "human_action_required": True, "reason": EscalationReason.MANUAL_REVIEW_REQUIRED,
                "explanation": "Delivery refused: customer KYC is not verified."})
            return self._final(run, ResponseStatus.ESCALATED, HOLDING_MESSAGE.format(name=run.greeting()), intake=intake, context=context,
                               validation=validation, escalation=escalation, draft=draft, gate=gate, notification=notification,
                               next_action="Complete KYC verification before account details can be sent.")
        return self._final(run, ResponseStatus.READY, draft.message, intake=intake, context=context, validation=validation,
                           escalation=escalation, draft=draft, gate=gate, notification=notification, sendable=True)

    # ------------------------------------------------------------------------- failure helpers

    @trace(logger)
    def _rejected(self, run: _Run, error: IntakeError) -> FinalResponse:
        code = {
            IntakeErrorCode.PROMPT_INJECTION: PipelineErrorCode.PROMPT_INJECTION,
            IntakeErrorCode.UNKNOWN_CUSTOMER: PipelineErrorCode.UNKNOWN_CUSTOMER,
        }.get(error.code, PipelineErrorCode.INVALID_INTAKE)
        status = ResponseStatus.BLOCKED if code == PipelineErrorCode.PROMPT_INJECTION else ResponseStatus.REJECTED
        return self._final(run, status, error.message, error=PipelineError(
            code=code, stage="intake", message=error.message, detail_code=error.code.value), audit=False)

    @trace(logger)
    def _try_escalate(self, run: _Run, intake: IntakeRequest, confidence: float) -> EscalationResult | None:
        started = time.perf_counter()
        try:
            result = self.escalator.decide(intake, confidence, validation_passed=False)
        except (EscalationUnavailable, EscalationInvariantViolation) as exc:
            run.step("escalate", "EscalationAgent", started, "failed", code=getattr(exc, "code", "invariant_violation"))
            return None
        run.step("escalate", "EscalationAgent", started, "escalate" if result.should_escalate else "no_escalation",
                 reason=result.reason.value, server_reason=result.server_reason, decision_id=result.decision_id,
                 hard=result.hard_escalation_triggered)
        return result

    @trace(logger)
    def _incomplete(self, run: _Run, intake: IntakeRequest, context: GroundedContext) -> FinalResponse:
        """No customer-ready draft from incomplete context: typed failure or safe hand-off."""
        escalation = self._try_escalate(run, intake, 0.0)
        if context.service_errors:
            first = context.service_errors[0]
            unavailable = first.code in {"service_unavailable", "primitive_error"}
            error = PipelineError(
                code=PipelineErrorCode.SERVICE_UNAVAILABLE if unavailable else PipelineErrorCode.MCP_TOOL_ERROR,
                stage="retrieve", server=first.server, tool=first.tool, detail_code=first.code, retryable=first.retryable,
                message=f"Required {first.server} data could not be retrieved ({first.code}); no answer was guessed.",
            )
            message = UNAVAILABLE_MESSAGE.format(name=run.greeting(), what=SERVER_LABELS.get(first.server, first.server))
            return self._final(run, ResponseStatus.FAILED, message, intake=intake, context=context, error=error, escalation=escalation)
        error = PipelineError(code=PipelineErrorCode.RETRIEVAL_INCOMPLETE, stage="retrieve",
                              message="Could not verify: " + ", ".join(context.missing_requirements))
        return self._final(run, ResponseStatus.ESCALATED if escalation else ResponseStatus.FAILED, HOLDING_MESSAGE.format(name=run.greeting()),
                           intake=intake, context=context, error=error, escalation=escalation,
                           next_action="Specialist review: " + ", ".join(context.missing_requirements))

    @trace(logger)
    def _unavailable(self, run: _Run, server: str, tool: str, stage: str, code: str, intake: IntakeRequest | None) -> FinalResponse:
        return self._final(run, ResponseStatus.FAILED, UNAVAILABLE_MESSAGE.format(name=run.greeting(), what=SERVER_LABELS.get(server, server)),
                           intake=intake, error=PipelineError(code=PipelineErrorCode.SERVICE_UNAVAILABLE, stage=stage, server=server, tool=tool,
                                                              detail_code=code, retryable=True,
                                                              message=f"The {server} service is unavailable; no answer was guessed."), audit=False)

    # ----------------------------------------------------------------------------- MCP writes

    @trace(logger)
    def _deliver(self, run: _Run, message: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            envelope = self.gateway.call_tool("send_customer_notification", {
                "customer_id": run.customer_id, "channel": run.channel, "message": message})
        except (MCPUnavailable, MCPPrimitiveError):
            run.step("deliver", "mcp", started, "failed", code="service_unavailable")
            return {"ok": False, "code": "service_unavailable"}
        if not envelope.get("ok"):
            code = (envelope.get("error") or {}).get("code")
            run.step("deliver", "mcp", started, "refused", code=code)
            return {"ok": False, "code": code}
        data = envelope["data"]
        run.step("deliver", "mcp", started, data["outcome"], delivery_confirmed=data.get("delivery_confirmed", False))
        return {"ok": True, "outcome": data["outcome"], "delivery_confirmed": data.get("delivery_confirmed", False),
                "channel": data["channel"], "audit_id": data.get("audit_id")}

    @trace(logger)
    def _audit(self, run: _Run, status: ResponseStatus, detail: dict[str, Any]) -> dict[str, Any] | None:
        if not self.audit_enabled or run.customer_id is None:
            return None
        started = time.perf_counter()
        try:
            envelope = self.gateway.call_tool("write_audit_log", {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "action_type": "nexus_request_outcome",
                "performed_by": "nexus_orchestrator",
                "customer_id": run.customer_id,
                "outcome": status.value,
                "detail": json.dumps(detail, separators=(",", ":"), sort_keys=True),
            })
        except (MCPUnavailable, MCPPrimitiveError):
            run.step("audit", "mcp", started, "failed", code="service_unavailable")
            return {"recorded": False, "code": "service_unavailable"}
        if not envelope.get("ok"):
            run.step("audit", "mcp", started, "failed", code=(envelope.get("error") or {}).get("code"))
            return {"recorded": False, "code": (envelope.get("error") or {}).get("code")}
        run.step("audit", "mcp", started, "recorded", audit_id=envelope["data"]["audit_id"])
        return {"recorded": True, "audit_id": envelope["data"]["audit_id"]}

    # ----------------------------------------------------------------------------- assembly

    @trace(logger)
    def _final(
        self,
        run: _Run,
        status: ResponseStatus,
        message: str,
        *,
        intake: IntakeRequest | None = None,
        context: GroundedContext | None = None,
        validation: ValidationResult | None = None,
        escalation: EscalationResult | None = None,
        draft: DraftResponse | None = None,
        error: PipelineError | None = None,
        gate: dict[str, Any] | None = None,
        notification: dict[str, Any] | None = None,
        next_action: str | None = None,
        sendable: bool = False,
        audit: bool = True,
    ) -> FinalResponse:
        if next_action is None:
            next_action = {
                ResponseStatus.READY: draft.proposed_action if draft else None,
                ResponseStatus.ESCALATED: "Human specialist review",
                ResponseStatus.BLOCKED: "Human specialist review",
                ResponseStatus.FAILED: "Retry when services recover; specialist follow-up",
                ResponseStatus.REJECTED: "Correct the request and resubmit",
            }.get(status)
        audit_record = None
        if audit:
            audit_record = self._audit(run, status, {
                "request_id": run.request_id,
                "request_type": intake.request_type.value if intake else None,
                "product_line": intake.product_line.value if intake else None,
                "escalation_category": intake.escalation_category.value if intake else None,
                "escalation_reason": escalation.reason.value if escalation else None,
                "decision_id": escalation.decision_id if escalation else None,
                "error": error.code.value if error else None,
                "conflicts": len(context.conflicts) if context else 0,
                "blocked_sources": len(context.blocked_sources) if context else 0,
                "kyc_status": run.kyc_status,
                "notification_outcome": (notification or {}).get("outcome") or (notification or {}).get("code"),
                "notification_audit_id": (notification or {}).get("audit_id"),
            })
        return FinalResponse(
            request_id=run.request_id,
            customer_id=mask_identifier(run.customer_id) if run.customer_id else None,
            message=message,
            status=status,
            validation=validation,
            escalation=escalation,
            error=error,
            can_be_sent_to_customer=sendable and status == ResponseStatus.READY,
            next_action=next_action,
            evidence_ids=draft.evidence_ids if draft else [],
            conflicts=[f"{c.conflict_id}: {c.field} (KB vs {c.mcp_server}.{c.mcp_tool}; MCP authoritative)" for c in context.conflicts] if context else [],
            blocked_sources=[f"{b.source_reference}: {b.reason}" for b in context.blocked_sources] if context else [],
            output_gate=gate or {},
            notification=notification,
            audit=audit_record,
            steps=run.steps,
            mcp_calls=list(self.gateway.calls),
        )


# ------------------------------------------------------------------------------------ wiring


@trace(logger)
def build_production_orchestrator(*, deliver_notifications: bool = True, gateway: MCPGateway | None = None) -> NexusOrchestrator:
    """HTTP MCP clients to the four running servers, CrewAI for model steps, Chroma for the KB."""
    from app.agents.llm import CrewAIBackend
    from app.mcp_client.http_gateway import HttpMCPGateway
    from app.retrieval.retrieval import UnstructuredRetriever
    from app.runtime import bootstrap

    bootstrap()
    return NexusOrchestrator(
        gateway=gateway or HttpMCPGateway(),
        llm=CrewAIBackend(),
        kb_search=UnstructuredRetriever.from_chroma(),
        deliver_notifications=deliver_notifications,
    )


@trace(logger)
def summarize(final: FinalResponse) -> str:
    """Concise masked trace: statuses, evidence, decision source, sendability."""
    lines = [f"request {final.request_id} customer {final.customer_id} -> {final.status.value.upper()} (sendable={final.can_be_sent_to_customer})"]
    for step in final.steps:
        keep = {k: v for k, v in step.detail.items() if v not in (None, [], {})}
        lines.append(f"  {step.step:<14} {step.role:<16} {step.status:<18} {json.dumps(keep, default=str)[:220]}")
    if final.escalation:
        lines.append(f"  decision: should_escalate={final.escalation.should_escalate} reason={final.escalation.reason.value} "
                     f"source={final.escalation.decision_source} confidence={final.escalation.draft_confidence}")
    if final.error:
        lines.append(f"  error: {final.error.code.value} at {final.error.stage} ({final.error.server or '-'}.{final.error.tool or '-'})")
    calls = [f"{c.server}.{c.name}" + ("" if c.ok else f"!{c.error_code}") for c in final.mcp_calls]
    lines.append("  mcp calls: " + (", ".join(calls) or "none"))
    return "\n".join(lines)


class _Handler(BaseHTTPRequestHandler):
    """Loopback JSON API. The model/KB wiring is built on the first POST, so ``/health`` (which
    lists every server's primitives over MCP) works as a liveness check without model credentials."""

    gateway: Any = None
    orchestrator: NexusOrchestrator | None = None
    _build_lock = threading.Lock()

    @classmethod
    @trace(logger)
    def pipeline(cls) -> NexusOrchestrator:
        with cls._build_lock:
            if cls.orchestrator is None:
                cls.orchestrator = build_production_orchestrator(gateway=cls.gateway)
            return cls.orchestrator

    @trace(logger)
    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @trace(logger)
    def log_message(self, format: str, *args: Any) -> None:  # silence default stderr access log
        logger.info({"event": "HTTP", "path": self.path.split("?")[0], "status": args[1] if len(args) > 1 else None})

    @trace(logger)
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            return self._send(404, {"error": "not_found"})
        try:
            primitives = self.gateway.list_primitives()
            counts = {s: {k: len(v) for k, v in p.items()} for s, p in primitives.items()}
            return self._send(200, {"status": "ok", "servers": counts})
        except Exception:
            return self._send(503, {"status": "degraded"})

    @trace(logger)
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/requests":
            return self._send(404, {"error": "not_found"})
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 20_000)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "invalid_json"})
        final = self.pipeline().handle(payload)
        return self._send(200, final.model_dump(mode="json"))


@trace(logger)
def serve(host: str, port: int, orchestrator: NexusOrchestrator | None = None) -> None:
    from app.mcp_client.http_gateway import HttpMCPGateway
    from app.runtime import bootstrap

    bootstrap()
    _Handler.orchestrator = orchestrator
    _Handler.gateway = orchestrator.gateway if orchestrator else HttpMCPGateway()
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Nexus orchestrator listening on http://{host}:{port} (POST /v1/requests, GET /health)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@trace(logger)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nexus_orchestrator", description="Nexus orchestrator (MCP client process)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_serve = sub.add_parser("serve", help="run the orchestrator HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8100)
    p_ask = sub.add_parser("ask", help="process one request and print the masked trace")
    p_ask.add_argument("message")
    p_ask.add_argument("--customer-id", required=True)
    for ref in ("account", "policy", "portfolio", "product", "claim"):
        p_ask.add_argument(f"--{ref}-id")
    p_ask.add_argument("--channel", default="chat")
    p_ask.add_argument("--no-deliver", action="store_true", help="do not call send_customer_notification")
    p_ask.add_argument("--json", action="store_true", help="also print the full FinalResponse JSON")
    sub.add_parser("check", help="list Tools/Resources/Prompts on each running server")
    args = parser.parse_args(argv)

    if args.command == "serve":
        serve(args.host, args.port)
        return 0
    if args.command == "check":
        from app.mcp_client.http_gateway import HttpMCPGateway
        from app.runtime import bootstrap

        bootstrap()
        gateway = HttpMCPGateway()
        try:
            listing = gateway.list_primitives()
        except MCPUnavailable as exc:
            print(f"unavailable: {exc}")
            return 1
        print(json.dumps(listing, indent=2))
        totals = [sum(len(p[k]) for p in listing.values()) for k in ("tools", "resources", "prompts")]
        print(f"TOTAL tools={totals[0]} resources={totals[1]} prompts={totals[2]}")
        return 0 if totals == [13, 4, 2] else 1
    orchestrator = build_production_orchestrator(deliver_notifications=not args.no_deliver)
    references = {f"{r}_id": getattr(args, f"{r}_id") for r in ("account", "policy", "portfolio", "product", "claim") if getattr(args, f"{r}_id")}
    final = orchestrator.handle({"message": args.message, "customer_id": args.customer_id, "references": references, "channel": args.channel})
    print(summarize(final))
    print(f"\nmessage: {final.message}")
    if args.json:
        print(json.dumps(final.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
