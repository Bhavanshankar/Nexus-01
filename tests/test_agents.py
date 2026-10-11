"""Each agent role in isolation: output contracts, deterministic checks, decision verification."""

import pytest

from app.agents import DraftingAgent, EscalationAgent, RetrieverAgent, ValidationAgent
from app.agents._shared import AgentStepError
from app.agents.escalation_agent import EscalationInvariantViolation, EscalationUnavailable
from app.agents.scripted import ScriptedLLM
from app.mcp_client.gateway import MCPGateway, MCPUnavailable
from app.models.intake import IntakeRequest
from app.models.response import AssertedFact, DraftResponse
from app.models.retrieval import GroundedContext
from app.retrieval.hybrid import HybridRetriever


def intake(**overrides):
    base = dict(request_id="REQ-ABCDEF123456", customer_id="CUS-20077", product_line="cross_product", request_type="auto_debit_setup",
                urgency="routine", customer_stated_intent="Pay the premium by auto-debit from savings.",
                referenced_ids={"account_id": "ACC-20077", "policy_id": "POL-IN-30091"})
    base.update(overrides)
    return IntakeRequest(**base)


@pytest.fixture
def context(gateway, kb_retriever):
    return RetrieverAgent(HybridRetriever(kb_retriever, gateway)).retrieve(intake())


@pytest.fixture
def conflict_context(gateway, kb_retriever):
    return RetrieverAgent(HybridRetriever(kb_retriever, gateway)).retrieve(intake(
        customer_id="CUS-30142", referenced_ids={"account_id": "ACC-30142", "policy_id": "POL-GE-30250"}))


def draft_for(context, message, cited=None, asserted=None, holding=False):
    facts = [f.fact_id for f in context.authoritative_facts]
    return DraftResponse(request_id="REQ-ABCDEF123456", message=message, evidence_ids=cited if cited is not None else facts[:3],
                         asserted_facts=asserted or [], holding_reply=holding)


def test_retriever_agent_returns_grounded_context(context):
    assert isinstance(context, GroundedContext) and context.request_id == "REQ-ABCDEF123456" and context.retrieval_complete


def test_drafting_agent_cites_known_evidence_via_mcp_prompt(gateway, context):
    gateway.reset_calls()
    draft = DraftingAgent(ScriptedLLM(), gateway).draft(intake(), context, customer_name="Arjun")
    assert [c.name for c in gateway.calls] == ["response_drafting_prompt"]
    assert draft.evidence_ids and set(draft.evidence_ids) <= context.evidence_ids()
    assert {a.fact_id for a in draft.asserted_facts} <= {f.fact_id for f in context.authoritative_facts}


def test_drafting_agent_writes_holding_reply_for_hard_categories(gateway, context):
    draft = DraftingAgent(ScriptedLLM(), gateway).draft(intake(escalation_category="financial_hardship"), context, customer_name="Arjun")
    assert draft.holding_reply and not draft.asserted_facts and "specialist" in draft.message


def test_drafting_agent_rejects_invalid_model_output(gateway, context):
    with pytest.raises(AgentStepError):
        DraftingAgent(ScriptedLLM(overrides={"drafter": lambda c: {"message": ""}}), gateway).draft(intake(), context, customer_name=None)


def test_validation_passes_grounded_draft(gateway, context):
    draft = DraftingAgent(ScriptedLLM(), gateway).draft(intake(), context, customer_name="Arjun")
    result = ValidationAgent(ScriptedLLM()).validate(draft, context, intake())
    assert result.passed and result.grounding_valid and result.confidence == min(result.deterministic_confidence, result.model_confidence)
    assert all(result.checks.values())


@pytest.mark.parametrize("message,kwargs,issue", [
    ("Your account ****0077 is active.", {"cited": ["mcp:made.up"]}, "unknown_citation"),
    ("Your account ****0077 is active.", {"cited": []}, "no_authoritative_citation"),
    ("Your account is restricted.", {"asserted": [AssertedFact(fact_id="mcp:banking.account.status", value="restricted")]}, "contradicts_live_record"),
    ("Your account is active.", {"asserted": [AssertedFact(fact_id="mcp:banking.account.nickname", value="x")]}, "unknown_fact"),
    ("Your premium is 99.00 USD.", {}, "ungrounded_number"),
    ("Your account ACC-20077 is active.", {}, "pii_in_draft"),
    ("I have set up the auto-debit for you.", {}, "claims_completed_action"),
    ("Returns are guaranteed.", {}, "guarantee_language"),
])
def test_validation_catches_what_a_model_could_miss(context, message, kwargs, issue):
    result = ValidationAgent(ScriptedLLM()).validate(draft_for(context, message, **kwargs), context, intake())
    assert not result.passed and issue in [i.code for i in result.issues]
    assert result.confidence <= 0.5


def test_validation_requires_conflicts_to_be_respected(conflict_context):
    assert conflict_context.conflicts
    trusting = draft_for(conflict_context, "Yes, your policy permits auto-debit so you can set it up.",
                         asserted=[AssertedFact(fact_id="mcp:insurance.policy.auto_debit_permitted", value=True)])
    result = ValidationAgent(ScriptedLLM()).validate(trusting, conflict_context, intake())
    assert result.conflicts_detected and not result.conflicts_acknowledged and not result.passed
    respectful = draft_for(conflict_context, "Your account is restricted and does not allow auto-debit, so it cannot be set up.",
                           asserted=[AssertedFact(fact_id=c.mcp_fact_id, value=conflict_context.fact(c.mcp_fact_id).value) for c in conflict_context.conflicts])
    assert ValidationAgent(ScriptedLLM()).validate(respectful, conflict_context, intake()).passed


def test_model_cannot_raise_confidence_above_deterministic_cap(context):
    overconfident = ScriptedLLM(overrides={"validator": lambda c: {"factually_supported": True, "confidence": 1.0}})
    result = ValidationAgent(overconfident).validate(draft_for(context, "Your premium is 99.00 USD."), context, intake())
    assert result.model_confidence == 1.0 and result.confidence <= 0.4


def test_model_review_failure_fails_validation(context):
    result = ValidationAgent(ScriptedLLM(fail_roles={"validator"})).validate(draft_for(context, "Your account is active."), context, intake())
    assert not result.passed and result.confidence == 0.0


def test_inconclusive_decisive_fact_lowers_confidence(gateway, kb_retriever):
    ctx = HybridRetriever(kb_retriever, gateway).retrieve(intake(product_line="insurance", request_type="coverage_question",
                                                                 customer_stated_intent="Is flood damage covered?", referenced_ids={"policy_id": "POL-IN-30091"}))
    draft = DraftingAgent(ScriptedLLM(), gateway).draft(intake(request_type="coverage_question", referenced_ids={"policy_id": "POL-IN-30091"}), ctx, customer_name=None)
    result = ValidationAgent(ScriptedLLM()).validate(draft, ctx, intake())
    assert result.passed and result.confidence < 0.70


# ------------------------------------------------------------------------------- escalation

def test_escalation_agent_uses_the_mcp_tool(gateway):
    gateway.reset_calls()
    result = EscalationAgent(gateway).decide(intake(), 0.93)
    assert [c.name for c in gateway.calls] == ["run_escalation_check"]
    assert not result.should_escalate and result.decision_source == "concierge_ops.run_escalation_check" and result.customer_id == "****0077"


def test_escalation_agent_hard_and_low_confidence(gateway):
    hard = EscalationAgent(gateway).decide(intake(escalation_category="compliance_override_request"), 1.0)
    low = EscalationAgent(gateway).decide(intake(), 0.4)
    assert hard.should_escalate and hard.hard_escalation_triggered and hard.reason.value == "hard_category"
    assert low.should_escalate and low.reason.value == "low_confidence"


def test_failed_validation_escalates_even_if_server_would_not(gateway):
    result = EscalationAgent(gateway).decide(intake(), 0.95, validation_passed=False)
    assert result.should_escalate and result.reason.value == "validation_failure" and result.server_reason == "none"


class FakeGateway(MCPGateway):
    def __init__(self, envelope=None, raise_unavailable=False):
        super().__init__()
        self.envelope, self.raise_unavailable = envelope, raise_unavailable

    def _call_tool(self, server, tool, arguments):
        if self.raise_unavailable:
            raise MCPUnavailable(server, "tool", tool)
        return self.envelope

    def _read_resource(self, server, uri):
        raise NotImplementedError

    def _get_prompt(self, server, name, arguments):
        raise NotImplementedError


def test_tool_failure_never_becomes_a_decision():
    with pytest.raises(EscalationUnavailable):
        EscalationAgent(FakeGateway(raise_unavailable=True)).decide(intake(), 0.9)
    with pytest.raises(EscalationUnavailable):
        EscalationAgent(FakeGateway({"ok": False, "error": {"code": "database_error"}})).decide(intake(), 0.9)


def test_decision_that_violates_hard_rule_is_rejected():
    lying = {"ok": True, "data": {"customer_id": "CUS-20077", "request_category": "financial_hardship", "should_escalate": False,
                                  "hard_escalation_triggered": False, "reason": "none"}}
    with pytest.raises(EscalationInvariantViolation):
        EscalationAgent(FakeGateway(lying)).decide(intake(escalation_category="financial_hardship"), 1.0)
    mismatched = {"ok": True, "data": {"customer_id": "CUS-30142", "request_category": "none", "should_escalate": False, "reason": "none"}}
    with pytest.raises(EscalationInvariantViolation):
        EscalationAgent(FakeGateway(mismatched)).decide(intake(), 0.9)
