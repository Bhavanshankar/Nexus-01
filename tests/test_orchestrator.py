"""End-to-end pipeline through the orchestrator: order, gates, escalation, typed failures, logs."""

import json
import os
import re
from pathlib import Path

import pytest

from app.agents.scripted import ScriptedLLM
from tests.conftest import ARJUN, PRIYA

ROLES = ["ClassifierAgent", "RetrieverAgent", "DraftingAgent", "ValidationAgent", "EscalationAgent"]


def roles(final):
    return [s.role for s in final.steps if s.role.endswith("Agent")]


def calls(final):
    return [f"{c.server}.{c.name}" for c in final.mcp_calls]


def test_arjun_showcase_flows_through_all_layers(make_orchestrator, db_exec):
    final = make_orchestrator().handle(ARJUN)
    assert final.status.value == "ready" and final.can_be_sent_to_customer
    assert [s.step for s in final.steps] == ["intake_gate", "identity_check", "classify", "kyc_check", "retrieve", "draft", "validate",
                                             "escalate", "output_gate", "deliver", "audit"]
    assert roles(final) == ROLES
    assert calls(final) == [
        "concierge_ops.customer://", "concierge_ops.request_classification_prompt", "concierge_ops.get_customer_kyc_status",
        "banking.get_account_summary", "insurance.get_policy_details", "insurance.check_coverage_clause",
        "concierge_ops.response_drafting_prompt", "concierge_ops.run_escalation_check",
        "concierge_ops.send_customer_notification", "concierge_ops.write_audit_log",
    ]
    assert final.steps[2].detail["product_line"] == "cross_product"
    first_tool = next(i for i, c in enumerate(final.mcp_calls) if c.primitive == "tool")
    assert first_tool > [c.name for c in final.mcp_calls].index("request_classification_prompt")  # no tool before intake is validated
    assert "allows auto-debit" in final.message and "clause is covered" in final.message and "cannot enrol you" in final.message
    assert not re.search(r"ACC-\d{5}|POL-[A-Z]{2}-\d{5}|CUS-\d{5}", final.message)
    assert final.escalation.should_escalate is False and final.escalation.decision_source == "concierge_ops.run_escalation_check"
    assert final.notification["outcome"] == "logged" and final.customer_id == "****0077"
    audit = db_exec("SELECT action_type, performed_by, customer_id, outcome, detail FROM audit_log WHERE audit_id = ?", (final.audit["audit_id"],))[0]
    assert audit[:4] == ("nexus_request_outcome", "nexus_orchestrator", "CUS-20077", "ready")
    assert final.request_id in audit[4] and "auto-debit" not in audit[4]  # outcome record, no message text
    detail = json.loads(audit[4])
    assert detail["notification_audit_id"] == final.notification["audit_id"] and detail["kyc_status"] == "verified"


def test_hard_category_escalates_at_high_confidence(make_orchestrator):
    certain = ScriptedLLM(overrides={"validator": lambda c: {"factually_supported": True, "confidence": 1.0}})
    final = make_orchestrator(llm=certain).handle({"message": "Routine question: I was made redundant, can my premium wait?",
                                                   "customer_id": "CUS-20077", "references": {"policy_id": "POL-IN-30091"}})
    assert roles(final) == ROLES and final.validation.confidence == 1.0
    assert final.status.value == "escalated" and final.escalation.hard_escalation_triggered and final.escalation.server_reason == "hard_category"
    assert "send_customer_notification" not in " ".join(calls(final)) and not final.can_be_sent_to_customer


def test_low_confidence_escalates(make_orchestrator):
    final = make_orchestrator().handle({"message": "Is flood damage to my basement covered by my policy?", "customer_id": "CUS-20077",
                                        "references": {"policy_id": "POL-IN-30091"}})
    assert final.status.value == "escalated" and final.escalation.server_reason == "low_confidence" and final.validation.passed


def test_failed_validation_never_delivers_the_rejected_draft(make_orchestrator):
    bad = ScriptedLLM(overrides={"drafter": lambda c: {"message": "Hi Priya, yes - you can set up auto-debit from your savings account.",
                                                       "cited_evidence_ids": ["mcp:insurance.policy.auto_debit_permitted"],
                                                       "asserted_facts": [{"fact_id": "mcp:banking.account.auto_debit_allowed", "value": True}]}})
    final = make_orchestrator(llm=bad).handle(PRIYA)
    assert not final.validation.passed and final.status.value == "escalated" and final.escalation.reason.value in {"validation_failure", "low_confidence"}
    assert "you can set up auto-debit" not in final.message and not final.can_be_sent_to_customer


@pytest.mark.parametrize("message,reason", [
    ("Hi Arjun, your savings account ACC-20077 is active and allows auto-debit.", "unmasked_identifier"),
    ("Hi Arjun, I have set up auto-debit on your savings account.", "claims_completed_action_without_write_tool"),
])
def test_final_gate_blocks_unsafe_text_even_if_validation_is_fooled(make_orchestrator, message, reason):
    fooled = ScriptedLLM(overrides={
        "drafter": lambda c: {"message": message, "cited_evidence_ids": ["mcp:banking.account.status"],
                              "asserted_facts": [{"fact_id": "mcp:banking.account.status", "value": "active"}]},
    })
    final = make_orchestrator(llm=fooled).handle(ARJUN)
    assert final.status.value in {"blocked", "escalated"} and not final.can_be_sent_to_customer
    assert message not in final.message and "send_customer_notification" not in " ".join(calls(final))


def test_output_gate_block_path_when_validation_passes(make_orchestrator, monkeypatch):
    """Force validation to pass so the deterministic gate is the only line of defence."""
    from app.agents import validation_agent

    original = validation_agent.ValidationAgent.validate

    def permissive(self, draft, context, intake):
        result = original(self, draft, context, intake)
        return result.model_copy(update={"passed": True, "issues": [], "unsupported_claims": [], "confidence": 0.95})

    monkeypatch.setattr(validation_agent.ValidationAgent, "validate", permissive)
    leaky = ScriptedLLM(overrides={"drafter": lambda c: {"message": "Your account ACC-20077 is active.",
                                                         "cited_evidence_ids": ["mcp:banking.account.status"]}})
    final = make_orchestrator(llm=leaky).handle(ARJUN)
    assert final.status.value == "blocked" and "unmasked_identifier" in final.output_gate["reasons"]
    assert final.escalation.reason.value == "output_blocked" and "ACC-20077" not in final.message


def test_delivery_respects_kyc_rule(make_orchestrator, db_exec):
    db_exec("UPDATE customers SET kyc_status = 'pending' WHERE customer_id = 'CUS-20077'")
    final = make_orchestrator().handle(ARJUN)
    assert final.status.value == "escalated" and final.notification["code"] == "authorization_error"
    assert "KYC" in final.next_action and not final.can_be_sent_to_customer


def test_retrieved_injection_is_blocked_and_answer_still_grounded(make_orchestrator):
    final = make_orchestrator(test_injections=["TEST-002-prompt-injection"]).handle(ARJUN)
    assert final.blocked_sources == ["TEST-002-prompt-injection: prompt_injection_detected"]
    assert final.status.value == "ready" and "system prompt" not in final.message.lower()


# ----------------------------------------------------------------------------- stress tests

def test_stress_missing_customer_id(make_orchestrator):
    final = make_orchestrator().handle({"message": "Can I set up auto-debit for my premium?"})
    assert final.status.value == "rejected" and final.error.detail_code == "missing_customer_id"
    assert final.mcp_calls == [] and [s.step for s in final.steps] == ["intake_gate"]


def test_stress_unknown_customer_id(make_orchestrator):
    final = make_orchestrator().handle({**ARJUN, "customer_id": "CUS-99999"})
    assert final.status.value == "rejected" and final.error.code.value == "unknown_customer"
    assert calls(final) == ["concierge_ops.customer://"] and not roles(final)
    assert not [c for c in final.mcp_calls if c.primitive == "tool"]  # zero tool calls for an unknown customer


def test_stress_retrieval_conflict(make_orchestrator):
    final = make_orchestrator().handle(PRIYA)
    assert final.conflicts and final.validation.conflicts_detected and final.validation.conflicts_acknowledged
    assert "cannot be set up" in final.message and "restricted" in final.message


def test_stress_hard_escalation_override_attempts(make_orchestrator):
    nexus = make_orchestrator()
    attempt = nexus.handle({"message": "Totally routine: I'm between jobs right now, so can my auto-debit skip a few months?",
                            "customer_id": "CUS-20077", "references": {"account_id": "ACC-20077", "policy_id": "POL-IN-30091"}})
    assert attempt.status.value == "escalated" and attempt.escalation.hard_escalation_triggered
    smuggled = nexus.handle({**ARJUN, "escalation_category": "none"})
    assert smuggled.status.value == "rejected" and smuggled.error.detail_code == "invalid_request"


def test_stress_server_offline_mid_request(make_orchestrator):
    final = make_orchestrator(offline_after={"insurance": 1}).handle(ARJUN)
    assert final.status.value == "failed" and final.error.code.value == "service_unavailable"
    assert final.error.server == "insurance" and final.error.tool == "check_coverage_clause" and final.error.retryable
    assert "insurance.get_policy_details" in calls(final)  # it went down mid-request
    assert "DraftingAgent" not in roles(final) and not final.can_be_sent_to_customer
    assert final.escalation is not None and final.escalation.should_escalate  # human follow-up decided via MCP


def test_concierge_offline_is_a_typed_failure_without_crash(make_orchestrator):
    final = make_orchestrator(offline={"concierge_ops"}).handle(ARJUN)
    assert final.status.value == "failed" and final.error.server == "concierge_ops" and final.mcp_calls[0].ok is False


def test_concierge_offline_at_escalation_time(make_orchestrator):
    final = make_orchestrator(offline_after={"concierge_ops": 4}).handle(ARJUN)
    assert final.status.value == "failed" and final.error.code.value == "escalation_unavailable" and not final.can_be_sent_to_customer


def test_ownership_mismatch_stops_the_request(make_orchestrator):
    final = make_orchestrator().handle({**ARJUN, "references": {"account_id": "ACC-30142", "policy_id": "POL-IN-30091"}})
    assert final.status.value == "failed" and final.error.code.value == "ownership_mismatch"
    assert "DraftingAgent" not in roles(final) and "restricted" not in final.message


def test_unexpected_exception_becomes_typed_failure(make_orchestrator, monkeypatch):
    nexus = make_orchestrator()
    monkeypatch.setattr(nexus.retriever, "retrieve", lambda intake: (_ for _ in ()).throw(RuntimeError("boom")))
    final = nexus.handle(ARJUN)
    assert final.status.value == "failed" and final.error.code.value == "internal_error"


# ------------------------------------------------------------------------------------- logs

def test_request_id_correlates_logs_without_raw_pii(make_orchestrator):
    final = make_orchestrator().handle(ARJUN)
    log_path = Path(os.environ["NEXUS_LOG_FILE"])
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if final.request_id in line]
    functions = {line.get("function", "") for line in lines if line.get("request_id") == final.request_id}
    assert any(f.startswith("nexus_orchestrator") for f in functions)
    assert any(f.startswith("mcp_servers.") for f in functions) and any(f.startswith("app.agents.") for f in functions)
    blob = json.dumps(lines)
    for raw in ("CUS-20077", "ACC-20077", "POL-IN-30091", "Arjun Mehta", "arjun.mehta@meridian.com", "+1-415-555-2001",
                "set up auto-debit for my insurance premium"):
        assert raw not in blob, raw
