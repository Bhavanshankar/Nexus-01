"""Deterministic ``LLMBackend`` for the local demo and the test suite.

Implements the same ``structured()`` interface as ``CrewAIBackend`` so every agent, gate and MCP
call runs unchanged; only text generation is scripted. It is deliberately naive in one respect:
the scripted classifier never assigns a hard escalation category, which demonstrates that the
deterministic intake signals - not the model - enforce hard escalation.

Test hooks: ``overrides`` replaces a role's output (e.g. a fooled classifier, a draft leaking an
identifier), ``fail_roles`` makes a role return invalid output.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel

from app.agents.llm import ModelOutputError
from app.guardrails import COMPLETED_ACTION_CLAIMS
from app.logging_config import get_logger, trace
from app.models.intake import ClassificationProposal
from app.models.response import AssertedFact, DraftProposal
from app.models.retrieval import GroundedContext, StructuredFact
from app.models.validation import ValidationReview

logger = get_logger(__name__)

_LINE_WORDS = {
    "insurance": re.compile(r"\b(?:insurance|premium|policy|claim|cover(?:ed|age)?)\b", re.I),
    "banking": re.compile(r"\b(?:bank|account|savings|checking|balance|transactions?|debit\w*)\b", re.I),
    "wealth": re.compile(r"\b(?:portfolio|invest\w*|fund|etf|bond|annuity|suitab\w*|risk\s+profile)\b", re.I),
}
_TYPE_RULES = (
    (re.compile(r"auto[-\s]?debit|automatic(?:ally)?\s+pa(?:y|id)|direct\s+debit", re.I), "auto_debit_setup"),
    (re.compile(r"suitab", re.I), "suitability_check"),
    (re.compile(r"\bclaim\b", re.I), "claim_status"),
    (re.compile(r"\bbalance\b", re.I), "balance_inquiry"),
    (re.compile(r"\btransactions?\b", re.I), "transaction_history"),
    (re.compile(r"\bcover(?:ed|age)?\b", re.I), "coverage_question"),
    (re.compile(r"payment\s+terms|\bdue\b", re.I), "payment_terms"),
    (re.compile(r"\bportfolio\b", re.I), "portfolio_query"),
    (re.compile(r"\b(?:fund|etf|bond|annuity)\b", re.I), "investment_product_query"),
    (re.compile(r"\bcomplain", re.I), "complaint"),
)
_INTENTS = {
    "auto_debit_setup": "Customer wants to know whether a premium can be paid by auto-debit and what the payment terms allow.",
    "suitability_check": "Customer asks whether an investment product is suitable for their portfolio.",
    "claim_status": "Customer asks for the status of an insurance claim.",
    "balance_inquiry": "Customer asks for an account balance.",
    "transaction_history": "Customer asks about recent account transactions.",
    "coverage_question": "Customer asks what their insurance policy covers.",
    "payment_terms": "Customer asks about insurance premium payment terms.",
    "portfolio_query": "Customer asks about the status of their investment portfolio.",
    "investment_product_query": "Customer asks about an investment product.",
    "complaint": "Customer wants to raise a complaint.",
    "other": "Customer has a general question.",
}


class ScriptedLLM:
    name = "scripted"

    @trace(logger)
    def __init__(
        self,
        *,
        overrides: dict[str, Callable[[dict[str, Any]], BaseModel | dict[str, Any]]] | None = None,
        fail_roles: set[str] | None = None,
    ) -> None:
        self.overrides = dict(overrides or {})
        self.fail_roles = set(fail_roles or ())
        self.calls: list[str] = []

    @trace(logger)
    def structured(self, *, role: str, goal: str, backstory: str, prompt: str, output_model: type[BaseModel], context: dict[str, Any] | None = None) -> BaseModel:
        key = {ClassificationProposal: "classifier", DraftProposal: "drafter", ValidationReview: "validator"}.get(output_model)
        if key is None:
            raise ModelOutputError(f"scripted backend has no script for {output_model.__name__}")
        self.calls.append(key)
        context = context or {}
        if key in self.fail_roles:
            raise ModelOutputError(f"{key}: scripted invalid output")
        if key in self.overrides:
            return output_model.model_validate(self.overrides[key](context))
        return {"classifier": self._classify, "drafter": self._draft, "validator": self._review}[key](context)

    # ------------------------------------------------------------------------------ classifier

    @trace(logger)
    def _classify(self, context: dict[str, Any]) -> ClassificationProposal:
        message = context.get("message", "")
        references = context.get("references", {})
        lines = [line for line, pattern in _LINE_WORDS.items() if pattern.search(message)]
        if {"account_id"} & set(references):
            lines.append("banking")
        if {"policy_id", "claim_id"} & set(references):
            lines.append("insurance")
        if {"portfolio_id", "product_id"} & set(references):
            lines.append("wealth")
        lines = sorted(set(lines))
        product_line = "cross_product" if len(lines) > 1 else (lines[0] if lines else "banking")
        request_type = next((t for pattern, t in _TYPE_RULES if pattern.search(message)), "other")
        urgency = "high" if re.search(r"\b(?:urgent\w*|immediately|asap|today)\b", message, re.I) else "routine"
        intent = _INTENTS[request_type]
        if request_type in {"other", "complaint"}:
            intent = f"{intent} {message[:180]}"  # already sanitised and identifier-masked
        return ClassificationProposal(
            product_line=product_line, request_type=request_type, urgency=urgency,
            customer_stated_intent=intent, escalation_category="none",
        )

    # --------------------------------------------------------------------------------- drafter

    @trace(logger)
    def _draft(self, context: dict[str, Any]) -> DraftProposal:
        name = context.get("customer_name") or "there"
        if context.get("holding_reply"):
            return DraftProposal(
                message=(f"Hi {name}, thank you for telling us about this. A Meridian specialist will review your request "
                         "personally and contact you shortly to talk through your options."),
                cited_evidence_ids=[], asserted_facts=[], proposed_next_step="Specialist follow-up",
            )
        ctx: GroundedContext = context["grounded_context"]
        intake = context["intake"]
        facts = {f.fact_id: f for f in ctx.authoritative_facts}
        kb_ids = [e.evidence_id for e in ctx.evidence if not e.authoritative][:2]
        builder = getattr(self, f"_draft_{intake.request_type.value}", self._draft_general)
        message, used, next_step = builder(name, facts, ctx)
        return DraftProposal(
            message=message,
            cited_evidence_ids=[f.fact_id for f in used] + kb_ids,
            asserted_facts=[AssertedFact(fact_id=f.fact_id, value=f.value) for f in used],
            proposed_next_step=next_step,
        )

    @staticmethod
    @trace(logger)
    def _pick(facts: dict[str, StructuredFact], *ids: str) -> list[StructuredFact]:
        return [facts[i] for i in ids if i in facts]

    @trace(logger)
    def _draft_auto_debit_setup(self, name, facts, ctx):
        a = lambda f: facts.get(f"mcp:banking.account.{f}")
        p = lambda f: facts.get(f"mcp:insurance.policy.{f}")
        covered = facts.get("mcp:insurance.coverage.covered")
        used = self._pick(facts, *(f"mcp:banking.account.{f}" for f in ("account_type", "status", "auto_debit_allowed")),
                          *(f"mcp:insurance.policy.{f}" for f in ("policy_type", "status", "auto_debit_permitted", "premium", "payment_frequency", "payment_terms")),
                          "mcp:insurance.coverage.covered")
        account = f"{a('account_type').value} account ending {a('account_type').object_ref}" if a("account_type") else "account"
        policy_kind = p("policy_type").value if p("policy_type") else "insurance"
        policy = f"{'general insurance' if policy_kind == 'general' else policy_kind + ' insurance'} policy ending {p('policy_type').object_ref}" if p("policy_type") else "policy"
        account_ok = a("auto_debit_allowed") and a("auto_debit_allowed").value is True and a("status") and a("status").value == "active"
        policy_ok = p("auto_debit_permitted") is not None and p("auto_debit_permitted").value is True and p("status") and p("status").value == "active"
        clause_ok = covered is None or covered.value is True
        terms = ""
        if p("premium") and p("payment_frequency"):
            amount = str(p("premium").value).split(" (")[0]
            terms = f" The premium is {amount}, billed {p('payment_frequency').value}"
            terms += f" ({p('payment_terms').value.rstrip('.')})." if p("payment_terms") else "."
        if account_ok and policy_ok and clause_ok:
            message = (
                f"Hi {name}, here is what your live records show. Your {account} is active and allows auto-debit. "
                f"Your {policy} is active and its payment terms permit auto-debit - the auto-debit payment clause is covered.{terms} "
                "So yes, this account is eligible to pay the premium by auto-debit. Nexus cannot enrol you itself: to start "
                "auto-debit, please give your authorisation in the Meridian app or with an agent."
            )
            return message, used, "Customer authorises auto-debit enrolment"
        reasons = []
        if not account_ok:
            state = a("status").value if a("status") else "unverified"
            reasons.append(f"your {account} is currently {state} and does not allow auto-debit")
        if not policy_ok:
            reasons.append(f"your {policy} does not currently permit auto-debit")
        if not clause_ok:
            reasons.append("the policy's auto-debit payment clause is not covered")
        lead = f"your {policy} permits auto-debit, but " if policy_ok else "unfortunately "
        message = (
            f"Hi {name}, {lead}{' and '.join(reasons)}, so auto-debit cannot be set up from this account right now. "
            f"Your live account and policy records take precedence over general guidance.{terms} You could use another "
            "eligible Meridian account, or speak to a specialist about the restriction."
        )
        return message, used, "Offer an alternative eligible account or specialist review"

    @trace(logger)
    def _draft_payment_terms(self, name, facts, ctx):
        used = self._pick(facts, "mcp:insurance.policy.premium", "mcp:insurance.policy.payment_frequency",
                          "mcp:insurance.policy.payment_terms", "mcp:insurance.policy.status", "mcp:insurance.coverage.coverage_status")
        parts = [f.statement.split(": ", 1)[1] for f in used]
        return f"Hi {name}, your live policy record shows: " + "; ".join(parts) + ".", used, None

    _draft_coverage_question = _draft_payment_terms

    @trace(logger)
    def _draft_balance_inquiry(self, name, facts, ctx):
        used = self._pick(facts, "mcp:banking.account_balance.balance", "mcp:banking.account.status")
        balance = str(facts["mcp:banking.account_balance.balance"].value).split(" (")[0] if used else "not available"
        return f"Hi {name}, the current balance on your account is {balance}.", used, None

    @trace(logger)
    def _draft_general(self, name, facts, ctx):
        used = list(facts.values())[:6]
        if used:
            return f"Hi {name}, here is what your live records show: " + "; ".join(f.statement for f in used) + ".", used, None
        guidance = next((e for e in ctx.evidence if not e.authoritative), None)
        if guidance is None:
            return (f"Hi {name}, I could not find verified information to answer this question, so a specialist will help."), [], None
        sentence = re.split(r"(?<=[.!?])\s+", re.sub(r"^Section:[^\n]*\n+", "", guidance.content).strip())[0]
        return f"Hi {name}, based on Meridian's general guidance: {sentence}", [], None

    # ------------------------------------------------------------------------------- validator

    @trace(logger)
    def _review(self, context: dict[str, Any]) -> ValidationReview:
        draft = context["draft"]
        ctx: GroundedContext = context["grounded_context"]
        facts = {f.fact_id: f for f in ctx.authoritative_facts}
        unsupported = [a.fact_id for a in draft.asserted_facts if a.fact_id in facts and a.value is not None and str(a.value).lower() != str(facts[a.fact_id].value).lower()]
        confidence = 0.93
        if ctx.kb_quality is not None and ctx.kb_quality.status == "SUCCESS_AFTER_EXPANSION":
            confidence = 0.62
        if not ctx.evidence and not draft.holding_reply:
            confidence = 0.45
        return ValidationReview(
            factually_supported=not unsupported,
            tone_appropriate=True,
            compliance_ok=not COMPLETED_ACTION_CLAIMS.search(draft.message),
            acknowledges_conflicts=True,
            unsupported_claims=unsupported,
            issues=[],
            confidence=confidence if not unsupported else 0.3,
        )
