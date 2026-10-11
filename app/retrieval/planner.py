"""Retrieval planning: the validated intake decides what to retrieve (PDF 3.3).

Product line, request type, intent and the supplied object IDs select (a) knowledge-base searches
restricted to the relevant product lines and (b) the MCP calls needed for the customer's live facts.
A cross-product auto-debit question therefore plans banking *and* insurance guidance plus the live
account, policy and payment-clause records; a suitability question plans the live portfolio risk
profile, the product record and the deterministic suitability check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.logging_config import get_logger, trace
from app.models.intake import IntakeRequest, ProductLine, RequestType

logger = get_logger(__name__)

# Agent-scope reads establish ownership (they return customer_id); customer scope adds balances.
AGENT_SCOPE = "authorized_agent"
CUSTOMER_SCOPE = "customer"


@dataclass
class KBQuery:
    text: str
    product_lines: tuple[str, ...] | None
    required: bool = True


@dataclass
class PlannedCall:
    tool: str
    arguments: dict[str, Any] | Callable[[dict[str, dict[str, Any]]], dict[str, Any]]
    key: str  # name used for the result, e.g. "account", "policy", "coverage"
    owner_check: bool = False  # result carries customer_id that must equal the intake customer
    required: bool = True


@dataclass
class RetrievalPlan:
    kb_queries: list[KBQuery] = field(default_factory=list)
    calls: list[PlannedCall] = field(default_factory=list)
    missing_references: list[str] = field(default_factory=list)
    needs_kb: bool = True
    customer_specific: bool = True
    product_document: str | None = None  # product_id whose KB product document is compared to MCP


# Reference objects each request type needs before it can be answered for this customer.
REQUIRED_REFERENCES: dict[RequestType, tuple[str, ...]] = {
    RequestType.BALANCE_INQUIRY: ("account_id",),
    RequestType.TRANSACTION_HISTORY: ("account_id",),
    RequestType.LINKED_ACCOUNTS: (),
    RequestType.AUTO_DEBIT_SETUP: ("account_id",),
    RequestType.PAYMENT_TERMS: ("policy_id",),
    RequestType.COVERAGE_QUESTION: ("policy_id",),
    RequestType.CLAIM_STATUS: ("claim_id",),
    RequestType.PORTFOLIO_QUERY: ("portfolio_id",),
    RequestType.INVESTMENT_PRODUCT_QUERY: ("product_id",),
    RequestType.SUITABILITY_CHECK: ("product_id", "portfolio_id"),
    RequestType.COMPLAINT: (),
    RequestType.OTHER: (),
}
KB_REQUIRED_TYPES = {
    RequestType.AUTO_DEBIT_SETUP,
    RequestType.PAYMENT_TERMS,
    RequestType.COVERAGE_QUESTION,
    RequestType.SUITABILITY_CHECK,
    RequestType.COMPLAINT,
    RequestType.OTHER,
}
COVERAGE_TAGS = (
    (re.compile(r"auto[-\s]?debit|automatic(?:ally)?\s+(?:pay|collect|debit)|direct\s+debit", re.I), "auto_debit_payment"),
    (re.compile(r"payment\s+terms?|premium\s+(?:due|schedule|frequency)|how\s+often|due\s+date", re.I), "general_payment_terms"),
)


@trace(logger)
def coverage_tag_for(intent: str) -> str:
    """Deterministic scenario tag for check_coverage_clause; unknown scenarios get a tag that
    matches no clause, so coverage is never inferred."""
    for pattern, tag in COVERAGE_TAGS:
        if pattern.search(intent):
            return tag
    return "unspecified_scenario"


@trace(logger)
def plan_retrieval(intake: IntakeRequest) -> RetrievalPlan:
    refs = intake.referenced_ids
    rtype = intake.request_type
    line = intake.product_line
    plan = RetrievalPlan(needs_kb=rtype in KB_REQUIRED_TYPES)

    required = list(REQUIRED_REFERENCES[rtype])
    if rtype == RequestType.AUTO_DEBIT_SETUP and (line in {ProductLine.INSURANCE, ProductLine.CROSS_PRODUCT} or refs.policy_id):
        required.append("policy_id")
    plan.missing_references = [name for name in required if not getattr(refs, name)]
    plan.customer_specific = bool(required) or bool(refs.present())

    # --- live structured facts: every referenced object is read (with an ownership check) ------
    if refs.account_id:
        plan.calls.append(PlannedCall("get_account_summary", {"account_id": refs.account_id, "caller_scope": AGENT_SCOPE}, "account", owner_check=True))
    if refs.policy_id:
        plan.calls.append(PlannedCall("get_policy_details", {"policy_id": refs.policy_id}, "policy", owner_check=True))
    if refs.portfolio_id:
        plan.calls.append(PlannedCall("get_portfolio_summary", {"portfolio_id": refs.portfolio_id, "caller_scope": AGENT_SCOPE}, "portfolio", owner_check=True))
    if refs.claim_id:
        plan.calls.append(PlannedCall("get_claim_status", {"claim_id": refs.claim_id}, "claim", owner_check=True))
    if refs.product_id:
        plan.calls.append(PlannedCall("get_investment_product_details", {"product_id": refs.product_id, "caller_scope": CUSTOMER_SCOPE}, "product"))
        plan.product_document = refs.product_id

    # --- request-type specific calls (run after ownership is established) -----------------------
    if rtype == RequestType.BALANCE_INQUIRY and refs.account_id:
        plan.calls.append(PlannedCall("get_account_summary", {"account_id": refs.account_id, "caller_scope": CUSTOMER_SCOPE}, "account_balance"))
    if rtype == RequestType.TRANSACTION_HISTORY and refs.account_id:
        plan.calls.append(PlannedCall("get_transaction_history", {"account_id": refs.account_id, "limit": 5, "caller_scope": CUSTOMER_SCOPE}, "transactions"))
    if rtype == RequestType.LINKED_ACCOUNTS:
        plan.calls.append(PlannedCall("get_linked_accounts", {"customer_id": intake.customer_id, "caller_scope": CUSTOMER_SCOPE}, "linked_accounts"))
    if refs.policy_id and rtype in {RequestType.AUTO_DEBIT_SETUP, RequestType.PAYMENT_TERMS, RequestType.COVERAGE_QUESTION}:
        tag = "auto_debit_payment" if rtype == RequestType.AUTO_DEBIT_SETUP else coverage_tag_for(intake.customer_stated_intent)
        if rtype == RequestType.PAYMENT_TERMS and tag == "unspecified_scenario":
            tag = "general_payment_terms"
        plan.calls.append(PlannedCall("check_coverage_clause", {"policy_id": refs.policy_id, "scenario_tag": tag}, "coverage"))
    if rtype == RequestType.SUITABILITY_CHECK and refs.product_id and refs.portfolio_id:
        product_id = refs.product_id
        plan.calls.append(PlannedCall(
            "check_product_suitability",
            lambda results: {
                "product_id": product_id,
                "risk_profile": results["portfolio"]["risk_profile"],  # live profile, never customer-asserted
                "caller_scope": AGENT_SCOPE,
            },
            "suitability",
        ))

    # --- knowledge-base searches ------------------------------------------------------------------
    plan.kb_queries = _kb_queries(intake)
    return plan


@trace(logger)
def _kb_queries(intake: IntakeRequest) -> list[KBQuery]:
    rtype, line = intake.request_type, intake.product_line
    q: list[KBQuery] = []
    if rtype == RequestType.AUTO_DEBIT_SETUP:
        q.append(KBQuery("auto-debit eligibility account status permission", ("banking",)))
        if line in {ProductLine.INSURANCE, ProductLine.CROSS_PRODUCT} or intake.referenced_ids.policy_id:
            q.append(KBQuery("insurance premium auto-debit payment terms policy permission", ("insurance",)))
            q.append(KBQuery("bank account auto-debit eligibility and insurance premium payment", ("cross_product",)))
    elif rtype == RequestType.PAYMENT_TERMS:
        q.append(KBQuery("premium payment terms due date frequency", ("insurance",)))
    elif rtype == RequestType.COVERAGE_QUESTION:
        q.append(KBQuery(f"policy coverage clause terms {intake.customer_stated_intent}", ("insurance",)))
    elif rtype == RequestType.CLAIM_STATUS:
        q.append(KBQuery("claim status meaning", ("insurance",), required=False))
    elif rtype == RequestType.BALANCE_INQUIRY:
        q.append(KBQuery("account balance status", ("banking",), required=False))
    elif rtype == RequestType.TRANSACTION_HISTORY:
        q.append(KBQuery("transaction history processing", ("banking",), required=False))
    elif rtype == RequestType.LINKED_ACCOUNTS:
        q.append(KBQuery("account types and recurring payment use", ("banking",), required=False))
    elif rtype == RequestType.PORTFOLIO_QUERY:
        q.append(KBQuery("portfolio status frozen meaning", ("wealth",), required=False))
    elif rtype == RequestType.INVESTMENT_PRODUCT_QUERY:
        q.append(KBQuery(f"investment product {intake.customer_stated_intent}", ("wealth",), required=False))
    elif rtype == RequestType.SUITABILITY_CHECK:
        q.append(KBQuery("product suitability policy requirement eligibility risk profile minimum", ("wealth",)))
    else:  # complaint / other: general guidance across all lines, tied to the stated intent
        lines = None if line == ProductLine.CROSS_PRODUCT else (line.value, "cross_product")
        q.append(KBQuery(intake.customer_stated_intent, lines))
    return q
