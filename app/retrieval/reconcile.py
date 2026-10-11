"""Deterministic reconciliation of live MCP facts with knowledge-base guidance (PDF 3.3).

* Structured MCP facts describe the customer's current state and are authoritative.
* KB documents describe general rules and are never treated as customer facts.
* When the two disagree, a ``RetrievalConflict`` preserving both sides is produced and passed to the
  Validation Agent. Conflicts are detected here in code; no model is asked to find them.
"""

from __future__ import annotations

import re
from typing import Any

from app import knowledge_base
from app.guardrails import PromptInjectionDetected, InputValidationError, mask_identifier, sanitize_free_text
from app.logging_config import get_logger, trace
from app.models.intake import IntakeRequest, RequestType
from app.models.retrieval import (
    BlockedSource,
    GroundedEvidence,
    RetrievalConflict,
    RetrievalSource,
    RetrievedDocument,
    StructuredFact,
)

logger = get_logger(__name__)

# result key -> (server, tool, record label, fields to expose as facts)
FACT_FIELDS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "account": ("banking", "get_account_summary", "Banking account record", ("account_type", "status", "auto_debit_allowed", "currency")),
    "account_balance": ("banking", "get_account_summary", "Banking account balance", ("balance",)),
    "policy": ("insurance", "get_policy_details", "Insurance policy record", (
        "policy_type", "status", "payment_frequency", "premium", "payment_terms", "auto_debit_permitted")),
    "coverage": ("insurance", "check_coverage_clause", "Policy clause check", ("resolved_clause_id", "coverage_status", "covered")),
    "claim": ("insurance", "get_claim_status", "Claim record", ("status", "filed_at", "last_updated_at")),
    "portfolio": ("wealth", "get_portfolio_summary", "Portfolio record", ("risk_profile", "status")),
    "product": ("wealth", "get_investment_product_details", "Investment product record", ("name", "product_type", "min_risk_profile", "description")),
    "suitability": ("wealth", "check_product_suitability", "Suitability check", ("risk_profile", "minimum_risk_profile", "suitable", "reason_code")),
}
OBJECT_REF_FIELDS = {"account": "account_id", "account_balance": "account_id", "policy": "policy_id", "coverage": "policy_id",
                     "claim": "claim_id", "portfolio": "portfolio_id", "product": "product_id", "suitability": "product_id"}
TEXT_FIELDS = {"payment_terms", "description", "name"}
_AUTO_DEBIT = re.compile(r"auto[-\s]?debit|automatic\s+(?:collection|payment)", re.I)
_PERMITS = re.compile(r"\b(?:permit\w*|allow\w*|support\w*|accept\w*|available|eligible)\b", re.I)
_CAVEAT = re.compile(r"\b(?:not|n't|unless|while|but|only|restricted|disallow\w*|must)\b", re.I)
AUTO_DEBIT_TOPICS = {"auto_debit", "premium_payment", "recurring_payments", "bank_insurance", "eligibility", "payment", "retrieval_conflict"}
ACTIVE = "active"
_WHITESPACE = re.compile(r"\s+")


@trace(logger, redact_result=True)
def render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict) and "amount_minor" in value:
        return f"{value['amount_minor'] / 100:.2f} {value.get('currency', 'USD')} ({value['amount_minor']} minor units)"
    if value is None:
        return "not recorded"
    return str(value)


@trace(logger)
def facts_from_result(key: str, data: dict[str, Any]) -> tuple[list[StructuredFact], list[BlockedSource]]:
    """Turn one MCP result into authoritative facts; text fields are sanitised before any model
    sees them and withheld (blocked) if they fail."""
    server, tool, label, fields = FACT_FIELDS[key]
    ref_value = data.get(OBJECT_REF_FIELDS.get(key, ""))
    object_ref = mask_identifier(ref_value) if isinstance(ref_value, str) and not ref_value.startswith("PROD-") else ref_value
    facts: list[StructuredFact] = []
    blocked: list[BlockedSource] = []
    for name in fields:
        if name not in data:
            continue
        fact_id = f"mcp:{server}.{key}.{name}"
        value = data[name]
        if name in TEXT_FIELDS and isinstance(value, str):
            try:
                value = sanitize_free_text(value, field=name, max_length=1000, source="mcp")
            except (PromptInjectionDetected, InputValidationError) as exc:
                blocked.append(BlockedSource(
                    source_reference=fact_id, source=RetrievalSource.MCP, reason="failed_sanitization",
                    signatures=list(getattr(exc, "signatures", [])),
                ))
                continue
        stored: str | int | float | bool | None
        if isinstance(value, dict):
            stored = render_value(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            stored = value
        else:
            stored = str(value)
        facts.append(StructuredFact(
            server_name=server, tool_name=tool, fact_id=fact_id, field=f"{key}.{name}", value=stored,
            object_ref=object_ref, statement=f"{label} ({object_ref}): {name.replace('_', ' ')} = {render_value(value)}",
        ))
    return facts, blocked


@trace(logger)
def facts_from_list(key: str, server: str, tool: str, items: list[dict[str, Any]], label: str) -> list[StructuredFact]:
    """Facts for list results (transactions, linked accounts)."""
    facts = [StructuredFact(
        server_name=server, tool_name=tool, fact_id=f"mcp:{server}.{key}.count", field=f"{key}.count",
        value=len(items), statement=f"{label}: {len(items)} record(s) returned",
    )]
    for index, item in enumerate(items):
        parts = []
        for k, v in item.items():
            if k == "transaction_id":
                continue
            if isinstance(v, str) and k in TEXT_FIELDS:
                try:
                    v = sanitize_free_text(v, field=k, max_length=300, source="mcp")
                except (PromptInjectionDetected, InputValidationError):
                    v = "[withheld by safety checks]"
            parts.append(f"{k.replace('_', ' ')} = {render_value(v)}")
        facts.append(StructuredFact(
            server_name=server, tool_name=tool, fact_id=f"mcp:{server}.{key}.{index}", field=f"{key}.{index}",
            value="; ".join(parts), object_ref=item.get("account_id"), statement=f"{label} #{index + 1}: " + "; ".join(parts),
        ))
    return facts


@trace(logger)
def _auto_debit_guidance(documents: list[RetrievedDocument]) -> RetrievedDocument | None:
    """The retrieved KB document presenting auto-debit as an available payment method: a
    payment-method guidance document (by topic), or a chunk with an affirmative, un-negated
    permission sentence. Descriptive caveats ("not guaranteed", "while ... restricted") don't count."""
    for document in documents:
        if (document.topic or "").lower() in AUTO_DEBIT_TOPICS:
            return document
    for document in documents:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", document.content):
            if _AUTO_DEBIT.search(sentence) and _PERMITS.search(sentence) and not _CAVEAT.search(sentence):
                return document
    return None


@trace(logger)
def detect_conflicts(
    intake: IntakeRequest,
    documents: list[RetrievedDocument],
    facts: list[StructuredFact],
) -> list[RetrievalConflict]:
    by_id = {f.fact_id: f for f in facts}
    conflicts: list[RetrievalConflict] = []

    def add(field: str, kb_value: str, fact: StructuredFact, kb_document: str | None) -> None:
        conflicts.append(RetrievalConflict(
            conflict_id=f"conflict-{len(conflicts) + 1}",
            field=field,
            kb_value=kb_value,
            mcp_value=f"{fact.field} = {render_value(fact.value)}",
            kb_document_id=kb_document,
            mcp_fact_id=fact.fact_id,
            mcp_server=fact.server_name,
            mcp_tool=fact.tool_name,
        ))

    # Rule A - payment method: KB guidance says auto-debit is a permitted/available payment method,
    # but a live gating control for this customer says otherwise.
    if intake.request_type in {RequestType.AUTO_DEBIT_SETUP, RequestType.PAYMENT_TERMS, RequestType.COVERAGE_QUESTION}:
        source = _auto_debit_guidance(documents)
        if source is not None:
            kb_claim = f"General guidance ({source.document_id}) presents auto-debit as an available premium payment method"
            gates = (
                ("mcp:banking.account.auto_debit_allowed", lambda v: v is False, "account auto-debit permission"),
                ("mcp:banking.account.status", lambda v: v != ACTIVE, "account status"),
                ("mcp:insurance.policy.auto_debit_permitted", lambda v: v is False, "policy auto-debit permission"),
                ("mcp:insurance.policy.status", lambda v: v != ACTIVE, "policy status"),
                ("mcp:insurance.coverage.covered", lambda v: v is False, "payment clause coverage"),
            )
            clause = by_id.get("mcp:insurance.coverage.resolved_clause_id")
            for fact_id, contradicts, label in gates:
                if fact_id == "mcp:insurance.coverage.covered" and (clause is None or clause.value != "auto_debit_payment"):
                    continue  # only the auto-debit clause speaks to the payment method
                fact = by_id.get(fact_id)
                if fact is not None and contradicts(fact.value):
                    add(label, kb_claim, fact, source.document_id)

    # Rule B - product document vs live product record: the minimum risk profile stated in the
    # KB product document must match the wealth server's record.
    product_fact = by_id.get("mcp:wealth.product.min_risk_profile")
    if product_fact is not None and intake.referenced_ids.product_id:
        document = knowledge_base.product_document_for(intake.referenced_ids.product_id)
        documented = knowledge_base.product_minimum_risk_profile(document) if document else None
        if documented and documented != product_fact.value:
            add("product minimum risk profile", f"Product document ({document.document_id}) states minimum risk profile '{documented}'",
                product_fact, document.document_id)
    return conflicts


@trace(logger)
def evidence_from(facts: list[StructuredFact], documents: list[RetrievedDocument]) -> list[GroundedEvidence]:
    evidence = [
        GroundedEvidence(
            evidence_id=f.fact_id, source=RetrievalSource.MCP, content=f.statement,
            source_reference=f"mcp://{f.server_name}/{f.tool_name}", authoritative=True,
        )
        for f in facts
    ]
    for document in documents:
        index = document.metadata.get("chunk_index", document.chunk_id.rsplit(":", 1)[-1])
        evidence.append(GroundedEvidence(
            evidence_id=f"kb:{document.document_id}#{index}", source=RetrievalSource.VECTOR_KB,
            content=document.content[:1200], source_reference=document.source_path, authoritative=False,
        ))
    return evidence


@trace(logger)
def grounding_summary(
    evidence: list[GroundedEvidence],
    conflicts: list[RetrievalConflict],
    blocked: list[BlockedSource],
    missing: list[str],
) -> str:
    """The evidence block given to the Drafting prompt (identifiers already masked)."""
    lines = ["AUTHORITATIVE LIVE FACTS (MCP - current customer state):"]
    lines += [f"[{e.evidence_id}] {e.content}" for e in evidence if e.authoritative] or ["(none)"]
    if conflicts:
        lines.append("\nCONFLICTS (live MCP data is authoritative; say what it means for the customer):")
        lines += [f"[{c.conflict_id}] {c.field}: knowledge base - {c.kb_value}; live record - {c.mcp_value}" for c in conflicts]
    lines.append("\nGENERAL GUIDANCE (knowledge base - rules, not customer facts):")
    lines += [f"[{e.evidence_id}] {_WHITESPACE.sub(' ', e.content)[:700]}" for e in evidence if not e.authoritative] or ["(none)"]
    if blocked:
        lines.append(f"\nWITHHELD: {len(blocked)} retrieved source(s) failed safety checks and were removed.")
    if missing:
        lines.append("\nCOULD NOT VERIFY: " + ", ".join(missing))
    return "\n".join(lines)
