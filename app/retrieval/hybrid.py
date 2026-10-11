"""Hybrid retrieval: KB search + live MCP facts, reconciled into one ``GroundedContext``.

Guarantees:
* Every KB chunk and every MCP text field is sanitised before any model sees it; content that fails
  (e.g. an injected instruction in a retrieved document) is withheld and recorded as a
  ``BlockedSource`` - its text never reaches drafting.
* Every customer-specific record is checked to belong to the intake customer (``customer_id`` from
  an agent-scope read). A mismatch stops the request with ``OwnershipMismatch``.
* A missing reference, an MCP error envelope, an unreachable server or an insufficient KB result
  where KB guidance is required sets ``retrieval_complete = False``; nothing is guessed.
* Test/adversarial documents are excluded from ordinary search; ``test_injections`` is an explicit
  test mode that adds a named test document's payload as if it had been retrieved.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from app import knowledge_base
from app.guardrails import (
    InputValidationError,
    PromptInjectionDetected,
    mask_identifiers_in_text,
    sanitize_free_text,
)
from app.logging_config import get_logger, trace
from app.mcp_client.gateway import TOOL_SERVER, MCPGateway, MCPPrimitiveError, MCPUnavailable
from app.models.intake import IntakeRequest
from app.models.retrieval import (
    BlockedSource,
    GroundedContext,
    KBQuality,
    RetrievalSource,
    RetrievedDocument,
    ServiceError,
    StructuredFact,
)
from app.retrieval import reconcile
from app.retrieval.adapter import to_kb_quality, to_retrieved_documents
from app.retrieval.planner import PlannedCall, RetrievalPlan, plan_retrieval
from app.retrieval.retrieval import KBSearchResult

logger = get_logger(__name__)
KBSearch = Callable[[str, Iterable[str] | None], KBSearchResult]


class OwnershipMismatch(RuntimeError):
    """A referenced record does not belong to the intake customer."""

    def __init__(self, key: str, server: str, tool: str) -> None:
        super().__init__(f"{key} record from {server}.{tool} does not belong to the requesting customer")
        self.key = key
        self.server = server
        self.tool = tool


class HybridRetriever:
    @trace(logger)
    def __init__(
        self,
        kb_search: KBSearch | Any,
        gateway: MCPGateway,
        *,
        test_injections: Iterable[str] = (),
        docs_per_query: int = 3,
        max_kb_documents: int = 6,
    ) -> None:
        self._kb_search = kb_search.retrieve if hasattr(kb_search, "retrieve") else kb_search
        self.gateway = gateway
        self.test_injections = tuple(test_injections)
        self.docs_per_query = docs_per_query
        self.max_kb_documents = max_kb_documents

    # ----------------------------------------------------------------------------- public

    @trace(logger)
    def retrieve(self, intake: IntakeRequest) -> GroundedContext:
        plan = plan_retrieval(intake)
        documents, kb_quality, kb_errors = self._search_kb(plan)
        documents += self._injected_documents()
        documents, blocked = self._sanitize_documents(documents)

        facts, fact_blocked, service_errors = self._call_mcp(intake, plan)
        blocked += fact_blocked
        service_errors = kb_errors + service_errors

        # Hard-category requests get a holding reply that states no facts, so missing references or
        # weak KB guidance cannot make them "incomplete"; only backend failures still do.
        answer_required = not intake.is_hard_category
        missing = [f"{name.replace('_id', '')} reference" for name in plan.missing_references] if answer_required else []
        if answer_required and plan.needs_kb and not (kb_quality.passed and documents):
            missing.append("knowledge-base guidance")
        if fact_blocked:
            missing.append("structured record text (withheld by safety checks)")
        for error in service_errors:
            missing.append(f"{error.server}.{error.tool} ({error.code})")

        conflicts = reconcile.detect_conflicts(intake, documents, facts)
        evidence = reconcile.evidence_from(facts, documents)
        complete = not missing and not service_errors
        return GroundedContext(
            request_id=intake.request_id,
            customer_id=intake.customer_id,
            evidence=evidence,
            conflicts=conflicts,
            authoritative_facts=facts,
            blocked_sources=blocked,
            service_errors=service_errors,
            missing_requirements=missing,
            kb_quality=kb_quality,
            customer_specific=plan.customer_specific,
            retrieval_complete=complete,
            grounding_summary=reconcile.grounding_summary(evidence, conflicts, blocked, missing),
        )

    # -------------------------------------------------------------------------- knowledge base

    @trace(logger)
    def _search_kb(self, plan: RetrievalPlan) -> tuple[list[RetrievedDocument], KBQuality, list[ServiceError]]:
        results: list[KBSearchResult] = []
        required_results: list[KBSearchResult] = []
        documents: dict[str, RetrievedDocument] = {}
        errors: list[ServiceError] = []
        for query in plan.kb_queries:
            try:
                result = self._kb_search(query.text, query.product_lines)
            except Exception:
                errors.append(ServiceError(server="knowledge_base", tool="vector_search", code="service_unavailable",
                                           message="The knowledge-base search could not be completed.", retryable=True))
                continue
            results.append(result)
            if query.required:
                required_results.append(result)
            if not result.safe_to_answer and query.required:
                continue
            for document in to_retrieved_documents(result)[: self.docs_per_query]:
                documents.setdefault(document.chunk_id, document)
        quality = to_kb_quality(required_results or results)
        if errors and plan.needs_kb:
            quality = KBQuality(status="UNAVAILABLE", confidence=0.0, passed=False, queries=quality.queries)
        ranked = sorted(documents.values(), key=lambda d: d.score, reverse=True)[: self.max_kb_documents]
        if not plan.needs_kb and errors:
            errors = []  # optional guidance; its absence does not make the answer incomplete
        return ranked, quality, errors

    @trace(logger)
    def _injected_documents(self) -> list[RetrievedDocument]:
        injected = []
        for document_id in self.test_injections:
            document = knowledge_base.get_document(document_id, include_test=True)
            if document is None:
                continue
            match = re.search(r"TEST-SPECIFIC PAYLOAD\s*(.*?)(?:\n\s*13\.|\Z)", document.body, re.DOTALL)
            payload = (match.group(1) if match else document.body[:1500]).strip()
            injected.append(RetrievedDocument(
                document_id=document.document_id, chunk_id=f"{document.document_id}:payload",
                product_line=document.product_line or "cross_product", document_type="test", topic=document.subtopic,
                title=document.title, section="TEST-SPECIFIC PAYLOAD", source_path=document.relative_path,
                content=payload, score=0.99, metadata={"test_injection": "true"},
            ))
        return injected

    @trace(logger)
    def _sanitize_documents(self, documents: list[RetrievedDocument]) -> tuple[list[RetrievedDocument], list[BlockedSource]]:
        clean: list[RetrievedDocument] = []
        blocked: list[BlockedSource] = []
        for document in documents:
            try:
                text = sanitize_free_text(
                    mask_identifiers_in_text(document.content), field="kb_chunk", max_length=6000,
                    source="knowledge_base", keep_newlines=True,
                )
            except PromptInjectionDetected as exc:
                blocked.append(BlockedSource(source_reference=document.document_id, source=RetrievalSource.VECTOR_KB,
                                             reason="prompt_injection_detected", signatures=exc.signatures))
                continue
            except InputValidationError:
                blocked.append(BlockedSource(source_reference=document.document_id, source=RetrievalSource.VECTOR_KB,
                                             reason="failed_validation"))
                continue
            clean.append(document.model_copy(update={"content": text}))
        return clean, blocked

    # ---------------------------------------------------------------------------------- MCP

    @trace(logger)
    def _call_mcp(self, intake: IntakeRequest, plan: RetrievalPlan) -> tuple[list[StructuredFact], list[BlockedSource], list[ServiceError]]:
        results: dict[str, dict[str, Any]] = {}
        facts: list[StructuredFact] = []
        blocked: list[BlockedSource] = []
        errors: list[ServiceError] = []
        for call in plan.calls:
            server = TOOL_SERVER[call.tool]
            try:
                arguments = call.arguments(results) if callable(call.arguments) else call.arguments
            except KeyError:
                continue  # a prerequisite result is missing; its error is already recorded
            try:
                envelope = self.gateway.call_tool(call.tool, arguments)
            except MCPUnavailable:
                errors.append(ServiceError(server=server, tool=call.tool, code="service_unavailable",
                                           message=f"The {server} service is unavailable.", retryable=True))
                break  # do not continue with a partially reachable backend
            except MCPPrimitiveError:
                errors.append(ServiceError(server=server, tool=call.tool, code="primitive_error",
                                           message=f"The {server} service returned an invalid response."))
                break
            if not envelope.get("ok"):
                error = envelope.get("error") or {}
                errors.append(ServiceError(server=server, tool=call.tool, code=str(error.get("code", "unknown_error")),
                                           message=str(error.get("message", "The request could not be completed."))[:200],
                                           retryable=bool(error.get("retryable", False))))
                continue
            data = envelope.get("data") or {}
            if call.owner_check and data.get("customer_id") != intake.customer_id:
                raise OwnershipMismatch(call.key, server, call.tool)
            results[call.key] = data
            new_facts, new_blocked = self._facts_for(call, server, data)
            facts += new_facts
            blocked += new_blocked
        return facts, blocked, errors

    @trace(logger)
    def _facts_for(self, call: PlannedCall, server: str, data: dict[str, Any]) -> tuple[list[StructuredFact], list[BlockedSource]]:
        if call.key == "transactions":
            return reconcile.facts_from_list("transactions", server, call.tool, data.get("transactions", []), "Recent transaction"), []
        if call.key == "linked_accounts":
            return reconcile.facts_from_list("linked_accounts", server, call.tool, data.get("accounts", []), "Linked account"), []
        return reconcile.facts_from_result(call.key, data)
