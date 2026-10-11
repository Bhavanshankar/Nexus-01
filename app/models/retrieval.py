"""Hybrid retrieval and grounding contracts (PDF section 3.3).

The unstructured retriever's own dataclass result is ``app.retrieval.retrieval.KBSearchResult``;
the adapter in ``app.retrieval.adapter`` converts it into ``RetrievedDocument`` objects here.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievalSource(str, Enum):
    VECTOR_KB = "vector_kb"
    MCP = "mcp"


class RetrievedDocument(BaseModel):
    """
    A document/chunk retrieved from the local vector knowledge base.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    product_line: str | None = None
    document_type: str
    topic: str | None = None
    title: str | None = None
    section: str | None = None
    source_path: str

    content: str = Field(min_length=1)

    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, str] = Field(default_factory=dict)

    source: RetrievalSource = RetrievalSource.VECTOR_KB


class KBQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    passed: bool
    queries: list[str] = Field(default_factory=list)


class StructuredFact(BaseModel):
    """
    A fact retrieved live through an MCP server.

    MCP structured data is authoritative over conflicting KB data.
    """

    model_config = ConfigDict(extra="forbid")

    server_name: str
    tool_name: str

    fact_id: str

    field: str
    value: str | int | float | bool | None

    object_ref: str | None = None  # masked identifier of the record the fact belongs to
    statement: str = ""  # human-readable rendering used as drafting evidence

    source: RetrievalSource = RetrievalSource.MCP

    authoritative: bool = True

    masked: bool = True


class RetrievalConflict(BaseModel):
    """
    Represents a disagreement between unstructured KB content
    and live structured MCP data.
    """

    model_config = ConfigDict(extra="forbid")

    conflict_id: str

    field: str

    kb_value: str
    mcp_value: str

    kb_document_id: str | None = None
    mcp_fact_id: str | None = None

    mcp_server: str
    mcp_tool: str

    authoritative_source: RetrievalSource = RetrievalSource.MCP

    resolution: str = "structured_mcp_data_wins"

    surfaced_to_validation: bool = True


class BlockedSource(BaseModel):
    """Retrieved content withheld from every model because it failed sanitisation."""

    model_config = ConfigDict(extra="forbid")

    source_reference: str
    source: RetrievalSource
    reason: str
    signatures: list[str] = Field(default_factory=list)


class ServiceError(BaseModel):
    """A typed MCP failure observed during retrieval (no raw exception text)."""

    model_config = ConfigDict(extra="forbid")

    server: str
    tool: str
    code: str
    message: str
    retryable: bool = False


class HybridRetrievalResult(BaseModel):
    """
    Raw hybrid retrieval result before reconciliation.
    """

    model_config = ConfigDict(extra="forbid")

    documents: list[RetrievedDocument] = Field(default_factory=list)

    structured_facts: list[StructuredFact] = Field(
        default_factory=list
    )

    retrieval_errors: list[ServiceError] = Field(
        default_factory=list
    )

    blocked_sources: list[BlockedSource] = Field(default_factory=list)

    kb_quality: KBQuality | None = None

    raw_tool_results: dict[str, Any] = Field(default_factory=dict, exclude=True)


class GroundedEvidence(BaseModel):
    """
    Evidence that the Drafting Agent is allowed to use.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str

    source: RetrievalSource

    content: str

    source_reference: str

    authoritative: bool = False


class GroundedContext(BaseModel):
    """
    Final reconciled evidence package passed to the Drafting Agent.

    Customer-facing claims should be derived only from this context.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    customer_id: str

    evidence: list[GroundedEvidence] = Field(
        default_factory=list
    )

    conflicts: list[RetrievalConflict] = Field(
        default_factory=list
    )

    authoritative_facts: list[StructuredFact] = Field(
        default_factory=list
    )

    blocked_sources: list[BlockedSource] = Field(default_factory=list)

    service_errors: list[ServiceError] = Field(default_factory=list)

    missing_requirements: list[str] = Field(default_factory=list)

    kb_quality: KBQuality | None = None

    customer_specific: bool = True

    retrieval_complete: bool = True

    grounding_summary: str = ""

    def fact(self, fact_id: str) -> StructuredFact | None:
        return next((f for f in self.authoritative_facts if f.fact_id == fact_id), None)

    def evidence_ids(self) -> set[str]:
        return {e.evidence_id for e in self.evidence}
