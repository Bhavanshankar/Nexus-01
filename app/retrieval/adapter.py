"""Adapter: ``app.retrieval.retrieval.KBSearchResult`` -> ``app.models.retrieval`` contracts.

Preserves the KB document ID, chunk ID, source path, section, score and metadata so every piece of
evidence the drafting agent sees can be traced back to its file.
"""

from __future__ import annotations

from app.logging_config import get_logger, trace
from app.models.retrieval import KBQuality, RetrievedDocument
from app.retrieval.retrieval import Chunk, KBSearchResult

logger = get_logger(__name__)
_METADATA_KEYS = ("document_hash", "chunk_index", "index_version", "embedding_model", "category", "subtopic")


@trace(logger)
def chunk_to_document(chunk: Chunk) -> RetrievedDocument:
    md = chunk.metadata
    source_path = str(md.get("source_path") or md.get("document_id") or chunk.id)
    document_id = str(md.get("source_document_id") or source_path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    return RetrievedDocument(
        document_id=document_id,
        chunk_id=str(md.get("chunk_id") or chunk.id),
        product_line=str(md.get("product_line") or "") or None,
        document_type=str(md.get("document_type") or "general"),
        topic=str(md.get("topic") or "") or None,
        title=str(md.get("title") or "") or None,
        section=str(md.get("section") or "") or None,
        source_path=source_path,
        content=chunk.text,
        score=max(0.0, min(1.0, float(chunk.rerank_score))),
        metadata={k: str(md[k]) for k in _METADATA_KEYS if k in md and md[k] is not None},
    )


@trace(logger)
def to_retrieved_documents(result: KBSearchResult) -> list[RetrievedDocument]:
    return [chunk_to_document(chunk) for chunk in result.chunks if chunk.text.strip()]


@trace(logger)
def to_kb_quality(results: list[KBSearchResult]) -> KBQuality:
    """Aggregate several searches: passed only if every required search passed."""
    if not results:
        return KBQuality(status="NOT_REQUESTED", confidence=0.0, passed=False, queries=[])
    passed = all(r.safe_to_answer for r in results)
    confidence = min(float(r.confidence) for r in results)
    statuses = {r.retrieval_status for r in results}
    status = "SUCCESS" if statuses == {"SUCCESS"} else ("INSUFFICIENT_CONTEXT" if not passed else "SUCCESS_AFTER_EXPANSION")
    return KBQuality(status=status, confidence=max(0.0, min(1.0, confidence)), passed=passed, queries=[r.query for r in results])
