"""
Nexus - Unstructured Knowledge Retrieval
-----------------------------------------
Hybrid search over the knowledge-base chunks built by ``build_index.py``.

Pipeline:
1. Dense semantic search using the same embedding model used to build the index.
2. BM25 lexical search over the same chunks.
3. Reciprocal Rank Fusion (RRF).
4. MMR diversification.
5. Lightweight deterministic reranking.
6. Quality gate.
7. Deterministic query expansion + second search if the first result is weak.
8. Safe INSUFFICIENT_CONTEXT response if evidence is still weak.

This module does NOT use MCP or structured data. Its result type is ``KBSearchResult``; use
``app.retrieval.adapter`` to convert it into ``app.models.retrieval.RetrievedDocument``.

Production uses ``UnstructuredRetriever.from_chroma()`` (OpenAI query embeddings, manifest
checked for model compatibility). Tests and the local demo use ``from_documents()`` with the
deterministic ``HashingEmbedder`` so they run offline and repeatably.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from app.logging_config import get_logger, trace

logger = get_logger(__name__)

# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_PATH = BASE_DIR / "vector_store" / "chroma"
DEFAULT_MANIFEST_PATH = BASE_DIR / "vector_store" / "manifest.json"
COLLECTION_NAME = os.getenv("NEXUS_COLLECTION_NAME", "nexus_knowledge")
EXPECTED_INDEX_VERSION = "1.0.0"

EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

DENSE_K = int(os.getenv("DENSE_K", "12"))
BM25_K = int(os.getenv("BM25_K", "12"))
HYBRID_K = int(os.getenv("HYBRID_K", "15"))
MMR_K = int(os.getenv("MMR_K", "8"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.72"))
RRF_K = int(os.getenv("RRF_K", "60"))


@trace(logger)
def chroma_path() -> Path:
    """Repository-relative index location, overridable with ``NEXUS_CHROMA_PATH``."""
    return Path(os.getenv("NEXUS_CHROMA_PATH", str(DEFAULT_CHROMA_PATH)))


class IndexCompatibilityError(RuntimeError):
    """The persisted index cannot be used with the configured embedding model/collection."""


@dataclass(frozen=True)
class QualityThresholds:
    """Quality-gate thresholds. Defaults are tuned for OpenAI embeddings; the hashing embedder
    produces lower cosine similarities and uses ``for_hashing_embedder()``."""

    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.65"))
    min_coverage: float = float(os.getenv("MIN_COVERAGE", "0.60"))
    min_authority: float = float(os.getenv("MIN_AUTHORITY", "0.60"))
    min_specificity: float = float(os.getenv("MIN_SPECIFICITY", "0.50"))
    strong_score: float = float(os.getenv("STRONG_SCORE", "0.80"))

    @classmethod
    def for_hashing_embedder(cls) -> "QualityThresholds":
        return cls(min_relevance=0.45, min_coverage=0.55, min_authority=0.60, min_specificity=0.35, strong_score=0.72)


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    dense_distance: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    mmr_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class QualityResult:
    relevance: float
    coverage: float
    authority: float
    specificity: float
    overall: float
    passed: bool
    reason: str


@dataclass
class KBSearchResult:
    retrieval_status: str
    query: str
    confidence: float
    safe_to_answer: bool
    reason: str
    chunks: List[Chunk]
    quality: QualityResult
    expanded_queries: List[str] = field(default_factory=list)


# -----------------------------
# Embedders
# -----------------------------

class OpenAIEmbedder:
    """Query embeddings from the OpenAI model the index was built with."""

    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; configure it in the environment or .env.")
        self.model = model
        self._client = OpenAI(api_key=api_key)

    @trace(logger)
    def __call__(self, text: str) -> List[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


class HashingEmbedder:
    """Deterministic offline embedder (feature hashing over word unigrams and bigrams)."""

    model = "nexus-hashing-v1"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    @trace(logger)
    def __call__(self, text: str) -> List[float]:
        tokens = UnstructuredRetriever._tokenize(text)
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


# -----------------------------
# Retriever
# -----------------------------

@trace(logger)
def is_test_chunk(metadata: Dict[str, Any]) -> bool:
    """Test/adversarial documents are excluded from ordinary retrieval."""
    doc_type = str(metadata.get("document_type", "")).lower()
    source = str(metadata.get("source_path", "")).lower().replace("\\", "/")
    category = str(metadata.get("category", "")).lower()
    return doc_type == "test" or "/test/" in source or "faqs/test" in source or category == "test"


class UnstructuredRetriever:
    @trace(logger, redact=lambda a: {"chunks": f"<{len(a.get('chunks') or [])} chunks>", "include_test_docs": a.get("include_test_docs")})
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: Dict[str, List[float]],
        embed: Callable[[str], List[float]],
        *,
        thresholds: QualityThresholds | None = None,
        include_test_docs: bool = False,
    ) -> None:
        self.embed = embed
        self.thresholds = thresholds or QualityThresholds()
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, List[float]] = {}
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            if not include_test_docs and is_test_chunk(chunk.metadata):
                continue
            self.chunks[chunk.id] = chunk
            if chunk.id in embeddings:
                self.embeddings[chunk.id] = list(embeddings[chunk.id])
        if not self.chunks:
            raise RuntimeError("No usable chunks remain after filtering test documents.")
        self.ids = list(self.chunks.keys())
        self.corpus = [self._tokenize(self.chunks[i].text) for i in self.ids]
        self.bm25 = BM25Okapi(self.corpus)

    # ---------- construction ----------

    @classmethod
    @trace(logger)
    def from_chroma(
        cls,
        *,
        path: Path | None = None,
        collection_name: str = COLLECTION_NAME,
        embedder: Callable[[str], List[float]] | None = None,
        include_test_docs: bool = False,
        manifest_path: Path | None = None,
    ) -> "UnstructuredRetriever":
        """Open the persisted Chroma index after checking its manifest matches this retriever."""
        import chromadb
        from chromadb.config import Settings

        path = path or chroma_path()
        if not path.exists():
            raise FileNotFoundError(f"Chroma index not found at {path}. Run: python build_index.py")
        embedder = embedder or OpenAIEmbedder()
        validate_manifest(manifest_path or path.parent / "manifest.json", getattr(embedder, "model", EMBEDDING_MODEL), collection_name)

        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        try:
            collection = client.get_collection(collection_name, embedding_function=None)
        except Exception as exc:
            raise IndexCompatibilityError(f"Could not open Chroma collection '{collection_name}'.") from exc
        stored_model = (collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != getattr(embedder, "model", stored_model):
            raise IndexCompatibilityError(f"Index embedded with {stored_model}, query embedder is {embedder.model}.")
        data = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = list(data.get("ids") if data.get("ids") is not None else [])
        documents = list(data.get("documents") if data.get("documents") is not None else [])
        metadatas = list(data.get("metadatas") if data.get("metadatas") is not None else [])
        vectors = data.get("embeddings")
        vectors = list(vectors) if vectors is not None else []
        if not ids:
            raise RuntimeError(f"Collection '{collection_name}' is empty.")
        chunks = [
            Chunk(id=str(cid), text=documents[i] or "", metadata=dict(metadatas[i] or {}))
            for i, cid in enumerate(ids)
        ]
        embeddings = {str(cid): list(vectors[i]) for i, cid in enumerate(ids) if i < len(vectors) and vectors[i] is not None}
        return cls(chunks, embeddings, embedder, include_test_docs=include_test_docs)

    @classmethod
    @trace(logger)
    def from_documents(
        cls,
        chunks: Iterable[dict[str, Any]],
        *,
        embedder: Callable[[str], List[float]] | None = None,
        thresholds: QualityThresholds | None = None,
        include_test_docs: bool = False,
    ) -> "UnstructuredRetriever":
        """Build an in-memory retriever from ``{"id", "text", "metadata"}`` chunk dicts."""
        embedder = embedder or HashingEmbedder()
        chunk_objects = [Chunk(id=c["id"], text=c["text"], metadata=dict(c["metadata"])) for c in chunks]
        embeddings = {c.id: embedder(c.text) for c in chunk_objects}
        return cls(
            chunk_objects,
            embeddings,
            embedder,
            thresholds=thresholds or (QualityThresholds.for_hashing_embedder() if isinstance(embedder, HashingEmbedder) else None),
            include_test_docs=include_test_docs,
        )

    # ---------- utilities ----------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9_./:-]*", text.lower())

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    def _fresh(self, chunk_id: str) -> Chunk:
        # Per-query copy: scores are never accumulated on the shared index objects.
        base = self.chunks[chunk_id]
        return Chunk(id=base.id, text=base.text, metadata=base.metadata)

    def _allowed(self, chunk_id: str, product_lines: frozenset[str] | None) -> bool:
        if not product_lines:
            return True
        return str(self.chunks[chunk_id].metadata.get("product_line", "")).lower() in product_lines

    # ---------- dense search ----------

    @trace(logger)
    def dense_search(self, query: str, product_lines: frozenset[str] | None = None) -> Tuple[List[Chunk], List[float]]:
        query_vector = self.embed(query)
        scored: List[Tuple[float, str]] = []
        for chunk_id, vector in self.embeddings.items():
            if not self._allowed(chunk_id, product_lines):
                continue
            scored.append((self._cosine(query_vector, vector), chunk_id))
        scored.sort(reverse=True)
        results = []
        for similarity, chunk_id in scored[:DENSE_K]:
            c = self._fresh(chunk_id)
            c.dense_score = similarity
            c.dense_distance = 1.0 - similarity
            results.append(c)
        return results, query_vector

    # ---------- BM25 ----------

    @trace(logger)
    def bm25_search(self, query: str, product_lines: frozenset[str] | None = None) -> List[Chunk]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            ((score, chunk_id) for score, chunk_id in zip(scores, self.ids) if self._allowed(chunk_id, product_lines)),
            key=lambda x: x[0],
            reverse=True,
        )[:BM25_K]
        results = []
        for score, chunk_id in ranked:
            c = self._fresh(chunk_id)
            c.bm25_score = float(score)
            results.append(c)
        return results

    # ---------- RRF ----------

    @trace(logger)
    def rrf(self, dense: List[Chunk], lexical: List[Chunk]) -> List[Chunk]:
        by_id: Dict[str, Chunk] = {}
        for rank, chunk in enumerate(dense, start=1):
            merged = by_id.setdefault(chunk.id, chunk)
            merged.rrf_score += 1.0 / (RRF_K + rank)
        for rank, chunk in enumerate(lexical, start=1):
            merged = by_id.setdefault(chunk.id, chunk)
            if merged is not chunk and merged.bm25_score is None:
                merged.bm25_score = chunk.bm25_score
            merged.rrf_score += 1.0 / (RRF_K + rank)
        return sorted(by_id.values(), key=lambda c: c.rrf_score, reverse=True)[:HYBRID_K]

    # ---------- MMR ----------

    @trace(logger)
    def mmr(self, candidates: List[Chunk], query_vector: List[float]) -> List[Chunk]:
        selected: List[Chunk] = []
        remaining = list(candidates)

        while remaining and len(selected) < MMR_K:
            best = None
            best_score = -float("inf")

            for candidate in remaining:
                relevance = (
                    self._cosine(query_vector, self.embeddings.get(candidate.id, []))
                    if candidate.id in self.embeddings
                    else 0.0
                )
                if not selected:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = max(
                        self._cosine(
                            self.embeddings.get(candidate.id, []),
                            self.embeddings.get(other.id, []),
                        )
                        for other in selected
                    )
                score = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * diversity_penalty
                if score > best_score:
                    best_score = score
                    best = candidate

            if best is None:
                break
            best.mmr_score = best_score
            selected.append(best)
            remaining.remove(best)

        return selected

    # ---------- deterministic reranker ----------

    @trace(logger)
    def rerank(self, query: str, candidates: List[Chunk]) -> List[Chunk]:
        q_tokens = set(self._tokenize(query))

        for chunk in candidates:
            text_tokens = set(self._tokenize(chunk.text))
            lexical_overlap = len(q_tokens & text_tokens) / len(q_tokens) if q_tokens else 0.0
            dense = chunk.dense_score if chunk.dense_score is not None else 0.0
            rrf_norm = min(chunk.rrf_score / (1.0 / (RRF_K + 1) * 2), 1.0)
            doc_type = str(chunk.metadata.get("document_type", "")).lower()
            authority = self._authority_score(chunk.metadata)
            type_boost = {
                "policy": 0.10,
                "product": 0.07,
                "faq": 0.05,
                "regulatory": 0.10,
            }.get(doc_type, 0.0)

            chunk.rerank_score = self._clip(
                0.50 * dense
                + 0.20 * lexical_overlap
                + 0.15 * rrf_norm
                + 0.15 * authority
                + type_boost
            )

        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

    # ---------- quality gate ----------

    @trace(logger)
    def _authority_score(self, metadata: Dict[str, Any]) -> float:
        authority = str(metadata.get("authority", "")).lower()
        source_class = str(metadata.get("source_class", "")).lower()
        doc_type = str(metadata.get("document_type", "")).lower()

        score = 0.45
        if authority:
            if any(x in authority for x in ["official", "authoritative", "approved"]):
                score += 0.30
            elif any(x in authority for x in ["internal", "controlled"]):
                score += 0.20
        if source_class:
            if any(x in source_class for x in ["policy", "regulatory", "official"]):
                score += 0.15
        if doc_type in {"policy", "regulatory"}:
            score += 0.10
        return self._clip(score)

    @trace(logger)
    def quality_gate(self, query: str, chunks: List[Chunk]) -> QualityResult:
        t = self.thresholds
        if not chunks:
            return QualityResult(0.0, 0.0, 0.0, 0.0, 0.0, False, "No evidence was retrieved.")

        top = chunks[:MMR_K]
        top_scores = [c.rerank_score for c in top]
        relevance = sum(top_scores[:3]) / max(1, min(3, len(top_scores)))

        q_tokens = {tok for tok in self._tokenize(query) if len(tok) >= 3}
        evidence_tokens = set()
        for c in top:
            evidence_tokens.update(self._tokenize(c.text))
        coverage = len(q_tokens & evidence_tokens) / len(q_tokens) if q_tokens else 0.0

        authority = sum(self._authority_score(c.metadata) for c in top) / len(top)

        specificity = 0.0
        for c in top[:3]:
            if any(
                token in c.text.lower()
                for token in [
                    "policy_id", "document_id", "product_line",
                    "effective_date", "procedure", "eligibility",
                    "requirement", "allowed", "not allowed", "must", "permit",
                ]
            ):
                specificity += 0.35
        specificity = self._clip(specificity)

        overall = 0.35 * relevance + 0.25 * coverage + 0.20 * authority + 0.20 * specificity

        passed = (
            relevance >= t.min_relevance
            and coverage >= t.min_coverage
            and authority >= t.min_authority
            and specificity >= t.min_specificity
        ) or overall >= t.strong_score

        reason = (
            "Evidence passed the retrieval quality gate."
            if passed
            else "Evidence did not meet the minimum relevance/coverage/authority/specificity thresholds."
        )
        return QualityResult(
            float(round(float(relevance), 4)),
            float(round(float(coverage), 4)),
            float(round(float(authority), 4)),
            float(round(float(specificity), 4)),
            float(round(float(overall), 4)),
            bool(passed),
            str(reason),
        )

    # ---------- query expansion ----------

    @trace(logger, redact_result=True)
    def expand_query(self, query: str) -> List[str]:
        """Cheap deterministic expansion: no extra model call, so the fallback is predictable."""
        q = query.strip()
        expansions = [q, f"{q} policy requirements eligibility procedure", f"{q} official policy document rules terms conditions"]
        identifiers = re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b|\b[A-Z]{2,}-\d{3,}\b", q)
        if identifiers:
            expansions.append(f"{' '.join(identifiers)} policy official requirements")
        output: List[str] = []
        seen = set()
        for item in expansions:
            normalized = item.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                output.append(item)
        return output[:3]

    # ---------- complete stages ----------

    @trace(logger)
    def _stage(self, query: str, product_lines: frozenset[str] | None = None) -> List[Chunk]:
        dense, qvec = self.dense_search(query, product_lines)
        lexical = self.bm25_search(query, product_lines)
        hybrid = self.rrf(dense, lexical)
        diversified = self.mmr(hybrid, qvec)
        return self.rerank(query, diversified)

    @trace(logger)
    def retrieve(self, query: str, product_lines: Iterable[str] | None = None) -> KBSearchResult:
        """Search the knowledge base, optionally restricted to chunks of the given product lines."""
        query = query.strip()
        lines = frozenset(p.lower() for p in product_lines) if product_lines else None

        if not query:
            quality = QualityResult(0, 0, 0, 0, 0, False, "Empty query.")
            return KBSearchResult("INSUFFICIENT_CONTEXT", query, 0.0, False, "Query is empty.", [], quality, [])

        first = self._stage(query, lines)
        q1 = self.quality_gate(query, first)
        if q1.passed:
            return KBSearchResult("SUCCESS", query, q1.overall, True, q1.reason, first, q1, [])

        expanded = self.expand_query(query)
        candidates: Dict[str, Chunk] = {}
        for expanded_query in expanded:
            for c in self._stage(expanded_query, lines):
                old = candidates.get(c.id)
                if old is None or c.rerank_score > old.rerank_score:
                    candidates[c.id] = c

        final_chunks = sorted(candidates.values(), key=lambda c: c.rerank_score, reverse=True)[:MMR_K]
        q2 = self.quality_gate(query, final_chunks)
        if q2.passed:
            return KBSearchResult("SUCCESS_AFTER_EXPANSION", query, q2.overall, True, q2.reason, final_chunks, q2, expanded)

        return KBSearchResult(
            "INSUFFICIENT_CONTEXT",
            query,
            q2.overall,
            False,
            "The unstructured knowledge base did not provide sufficient authoritative evidence to answer safely.",
            final_chunks,
            q2,
            expanded,
        )


@trace(logger)
def validate_manifest(manifest_path: Path, embedding_model: str, collection_name: str = COLLECTION_NAME) -> dict[str, Any]:
    """Refuse an index built with a different embedding model, collection or index version."""
    if not manifest_path.exists():
        raise IndexCompatibilityError(f"Index manifest not found at {manifest_path}. Run: python build_index.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("embedding_model") != embedding_model:
        problems.append(f"embedding_model {manifest.get('embedding_model')} != {embedding_model}")
    if manifest.get("collection_name") != collection_name:
        problems.append(f"collection_name {manifest.get('collection_name')} != {collection_name}")
    if manifest.get("index_version") != EXPECTED_INDEX_VERSION:
        problems.append(f"index_version {manifest.get('index_version')} != {EXPECTED_INDEX_VERSION}")
    if problems:
        raise IndexCompatibilityError("Index manifest is incompatible: " + "; ".join(problems))
    return manifest


# -----------------------------
# Output helpers
# -----------------------------

@trace(logger, redact_result=True)
def format_context(result: KBSearchResult) -> str:
    if not result.chunks:
        return "INSUFFICIENT_CONTEXT"

    blocks = []
    for i, chunk in enumerate(result.chunks, start=1):
        md = chunk.metadata
        source = md.get("source_path") or md.get("document_id") or chunk.id
        blocks.append(
            f"[SOURCE {i}]\n"
            f"Document: {md.get('source_document_id') or md.get('document_id', 'unknown')}\n"
            f"Title: {md.get('title', 'unknown')}\n"
            f"Type: {md.get('document_type', 'unknown')}\n"
            f"Product: {md.get('product_line', 'unknown')}\n"
            f"Source: {source}\n"
            f"Score: {chunk.rerank_score:.4f}\n"
            f"Content:\n{chunk.text}"
        )
    return "\n\n".join(blocks)


@trace(logger)
def print_result(result: KBSearchResult) -> None:
    print("\n" + "=" * 80)
    print("NEXUS UNSTRUCTURED RETRIEVAL")
    print("=" * 80)
    print(f"Status      : {result.retrieval_status}")
    print(f"Confidence  : {result.confidence:.4f}")
    print(f"Safe answer : {result.safe_to_answer}")
    print(f"Reason      : {result.reason}")
    print("\nQuality:")
    print(json.dumps(asdict(result.quality), indent=2))
    if result.expanded_queries:
        print("\nExpanded queries:")
        for q in result.expanded_queries:
            print(f" - {q}")
    print("\nRetrieved evidence:")
    if not result.chunks:
        print("  NONE")
    else:
        for i, c in enumerate(result.chunks, 1):
            md = c.metadata
            print(f"\n[{i}] {md.get('source_document_id') or md.get('document_id', c.id)} | {md.get('title', 'unknown')} | score={c.rerank_score:.4f}")
            print(c.text[:900].replace("\n", " "))
    print("\n" + "=" * 80)


@trace(logger)
def main() -> None:
    from app.runtime import bootstrap

    bootstrap()
    parser = argparse.ArgumentParser(description="Query the Nexus knowledge-base index.")
    parser.add_argument("query", nargs="*", help="Question to retrieve evidence for.")
    parser.add_argument("--include-test-docs", action="store_true", help="explicit test mode: include adversarial test documents")
    args = parser.parse_args()

    query = " ".join(args.query).strip() or input("Enter your question: ").strip()
    print(f"Chroma path : {chroma_path()}")
    print(f"Collection  : {COLLECTION_NAME}")
    print(f"Embedding   : {EMBEDDING_MODEL}")
    retriever = UnstructuredRetriever.from_chroma(include_test_docs=args.include_test_docs)
    result = retriever.retrieve(query)
    print_result(result)
    print("\nContext for downstream answer generation:\n")
    print(format_context(result))


if __name__ == "__main__":
    main()
