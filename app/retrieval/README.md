# Nexus retrieval

| File | Purpose |
| --- | --- |
| `retrieval.py` | Unstructured KB search: dense (embeddings) + BM25 -> RRF -> MMR -> deterministic rerank -> quality gate -> query expansion -> `INSUFFICIENT_CONTEXT` abstention. Result type `KBSearchResult`. |
| `adapter.py` | `KBSearchResult` -> `app.models.retrieval.RetrievedDocument` / `KBQuality` (keeps document ID, chunk ID, source path, section, score, metadata) |
| `planner.py` | Validated intake -> KB queries (per product line) + MCP calls (agent-scope ownership reads first) |
| `hybrid.py` | Executes the plan: KB search, explicit test-document injection, sanitisation, MCP calls, ownership checks, typed service errors, `GroundedContext` |
| `reconcile.py` | MCP results -> authoritative `StructuredFact`s; KB-vs-MCP `RetrievalConflict`s (MCP authoritative); evidence and the drafting summary |
| `demo.py`, `ragas_test.py` | Interactive query tool and evaluation scaffold over the real index |

## Index

Built by `python build_index.py` into `vector_store/chroma` (repository-relative; override with
`NEXUS_CHROMA_PATH`) with `vector_store/manifest.json`. `UnstructuredRetriever.from_chroma()` refuses an
index whose manifest names a different embedding model, collection or index version. Test/adversarial
documents (`faqs/test/`) are excluded unless `include_test_docs=True`; `HybridRetriever(test_injections=[...])`
is the explicit test mode that injects a named test document as if it had been retrieved.

Query embeddings use OpenAI `text-embedding-3-small` (the model the index was built with). Tests and the
offline demo use `UnstructuredRetriever.from_documents(chunks, embedder=HashingEmbedder())` over the same
chunks `build_index.build_chunks()` produces, with thresholds calibrated for that embedder.

```bash
python -m app.retrieval.retrieval "How does auto-debit work for insurance premiums?"
```

```bash
python -m app.retrieval.demo
```

Outbound TLS uses the operating-system certificate store (`app/runtime.py`), which is required behind
TLS-inspecting proxies; verification is never disabled.
