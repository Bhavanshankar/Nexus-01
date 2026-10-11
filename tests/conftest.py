"""Shared fixtures. Tests never touch data/meridian.db: every test gets its own seeded copy."""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_SESSION_DIR = Path(tempfile.mkdtemp(prefix="nexus-tests-"))
# Route DEBUG logs to a file before any project module creates its logger.
os.environ.setdefault("NEXUS_LOG_FILE", str(_SESSION_DIR / "pytest-nexus.jsonl"))
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.pop("NEXUS_TRUST_INTERNAL_SCOPE", None)

import build_index  # noqa: E402
from database import seed_all  # noqa: E402

TEMPLATE_DB = _SESSION_DIR / "template.db"
seed_all(TEMPLATE_DB)

ARJUN = {
    "message": "Can I set up auto-debit for my insurance premium from my savings account, and is auto-debit actually covered under my policy's payment terms?",
    "customer_id": "CUS-20077",
    "references": {"account_id": "ACC-20077", "policy_id": "POL-IN-30091"},
}
PRIYA = {
    "message": "Can I set up auto-debit for my insurance premium from my savings account?",
    "customer_id": "CUS-30142",
    "references": {"account_id": "ACC-30142", "policy_id": "POL-GE-30250"},
}


@pytest.fixture
def db_path(tmp_path, monkeypatch) -> Path:
    """A fresh seeded database for one test; servers resolve it through NEXUS_DB_PATH."""
    path = tmp_path / "meridian.db"
    shutil.copy2(TEMPLATE_DB, path)
    monkeypatch.setenv("NEXUS_DB_PATH", str(path))
    return path


@pytest.fixture
def db_exec(db_path):
    """Run raw SQL against the test database (fixture set-up only - never used by code under test)."""

    def run(sql: str, params: tuple = ()) -> list[tuple]:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
            conn.commit()
            return rows
        finally:
            conn.close()

    return run


@pytest.fixture(scope="session")
def kb_retriever():
    """Offline KB index over the same chunks build_index.py produces (deterministic embedder)."""
    from app.retrieval.retrieval import HashingEmbedder, UnstructuredRetriever

    build_index.QUIET = True
    _, chunks, _ = build_index.build_chunks()
    return UnstructuredRetriever.from_documents(chunks, embedder=HashingEmbedder())


@pytest.fixture(scope="session")
def kb_chunks():
    build_index.QUIET = True
    _, chunks, _ = build_index.build_chunks()
    return chunks


@pytest.fixture
def gateway(db_path):
    from app.mcp_client.in_process import InProcessMCPGateway

    return InProcessMCPGateway(db_path)


@pytest.fixture
def make_orchestrator(db_path, kb_retriever):
    """Factory: orchestrator wired with the in-process gateway and scripted model."""
    from app.agents.scripted import ScriptedLLM
    from app.mcp_client.in_process import InProcessMCPGateway
    from nexus_orchestrator import NexusOrchestrator

    def build(*, llm=None, test_injections=(), gateway=None, **faults):
        gw = gateway or InProcessMCPGateway(db_path, **faults)
        return NexusOrchestrator(gateway=gw, llm=llm or ScriptedLLM(), kb_search=kb_retriever, test_injections=test_injections)

    return build
