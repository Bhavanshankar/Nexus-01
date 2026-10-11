"""Retriever role: produce the reconciled ``GroundedContext`` (PDF 3.3 / 3.4).

Deliberately model-free. Retrieval planning, MCP calls, ownership checks, sanitisation and conflict
detection are deterministic code in ``app.retrieval.hybrid``; a model is never asked to invent or
find conflicts. The role's output is always a validated ``GroundedContext`` - never the raw
vector-search dataclass.
"""
from __future__ import annotations

from app.logging_config import get_logger, trace
from app.models.intake import IntakeRequest
from app.models.retrieval import GroundedContext
from app.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


class RetrieverAgent:
    """Role 2 of 5: validated intake -> grounded, reconciled context."""

    @trace(logger)
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    @trace(logger)
    def retrieve(self, intake: IntakeRequest) -> GroundedContext:
        return GroundedContext.model_validate(self.retriever.retrieve(intake).model_dump())
