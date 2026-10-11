"""Model backends for the LLM-assisted agent roles.

``LLMBackend.structured()`` runs one role-specific task and returns a validated Pydantic object.
Production uses ``CrewAIBackend`` (one CrewAI Agent + Task with ``output_pydantic`` per call).
The demo and tests use ``app.agents.scripted.ScriptedLLM``, which implements the same interface
deterministically. The orchestrator never trusts backend output directly: every agent validates it
at the boundary and applies deterministic checks.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.logging_config import get_logger, trace

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class ModelOutputError(RuntimeError):
    """The model returned nothing usable for the requested schema."""


class LLMBackend(Protocol):
    name: str

    def structured(
        self,
        *,
        role: str,
        goal: str,
        backstory: str,
        prompt: str,
        output_model: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        """Run one task. ``context`` carries already-validated structured inputs; production
        backends rely on ``prompt`` alone, deterministic fakes may read ``context``."""


class CrewAIBackend:
    """CrewAI execution: an Agent with no tools and no delegation, one Task, structured output."""

    name = "crewai"

    @trace(logger)
    def __init__(self, model: str | None = None, *, temperature: float = 0.0, max_iter: int = 3) -> None:
        self.model = model or os.getenv("NEXUS_CREWAI_MODEL") or os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
        self.temperature = temperature
        self.max_iter = max_iter
        self._llm = None

    @trace(logger)
    def _crewai(self):
        try:
            from crewai import LLM, Agent, Crew, Process, Task
        except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements.txt
            raise RuntimeError("CrewAI is required for the production backend; install requirements.txt") from exc
        return LLM, Agent, Crew, Process, Task

    @trace(logger)
    def structured(
        self,
        *,
        role: str,
        goal: str,
        backstory: str,
        prompt: str,
        output_model: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        LLM, Agent, Crew, Process, Task = self._crewai()
        if self._llm is None:
            self._llm = LLM(model=self.model, temperature=self.temperature)
        agent = Agent(
            role=role, goal=goal, backstory=backstory, llm=self._llm,
            allow_delegation=False, verbose=False, max_iter=self.max_iter,
        )
        task = Task(
            description=prompt,
            expected_output=f"One valid {output_model.__name__} object.",
            agent=agent,
            output_pydantic=output_model,
        )
        # kickoff() without inputs: prompt text is never template-interpolated.
        result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
        structured = getattr(result, "pydantic", None)
        if structured is None:
            raw = getattr(result, "raw", None)
            if not isinstance(raw, str):
                raise ModelOutputError(f"{role}: no structured output")
            try:
                structured = output_model.model_validate_json(raw)
            except ValidationError as exc:
                raise ModelOutputError(f"{role}: output did not match {output_model.__name__}") from exc
        return output_model.model_validate(structured.model_dump() if isinstance(structured, BaseModel) else structured)
