"""Helpers shared by the Nexus agent roles.

Sanitisation is not implemented here: every role uses the shared ``app.guardrails`` module, which
replaced the earlier agent-local fallback.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.llm import LLMBackend, ModelOutputError
from app.logging_config import get_logger, trace

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)


class AgentStepError(RuntimeError):
    """A role could not produce a valid output; carries a safe code for the pipeline error."""

    def __init__(self, role: str, code: str, message: str) -> None:
        super().__init__(f"{role}: {message}")
        self.role = role
        self.code = code
        self.message = message


@trace(logger, redact=lambda a: {"role": a.get("role"), "output_model": getattr(a.get("output_model"), "__name__", None)})
def run_model(
    llm: LLMBackend,
    *,
    role: str,
    goal: str,
    backstory: str,
    prompt: str,
    output_model: type[T],
    context: dict[str, Any] | None = None,
) -> T:
    """Run one model task and validate the result at the boundary; never returns unvalidated data."""
    try:
        result = llm.structured(role=role, goal=goal, backstory=backstory, prompt=prompt, output_model=output_model, context=context)
        return output_model.model_validate(result.model_dump() if isinstance(result, BaseModel) else result)
    except (ModelOutputError, ValidationError, TypeError, ValueError) as exc:
        raise AgentStepError(role, "model_output_invalid", f"model output failed {output_model.__name__} validation") from exc


@trace(logger)
def timed(operation: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = operation()
    return result, round((time.perf_counter() - started) * 1000, 3)
