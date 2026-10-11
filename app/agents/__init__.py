"""Nexus agent roles.

Five distinct roles, run in sequence by ``nexus_orchestrator.NexusOrchestrator``:

1. ``ClassifierAgent``  - model-assisted, deterministic guards -> ``IntakeRequest``
2. ``RetrieverAgent``   - deterministic hybrid retrieval -> ``GroundedContext``
3. ``DraftingAgent``    - model-assisted -> ``DraftResponse``
4. ``ValidationAgent``  - deterministic checks + model review -> ``ValidationResult``
5. ``EscalationAgent``  - MCP ``run_escalation_check`` -> ``EscalationResult``

Model access goes through an injected ``LLMBackend`` (``CrewAIBackend`` in production,
``ScriptedLLM`` for the demo and tests); MCP access through an injected ``MCPGateway``.
"""

from app.agents.classifier_agent import ClassifierAgent
from app.agents.drafting_agent import DraftingAgent
from app.agents.escalation_agent import EscalationAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.validation_agent import ValidationAgent

__all__ = [
    "ClassifierAgent",
    "RetrieverAgent",
    "DraftingAgent",
    "ValidationAgent",
    "EscalationAgent",
]
