# Nexus agent roles

Five distinct roles, run in sequence by `nexus_orchestrator.NexusOrchestrator`. Each validates its
output at the boundary; models never touch MCP or data directly.

| File | Role | Model? | Output |
| --- | --- | --- | --- |
| `classifier_agent.py` | Classifier | yes (prompt fetched from MCP `request_classification_prompt`) | `IntakeRequest` - deterministic hard-category signals override the model (it may add a category, never remove one); `request_id`/`customer_id` come from the server/trusted metadata; references spanning two product lines force `cross_product`; invalid output -> `IntakeRejected(incomplete_extraction)` |
| `retriever_agent.py` | Retriever | no | `GroundedContext` from `app/retrieval/hybrid.py` (KB + live MCP facts, ownership check, sanitisation, code-detected conflicts) |
| `drafting_agent.py` | Drafting | yes (prompt from MCP `response_drafting_prompt`) | `DraftResponse` with cited evidence IDs and asserted facts; hard categories get a holding reply |
| `validation_agent.py` | Validation | deterministic checks + model review | `ValidationResult`; confidence = `min(deterministic caps, model)`; sees every conflict |
| `escalation_agent.py` | Escalation | no | `EscalationResult` from the real `run_escalation_check` MCP call, verified (echo + hard-rule invariant); tool failure raises `EscalationUnavailable` |

## Model backends (`llm.py`, `scripted.py`)

`LLMBackend.structured(role, goal, backstory, prompt, output_model, context)` is the only model
interface.

- `CrewAIBackend` (production): one CrewAI `Agent` (no tools, no delegation) + `Task` with
  `output_pydantic`, model from `NEXUS_CREWAI_MODEL` (default `gpt-4o-mini`). Telemetry is disabled in
  `app/runtime.py`. CrewAI is imported lazily.
- `ScriptedLLM` (demo and tests): deterministic, same interface. Its classifier never assigns a hard
  category, which demonstrates that the deterministic signals do. `overrides` / `fail_roles` inject
  misbehaving outputs for tests.

## Usage

```python
from app.agents import ClassifierAgent, RetrieverAgent, DraftingAgent, ValidationAgent, EscalationAgent
```

Normally you call `NexusOrchestrator.handle(request)` instead, which wires all five with a gateway, a
model backend and the KB retriever, and adds the intake gate, identity check, output gate, delivery and
audit steps.
