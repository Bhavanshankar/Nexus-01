"""Drafting role: propose a reply from the grounded context only (PDF 3.4).

The prompt is the ``response_drafting_prompt`` from Concierge Ops MCP, filled with the grounding
summary (masked identifiers, sanitised text). For hard-category requests the escalation flag turns
the task into a holding reply that answers nothing substantive. The model's draft is validated into
``DraftResponse``; whether it is actually supported is decided later by the Validation Agent and the
final output gate, not by the drafter's own claims.
"""
from __future__ import annotations

from app.agents._shared import run_model
from app.agents.llm import LLMBackend
from app.logging_config import get_logger, trace
from app.mcp_client.gateway import MCPGateway
from app.models.intake import IntakeRequest
from app.models.response import DraftProposal, DraftResponse
from app.models.retrieval import GroundedContext

logger = get_logger(__name__)


class DraftingAgent:
    """Role 3 of 5: grounded context -> customer-facing draft."""

    role = "Grounded Financial Services Response Writer"
    goal = "Answer the customer's request using only the supplied evidence."
    backstory = (
        "You write concise, plain-language financial-services replies. You never invent customer, account, "
        "policy, investment or regulatory facts and never claim an action was performed."
    )

    @trace(logger)
    def __init__(self, llm: LLMBackend, gateway: MCPGateway) -> None:
        self.llm = llm
        self.gateway = gateway

    @trace(logger)
    def draft(self, intake: IntakeRequest, context: GroundedContext, *, customer_name: str | None) -> DraftResponse:
        holding = intake.is_hard_category
        prompt = self.gateway.get_prompt("response_drafting_prompt", {
            "classified_intent": f"{intake.request_type.value} ({intake.product_line.value}): {intake.customer_stated_intent}",
            "retrieved_context_summary": "" if holding else context.grounding_summary,
            "customer_name": customer_name or "there",
            "escalation_flag": "true" if holding else "false",
        })
        proposal = run_model(
            self.llm, role=self.role, goal=self.goal, backstory=self.backstory, prompt=prompt,
            output_model=DraftProposal,
            context={"intake": intake, "grounded_context": context, "customer_name": customer_name, "holding_reply": holding},
        )
        return DraftResponse(
            request_id=intake.request_id,
            message=proposal.message,
            # Unknown citations are kept so the Validation Agent can see and penalise them.
            evidence_ids=list(dict.fromkeys(proposal.cited_evidence_ids)),
            asserted_facts=proposal.asserted_facts,
            proposed_action=proposal.proposed_next_step,
            holding_reply=holding,
        )
