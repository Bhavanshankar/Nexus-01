"""Insurance MCP server: policy, coverage-clause and claim-status read tools plus ``policy://``.

Database scope: ``policies`` and ``claims`` only - structurally unable to read accounts,
transactions, portfolios or customers. Policy (and claim) numbers are masked in every response.
"""

import re
from typing import Any

from fastmcp import FastMCP

from app.guardrails import mask_identifier, minimize_free_text_pii
from app.logging_config import get_logger, trace
from mcp_servers.common import (
    connection,
    error,
    id_error,
    iso_utc,
    money,
    parse_coverage_clauses,
    register_resource,
    register_tool,
    serve,
    success,
    tool_failure,
)

logger = get_logger(__name__)
mcp = FastMCP("Nexus Insurance MCP")
SERVER = "insurance"
_POLICY_COLUMNS = (
    "policy_id, customer_id, policy_type, premium_cents, payment_frequency, payment_terms, "
    "auto_debit_permitted, status, coverage_clauses_json, started_at"
)
_SCENARIO_TAG = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
# Deterministic aliases so callers can use the natural tag for a scenario.
SCENARIO_ALIASES = {
    "auto_debit": "auto_debit_payment",
    "automatic_payment": "auto_debit_payment",
    "premium_payment": "general_payment_terms",
    "payment_terms": "general_payment_terms",
}
_COVERED_VALUES = {"permitted", "covered", "included", "allowed"}
_NOT_COVERED_VALUES = {"not_permitted", "excluded", "not_covered", "prohibited"}


@trace(logger)
def _policy_record(row: tuple) -> dict[str, Any]:
    return {
        "policy_id": mask_identifier(row[0]),
        "customer_id": row[1],
        "policy_type": row[2],
        "premium": money(row[3]),
        "payment_frequency": row[4],
        "payment_terms": minimize_free_text_pii(row[5]),
        "auto_debit_permitted": bool(row[6]),
        "status": row[7],
        "coverage_clauses": [
            {**clause, "text": minimize_free_text_pii(clause["text"])} for clause in parse_coverage_clauses(row[8])
        ],
        "started_at": iso_utc(row[9]),
    }


@trace(logger)
def _fetch_policy(policy_id: str) -> tuple | None:
    with connection(SERVER, read_tables={"policies"}) as conn:
        return conn.execute(f"SELECT {_POLICY_COLUMNS} FROM policies WHERE policy_id = ?", (policy_id,)).fetchone()


@trace(logger)
def get_policy_details(policy_id: str) -> dict[str, Any]:
    """Single policy lookup: premium, payment terms, auto-debit permission, status and clauses.
    The policy number is masked."""
    invalid = id_error("policy_id", policy_id)
    if invalid:
        return invalid
    try:
        row = _fetch_policy(policy_id)
        if row is None:
            return error("not_found", "The requested policy was not found.", field="policy_id")
        return success(_policy_record(row))
    except (ValueError, TypeError):  # malformed stored clause JSON; never echo the raw value
        return error("database_error", "Policy details could not be read.", retryable=False)
    except Exception as exc:
        return tool_failure(exc)


@trace(logger)
def check_coverage_clause(policy_id: str, scenario_tag: str) -> dict[str, Any]:
    """Deterministically check whether a scenario tag (e.g. auto_debit_payment) is covered by the
    policy's stored clauses. Never infers coverage that is not written in a clause."""
    invalid = id_error("policy_id", policy_id)
    if invalid:
        return invalid
    if not isinstance(scenario_tag, str) or not _SCENARIO_TAG.fullmatch(scenario_tag):
        return error(
            "validation_error",
            "scenario_tag must be 2-64 lowercase letters, digits or underscores.",
            field="scenario_tag",
        )
    resolved_tag = SCENARIO_ALIASES.get(scenario_tag, scenario_tag)
    try:
        row = _fetch_policy(policy_id)
        if row is None:
            return error("not_found", "The requested policy was not found.", field="policy_id")
        clauses = parse_coverage_clauses(row[8])
        matched = [clause for clause in clauses if clause["clause_id"] == resolved_tag]
        policy_status = row[7]
        if not matched:
            status, covered = "no_matching_clause", None
        else:
            value = matched[0]["text"].strip().lower()
            if policy_status != "active":
                status, covered = "policy_not_active", False
            elif value in _COVERED_VALUES:
                status, covered = "covered", True
            elif value in _NOT_COVERED_VALUES:
                status, covered = "not_covered", False
            elif value.startswith(("requires_", "subject_to_", "conditional")):
                status, covered = "conditional", False
            else:  # descriptive clause text (e.g. a billing schedule): the scenario is documented
                status, covered = "documented", True
        return success({
            "policy_id": mask_identifier(policy_id),
            "scenario_tag": scenario_tag,
            "resolved_clause_id": resolved_tag,
            "policy_status": policy_status,
            "coverage_status": status,
            "covered": covered,
            "matched_clauses": [{**c, "text": minimize_free_text_pii(c["text"])} for c in matched],
        })
    except (ValueError, TypeError):
        return error("database_error", "Policy clauses could not be read.", retryable=False)
    except Exception as exc:
        return tool_failure(exc)


@trace(logger)
def get_claim_status(claim_id: str) -> dict[str, Any]:
    """Claim status lookup only - no settlement authority. Returns the owning customer so callers
    can verify ownership; claim and policy numbers are masked."""
    invalid = id_error("claim_id", claim_id)
    if invalid:
        return invalid
    try:
        with connection(SERVER, read_tables={"claims", "policies"}) as conn:
            row = conn.execute(
                "SELECT claims.claim_id, claims.policy_id, policies.customer_id, claims.status, "
                "claims.filed_at, claims.last_updated_at FROM claims "
                "JOIN policies ON policies.policy_id = claims.policy_id WHERE claims.claim_id = ?",
                (claim_id,),
            ).fetchone()
        if row is None:
            return error("not_found", "The requested claim was not found.", field="claim_id")
        return success({
            "claim_id": mask_identifier(row[0]),
            "policy_id": mask_identifier(row[1]),
            "customer_id": row[2],
            "status": row[3],
            "filed_at": iso_utc(row[4]),
            "last_updated_at": iso_utc(row[5]),
        })
    except Exception as exc:
        return tool_failure(exc)


@trace(logger)
def policy_summary_resource(policy_id: str) -> dict[str, Any]:
    """policy://{policy_id}/summary - masked policy summary without ownership or clause text."""
    result = get_policy_details(policy_id)
    if not result.get("ok"):
        return result
    data = result["data"]
    return success({key: data[key] for key in (
        "policy_id", "policy_type", "status", "payment_frequency", "premium", "auto_debit_permitted",
    )})


for _tool in (get_policy_details, check_coverage_clause, get_claim_status):
    register_tool(mcp, _tool)
register_resource(mcp, "policy://{policy_id}/summary", policy_summary_resource)


if __name__ == "__main__":
    serve(mcp, "NEXUS_INSURANCE_PORT", 8102)
