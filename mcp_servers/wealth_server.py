"""Wealth MCP server: portfolio and investment-product read tools.

Database scope: ``portfolios`` and ``investment_products`` only - no account or policy data.
Every tool takes an explicit ``caller_scope`` (PDF section 4); portfolio numbers are masked.
"""

from typing import Any

from fastmcp import FastMCP

from app.guardrails import minimize_free_text_pii, minimize_portfolio_fields
from app.logging_config import get_logger, trace
from mcp_servers.common import (
    connection,
    error,
    id_error,
    iso_utc,
    money,
    register_tool,
    scope_error,
    serve,
    success,
    tool_failure,
)

logger = get_logger(__name__)
mcp = FastMCP("Nexus Wealth MCP")
SERVER = "wealth"
RISK_ORDER = {"conservative": 0, "moderate": 1, "aggressive": 2}


@trace(logger)
def get_portfolio_summary(portfolio_id: str, caller_scope: str) -> dict[str, Any]:
    """Portfolio lookup with fields minimised per caller_scope; the portfolio number is masked and
    the total value is returned only to scopes that need it."""
    invalid = id_error("portfolio_id", portfolio_id) or scope_error(caller_scope)
    if invalid:
        return invalid
    try:
        with connection(SERVER, read_tables={"portfolios"}) as conn:
            row = conn.execute(
                "SELECT portfolio_id, customer_id, risk_profile, total_value_cents, status, opened_at "
                "FROM portfolios WHERE portfolio_id = ?",
                (portfolio_id,),
            ).fetchone()
        if row is None:
            return error("not_found", "The requested portfolio was not found.", field="portfolio_id")
        record = {
            "portfolio_id": row[0],
            "customer_id": row[1],
            "risk_profile": row[2],
            "total_value": money(row[3]),
            "status": row[4],
            "opened_at": iso_utc(row[5]),
        }
        return success(minimize_portfolio_fields(record, caller_scope))
    except Exception as exc:
        return tool_failure(exc)


@trace(logger)
def _product_row(product_id: str) -> tuple | None:
    with connection(SERVER, read_tables={"investment_products"}) as conn:
        return conn.execute(
            "SELECT product_id, name, product_type, min_risk_profile, description "
            "FROM investment_products WHERE product_id = ?",
            (product_id,),
        ).fetchone()


@trace(logger)
def get_investment_product_details(product_id: str, caller_scope: str) -> dict[str, Any]:
    """Single investment product lookup (catalogue data only, no customer data)."""
    invalid = id_error("product_id", product_id) or scope_error(caller_scope)
    if invalid:
        return invalid
    try:
        row = _product_row(product_id)
        if row is None:
            return error("not_found", "The requested product was not found.", field="product_id")
        return success({
            "product_id": row[0],
            "name": row[1],
            "product_type": row[2],
            "min_risk_profile": row[3],
            "description": minimize_free_text_pii(row[4]),
        })
    except Exception as exc:
        return tool_failure(exc)


@trace(logger)
def check_product_suitability(product_id: str, risk_profile: str, caller_scope: str) -> dict[str, Any]:
    """Suitability check only - no write and no customer data beyond the risk profile passed in.
    Suitable when risk_profile is at or above the product's minimum (conservative < moderate <
    aggressive). This is a rule check, not investment advice."""
    invalid = id_error("product_id", product_id) or scope_error(caller_scope)
    if invalid:
        return invalid
    if risk_profile not in RISK_ORDER:
        return error("validation_error", "risk_profile must be conservative, moderate or aggressive.", field="risk_profile")
    try:
        row = _product_row(product_id)
        if row is None:
            return error("not_found", "The requested product was not found.", field="product_id")
        minimum = row[3]
        suitable = RISK_ORDER[risk_profile] >= RISK_ORDER[minimum]
        return success({
            "product_id": product_id,
            "risk_profile": risk_profile,
            "minimum_risk_profile": minimum,
            "suitable": suitable,
            "reason_code": "meets_minimum_risk_profile" if suitable else "below_minimum_risk_profile",
        })
    except Exception as exc:
        return tool_failure(exc)


for _tool in (get_portfolio_summary, get_investment_product_details, check_product_suitability):
    register_tool(mcp, _tool)


if __name__ == "__main__":
    serve(mcp, "NEXUS_WEALTH_PORT", 8103)
