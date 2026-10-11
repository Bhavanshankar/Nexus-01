"""Banking MCP server: three read-only tools and the ``account://`` Resource.

Database scope: ``accounts`` and ``transactions`` only (enforced by the SQLite authorizer).
Every tool takes an explicit ``caller_scope`` (PDF section 4) and returns masked account numbers.
"""

from typing import Any

from fastmcp import FastMCP

from app.guardrails import minimize_account_fields, minimize_transaction_fields
from app.logging_config import get_logger, trace
from mcp_servers.common import (
    connection,
    error,
    id_error,
    iso_utc,
    money,
    register_resource,
    register_tool,
    scope_error,
    serve,
    success,
    tool_failure,
)

logger = get_logger(__name__)
mcp = FastMCP("Nexus Banking MCP")
SERVER = "banking"
_ACCOUNT_COLUMNS = "account_id, customer_id, account_type, balance_cents, currency, status, auto_debit_allowed, opened_at"


@trace(logger)
def _account_record(row: tuple) -> dict[str, Any]:
    return {
        "account_id": row[0],
        "customer_id": row[1],
        "account_type": row[2],
        "balance": money(row[3], row[4]),
        "currency": row[4],
        "status": row[5],
        "auto_debit_allowed": bool(row[6]),
        "opened_at": iso_utc(row[7]),
    }


@trace(logger)
def get_account_summary(account_id: str, caller_scope: str) -> dict[str, Any]:
    """Get one account summary. The account number is masked; fields are minimised per caller_scope
    (customer | authorized_agent | internal_service)."""
    invalid = id_error("account_id", account_id) or scope_error(caller_scope)
    if invalid:
        return invalid
    try:
        with connection(SERVER, read_tables={"accounts"}) as conn:
            row = conn.execute(f"SELECT {_ACCOUNT_COLUMNS} FROM accounts WHERE account_id = ?", (account_id,)).fetchone()
        if row is None:
            return error("not_found", "The requested account was not found.", field="account_id")
        return success(minimize_account_fields(_account_record(row), caller_scope))
    except Exception as exc:  # never send database diagnostics over MCP
        return tool_failure(exc)


@trace(logger)
def get_transaction_history(account_id: str, limit: int, caller_scope: str) -> dict[str, Any]:
    """Get up to `limit` (1-100) most recent transactions for an account, newest first, with
    descriptions minimised and fields filtered per caller_scope."""
    invalid = id_error("account_id", account_id) or scope_error(caller_scope)
    if invalid:
        return invalid
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        return error("validation_error", "limit must be an integer from 1 to 100.", field="limit")
    try:
        with connection(SERVER, read_tables={"accounts", "transactions"}) as conn:
            account = conn.execute("SELECT currency FROM accounts WHERE account_id = ?", (account_id,)).fetchone()
            if account is None:
                return error("not_found", "The requested account was not found.", field="account_id")
            rows = conn.execute(
                "SELECT transaction_id, amount_cents, direction, description, occurred_at "
                "FROM transactions WHERE account_id = ? ORDER BY occurred_at DESC, transaction_id DESC LIMIT ?",
                (account_id, limit),
            ).fetchall()
        transactions = [
            minimize_transaction_fields(
                {
                    "transaction_id": row[0],
                    "amount": money(-abs(row[1]) if row[2] == "debit" else abs(row[1]), account[0]),
                    "direction": row[2],
                    "description": row[3],
                    "occurred_at": iso_utc(row[4]),
                },
                caller_scope,
            )
            for row in rows
        ]
        return success({
            "account_id": minimize_account_fields({"account_id": account_id}, caller_scope)["account_id"],
            "transactions": transactions,
            "returned_count": len(transactions),
        })
    except Exception as exc:
        return tool_failure(exc)


@trace(logger)
def get_linked_accounts(customer_id: str, caller_scope: str) -> dict[str, Any]:
    """Get all accounts linked to a customer with the same scoped minimisation as get_account_summary.

    The banking server cannot read the customers table, so an unknown customer and a customer with
    no accounts both return an empty list; identity is verified by concierge_ops."""
    invalid = id_error("customer_id", customer_id) or scope_error(caller_scope)
    if invalid:
        return invalid
    try:
        with connection(SERVER, read_tables={"accounts"}) as conn:
            rows = conn.execute(
                f"SELECT {_ACCOUNT_COLUMNS} FROM accounts WHERE customer_id = ? ORDER BY account_id",
                (customer_id,),
            ).fetchall()
        accounts = []
        for row in rows:
            item = minimize_account_fields(_account_record(row), caller_scope)
            item.pop("customer_id", None)  # implied by the request; not repeated per item
            accounts.append(item)
        return success({"customer_id": customer_id, "accounts": accounts, "returned_count": len(accounts)})
    except Exception as exc:
        return tool_failure(exc)


@trace(logger)
def account_summary_resource(account_id: str) -> dict[str, Any]:
    """account://{account_id}/summary - customer-scoped, masked account summary (no balance history,
    no ownership fields)."""
    return get_account_summary(account_id, "customer")


for _tool in (get_account_summary, get_transaction_history, get_linked_accounts):
    register_tool(mcp, _tool)
register_resource(mcp, "account://{account_id}/summary", account_summary_resource)


if __name__ == "__main__":
    serve(mcp, "NEXUS_BANKING_PORT", 8101)
