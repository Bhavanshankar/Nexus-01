# Nexus MCP servers

Four independent processes implementing [`../MCP_CONTRACTS.md`](../MCP_CONTRACTS.md) (v2.0.0):
13 Tools, 4 Resources, 2 Prompts over Streamable HTTP at `/mcp`, bound to `127.0.0.1`. The
orchestrator reaches them only through an MCP client; nothing else may query SQLite.

## Run

From the repository root, after `python database.py --seed` (`python -m banking_server` etc. are equivalent root entry points;
set `NEXUS_MCP_TRANSPORT=stdio` to serve over stdio for MCP hosts such as Claude Desktop):

```bash
python -m mcp_servers.banking_server
```

```bash
python -m mcp_servers.insurance_server
```

```bash
python -m mcp_servers.wealth_server
```

```bash
python -m mcp_servers.concierge_ops_server
```

| Server | Endpoint | Port variable | Tools | Resources / Prompts |
| --- | --- | --- | --- | --- |
| Banking | `http://127.0.0.1:8101/mcp` | `NEXUS_BANKING_PORT` | `get_account_summary`, `get_transaction_history`, `get_linked_accounts` | `account://{account_id}/summary` |
| Insurance | `http://127.0.0.1:8102/mcp` | `NEXUS_INSURANCE_PORT` | `get_policy_details`, `check_coverage_clause`, `get_claim_status` | `policy://{policy_id}/summary` |
| Wealth | `http://127.0.0.1:8103/mcp` | `NEXUS_WEALTH_PORT` | `get_portfolio_summary`, `get_investment_product_details`, `check_product_suitability` | - |
| Concierge Ops | `http://127.0.0.1:8104/mcp` | `NEXUS_CONCIERGE_OPS_PORT` | `get_customer_kyc_status`, `run_escalation_check`, `send_customer_notification`, `write_audit_log` | `customer://{customer_id}/profile`, `knowledgebase://{document_id}/excerpt`; Prompts `request_classification_prompt`, `response_drafting_prompt` |

`NEXUS_DB_PATH` selects the database file (default `data/meridian.db`); `NEXUS_LOG_FILE` routes the
JSON trace lines to a file.

## Behaviour

- **Isolation**: each operation opens a connection scoped to its own tables with a SQLite authorizer;
  banking/insurance/wealth are also `query_only`. Only concierge_ops reads `customers` and only it writes
  (`interaction_log`, `audit_log`).
- **Validation**: exact identifier formats, `caller_scope` on every banking/wealth tool (unknown scopes
  rejected; `internal_service` requires `NEXUS_TRUST_INTERNAL_SCOPE=1`), bounded/sanitised free text.
- **Masking**: account, policy, portfolio and claim numbers are `****` + last four in every response.
- **Escalation**: `run_escalation_check` is read-only; hard categories always escalate; the 0.70
  confidence threshold is a server constant.
- **Writes**: `send_customer_notification` enforces the KYC rule and writes the interaction and its
  audit row atomically; outcome is `logged` (no delivery is claimed). `write_audit_log` inserts one
  minimised audit row.
- **Errors**: the shared typed envelope; no SQL, paths or tracebacks leave a server.
- **Tracing**: every primitive is `@trace`d; the caller's `X-Nexus-Request-Id` header becomes the
  envelope `request_id` and appears in each log line.

Implementation helpers shared by all servers are in `common.py` (envelopes, scoped connections,
request context, registration).
