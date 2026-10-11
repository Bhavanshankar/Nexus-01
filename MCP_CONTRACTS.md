# Nexus MCP Interface Contract

**MCP Contract Version: 2.0.0** (2026-10-09). Supersedes 1.0.0; see "Changes from 1.0.0" at the end.

Authority: the assignment PDF (`week5-01-nexus-final.pdf`, v1.0, 2026-10-05). Where this contract
and the PDF disagree, the PDF wins and the resolution is recorded in
[`docs/ARCHITECTURE.md` section 6](docs/ARCHITECTURE.md#6-contract-tensions-and-how-they-were-resolved).

Four independent MCP servers expose exactly **13 Tools, 4 Resources and 2 Prompts**. The orchestrator
obtains every backend fact through these primitives; it never opens SQLite.

## Shared rules

- Each server is a separate local Python process (`python -m banking_server`, or equivalently
  `python -m mcp_servers.banking_server`), Streamable HTTP at `/mcp` bound to `127.0.0.1`; stdio when
  started with `NEXUS_MCP_TRANSPORT=stdio` (for MCP hosts such as Claude Desktop).
- **Table scope is structural** (SQLite authorizer + `PRAGMA query_only` for read-only servers):

  | Server | Reads | Writes |
  | --- | --- | --- |
  | banking | `accounts`, `transactions` | none |
  | insurance | `policies`, `claims` | none |
  | wealth | `portfolios`, `investment_products` | none |
  | concierge_ops | `customers`, `interaction_log`, `escalation_flags`, `audit_log` | `interaction_log`, `audit_log` |

  Only concierge_ops can read `customers` (the only table with names, e-mail addresses and phone numbers).
- **Identifier formats are exact** (`app/guardrails.py`): `CUS-\d{5}`, `ACC-\d{5}`, `POL-[A-Z]{2}-\d{5}`,
  `PORT-\d{5}`, `PROD-[A-Z]{2}-\d{2}`, `CLM-\d{5}`. A mismatch returns `validation_error` naming the field.
- **Masking**: account, policy, portfolio and claim numbers are returned as `****` + last four characters
  in every Tool and Resource response, including nested results. Customer IDs are returned only where a
  caller needs them to verify ownership (agent scope), and are masked in Resources.
- **`caller_scope`** is required on every banking and wealth tool (PDF section 4). Values:

  | Scope | Meaning | Account fields | Portfolio fields |
  | --- | --- | --- | --- |
  | `customer` | the authenticated customer viewing their own record | masked id, type, status, auto-debit flag, currency, balance | masked id, risk profile, status, total value |
  | `authorized_agent` | Nexus acting for a verified customer (ownership + eligibility) | masked id, customer_id, type, status, auto-debit flag, currency | masked id, customer_id, risk profile, status |
  | `internal_service` | back-office use | all of the above + balance + opened_at | all + opened_at |

  Any other value is `validation_error`. `internal_service` is refused with `authorization_error` unless
  the server process is started with `NEXUS_TRUST_INTERNAL_SCOPE=1` (documented loopback-demo trust
  boundary; the orchestrator never uses it).
- **Free text** in arguments (`message`, `detail`, Prompt arguments) passes `guardrails.sanitize_free_text`
  (type, length, HTML stripping, Unicode normalisation, injection signatures); failures are
  `validation_error` (Tools/Resources) or an MCP prompt error (Prompts).
- **Writes**: exactly two write-capable tools. Every persisted write and its audit row commit in one
  SQLite transaction.
- **Correlation**: the orchestrator sends `X-Nexus-Request-Id`; servers use it as the envelope
  `request_id` and in their JSON log lines.

### Envelope

Success: `{"ok": true, "data": {...}}`. Failure (no success payload):

```json
{"ok": false, "error": {"code": "not_found", "message": "The requested account was not found.",
                         "retryable": false, "request_id": "REQ-0123456789AB", "field": "account_id"}}
```

`code` is one of `validation_error`, `not_found`, `authorization_error`, `database_error`,
`service_unavailable`, `internal_error`. Messages never include SQL, file paths, stack traces or raw
exception text. Timestamps are ISO 8601 with `Z`; money is `{amount_minor, currency}`.

## Tools

### Banking - `mcp_servers/banking_server.py`

| Tool | Parameters | Returns (`data`) |
| --- | --- | --- |
| `get_account_summary` | `account_id`, `caller_scope` | account fields per scope table above |
| `get_transaction_history` | `account_id`, `limit` (int 1-100), `caller_scope` | `account_id` (masked), `transactions[]` newest first (`amount` signed, `direction`, minimised `description`, `occurred_at`; `transaction_id` only for internal scope), `returned_count` |
| `get_linked_accounts` | `customer_id`, `caller_scope` | `customer_id`, `accounts[]` (per-scope fields, no customer_id per item), `returned_count`. Banking cannot read `customers`, so an unknown customer and one without accounts both return an empty list; identity is concierge_ops' job. |

### Insurance - `mcp_servers/insurance_server.py`

| Tool | Parameters | Returns (`data`) |
| --- | --- | --- |
| `get_policy_details` | `policy_id` | masked `policy_id`, `customer_id`, `policy_type`, `premium`, `payment_frequency`, `payment_terms` (PII-minimised), `auto_debit_permitted`, `status`, `coverage_clauses[{clause_id,title,text}]`, `started_at`. Corrupt stored clause JSON -> `database_error` without the raw value. |
| `check_coverage_clause` | `policy_id`, `scenario_tag` (`^[a-z][a-z0-9_]{1,63}$`; aliases `auto_debit`, `automatic_payment` -> `auto_debit_payment`, `premium_payment`/`payment_terms` -> `general_payment_terms`) | masked `policy_id`, `scenario_tag`, `resolved_clause_id`, `policy_status`, `coverage_status` (`covered` / `not_covered` / `conditional` / `documented` / `policy_not_active` / `no_matching_clause`), `covered` (bool, or `null` when no clause matches), `matched_clauses[]`. Exact clause-ID match only; coverage is never inferred. |
| `get_claim_status` | `claim_id` | masked `claim_id`, masked `policy_id`, `customer_id` (owner, for ownership checks), `status`, `filed_at`, `last_updated_at`. Status only - no settlement authority. |

### Wealth - `mcp_servers/wealth_server.py`

| Tool | Parameters | Returns (`data`) |
| --- | --- | --- |
| `get_portfolio_summary` | `portfolio_id`, `caller_scope` | portfolio fields per scope table above |
| `get_investment_product_details` | `product_id`, `caller_scope` | `product_id`, `name`, `product_type`, `min_risk_profile`, `description` (catalogue data, no customer data) |
| `check_product_suitability` | `product_id`, `risk_profile` (`conservative`/`moderate`/`aggressive`), `caller_scope` | `product_id`, `risk_profile`, `minimum_risk_profile`, `suitable`, `reason_code`. Pure rule check (conservative < moderate < aggressive); no write, no customer data beyond the profile passed in. |

### Concierge Ops - `mcp_servers/concierge_ops_server.py`

| Tool | Op | Parameters | Returns (`data`) |
| --- | --- | --- | --- |
| `get_customer_kyc_status` | Read | `customer_id` | `customer_id`, `kyc_status`, `kyc_last_checked_at`. The only tool returning raw KYC data; also the narrow identity check for syntactically valid IDs. |
| `run_escalation_check` | **Read** | `customer_id`, `request_category` (`none` + 3 hard categories), `draft_confidence` (number 0.0-1.0, booleans rejected) | `customer_id`, `request_category`, `draft_confidence`, `should_escalate`, `reason` (`hard_category`/`low_confidence`/`none`), `hard_escalation_triggered`, `confidence_threshold` (fixed **0.70**), `policy_version`, `decision_id`, `persisted: false`. Hard categories escalate unconditionally *before* any confidence logic; the threshold is a server constant. Unknown customer -> `not_found`. |
| `send_customer_notification` | Write | `customer_id`, `channel` (`chat`/`phone`/`email`/`sms`/`branch`), `message` (1-5000 chars, sanitised, must not contain unmasked identifiers) | `customer_id`, `request_id`, `channel`, `outcome: "logged"`, `delivery_confirmed: false`, `interaction_id`, `audit_id`, `created_at`. **KYC rule**: only `kyc_status = verified` customers may be sent customer-specific notifications; others get `authorization_error` and the refusal is audited. Interaction row + audit row commit atomically (audit failure rolls back the interaction). Never claims delivery - no channel provider is configured. |
| `write_audit_log` | Write | `timestamp` (ISO 8601 with offset), `action_type`, `performed_by`, `outcome` (tokens `^[A-Za-z][A-Za-z0-9_.:-]{0,99}$`), `customer_id` (or null), `detail` (1-2000 chars, sanitised, PII-minimised) | `audit_id`, `timestamp`, `action_type`, `performed_by`, `customer_id`, `outcome`, `recorded: true` (detail is not echoed). |

## Resources

| URI template | Server | Content |
| --- | --- | --- |
| `customer://{customer_id}/profile` | concierge_ops | masked `customer_id`, `first_name`, `customer_tier`, sanitised `recent_interaction_summary` (withheld if it fails sanitisation), `recent_interaction_at`. No e-mail, phone, full name or KYC data (only `get_customer_kyc_status` returns KYC). Used by the orchestrator as the identity check before any tool call. |
| `account://{account_id}/summary` | banking | `get_account_summary(account_id, "customer")` - masked number, no ownership fields. |
| `policy://{policy_id}/summary` | insurance | masked `policy_id`, `policy_type`, `status`, `payment_frequency`, `premium`, `auto_debit_permitted` (no ownership, no clause text). |
| `knowledgebase://{document_id}/excerpt` | concierge_ops | one approved KB document's `document_id`, `title`, `product_line`, `category`, a bounded (<= 1200 chars, paragraph-cut) sanitised and masked `excerpt`, `truncated`. Never a bulk corpus; test/adversarial documents return `not_found`. |

Resources return the same envelope as Tools and validate the templated ID with the exact formats.

## Prompts (concierge_ops)

| Prompt | Arguments | Output |
| --- | --- | --- |
| `request_classification_prompt` | `customer_message`, `customer_tier`, `prior_interaction_summary`, `known_product_lines` (comma list) | Classifier instructions with the enums, the allowed product lines (adds `cross_product` when 2+ lines are known), tier-specific urgency guidance, hard-category definitions that classify the underlying situation, and the sanitised message inside `<customer_message>` data tags. |
| `response_drafting_prompt` | `classified_intent`, `retrieved_context_summary`, `customer_name`, `escalation_flag` (`true`/`false`) | Drafting instructions grounded in the supplied evidence block (answer every question with the deciding live facts, follow authoritative MCP values on conflict, cite `evidence_ids`, list `asserted_facts`, never claim an action was performed, masked identifiers only). With `escalation_flag=true` it becomes a holding-reply task that answers nothing substantive. |

All arguments are sanitised; an injection signature makes the prompt fail to render.

## Changes from 1.0.0

| 1.0.0 | 2.0.0 | Reason |
| --- | --- | --- |
| `check_policy_clause`, `get_portfolio`, `get_investment_product`, `get_kyc_status` | `check_coverage_clause`, `get_portfolio_summary`, `get_investment_product_details`, `get_customer_kyc_status` | PDF 3.1 names; no aliases kept (exactly 13 tools). |
| `caller_scope` only on two tools | required on all six banking/wealth tools | PDF section 4 ("every banking_server/wealth_server tool call"). |
| `run_escalation_check` "read; conditional write" (inserted `escalation_flags` + audit) | read-only decision; orchestrator records the outcome with `write_audit_log` | PDF 3.1 lists it as Read and names only two write tools. |
| Policy IDs returned unmasked | masked in every Tool and Resource | PDF section 4 hard requirement. |
| Generic identifier regex | exact per-type formats; product IDs migrated `PROD-001` -> `PROD-XX-XX` | PDF section 4. |
| Domain servers could read `customers` | only concierge_ops reads `customers` | PDF section 3 table scopes; least privilege for PII. |
| `get_claim_status` without owner | returns owning `customer_id` (via `policies` join inside insurance scope) | ownership must be verifiable without matching masked suffixes. |
| no Resources / Prompts | 4 Resources, 2 Prompts | PDF section 7. |
