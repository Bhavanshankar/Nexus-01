# Submission evidence (executed, 2026-10-09)

Everything below was actually run on Windows 11, Python 3.13.7 (project supports 3.11+; a tokenizer
test guards against 3.12-only syntax). Raw live record: [`evidence/live_verification.md`](evidence/live_verification.md).

## 1. Commands and exit codes

| Check | Command | Result | Exit |
| --- | --- | --- | --- |
| Clean environment install | `python -m venv <new>` then `pip install -r requirements.txt`, `pip check` | installed; "No broken requirements found" | 0 |
| Seed (clean env, temp path) | `python database.py --seed --db-path <tmp>` | "Seeded database" | 0 |
| Imports (clean env) | `import crewai, fastmcp, rank_bm25, chromadb, openai, dotenv, truststore, pydantic, pytest, nexus_orchestrator` | ok | 0 |
| Demo (clean env and dev env) | `python run_local_demo.py` | **43 checks passed, 0 failed** (~30 s) | 0 |
| Test suite (clean env and dev env) | `pytest tests/` | **317 passed** (~2 min 15 s) | 0 |
| Dev database | `python database.py --seed` | previous file backed up to `data/backups/meridian-20261009T030106Z.db`, reseeded | 0 |
| Index | `python build_index.py` | 64 documents, 335 vectors, `text-embedding-3-small`, consistency checks passed | 0 |
| Live, five processes | `python scripts/verify_live.py` | **ALL CHECKS PASSED** | 0 |

## 2. Live verification summary (real CrewAI + OpenAI + Chroma, five processes)

Servers started with the PDF's own commands (`python -m banking_server`, `python -m insurance_server`,
`python -m wealth_server`, `python -m concierge_ops_server`) plus
`python -m nexus_orchestrator serve --port 8100`, each a separate PID on 127.0.0.1.

- MCP listing over HTTP: **13 Tools, 4 Resources, 2 Prompts**; all 13 tools, 4 Resources and 2 Prompts
  invoked through a real FastMCP client; `internal_service` scope refused without the trust flag.
- Orchestrator `/health` reached all four servers.
- **Showcase (Arjun, `CUS-20077`)**: `READY`, sendable; `cross_product`; live
  `banking.get_account_summary`, `insurance.get_policy_details`, `insurance.check_coverage_clause`;
  decision from `concierge_ops.run_escalation_check`; no claim that auto-debit was set up.
- **Retrieval conflict (`CUS-30142`)**: two conflicts surfaced to validation, MCP authoritative; the
  answer never claims auto-debit is available (outcome `ready` in the latest run, `escalated` in an earlier
  run - model variance, both safe).
- **Hard escalation, rephrased as routine**: `ESCALATED`, `hard_category`.
- **Unknown customer**: `REJECTED` after one `customer://` Resource read, zero tool calls.
- **Invalid intake**: `REJECTED`, zero MCP calls.
- **Insurance stopped mid-request** (killed between `get_policy_details` and `check_coverage_clause`):
  `FAILED` / `service_unavailable` naming insurance, no draft, human follow-up still decided via MCP; the
  orchestrator *process* also returns the typed error while insurance is down.
- **Log audit**: every process log is valid JSON lines with no raw identifiers, names, contact details,
  customer messages or injected text.

## 3. Log lines inspected (PDF section 6)

Real lines from the live run (traceback shortened here):

```json
{"ts": "2026-10-09T03:48:17.350+00:00", "level": "DEBUG", "logger": "mcp_servers.banking_server", "request_id": "REQ-3B1470C70ABD", "event": "ENTER", "function": "mcp_servers.banking_server.get_account_summary", "call_id": "df16a104626146b4a6c0175000b2f8ea", "arguments": {"account_id": "****0077", "caller_scope": "authorized_agent"}}
{"ts": "2026-10-09T03:48:17.372+00:00", "level": "DEBUG", "logger": "mcp_servers.banking_server", "request_id": "REQ-3B1470C70ABD", "event": "EXIT", "function": "mcp_servers.banking_server.get_account_summary", "call_id": "df16a104626146b4a6c0175000b2f8ea", "duration_ms": 21.866, "return_preview": "{\"ok\": true, \"data\": {\"account_id\": \"****0077\", \"customer_id\": \"****0077\", \"account_type\": \"savings\", \"status\": \"active\", \"auto_debit_allowed\": true, \"currency\": \"USD\"}}"}
{"ts": "2026-10-09T03:49:00.214+00:00", "level": "ERROR", "logger": "app.intake.gate", "request_id": "REQ-CA2B997CA678", "event": "FAILED", "function": "app.intake.gate.validate_inbound", "call_id": "0ed20a8696b34a5ea52d92aa8e708ff1", "duration_ms": 13.283, "exception_type": "app.intake.gate.IntakeRejected", "exception_message": "malformed_customer_id: customer_id must use the CUS-XXXXX format.", "traceback": "Traceback (most recent call last): ... (full traceback in the log)"}
```

Inspection notes: one JSON object per line; ENTER/EXIT share `call_id`; EXIT has `duration_ms` and a
bounded preview; FAILED is ERROR with exception type and full redacted traceback (the exception is
re-raised and handled as a typed rejection); the orchestrator's request ID appears on the server's own
lines (propagated via `X-Nexus-Request-Id`); identifiers are masked even inside previews; function names
are fully qualified even when a module runs as `__main__`.

## 4. Requirement audit against the PDF

| PDF section | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Single front door; MCP-only backend; DEBUG JSON tracing; guardrails.py with sanitisation, injection defence, output filtering, PII-safe logging | Met | ARCHITECTURE 1, 5, 7 |
| 1 | One real request through all four layers | Met | live showcase; `test_arjun_showcase_flows_through_all_layers` |
| 2 | Plain local processes; `database.py --seed`; `build_index.py`; `python -m <server>`; `python -m nexus_orchestrator`; `run_local_demo.py`; `pytest tests/` | Met (exact commands work) | section 1 above |
| 2.3 | Orchestrator's MCP client pointed at each server; MCP host config | Met | `HttpMCPGateway`; stdio + `docs/claude_desktop_config.example.json` |
| 3 | Four servers with the PDF table scopes; isolation structural | Met | `test_database.py` isolation tests |
| 3.1 | 13 tools, exact names, two write-capable; escalation tool built and independent of model confidence | Met | `test_mcp_registration.py`, `test_mcp_tools.py` |
| 3.2 | Typed schema before retrieval, drafting **or any tool call**; server request_id; CUS format; enums; sanitised intent; intake-time escalation_category; invalid does not proceed | Met (identity via Resource; first tool after validation) | `test_intake.py`, `test_orchestrator.py` (first-tool ordering asserted), demo 1 and 3 |
| 3.3 | KB + live MCP reconciled; MCP authoritative on conflict; conflict surfaced to validation | Met | `test_retrieval.py`, `test_agents.py`, demo 2 and 8b |
| 3.4 | Five roles, distinct steps; drafting grounded only; validation with confidence; escalation calls `run_escalation_check`; hard rule enforced in the tool and unbypassable by rephrasing | Met | `test_orchestrator.py`, `test_agents.py`, `test_intake.py`, demo 5 and 8c |
| 4 | `minimize_account_fields` + portfolio equivalent with explicit `caller_scope`; source-level scoping | Met | `test_guardrails.py`, `test_database.py` |
| 4 | Mask account/policy/portfolio numbers on every Tool and Resource | Met | `test_mcp_tools.py`, `test_mcp_registration.py` |
| 4 | Exact ID formats on every tool/resource argument, typed errors | Met | `test_guardrails.py::test_exact_identifier_formats`, per-tool tests |
| 4 | Hard escalation before confidence logic | Met | `run_escalation_check`; 18 parametrised tests |
| 4 | `audit_log` table and `write_audit_log` for every write | Met | notification audited in-transaction; outcome record links its audit ID |
| 4 | `sanitize_free_text` on outbound messages, customer text, retrieved documents and tool output | Met | `test_retrieval.py` (KB + MCP text), `test_mcp_tools.py` (notification), demo 6 |
| 4 | Output filtering against grounded context | Met | `filter_final_output`; `test_orchestrator.py` gate tests |
| 4 | `redact_for_logging` masks PII-shaped names; data-minimisation documented | Met | `test_logging_config.py`; ARCHITECTURE 7 |
| 5 | `@trace(logger)` on every function; ENTER/EXIT/FAILED; call_id; redacted args; duration; preview; traceback; re-raise; JSON lines; LOG_LEVEL default DEBUG | Met | `test_logging_config.py`; `test_architecture.py::test_every_project_function_is_traced` |
| 6 | Unit coverage of logging_config/guardrails/database run and inspected | Met | section 3 above |
| 6 | All 13 tools' logic; real FastMCP registration (not only a stub) | Met | real package used throughout; no stub |
| 6 | Intake valid + invalid; hybrid agree + conflict; full chain with both escalations | Met | test files above |
| 6 | pytest actually executed | Met | 317 passed |
| 6 | Each process starts and responds to a live MCP client | Met | `test_live_processes.py`, live section 1-2 |
| 6 | Showcase and four stress tests with fixtures | Met | demo 7-8; live 3-4 |
| 7 | 4 templated Resources, no bulk data, no unmasked fields; 2 Prompts with 3+ dynamic args; guardrails reused | Met | `test_mcp_registration.py` |
| 8 | Demo covers the full sequence in order without a live MCP client | Met | `run_local_demo.py`, 43 checks |
| 8.1 | Arjun savings `ACC-20077` + general insurance `POL-IN-30091`, cross_product, resolved without human | Met | fixtures corrected; live showcase `READY` |
| 8.2 | Missing/unknown customer, conflict, override attempt, server offline mid-request | Met | demo 8a-8d; live 3-4 |

## 5. Rubric self-check

| Component (weight) | What assessors look for | Where it is demonstrated |
| --- | --- | --- |
| End-to-end integration (25%) | One real request through intake, retrieval, multi-agent and MCP in a single live call path | Live showcase through the orchestrator process to four server processes (live record section 3) |
| Structured intake quality (10%) | Validated before anything downstream; invalid/incomplete never proceeds silently | Gate with zero calls; first tool only after validation; typed `IntakeError`s; incomplete-extraction test |
| Grounded, reconciled retrieval (15%) | Account-specific answers reconciled with KB; structured authoritative on conflict | Code-detected `RetrievalConflict`s; validation fails drafts that ignore them (demo 8b) |
| Multi-agent orchestration and escalation (15%) | Five distinct roles; confidence and hard escalation; hard rule provably unbypassable | Step trace per role; low-confidence case; hard rule at confidence 1.0 and across rephrasings; invariant check |
| MCP-based tool access (15%) | All reads/writes through the four servers with scoped access | Architecture tests forbid SQLite/server imports outside servers; authorizer tests; call records |
| Guardrails stack (10%) | Sanitisation, injection defence (incl. retrieved), output filtering, PII redaction demonstrably active | Demo 1, 6 and log audit; gate tests; live log audit |
| System startup and process evidence (5%) | Four servers + orchestrator as independent processes, verified live | `test_live_processes.py`, `scripts/verify_live.py` |
| Demo and debrief (5%) | Narrate the trace and the trade-offs | `docs/DEMO_SCRIPT.md` with prepared answers |

## 6. Remaining limitations (not hidden)

- Real-model outputs vary; borderline cases (the conflict case) can route to `ready` or `escalated`
  between runs. Every safety property is enforced deterministically regardless.
- No notification delivery provider: outcomes are `logged`, never `sent`.
- No caller authentication on the servers (loopback-only demo assumption).
- Test functions in `tests/` are not wrapped in `@trace`; all runtime project code is.
- Recorded process exit codes are 1 because the harness stops processes with Windows `TerminateProcess`.
