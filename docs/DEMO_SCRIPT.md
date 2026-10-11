# 20-minute live demo and debrief

Audience: assessment panel. Goal: show one real request crossing all four layers, then prove each
guardrail and stress behaviour, then defend the design trade-offs.

## Before the session (5 minutes, not on the clock)

```bash
python database.py --seed
```

Start the four servers and the orchestrator in five terminals (see README "Run the system"). Check:

```bash
python -m nexus_orchestrator check
```

Expect `TOTAL tools=13 resources=4 prompts=2`. Keep `docs/ARCHITECTURE.md` (diagram) and
`docs/evidence/live_verification.md` open.

## Timeline

| Time | Segment | What to show | What to say |
| --- | --- | --- | --- |
| 0:00-2:00 | Problem and shape | ARCHITECTURE section 1 diagram | One front door; four layers; MCP is the only backend path; the orchestrator never opens SQLite (enforced by `tests/test_architecture.py`). |
| 2:00-6:00 | Showcase, live | `python -m nexus_orchestrator ask --customer-id CUS-20077 --account-id ACC-20077 --policy-id POL-IN-30091 "Can I set up auto-debit ..."` | Walk the printed trace line by line: intake gate -> identity via the `customer://` Resource (no tool yet) -> Classifier (`cross_product`, prompt from MCP) -> KYC tool -> Retriever (banking account facts, insurance policy + clause check, KB banking/insurance/cross-product guidance, 0 conflicts) -> Drafter -> Validator (deterministic + model confidence) -> `run_escalation_check` (none) -> output gate `send` -> notification `logged` -> audit. Read the answer: both questions answered; masked numbers; next step is to authorise - it does not claim auto-debit was set up. |
| 6:00-8:00 | Conflict | `ask` with `CUS-30142`, `ACC-30142`, `POL-GE-30250` | Policy permits, account is restricted. Two `RetrievalConflict`s, MCP authoritative, shown to validation. A draft that trusts the policy side fails validation (`run_local_demo.py` 8b shows it deterministically). |
| 8:00-10:00 | Hard escalation | `ask` with "Quick routine question: since I was laid off last month, can I push my premium payments back a few months?" | Signals `financial_hardship:job_loss` set the category at intake; the model cannot remove it; the server escalates `hard_category` regardless of confidence (demo 8c forces confidence 1.0). |
| 10:00-12:00 | Offline demo, fast | `python run_local_demo.py` | 43 assertions in ~30 s without live services: invalid intakes with zero MCP calls, low-confidence escalation (inconclusive clause check), injected KB text withheld and never in any prompt, unknown customer (one Resource read, zero tool calls), log audit. |
| 12:00-14:00 | Outage | Stop the insurance terminal (Ctrl+C), re-run the showcase `ask` | Typed `service_unavailable` naming insurance, no draft, human follow-up still decided via MCP. Mention the true mid-request version in `tests/test_live_processes.py::test_insurance_process_stopped_mid_request` and the live record section 4. Restart insurance. |
| 14:00-15:00 | Observability | Open a server log file / `logs/run_local_demo.jsonl` | One JSON object per line, ENTER/EXIT/FAILED with shared `call_id`, the same `request_id` in orchestrator and server logs, masked IDs, `<redacted:N chars>` for content. |
| 15:00-20:00 | Debrief / Q&A | ARCHITECTURE sections 3, 6, 9 | Use the prepared answers below. |

## Prepared answers for the debrief

- **Why is the Retriever not an LLM agent?** Conflicts and grounding must be provable. Planning, MCP
  calls, ownership, sanitisation and conflict detection are deterministic; models only classify, draft
  and review. A model asserting it is grounded proves nothing.
- **How is hard escalation unbypassable?** Three layers: deterministic intake signals (model can add, not
  remove), the server's hard gate before any confidence logic, and the Escalation Agent rejecting a
  decision that violates the rule. Callers cannot set the category (`extra="forbid"`).
- **Unknown customer vs "no tool call"?** Missing/malformed IDs: zero MCP calls. A well-formed ID needs one
  narrow identity lookup, done through the `customer://` Resource, so even then zero tools run; after
  `not_found` nothing else happens. No tool runs before the intake schema is validated. Hard-coding existence or reading SQLite
  would break the MCP-only rule.
- **Why is `run_escalation_check` read-only?** The PDF lists it as Read and names two write tools; the
  outcome (with `decision_id`) is persisted by `write_audit_log`.
- **Caller scope?** Required on every banking/wealth call; unknown scopes rejected; agent scope omits
  balances; `internal_service` needs an operator trust flag.
- **Block vs escalate?** Block when the text is unsafe (unmasked IDs, contact details, injection, false
  action claims); escalate when it is unproven (validation failure, unverifiable numbers, low confidence).
- **What changed in the data?** Product IDs moved to `PROD-XX-XX`; Arjun's policy is general insurance
  and his account savings, per the PDF; KB documents no longer embed customer records.
- **Limits?** Model variance on borderline cases (always on the safe side), no delivery provider, no
  server authentication outside loopback.

## Fallback if live model calls fail

Run `python run_local_demo.py` (no credentials) and show `docs/evidence/live_verification.md`, which
records a completed live run with commands, exit codes, tool lists and masked traces.
