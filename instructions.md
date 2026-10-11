# Nexus capstone: engineering handoff for the implementation agent

## Mission and source of truth

Finish the Nexus project in this repository so that one customer request genuinely passes through validated intake, hybrid retrieval, five distinct agent roles, and four independently running MCP servers. Deliver the working system, executed tests, architecture documentation, local demo, and evidence required by the assignment.

The assignment is `C:\Users\2531802\Downloads\week5-01-nexus-final.pdf` (version 1.0, October 5, 2026). Read it in full before changing code, especially sections 3 through 8. This file is a handoff from the user and is intended as implementation instructions. The PDF is the requirements source. If an instructor supplies a separate written specification, that specification takes precedence as the PDF itself states. `MCP_CONTRACTS.md` and component READMEs describe current code; they are not authority when they disagree with the assignment. Record and resolve each disagreement in the architecture document.

Do not stop when files merely exist or a mocked demo prints plausible output. Completion means that the acceptance checks at the end of this file have actually passed. Make reasonable implementation decisions, preserve the user's existing uncommitted work, and report genuine external blockers such as missing provider credentials rather than claiming unrun checks passed.

## Current baseline: verify before editing

The repository already contains database schema and seed data, 13 MCP tool functions, a knowledge base and Chroma index builder, an unstructured retriever, Pydantic models, and five separate CrewAI agent classes. These are starting points, not completed layers. `nexus_orchestrator.py` and `app/guardrails.py` are empty. `run_local_demo.py` is a stub. No MCP Resources or Prompts are registered. The retrieval dataclass does not match `app.models.retrieval.RetrievalResult`. The current virtual environment lacks CrewAI, FastMCP, and `rank_bm25`; `requirements.txt` also omits required dependencies. The current retrieval default points at `C:\week5\week-5-nexus`, not this checkout. Only two embedding-error tests existed and passed at the time of this handoff; that result does not verify the project.

Before work, inspect `git status`, the unstaged and staged diffs, and any `AGENTS.md`. Do not reset, overwrite, or discard existing edits. Keep tests isolated from `data/meridian.db`; seeding that path replaces its contents. Use temporary databases for automated tests and back up any existing development database before intentionally reseeding it.

## Non-negotiable system behavior

1. A request has one live call path: sanitize and validate intake; classify; retrieve KB and relevant customer-specific facts through MCP; reconcile; draft; validate; obtain deterministic escalation decision through MCP; apply a final output gate. All five roles must be observable as distinct steps. Invalid intake must stop before retrieval, drafting, or domain data calls.
2. The orchestrator and retrieval/agent layers must not open SQLite or import server business functions to obtain backend facts. All customer-specific backend reads and writes go through MCP clients. The local vector index is available to the retrieval layer.
3. For customer-specific state, current structured MCP facts outrank general KB text. Preserve source references and explicitly pass conflicts to validation. Never turn missing or offline structured data into a guessed answer.
4. Hard categories (`financial_hardship`, `safeguarding_concern`, `compliance_override_request`) always escalate, even at confidence 1.0 and after rephrasing. Enforce that rule in the server's `run_escalation_check`, then verify the returned decision in the orchestrator. Low-confidence escalation uses a fixed server-side threshold.
5. No unsupported claim, unmasked account/policy/portfolio identifier, or unsafe instruction from retrieved text reaches a customer. A model's self-reported confidence or assertion that it called a tool is not proof of grounding or of an MCP call.
6. Four servers and the orchestrator run as separate plain local Python processes. Containers are not part of the submission.

## Phase 1: reconcile contracts and make setup reproducible

Create a requirements-to-code matrix in the architecture documentation. For each PDF requirement, identify the owning file, test, and demo evidence. Resolve these known discrepancies first:

- The PDF's tool names are `check_coverage_clause`, `get_portfolio_summary`, `get_investment_product_details`, and `get_customer_kyc_status`. Current registrations use different names. Rename the public registrations and update all clients, tests, docs, and `MCP_CONTRACTS.md`. Do not keep extra public aliases if doing so changes the required count of 13 tools.
- PDF section 3.1 lists `caller_scope` explicitly on account and portfolio summaries; section 4 says every banking and wealth tool call must have explicit caller scope. Apply the stronger section 4 rule consistently and document the resulting signatures in `MCP_CONTRACTS.md` and architecture notes. Reject unsupported scopes rather than silently choosing a privileged default.
- The PDF labels `run_escalation_check` a read tool and says there are only two write-capable tools. Existing code conditionally persists `escalation_flags` and an audit entry. Prefer the PDF's two-write-tool inventory: make the decision operation read-only and document why existing flag persistence was removed. If an instructor has explicitly required persisted escalation flags, record that override and keep the flag and audit insert in one transaction. Never claim a conditional write is read-only.
- The PDF's missing/unknown-customer stress case says no downstream retrieval or tool call. A missing or malformed ID can be rejected at intake with zero tool calls. Existence of a syntactically valid ID cannot be established against the live database without a backend call. For a syntactically valid but unknown ID, use a narrowly scoped MCP identity check, then return a typed intake/identity error and make no subsequent retrieval, drafting, or domain calls. Document this unavoidable interpretation; do not hardcode customer existence in production or access SQLite directly to satisfy a test.

Pin compatible Python 3.11+ dependencies for all direct imports, including CrewAI, FastMCP, `rank-bm25`, Pydantic, ChromaDB, OpenAI, dotenv, and pytest. Make the Chroma path repository-relative by default while retaining an environment override. Provide `.env.example` with names and non-secret examples. A fresh environment must support the documented seed, index-build, server startup, retrieval, demo, and test commands. Do not place credentials in source, logs, tests, or committed example files.

**Exit check:** clean-environment imports and setup commands run without path edits or missing-package errors. Public MCP contract and requirement matrix reflect the PDF.

## Phase 2: shared guardrails, identity rules, and tracing

Implement `app/guardrails.py` as the common implementation used by servers, Resources, Prompts, retrieval, and agents:

- `sanitize_free_text`: type and length validation, HTML stripping, Unicode/whitespace normalization as needed, and deterministic regex-based injection-signature detection. Define a typed rejection mode for adversarial instructions; preserve harmless financial language. Apply it to customer free text, all retrieved KB and MCP text before any model sees it, notification bodies, and free-text MCP/Resource/Prompt inputs.
- `redact_for_logging`: recursively handle mappings, lists, Pydantic models, positional/bound arguments, nested results, and exception messages. Mask fields such as `customer_id`, `account_id`, `policy_id`, `portfolio_id`, `phone`, `email`, and message/description content. Do not log raw full IDs or raw customer content under a different key.
- `minimize_account_fields` and equivalent portfolio minimization: take explicit `caller_scope`; allow only fields needed at that scope. Centralize masking of account, policy, and portfolio numbers to the final four characters.
- Final output filtering: detect unmasked identifier patterns and enforce support for customer-specific factual statements through evidence references and validation results. Define what causes a block versus a human escalation. A prompt saying "be grounded" is not the filter.

Upgrade `app/logging_config.py` so `@trace(logger)` emits one JSON object per line, `LOG_LEVEL` defaults to DEBUG, and ENTER/EXIT/FAILED for one call share a `call_id`. Include fully qualified function name, redacted bound arguments, duration, bounded safe return preview, and exception type/full traceback on failure. Re-raise failures. Redact sensitive values from arguments, previews, exception strings, and traceback rendering while retaining useful stack frames. Apply trace coverage to every exposed Tool, Resource, Prompt, each agent step, orchestration steps, and project-written functions as required by PDF section 5. Arrange `@mcp.tool()`/`@trace()` order so real FastMCP still registers the intended public signature. Replace the agent-local sanitizer fallback with the shared guardrails implementation once available.

**Exit check:** unit tests cover format errors, HTML, long input, injection in a retrieved chunk, scoped minimization, masking, output blocking, JSON log shape, matching call IDs, exception re-raise, and the absence of raw PII in captured logs.

## Phase 3: finish all MCP primitives and server isolation

Implement exactly these 13 public tools, retaining the PDF's parameter meaning and typed safe error envelope:

| Server | Required tools |
| --- | --- |
| Banking | `get_account_summary`, `get_transaction_history`, `get_linked_accounts` |
| Insurance | `get_policy_details`, `check_coverage_clause`, `get_claim_status` |
| Wealth | `get_portfolio_summary`, `get_investment_product_details`, `check_product_suitability` |
| Concierge Ops | `get_customer_kyc_status`, `run_escalation_check`, `send_customer_notification`, `write_audit_log` |

Validate each identifier by its exact format: `CUS-XXXXX`, `ACC-XXXXX`, `POL-XX-XXXXX`, `PORT-XXXXX`, `PROD-XX-XX`, and `CLM-XXXXX`. Generic `validate_identifier` is insufficient. Validate non-ID arguments (limits, confidence, category, risk profile, scope, channel, timestamps, and free text) and return typed errors without SQL details or raw exception content. Mask account, policy, and portfolio numbers in **every** Tool and Resource response, including nested claim or clause results. Verify returned customer-specific objects belong to the intended customer before using them in the orchestrator. Where the public tool signature lacks a customer argument, establish ownership through a separate permitted MCP lookup; do not infer ownership from coincidentally matching numeric suffixes. Do not let a caller assert `internal_service` privilege without an explicit, documented trust boundary for the local demo.

Keep structural SQLite isolation through `app/db_access.py` and `mcp_servers/common.py`: banking cannot query policies/portfolios, insurance cannot query accounts/portfolios, wealth cannot query accounts/policies, and concierge cannot read other domains. Test the authorizer and server-specific scopes, including denied reads and writes. Use transactions for every persisted operation and its required audit record. Enforce the PDF's KYC-based notification rule. Report `logged` unless an actual channel confirms delivery; do not claim a notification was sent solely because an interaction row was written. If the orchestrator sends a notification, it must first pass the final response gate.

Register all four Resources, with templated URIs, narrow responses, exact ID validation, shared guardrails, and no unmasked sensitive fields:

- `customer://{customer_id}/profile`
- `account://{account_id}/summary`
- `policy://{policy_id}/summary`
- `knowledgebase://{document_id}/excerpt`

Register both Prompts, sanitize arguments, and provide meaningful dynamic output:

- `request_classification_prompt(customer_message, customer_tier, prior_interaction_summary, known_product_lines)`
- `response_drafting_prompt(classified_intent, retrieved_context_summary, customer_name, escalation_flag)`

Assign Resources and Prompts to appropriate servers, documenting their ownership. The knowledge-base Resource may read the local KB/index; it must return one bounded excerpt and must not expose a bulk corpus. Confirm that registration works against real FastMCP, not only a stub, and that MCP listing reports exactly 13 Tools, 4 Resources, and 2 Prompts.

**Exit check:** every tool business function has success, invalid-input, not-found, masking/scope, and failure tests as relevant. All primitives are listed and invoked by a real local MCP client. Cross-domain SQL access tests fail closed.

## Phase 4: typed intake and grounded hybrid retrieval

Keep `app/models/intake.py` as the validated contract: server-generated `request_id`, exact customer ID format, product-line/request-type/urgency enums, sanitized intent, and intake-time hard category. The customer ID may be supplied as trusted request metadata; do not force it into the natural-language message. Reject missing, conflicting, or malformed IDs and incomplete model extraction with a typed error. No retrieval or draft may start after such a failure. Test rephrased hardship/safeguarding/compliance requests; deterministic signals must override an unsafe model classification.

Create an adapter between `app/retrieval/retrieval.py`'s unstructured result and `app/models/retrieval.py`'s `RetrievedDocument`, `StructuredFact`, `RetrievalConflict`, and `GroundedContext`. Preserve source ID, source path/URI, section, score, and relevant metadata. Do not confuse the two classes currently named `RetrievalResult`. Validate the index manifest/model compatibility and exclude test/adversarial documents from ordinary retrieval while allowing an explicit test mode to inject them.

Use the validated product line, request type, intent, and supplied object IDs to plan MCP calls. A cross-product auto-debit question must retrieve both applicable banking and insurance guidance plus live banking account and insurance policy facts. A wealth suitability request needs relevant product/risk guidance and the appropriate live product/portfolio facts. Design other request types similarly; do not return a generic single-document answer for customer-specific requests.

Reconcile in code before drafting. Use structured current facts for the customer's actual eligibility/status, KB material for general rules, and an explicit `RetrievalConflict` when they disagree. Preserve both sides and mark MCP as authoritative. Pass conflicts to `ValidationAgent`; do not rely on the retriever LLM alone to invent or identify conflicts. If required MCP data is missing, times out, returns an error envelope, or goes offline mid-request, return a typed service failure. Set `retrieval_complete` accurately and prohibit customer-ready drafting from incomplete context.

**Exit check:** tests prove agreement, policy-permits/account-restricted conflict, unrelated or weak KB retrieval, attribution, ownership check, and one-server-offline handling. The conflict is visible to validation, and no guessed customer-specific answer is produced.

## Phase 5: run the five roles as a single orchestrated pipeline

Implement `nexus_orchestrator.py` as a documented callable interface and runnable local process. Wire `ClassifierAgent`, `RetrieverAgent`, `DraftingAgent`, `ValidationAgent`, and `EscalationAgent` in sequence. Prefer a small explicit controller with injected model, retriever, and MCP-client interfaces over agent-to-agent implicit side effects. Validate each Pydantic output at the boundary. Ensure `RetrieverAgent` returns `GroundedContext`, not a raw vector-search dataclass; `DraftingAgent` cites valid evidence IDs; `ValidationAgent` sees all conflicts and checks factual support, tone, compliance, and PII; `EscalationAgent` invokes the real server decision tool. A failing tool result must not become a successful LLM-shaped escalation result.

Construct `FinalResponse` only after deterministic post-validation checks. High-confidence routine Arjun should be `ready`/customer-sendable, provided evidence supports the statement. Hard categories should be escalated regardless of confidence. Low confidence should escalate. Failed validation should block or escalate, never deliver the rejected draft. Missing MCP data should yield a typed failure or safe handoff; no silent fallback. Correlate request IDs across intake, MCP calls, logs, and final response without logging the raw customer message.

Do not claim auto-debit was activated. The required inventory has no setup-payment write tool. The showcase response can explain eligibility/payment terms and a safe next step, based on live account/policy facts.

**Exit check:** one real request traverses all four layers in a single call; tests assert the order, each role's output, real MCP use, final gate behavior, confidence escalation, hard escalation, and typed failures.

## Phase 6: fixtures, demo, and exact scenario coverage

Correct seed fixture contradictions before using them as evidence: Arjun Mehta (`CUS-20077`) has savings account `ACC-20077`; the PDF calls `POL-IN-30091` a **general insurance** policy, while current seed data marks it health. The current prior-interaction text says checking account; correct it to savings. Keep or create an independently identifiable restricted-account fixture (currently `CUS-30142`/`ACC-30142`) for conflict testing. Keep policy wording, DB facts, KB documents, and expected assertions mutually consistent; rebuild the index if KB content changes. Avoid mutating the user's current DB during tests.

Replace `run_local_demo.py` with a fast deterministic script using injectable fake MCP/LLM adapters that implement the same interfaces as production. It must run without a live MCP client, **but it must not be the only integration evidence**. Exercise the PDF section 8 sequence in order:

1. Valid structured intake and deliberately invalid intake rejected before downstream work.
2. Retrieval combining policy/product text with live-style structured account/policy facts, including a surfaced conflict.
3. Full five-role chain for a routine request.
4. Low-confidence escalation.
5. Hard-category escalation at high confidence, including rephrased attempts.
6. Prompt injection arriving in retrieved content, blocked before it influences the answer.
7. Arjun showcase: classify as `cross_product`; consult banking and insurance evidence; answer the two questions; validate; resolve without human escalation when supported.
8. All four stress tests: missing/unknown customer, retrieval conflict, hard-escalation override attempt, and one MCP server offline mid-request.

Each case should assert its expected result and exit nonzero on failure. Print a concise trace of statuses, evidence references, decision source, and sendability with identifiers masked. Do not print credentials, full IDs, or raw injected text.

**Exit check:** `python run_local_demo.py` completes all assertions from a fresh local setup without live services.

## Phase 7: verification against real services and submission package

Write and execute pytest coverage for `logging_config.py`, `guardrails.py`, `database.py`, all 13 tool functions, exact ID formats, masks, Resources, Prompts, structured intake, hybrid retrieval/reconciliation, all five agent steps, final output gate, and the four stress tests. Use a temporary SQLite database and controlled fake embedding/LLM providers for repeatable unit tests. Also verify real FastMCP registration and client calls. Do not count a stub-only registration test as final evidence.

Run four servers in independent processes on loopback ports and the orchestrator in a fifth. Use a real MCP client to list/call each service and run the Arjun request through the orchestrator. Include a server-outage test that stops insurance mid-request and confirms a typed error without a crash or invented answer. Record commands, exit codes, observed tool lists, and the masked trace in a submission evidence file. If provider credentials are unavailable, finish all credential-independent work and tests, then explicitly identify the exact live check that could not run; never mark it passed.

Deliver a root `README.md`, an architecture document with a data-flow diagram, a requirement-to-test matrix, local startup/shutdown instructions, configuration and security assumptions, a 20-minute live demo/debrief script, and executed test evidence. Explain the documented contract tensions and the chosen implementation, especially caller scope, unknown-customer verification, and escalation persistence. The README should list the exact supported commands; module names in the PDF are illustrative, so use this repo's actual import paths.

### Final acceptance checklist

- [ ] Fresh Python 3.11+ setup installs all dependencies; seed and index commands work with documented paths.
- [ ] All 13 Tools, 4 Resources, and 2 Prompts are registered and callable through real MCP clients.
- [ ] Each server enforces its own table scope; all required ID formats, field scopes, masking, KYC rules, audits, and safe errors are tested.
- [ ] Shared sanitization, injection defense, output filtering, and PII-safe JSON tracing are demonstrably active on customer text, KB text, MCP text, and outbound messages.
- [ ] Invalid intake cannot trigger retrieval or domain tool calls; valid intake has a server-generated request ID and intake-time hard category.
- [ ] One customer request travels through intake, KB + MCP retrieval, reconciliation, five distinct roles, deterministic escalation, and final response filtering.
- [ ] Arjun showcase answers both payment questions from verified account, policy, and KB evidence without claiming the payment was set up.
- [ ] Missing customer, conflict, rephrased hard escalation, retrieved injection, low confidence, and server outage behave as specified.
- [ ] `python run_local_demo.py` and `pytest tests/` both pass; all four real servers and the orchestrator start and respond independently.
- [ ] Root README, architecture document, 20-minute demo/debrief, requirements mapping, and truthful executed verification record are complete.

When finished, report the changed files, exact checks executed and outcomes, key design decisions, and any genuine remaining limitation. Do not call the project complete while an unchecked item remains.
