# Live verification record

Run `20261009T055539Z` on win32, Python 3.13.7; working directory `logs/live-20261009T055539Z/` (fresh seeded database copy; `data/meridian.db` untouched).

## 1. Five independent processes

| Process | Command | PID | Endpoint | Ready after |
| --- | --- | --- | --- | --- |
| banking_server | `python -m banking_server` | 2052 | `http://127.0.0.1:60172/mcp` | 9.6s |
| insurance_server | `python -m insurance_server` | 11440 | `http://127.0.0.1:60174/mcp` | 0.0s |
| wealth_server | `python -m wealth_server` | 17368 | `http://127.0.0.1:60176/mcp` | 0.0s |
| concierge_ops_server | `python -m concierge_ops_server` | 30736 | `http://127.0.0.1:60178/mcp` | 0.0s |
| nexus_orchestrator | `python -m nexus_orchestrator serve --port 60201` | 20296 | `http://127.0.0.1:60201` | 3.8s |

## 2. Real MCP client: list and call every primitive

- **banking**: tools ['get_account_summary', 'get_linked_accounts', 'get_transaction_history']; resources ['account://{account_id}/summary']; prompts []
- **insurance**: tools ['check_coverage_clause', 'get_claim_status', 'get_policy_details']; resources ['policy://{policy_id}/summary']; prompts []
- **wealth**: tools ['check_product_suitability', 'get_investment_product_details', 'get_portfolio_summary']; resources []; prompts []
- **concierge_ops**: tools ['get_customer_kyc_status', 'run_escalation_check', 'send_customer_notification', 'write_audit_log']; resources ['customer://{customer_id}/profile', 'knowledgebase://{document_id}/excerpt']; prompts ['request_classification_prompt', 'response_drafting_prompt']
- PASS: MCP listing reports 13 Tools, 4 Resources, 2 Prompts (expected 13/4/2)

| Tool | Result (masked) |
| --- | --- |
| `get_account_summary` | `{"account_id": "****0077", "customer_id": "CUS-20077", "account_type": "savings", "status": "active", "auto_debit_allowed": true, "currency": "USD"}` |
| `get_transaction_history` | `{"account_id": "****0077", "transactions": [{"amount": {"amount_minor": 250000, "currency": "USD"}, "direction": "credit", "description": "Salary cred` |
| `get_linked_accounts` | `{"customer_id": "CUS-20077", "accounts": [{"account_id": "****0077", "account_type": "savings", "status": "active", "auto_debit_allowed": true, "curre` |
| `get_policy_details` | `{"policy_id": "****0091", "customer_id": "CUS-20077", "policy_type": "general", "premium": {"amount_minor": 12500, "currency": "USD"}, "payment_freque` |
| `check_coverage_clause` | `{"policy_id": "****0091", "scenario_tag": "auto_debit_payment", "resolved_clause_id": "auto_debit_payment", "policy_status": "active", "coverage_statu` |
| `get_claim_status` | `{"claim_id": "****0001", "policy_id": "****0091", "customer_id": "CUS-20077", "status": "submitted", "filed_at": "2023-12-16T13:20:00Z", "last_updated` |
| `get_portfolio_summary` | `{"portfolio_id": "****0001", "customer_id": "CUS-20077", "risk_profile": "conservative", "status": "frozen"}` |
| `get_investment_product_details` | `{"product_id": "PROD-MF-01", "name": "Meridian Balanced Growth Fund", "product_type": "mutual_fund", "min_risk_profile": "moderate", "description": "D` |
| `check_product_suitability` | `{"product_id": "PROD-MF-01", "risk_profile": "moderate", "minimum_risk_profile": "moderate", "suitable": true, "reason_code": "meets_minimum_risk_prof` |
| `get_customer_kyc_status` | `{"customer_id": "CUS-20077", "kyc_status": "verified", "kyc_last_checked_at": "2026-08-12T10:15:00Z"}` |
| `run_escalation_check` | `{"customer_id": "CUS-20077", "request_category": "financial_hardship", "draft_confidence": 1.0, "should_escalate": true, "reason": "hard_category", "h` |
| `send_customer_notification` | `{"customer_id": "CUS-20077", "request_id": "REQ-7DBD4A7E3682", "channel": "chat", "outcome": "logged", "delivery_confirmed": false, "interaction_id": ` |
| `write_audit_log` | `{"audit_id": 27, "timestamp": "2026-10-09T05:56:05.098004Z", "action_type": "live_verification", "performed_by": "verify_live", "customer_id": null, "` |
- PASS: all 13 tools answered ok through the HTTP MCP client
- PASS: internal_service scope refused without the documented trust flag
- resource `customer://` -> ok=True keys=['customer_id', 'customer_tier', 'first_name', 'recent_interaction_at', 'recent_interaction_summary']
- resource `account://` -> ok=True keys=['account_id', 'account_type', 'auto_debit_allowed', 'balance', 'currency', 'status']
- resource `policy://` -> ok=True keys=['auto_debit_permitted', 'payment_frequency', 'policy_id', 'policy_type', 'premium', 'status']
- resource `knowledgebase://` -> ok=True keys=['category', 'document_id', 'excerpt', 'product_line', 'title', 'truncated']
- PASS: all 4 Resources answered with masked, narrow data
- PASS: both Prompts render with their dynamic arguments

## 3. Requests through the orchestrator process (real CrewAI + OpenAI + Chroma)

- PASS: orchestrator /health reached all servers: {'banking': {'tools': 3, 'resources': 1, 'prompts': 0}, 'insurance': {'tools': 3, 'resources': 1, 'prompts': 0}, 'wealth': {'tools': 3, 'resources': 0, 'prompts': 0}, 'concierge_ops': {'tools': 4, 'resources': 2, 'prompts': 2}}

### Showcase - CUS-20077 auto-debit question (50.3s)

```text
request REQ-B2D7CF33D7A2 customer ****0077 -> READY (sendable=True)
  intake_gate    deterministic    passed             {"references": ["account_id", "policy_id"]}
  identity_check mcp_resource     passed             {"tier": "premium"}
  classify       ClassifierAgent  ok                 {"product_line": "cross_product", "request_type": "auto_debit_setup", "urgency": "routine", "escalation_category": "none"}
  kyc_check      mcp              ok                 {"kyc_status": "verified"}
  retrieve       RetrieverAgent   complete           {"kb_status": "SUCCESS", "evidence": ["mcp:banking.account.account_type", "mcp:banking.account.status", "mcp:banking.account.auto_debit_allowed", "mcp:banking.account.currency", "mcp:insurance.policy.policy_type", "mcp:i
  draft          DraftingAgent    ok                 {"holding_reply": false, "cited": ["mcp:banking.account.account_type", "mcp:banking.account.status", "mcp:banking.account.auto_debit_allowed", "mcp:insurance.policy.auto_debit_permitted", "mcp:insurance.policy.payment_te
  validate       ValidationAgent  passed             {"confidence": 1.0, "deterministic_confidence": 1.0, "model_confidence": 1.0, "conflicts_seen": 0}
  escalate       EscalationAgent  no_escalation      {"reason": "none", "server_reason": "none", "decision_id": "REQ-B2D7CF33D7A2", "hard": false}
  output_gate    deterministic    send               {}
  deliver        mcp              logged             {"delivery_confirmed": false}
  audit          mcp              recorded           {"audit_id": 29}
  decision: should_escalate=False reason=none source=concierge_ops.run_escalation_check confidence=1.0
  mcp calls: concierge_ops.customer://, concierge_ops.request_classification_prompt, concierge_ops.get_customer_kyc_status, banking.get_account_summary, insurance.get_policy_details, insurance.check_coverage_clause, concierge_ops.response_drafting_prompt, concierge_ops.run_escalation_check, concierge_ops.send_customer_notification, concierge_ops.write_audit_log
```

Customer-facing message: "Yes, you can set up auto-debit for your insurance premium from your savings account. Your savings account is active and allows auto-debit. Additionally, your insurance policy permits auto-debit payments, and it is covered under the payment terms, which state that electronic payments are accepted. The premium is due monthly on the 5th. 

To proceed, you may want to contact customer service to initiate the auto-debit setup process."

- PASS: Showcase - CUS-20077 auto-debit question: status ready (expected ready)
- PASS: showcase used live banking + insurance MCP facts and the MCP escalation tool
- PASS: showcase does not claim auto-debit was set up

### Stress - retrieval conflict (restricted account) (22.1s)

```text
request REQ-333ECE8023BB customer ****0142 -> ESCALATED (sendable=False)
  intake_gate    deterministic    passed             {"references": ["account_id", "policy_id"]}
  identity_check mcp_resource     passed             {"tier": "standard"}
  classify       ClassifierAgent  ok                 {"product_line": "cross_product", "request_type": "auto_debit_setup", "urgency": "routine", "escalation_category": "none"}
  kyc_check      mcp              ok                 {"kyc_status": "verified"}
  retrieve       RetrieverAgent   complete           {"kb_status": "SUCCESS", "evidence": ["mcp:banking.account.account_type", "mcp:banking.account.status", "mcp:banking.account.auto_debit_allowed", "mcp:banking.account.currency", "mcp:insurance.policy.policy_type", "mcp:i
  draft          DraftingAgent    ok                 {"holding_reply": false, "cited": ["mcp:banking.account.account_type", "mcp:banking.account.status", "mcp:banking.account.auto_debit_allowed", "mcp:insurance.policy.auto_debit_permitted", "mcp:insurance.policy.payment_fr
  validate       ValidationAgent  failed             {"confidence": 0.4, "deterministic_confidence": 1.0, "model_confidence": 0.4, "issues": ["model_found_unsupported_claims", "model_compliance_note"], "conflicts_seen": 2}
  escalate       EscalationAgent  escalate           {"reason": "low_confidence", "server_reason": "low_confidence", "decision_id": "REQ-333ECE8023BB", "hard": false}
  output_gate    deterministic    skipped_escalated  {}
  audit          mcp              recorded           {"audit_id": 30}
  decision: should_escalate=True reason=low_confidence source=concierge_ops.run_escalation_check confidence=0.4
  mcp calls: concierge_ops.customer://, concierge_ops.request_classification_prompt, concierge_ops.get_customer_kyc_status, banking.get_account_summary, insurance.get_policy_details, insurance.check_coverage_clause, concierge_ops.response_drafting_prompt, concierge_ops.run_escalation_check, concierge_ops.write_audit_log
```

Customer-facing message: "Thank you, Priya. Your request needs a Meridian specialist, who will review it personally and contact you. Nothing has been changed on your accounts or policies."

- PASS: Stress - retrieval conflict (restricted account): conflict surfaced to validation; outcome escalated, no auto-debit claimed: True

### Stress - hard escalation, rephrased as routine (12.0s)

```text
request REQ-A8982959AF59 customer ****0077 -> ESCALATED (sendable=False)
  intake_gate    deterministic    passed             {"references": ["policy_id"]}
  identity_check mcp_resource     passed             {"tier": "premium"}
  classify       ClassifierAgent  ok                 {"product_line": "insurance", "request_type": "payment_terms", "urgency": "high", "escalation_category": "financial_hardship", "signals": ["financial_hardship:job_loss"]}
  kyc_check      mcp              ok                 {"kyc_status": "verified"}
  retrieve       RetrieverAgent   complete           {"kb_status": "SUCCESS", "evidence": ["mcp:insurance.policy.policy_type", "mcp:insurance.policy.status", "mcp:insurance.policy.payment_frequency", "mcp:insurance.policy.premium", "mcp:insurance.policy.payment_terms", "mc
  draft          DraftingAgent    ok                 {"holding_reply": true}
  validate       ValidationAgent  passed             {"confidence": 1.0, "deterministic_confidence": 1.0, "model_confidence": 1.0, "conflicts_seen": 0}
  escalate       EscalationAgent  escalate           {"reason": "hard_category", "server_reason": "hard_category", "decision_id": "REQ-A8982959AF59", "hard": true}
  output_gate    deterministic    skipped_escalated  {}
  audit          mcp              recorded           {"audit_id": 31}
  decision: should_escalate=True reason=hard_category source=concierge_ops.run_escalation_check confidence=1.0
  mcp calls: concierge_ops.customer://, concierge_ops.request_classification_prompt, concierge_ops.get_customer_kyc_status, insurance.get_policy_details, insurance.check_coverage_clause, concierge_ops.response_drafting_prompt, concierge_ops.run_escalation_check, concierge_ops.write_audit_log
```

Customer-facing message: "Thank you, Arjun. Your request needs a Meridian specialist, who will review it personally and contact you. Nothing has been changed on your accounts or policies."

- PASS: Stress - hard escalation, rephrased as routine: status escalated (expected escalated)
- PASS: Stress - hard escalation, rephrased as routine: hard escalation via run_escalation_check

### Stress - unknown customer (0.7s)

```text
request REQ-5B0A2D5A1BD9 customer ****9999 -> REJECTED (sendable=False)
  intake_gate    deterministic    passed             {"references": ["account_id", "policy_id"]}
  identity_check mcp_resource     rejected           {"code": "not_found"}
  error: unknown_customer at intake (-.-)
  mcp calls: concierge_ops.customer://!not_found
```

Customer-facing message: "We could not find a customer with that ID."

- PASS: Stress - unknown customer: status rejected (expected rejected)

### Invalid intake - malformed customer ID (0.0s)

```text
request REQ-67F7FB757986 customer None -> REJECTED (sendable=False)
  intake_gate    deterministic    rejected           {"code": "malformed_customer_id"}
  error: invalid_intake at intake (-.-)
  mcp calls: none
```

Customer-facing message: "customer_id must use the CUS-XXXXX format."

- PASS: Invalid intake - malformed customer ID: status rejected (expected rejected)

## 4. Insurance server stopped mid-request (real process, real HTTP)

```text
request REQ-F3D69A2AA237 customer ****0077 -> FAILED (sendable=False)
  intake_gate    deterministic    passed             {"references": ["account_id", "policy_id"]}
  identity_check mcp_resource     passed             {"tier": "premium"}
  classify       ClassifierAgent  ok                 {"product_line": "cross_product", "request_type": "auto_debit_setup", "urgency": "routine", "escalation_category": "none"}
  kyc_check      mcp              ok                 {"kyc_status": "verified"}
  retrieve       RetrieverAgent   incomplete         {"kb_status": "SUCCESS", "evidence": ["mcp:banking.account.account_type", "mcp:banking.account.status", "mcp:banking.account.auto_debit_allowed", "mcp:banking.account.currency", "mcp:insurance.policy.policy_type", "mcp:i
  escalate       EscalationAgent  escalate           {"reason": "low_confidence", "server_reason": "low_confidence", "decision_id": "REQ-F3D69A2AA237", "hard": false}
  audit          mcp              recorded           {"audit_id": 32}
  decision: should_escalate=True reason=low_confidence source=concierge_ops.run_escalation_check confidence=0.0
  error: service_unavailable at retrieve (insurance.check_coverage_clause)
  mcp calls: concierge_ops.customer://, concierge_ops.request_classification_prompt, concierge_ops.get_customer_kyc_status, banking.get_account_summary, insurance.get_policy_details, insurance.check_coverage_clause!service_unavailable, concierge_ops.run_escalation_check, concierge_ops.write_audit_log
```

Insurance exit code after stop: 1; insurance calls attempted: ['get_policy_details', 'check_coverage_clause']

- PASS: typed service_unavailable naming insurance; no crash
- PASS: no draft and no guessed answer
- PASS: orchestrator process also returns a typed error while insurance is down

## 5. Shutdown

| Process | Exit code |
| --- | --- |
| banking | 1 |
| insurance | 1 |
| wealth | 1 |
| concierge_ops | 1 |
| orchestrator | 1 |

## 6. Log audit

- PASS: banking.jsonl: 291 JSON lines, raw PII found: False
- PASS: concierge_ops.jsonl: 1235 JSON lines, raw PII found: False
- PASS: insurance.jsonl: 381 JSON lines, raw PII found: False
- PASS: orchestrator.jsonl: 3197 JSON lines, raw PII found: False
- PASS: verify_live.jsonl: 1729 JSON lines, raw PII found: False
- PASS: wealth.jsonl: 81 JSON lines, raw PII found: False

**Result: ALL CHECKS PASSED**
