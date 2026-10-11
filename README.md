# Nexus - unified financial services concierge (Meridian Financial Group)

One customer request flows through **structured intake -> hybrid retrieval -> five agent roles ->
four MCP servers** in a single call path. Every backend read and write goes through MCP; every
function is traced as structured JSON; a shared guardrails module sanitises input, defends against
prompt injection (including in retrieved content), filters output and keeps PII out of logs.

- Architecture, data flow, requirement matrix, design tensions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- MCP contract (13 Tools, 4 Resources, 2 Prompts): [`MCP_CONTRACTS.md`](MCP_CONTRACTS.md)

- Executed verification record: [`docs/SUBMISSION_EVIDENCE.md`](docs/SUBMISSION_EVIDENCE.md)

## Setup (Python 3.11+)

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (needed only for the live path: index
build, query embeddings and the CrewAI model steps). The demo and the test suite need no credentials.

```bash
python database.py --seed
```

Creates `data/meridian.db` (an existing file is first copied to `data/backups/`). Use
`--db-path PATH` for another file.

```bash
python build_index.py
```

Builds `vector_store/chroma` + `vector_store/manifest.json` from `knowledge_base/` with OpenAI
`text-embedding-3-small` (64 documents, 335 chunks). The index is replaced atomically.

## Run the demo and the tests (no live services)

```bash
python run_local_demo.py
```

Runs the PDF section 8 sequence (valid/invalid intake, reconciliation and conflict, five-role chain,
confidence and hard escalation including rephrasings, injection in retrieved content, the Arjun
showcase, all four stress tests) with 43 assertions in about 30 s; exits non-zero on any failure.

```bash
pytest tests/
```

317 tests: guardrails, logging, database isolation, all 13 tools, real FastMCP registration,
Resources/Prompts, intake, retrieval/reconciliation, each agent role, the orchestrator and stress tests,
architecture rules (layer isolation, trace coverage), and real server/orchestrator **processes**.
Tests use temporary database copies and never modify `data/meridian.db`.

## Run the system as five local processes

Start each server in its own terminal (they bind to 127.0.0.1; `python -m mcp_servers.<name>` is equivalent):

```bash
python -m banking_server
```

```bash
python -m insurance_server
```

```bash
python -m wealth_server
```

```bash
python -m concierge_ops_server
```

Then the orchestrator (an MCP client of all four):

```bash
python -m nexus_orchestrator serve
```

It listens on `http://127.0.0.1:8100`: `GET /health` lists every server's primitives over MCP;
`POST /v1/requests` takes `{"message", "customer_id", "references": {...}, "channel"}` and returns the
`FinalResponse`. Other entry points:

```bash
python -m nexus_orchestrator check
```

```bash
python -m nexus_orchestrator ask --customer-id CUS-20077 --account-id ACC-20077 --policy-id POL-IN-30091 "Can I set up auto-debit for my insurance premium from my savings account, and is auto-debit actually covered under my policy's payment terms?"
```

Stop any process with Ctrl+C. Ports: banking 8101, insurance 8102, wealth 8103, concierge_ops 8104,
orchestrator 8100 (override with `NEXUS_*_PORT` / `NEXUS_*_URL`, see `.env.example`).

### One-command live verification (credentials required)

```bash
python scripts/verify_live.py
```

Starts all five processes on a fresh database copy, lists and calls every primitive through real MCP
clients, sends the showcase and stress requests to the orchestrator process (CrewAI + OpenAI + Chroma),
stops the insurance server mid-request, audits all logs and writes
[`docs/evidence/live_verification.md`](docs/evidence/live_verification.md).

### Using an MCP host instead of the orchestrator

Any MCP client can attach to the servers' Streamable HTTP endpoints, e.g. `http://127.0.0.1:8101/mcp`.

## Repository layout

| Path | Contents |
| --- | --- |
| `nexus_orchestrator.py` | Orchestrator: pipeline controller, HTTP API, CLI |
| `app/intake/` | Intake gate and hard-category signals |
| `app/agents/` | Classifier, Retriever, Drafting, Validation, Escalation roles; CrewAI and scripted backends |
| `app/retrieval/` | Hybrid KB retriever, adapter, planner, reconciliation, hybrid retrieval |
| `app/mcp_client/` | Gateway interface, HTTP gateway (production), in-process adapters (demo/tests) |
| `app/models/` | Pydantic contracts (intake, retrieval, validation, escalation, response) |
| `app/guardrails.py`, `app/logging_config.py` | Shared guardrails; structured tracing |
| `app/knowledge_base.py`, `app/db_access.py`, `app/runtime.py` | KB file access; least-privilege SQLite; process bootstrap |
| `mcp_servers/` | The four MCP servers and their shared helpers |
| `database.py`, `db/` | Schema, seed fixtures, seeding CLI |
| `build_index.py`, `knowledge_base/`, `vector_store/` | KB corpus and vector index |
| `run_local_demo.py`, `scripts/verify_live.py` | Offline demo; live verification |
| `tests/` | pytest suite |
| `docs/` | Architecture, demo script, evidence |

## Showcase fixture

Arjun Mehta `CUS-20077`: savings account `ACC-20077` (active, auto-debit allowed) and general insurance
policy `POL-IN-30091` (active, monthly, auto-debit clause permitted). Conflict fixture: `CUS-30142` with
restricted `ACC-30142` (auto-debit not allowed) and `POL-GE-30250` (auto-debit permitted).
