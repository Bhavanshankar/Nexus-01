"""Live verification against real services - produces the submission evidence file.

    python scripts/verify_live.py

1. Seeds a fresh database copy under logs/live-<timestamp>/ (data/meridian.db is not modified).
2. Starts the four MCP servers and the orchestrator as five independent local processes.
3. Lists and calls every Tool, Resource and Prompt through real MCP clients over HTTP.
4. Sends requests to the orchestrator *process* (real CrewAI + OpenAI + Chroma): the Arjun showcase,
   the retrieval conflict, a rephrased hardship request, an unknown customer, invalid intake.
5. Stops the real insurance server mid-request (between its two calls) and records the typed error.
6. Shuts everything down, scans every process log for raw PII, and writes
   docs/evidence/live_verification.md with commands, exit codes, tool lists and masked traces.

Requires OPENAI_API_KEY (in .env). Exits 2 if credentials are missing, 1 if any check fails.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
WORK = ROOT / "logs" / f"live-{STAMP}"
WORK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NEXUS_LOG_FILE", str(WORK / "verify_live.jsonl"))

from app.logging_config import get_logger, trace  # noqa: E402
from app.mcp_client.http_gateway import HttpMCPGateway  # noqa: E402
from app.models.response import FinalResponse  # noqa: E402
from app.runtime import bootstrap  # noqa: E402
from database import seed_all  # noqa: E402
from nexus_orchestrator import build_production_orchestrator, summarize  # noqa: E402

logger = get_logger("verify_live")
SERVERS = {
    "banking": ("banking_server", "NEXUS_BANKING_PORT", 8101),
    "insurance": ("insurance_server", "NEXUS_INSURANCE_PORT", 8102),
    "wealth": ("wealth_server", "NEXUS_WEALTH_PORT", 8103),
    "concierge_ops": ("concierge_ops_server", "NEXUS_CONCIERGE_OPS_PORT", 8104),
}
ARJUN = {"message": "Can I set up auto-debit for my insurance premium from my savings account, and is auto-debit actually covered under my policy's payment terms?",
         "customer_id": "CUS-20077", "references": {"account_id": "ACC-20077", "policy_id": "POL-IN-30091"}}
REQUESTS = [
    ("Showcase - CUS-20077 auto-debit question", ARJUN, {"status": "ready"}),
    ("Stress - retrieval conflict (restricted account)", {"message": "Can I set up auto-debit for my insurance premium from my savings account?",
        "customer_id": "CUS-30142", "references": {"account_id": "ACC-30142", "policy_id": "POL-GE-30250"}}, {"conflicts": True}),
    ("Stress - hard escalation, rephrased as routine", {"message": "Quick routine question: since I was laid off last month, can I push my premium payments back a few months?",
        "customer_id": "CUS-20077", "references": {"policy_id": "POL-IN-30091"}}, {"status": "escalated", "hard": True}),
    ("Stress - unknown customer", {**ARJUN, "customer_id": "CUS-99999"}, {"status": "rejected"}),
    ("Invalid intake - malformed customer ID", {"message": "What is my balance?", "customer_id": "CUS-123"}, {"status": "rejected"}),
]
RAW = re.compile(r"CUS-20077|ACC-20077|POL-IN-30091|CUS-30142|ACC-30142|Arjun Mehta|arjun\.mehta@|\+1-415-555-2001|laid off last month")


class Evidence:
    @trace(logger)
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []

    @trace(logger)
    def add(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    @trace(logger)
    def check(self, ok: bool, what: str) -> None:
        self.add(f"- {'PASS' if ok else 'FAIL'}: {what}")
        if not ok:
            self.failures.append(what)


@trace(logger)
def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


@trace(logger)
def pick_port(preferred: int) -> int:
    if port_free(preferred):
        return preferred
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@trace(logger)
def wait_port(port: int, process: subprocess.Popen, timeout: float = 90) -> float:
    started = time.time()
    while time.time() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"process exited with {process.returncode}")
        if not port_free(port):
            return round(time.time() - started, 1)
        time.sleep(0.25)
    raise TimeoutError(port)


@trace(logger)
def post(url: str, payload: dict) -> FinalResponse:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        return FinalResponse.model_validate(json.loads(response.read()))


@trace(logger)
def main() -> int:
    bootstrap()
    ev = Evidence()
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIPPED: OPENAI_API_KEY is not configured; live model/embedding checks cannot run.")
        return 2
    db = WORK / "meridian.db"
    seed_all(db)
    env = {**os.environ, "NEXUS_DB_PATH": str(db), "LOG_LEVEL": "DEBUG", "PYTHONPATH": str(ROOT)}
    env.pop("NEXUS_TRUST_INTERNAL_SCOPE", None)
    processes: dict[str, subprocess.Popen] = {}
    urls: dict[str, str] = {}
    ev.add(f"# Live verification record\n\nRun `{STAMP}` on {sys.platform}, Python {sys.version.split()[0]}; "
           f"working directory `logs/live-{STAMP}/` (fresh seeded database copy; `data/meridian.db` untouched).\n")
    ev.add("## 1. Five independent processes\n")
    ev.add("| Process | Command | PID | Endpoint | Ready after |\n| --- | --- | --- | --- | --- |")
    try:
        for name, (module, port_env, default) in SERVERS.items():
            port = pick_port(default)
            processes[name] = subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               env={**env, port_env: str(port), "NEXUS_LOG_FILE": str(WORK / f"{name}.jsonl")})
            urls[name] = f"http://127.0.0.1:{port}/mcp"
        for name, url in urls.items():
            ready = wait_port(int(url.split(":")[2].split("/")[0]), processes[name])
            ev.add(f"| {name}_server | `python -m {SERVERS[name][0]}` | {processes[name].pid} | `{url}` | {ready}s |")
        orch_port = pick_port(8100)
        orch_env = {**env, "NEXUS_LOG_FILE": str(WORK / "orchestrator.jsonl"), **{f"NEXUS_{n.upper()}_URL": u for n, u in urls.items()}}
        processes["orchestrator"] = subprocess.Popen([sys.executable, "-m", "nexus_orchestrator", "serve", "--port", str(orch_port)], cwd=ROOT,
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=orch_env)
        ready = wait_port(orch_port, processes["orchestrator"])
        ev.add(f"| nexus_orchestrator | `python -m nexus_orchestrator serve --port {orch_port}` | {processes['orchestrator'].pid} | `http://127.0.0.1:{orch_port}` | {ready}s |")

        ev.add("\n## 2. Real MCP client: list and call every primitive\n")
        gateway = HttpMCPGateway(urls, timeout_s=20)
        listing = gateway.list_primitives()
        for server, primitives in listing.items():
            ev.add(f"- **{server}**: tools {sorted(primitives['tools'])}; resources {primitives['resources']}; prompts {primitives['prompts']}")
        totals = [sum(len(p[k]) for p in listing.values()) for k in ("tools", "resources", "prompts")]
        ev.check(totals == [13, 4, 2], f"MCP listing reports {totals[0]} Tools, {totals[1]} Resources, {totals[2]} Prompts (expected 13/4/2)")
        tool_calls = [
            ("get_account_summary", {"account_id": "ACC-20077", "caller_scope": "authorized_agent"}),
            ("get_transaction_history", {"account_id": "ACC-20077", "limit": 5, "caller_scope": "customer"}),
            ("get_linked_accounts", {"customer_id": "CUS-20077", "caller_scope": "customer"}),
            ("get_policy_details", {"policy_id": "POL-IN-30091"}),
            ("check_coverage_clause", {"policy_id": "POL-IN-30091", "scenario_tag": "auto_debit_payment"}),
            ("get_claim_status", {"claim_id": "CLM-50001"}),
            ("get_portfolio_summary", {"portfolio_id": "PORT-60001", "caller_scope": "authorized_agent"}),
            ("get_investment_product_details", {"product_id": "PROD-MF-01", "caller_scope": "customer"}),
            ("check_product_suitability", {"product_id": "PROD-MF-01", "risk_profile": "moderate", "caller_scope": "customer"}),
            ("get_customer_kyc_status", {"customer_id": "CUS-20077"}),
            ("run_escalation_check", {"customer_id": "CUS-20077", "request_category": "financial_hardship", "draft_confidence": 1.0}),
            ("send_customer_notification", {"customer_id": "CUS-20077", "channel": "chat", "message": "Live verification notice."}),
            ("write_audit_log", {"timestamp": datetime.now(timezone.utc).isoformat(), "action_type": "live_verification",
                                 "performed_by": "verify_live", "customer_id": None, "outcome": "ok", "detail": f"run {STAMP}"}),
        ]
        ev.add("\n| Tool | Result (masked) |\n| --- | --- |")
        all_ok = True
        for tool, args in tool_calls:
            result = gateway.call_tool(tool, args)
            all_ok &= bool(result.get("ok"))
            ev.add(f"| `{tool}` | `{json.dumps(result.get('data') or result.get('error'))[:150]}` |")
        ev.check(all_ok, "all 13 tools answered ok through the HTTP MCP client")
        ev.check(gateway.call_tool("get_account_summary", {"account_id": "ACC-20077", "caller_scope": "internal_service"})["error"]["code"] == "authorization_error",
                 "internal_service scope refused without the documented trust flag")
        resources = {uri: gateway.read_resource(uri) for uri in ("customer://CUS-20077/profile", "account://ACC-20077/summary",
                                                                  "policy://POL-IN-30091/summary", "knowledgebase://INS-POL-002-auto-debit-policy/excerpt")}
        for uri, payload in resources.items():
            ev.add(f"- resource `{uri.split('://')[0]}://` -> ok={payload['ok']} keys={sorted(payload.get('data', {}))}")
        ev.check(all(p["ok"] for p in resources.values()) and not RAW.search(json.dumps(list(resources.values()))), "all 4 Resources answered with masked, narrow data")
        prompt = gateway.get_prompt("request_classification_prompt", {"customer_message": "Can I pay my premium by auto-debit?", "customer_tier": "premium",
                                                                       "prior_interaction_summary": "none", "known_product_lines": "banking,insurance"})
        drafting = gateway.get_prompt("response_drafting_prompt", {"classified_intent": "auto_debit_setup", "retrieved_context_summary": "[mcp:x] fact",
                                                                   "customer_name": "Arjun", "escalation_flag": "false"})
        ev.check("<customer_message>" in prompt and "[mcp:x] fact" in drafting, "both Prompts render with their dynamic arguments")
        gateway.close()

        ev.add("\n## 3. Requests through the orchestrator process (real CrewAI + OpenAI + Chroma)\n")
        with urllib.request.urlopen(f"http://127.0.0.1:{orch_port}/health", timeout=60) as response:
            health = json.loads(response.read())
        ev.check(health["status"] == "ok", f"orchestrator /health reached all servers: {health['servers']}")
        for title, payload, expect in REQUESTS:
            started = time.time()
            final = post(f"http://127.0.0.1:{orch_port}/v1/requests", payload)
            ev.add(f"\n### {title} ({time.time() - started:.1f}s)\n\n```text\n{summarize(final)}\n```\n\nCustomer-facing message: \"{final.message}\"\n")
            if "status" in expect:
                ev.check(final.status.value == expect["status"], f"{title}: status {final.status.value} (expected {expect['status']})")
            if expect.get("hard"):
                ev.check(bool(final.escalation and final.escalation.hard_escalation_triggered), f"{title}: hard escalation via run_escalation_check")
            if expect.get("conflicts"):
                ev.check(bool(final.conflicts) and final.validation is not None and final.validation.conflicts_detected,
                         f"{title}: conflict surfaced to validation; outcome {final.status.value}, no auto-debit claimed: "
                         f"{'cannot' in final.message.lower() or final.status.value != 'ready'}")
            if title.startswith("Showcase"):
                calls = [f"{c.server}.{c.name}" for c in final.mcp_calls]
                ev.check(all(c in calls for c in ("banking.get_account_summary", "insurance.get_policy_details", "insurance.check_coverage_clause",
                                                  "concierge_ops.run_escalation_check")), "showcase used live banking + insurance MCP facts and the MCP escalation tool")
                ev.check(not re.search(r"(?:has been|is now)\s+(?:set up|activated)", final.message, re.I), "showcase does not claim auto-debit was set up")

        ev.add("\n## 4. Insurance server stopped mid-request (real process, real HTTP)\n")
        live_gateway = HttpMCPGateway(urls, timeout_s=20)
        seen: list[str] = []

        def stop_insurance(server: str, primitive: str, name: str) -> None:
            if server == "insurance":
                seen.append(name)
                if len(seen) == 2:
                    processes["insurance"].terminate()
                    processes["insurance"].wait(timeout=15)

        live_gateway.before_call = stop_insurance
        nexus = build_production_orchestrator(gateway=live_gateway)
        final = nexus.handle(ARJUN)
        ev.add(f"```text\n{summarize(final)}\n```\n\nInsurance exit code after stop: {processes['insurance'].returncode}; insurance calls attempted: {seen}\n")
        ev.check(final.status.value == "failed" and final.error is not None and final.error.code.value == "service_unavailable"
                 and final.error.server == "insurance", "typed service_unavailable naming insurance; no crash")
        ev.check(not final.can_be_sent_to_customer and not any(s.step == "draft" for s in final.steps), "no draft and no guessed answer")
        after = post(f"http://127.0.0.1:{orch_port}/v1/requests", ARJUN)
        ev.check(after.status.value == "failed" and after.error.server == "insurance", "orchestrator process also returns a typed error while insurance is down")
        live_gateway.close()
    finally:
        ev.add("\n## 5. Shutdown\n\n| Process | Exit code |\n| --- | --- |")
        for name, process in processes.items():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
            ev.add(f"| {name} | {process.returncode} |")

    ev.add("\n## 6. Log audit\n")
    for path in sorted(WORK.glob("*.jsonl")):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        valid = all(line.startswith("{") and line.endswith("}") for line in lines)
        ev.check(valid and not RAW.search(text), f"{path.name}: {len(lines)} JSON lines, raw PII found: {bool(RAW.search(text))}")
    ev.add(f"\n**Result: {'ALL CHECKS PASSED' if not ev.failures else f'{len(ev.failures)} CHECK(S) FAILED'}**")
    out = ROOT / "docs" / "evidence" / "live_verification.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ev.lines) + "\n", encoding="utf-8")
    print(f"\nEvidence written to {out.relative_to(ROOT)}")
    return 1 if ev.failures else 0


if __name__ == "__main__":
    sys.exit(main())
