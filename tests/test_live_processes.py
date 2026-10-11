"""Real processes: the four MCP servers and the orchestrator each run as a separate local Python
process on loopback ports, reached through real FastMCP clients over Streamable HTTP.

No model credentials are needed: the orchestrator pipeline here uses the production HTTP gateway
with the scripted model and offline KB index, and the orchestrator *process* is checked through its
/health endpoint (which lists every server's primitives over MCP). The credentialed live run
(CrewAI + OpenAI + Chroma) is ``scripts/verify_live.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from app.agents.scripted import ScriptedLLM
from app.mcp_client.gateway import MCPUnavailable
from app.mcp_client.http_gateway import HttpMCPGateway
from nexus_orchestrator import NexusOrchestrator
from tests.conftest import ARJUN, ROOT, TEMPLATE_DB

SERVERS = {
    "banking": ("mcp_servers.banking_server", "NEXUS_BANKING_PORT"),
    "insurance": ("mcp_servers.insurance_server", "NEXUS_INSURANCE_PORT"),
    "wealth": ("mcp_servers.wealth_server", "NEXUS_WEALTH_PORT"),
    "concierge_ops": ("mcp_servers.concierge_ops_server", "NEXUS_CONCIERGE_OPS_PORT"),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, process: subprocess.Popen, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited early with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"port {port} did not open")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("live")
    db = workdir / "meridian.db"
    shutil.copy2(TEMPLATE_DB, db)
    env = {**os.environ, "NEXUS_DB_PATH": str(db), "LOG_LEVEL": "DEBUG", "PYTHONPATH": str(ROOT)}
    env.pop("NEXUS_TRUST_INTERNAL_SCOPE", None)
    processes, urls, logs = {}, {}, {}
    try:
        for name, (module, port_env) in SERVERS.items():
            port = free_port()
            logs[name] = workdir / f"{name}.jsonl"
            processes[name] = subprocess.Popen(
                [sys.executable, "-m", module], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={**env, port_env: str(port), "NEXUS_LOG_FILE": str(logs[name])},
            )
            urls[name] = f"http://127.0.0.1:{port}/mcp"
        for name, url in urls.items():
            wait_for_port(int(url.rsplit(":", 1)[1].split("/")[0]), processes[name])
        yield {"processes": processes, "urls": urls, "logs": logs, "db": db, "env": env, "workdir": workdir}
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


@pytest.fixture
def gateway(live):
    gw = HttpMCPGateway(live["urls"], timeout_s=10)
    yield gw
    gw.close()


def test_each_server_process_lists_its_primitives(gateway):
    listing = gateway.list_primitives()
    assert {s: sorted(p["tools"]) for s, p in listing.items()} == {
        "banking": ["get_account_summary", "get_linked_accounts", "get_transaction_history"],
        "insurance": ["check_coverage_clause", "get_claim_status", "get_policy_details"],
        "wealth": ["check_product_suitability", "get_investment_product_details", "get_portfolio_summary"],
        "concierge_ops": ["get_customer_kyc_status", "run_escalation_check", "send_customer_notification", "write_audit_log"],
    }
    totals = [sum(len(p[k]) for p in listing.values()) for k in ("tools", "resources", "prompts")]
    assert totals == [13, 4, 2]


def test_live_tool_resource_and_prompt_calls(gateway):
    assert gateway.call_tool("get_account_summary", {"account_id": "ACC-20077", "caller_scope": "authorized_agent"})["data"]["account_id"] == "****0077"
    assert gateway.call_tool("check_coverage_clause", {"policy_id": "POL-IN-30091", "scenario_tag": "auto_debit_payment"})["data"]["covered"] is True
    assert gateway.call_tool("check_product_suitability", {"product_id": "PROD-MF-01", "risk_profile": "moderate", "caller_scope": "customer"})["data"]["suitable"]
    hard = gateway.call_tool("run_escalation_check", {"customer_id": "CUS-20077", "request_category": "financial_hardship", "draft_confidence": 1.0})
    assert hard["data"]["should_escalate"] and hard["data"]["reason"] == "hard_category"
    assert gateway.read_resource("customer://CUS-20077/profile")["data"]["customer_id"] == "****0077"
    assert gateway.read_resource("policy://POL-IN-30091/summary")["data"]["policy_id"] == "****0091"
    assert "Customer tier: premium" in gateway.get_prompt("request_classification_prompt", {
        "customer_message": "auto-debit?", "customer_tier": "premium", "prior_interaction_summary": "none", "known_product_lines": "banking"})
    assert gateway.call_tool("get_account_summary", {"account_id": "ACC-1", "caller_scope": "customer"})["error"]["code"] == "validation_error"


def test_orchestrator_pipeline_over_real_server_processes(live, gateway, kb_retriever):
    final = NexusOrchestrator(gateway=gateway, llm=ScriptedLLM(), kb_search=kb_retriever).handle(ARJUN)
    assert final.status.value == "ready" and final.can_be_sent_to_customer
    assert [c.server for c in final.mcp_calls if c.primitive == "tool"][:3] == ["concierge_ops", "banking", "insurance"]
    # The orchestrator's request ID was propagated over HTTP and appears in the servers' own logs.
    for name in ("banking", "insurance", "concierge_ops"):
        text = live["logs"][name].read_text(encoding="utf-8")
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        correlated = {line["function"] for line in lines if line.get("request_id") == final.request_id}
        assert any(f.startswith(f"mcp_servers.{name}_server.") for f in correlated), name  # fully qualified, correlated
        assert "CUS-20077" not in text and "ACC-20077" not in text and "Arjun Mehta" not in text


def test_orchestrator_process_starts_and_reaches_all_servers(live):
    port = free_port()
    env = {**live["env"], "NEXUS_LOG_FILE": str(live["workdir"] / "orchestrator.jsonl"),
           "NEXUS_BANKING_URL": live["urls"]["banking"], "NEXUS_INSURANCE_URL": live["urls"]["insurance"],
           "NEXUS_WEALTH_URL": live["urls"]["wealth"], "NEXUS_CONCIERGE_OPS_URL": live["urls"]["concierge_ops"]}
    process = subprocess.Popen([sys.executable, "-m", "nexus_orchestrator", "serve", "--port", str(port)], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        wait_for_port(port, process)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=60) as response:
            health = json.loads(response.read())
        assert response.status == 200 and health["status"] == "ok"
        assert health["servers"] == {"banking": {"tools": 3, "resources": 1, "prompts": 0},
                                     "insurance": {"tools": 3, "resources": 1, "prompts": 0},
                                     "wealth": {"tools": 3, "resources": 0, "prompts": 0},
                                     "concierge_ops": {"tools": 4, "resources": 2, "prompts": 2}}
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_insurance_process_stopped_mid_request(live, gateway, kb_retriever):
    """Runs last: terminates the real insurance server between two of its calls in one request."""
    insurance = live["processes"]["insurance"]
    seen = []

    def stop_insurance_before_second_call(server, primitive, name):
        if server == "insurance":
            seen.append(name)
            if len(seen) == 2:
                insurance.terminate()
                insurance.wait(timeout=10)

    gateway.before_call = stop_insurance_before_second_call
    final = NexusOrchestrator(gateway=gateway, llm=ScriptedLLM(), kb_search=kb_retriever).handle(ARJUN)
    assert insurance.poll() is not None and seen == ["get_policy_details", "check_coverage_clause"]
    assert final.status.value == "failed" and final.error.code.value == "service_unavailable"
    assert final.error.server == "insurance" and final.error.tool == "check_coverage_clause"
    assert not final.can_be_sent_to_customer and "allows auto-debit" not in final.message
    assert final.escalation is not None and final.escalation.should_escalate  # concierge still reachable
    with pytest.raises(MCPUnavailable):
        gateway.call_tool("get_policy_details", {"policy_id": "POL-IN-30091"})


def test_pdf_style_entry_point_over_stdio_transport():
    """`python -m banking_server` (PDF 2.3) launched by an MCP host over stdio, as Claude Desktop does."""
    import asyncio

    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    async def run():
        transport = StdioTransport(command=sys.executable, args=["-m", "banking_server"], cwd=str(ROOT),
                                   env={**os.environ, "NEXUS_MCP_TRANSPORT": "stdio", "LOG_LEVEL": "ERROR", "PYTHONPATH": str(ROOT)})
        async with Client(transport) as client:
            tools = sorted(t.name for t in await client.list_tools())
            result = await client.call_tool("get_account_summary", {"account_id": "ACC-20077", "caller_scope": "customer"})
            return tools, result.structured_content

    tools, result = asyncio.run(run())
    assert tools == ["get_account_summary", "get_linked_accounts", "get_transaction_history"]
    assert result["data"]["account_id"] == "****0077"
