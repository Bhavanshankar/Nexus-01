"""Structural guarantees checked from source: layer isolation and @trace coverage (PDF section 5)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"nexusvenv", ".venv", "venv", "__pycache__", "vector_store", "tests", ".git", "data", "logs", "knowledge_base"}


def project_sources() -> list[Path]:
    found = []
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix == ".py":
            found.append(path)
        elif path.is_dir() and path.name not in SKIP_DIRS:
            found += [p for p in path.rglob("*.py") if not SKIP_DIRS & set(p.relative_to(ROOT).parts)]
    return sorted(found)


def imports_of(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# Layers that must reach backend data only through an MCP gateway.
ISOLATED = ["nexus_orchestrator.py", "app/agents", "app/retrieval", "app/intake", "app/models", "app/mcp_client/gateway.py",
            "app/mcp_client/http_gateway.py", "app/guardrails.py", "app/knowledge_base.py"]


def isolated_files() -> list[Path]:
    files = []
    for entry in ISOLATED:
        target = ROOT / entry
        files += sorted(target.rglob("*.py")) if target.is_dir() else [target]
    return files


@pytest.mark.parametrize("path", isolated_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_orchestration_layers_never_touch_sqlite_or_server_code(path):
    forbidden = {name for name in imports_of(path) if name == "sqlite3" or name.startswith("mcp_servers") or name in {"app.db_access", "database"}}
    assert not forbidden, f"{path.name} imports {forbidden}"


def test_production_wiring_does_not_use_in_process_adapters():
    source = (ROOT / "nexus_orchestrator.py").read_text(encoding="utf-8")
    assert "in_process" not in source and "ScriptedLLM" not in source
    assert "HttpMCPGateway" in source and "CrewAIBackend" in source


def test_only_mcp_servers_use_scoped_database_access():
    users = {p.relative_to(ROOT).as_posix() for p in project_sources() if "app.db_access" in imports_of(p)}
    assert users <= {"database.py", "mcp_servers/common.py"}


# ------------------------------------------------------------------------------ trace coverage

# qualified name -> reason it is exempt. Everything else project-written must carry @trace.
EXEMPT = {
    "app/logging_config.py:*": "the tracer itself; tracing it would recurse on every log line",
    "app/guardrails.py:redact_for_logging": "the tracer's own redactor (recursion)",
    "app/guardrails.py:redact_text": "the tracer's own redactor (recursion)",
    "app/db_access.py:_redact_initialize_arguments": "a redactor: tracing it would log the seed rows it hides",
    "app/guardrails.py:_normalize": "pure helper inside traced sanitize_free_text/detect_injection",
    "app/guardrails.py:GuardrailError.__init__": "exception constructors",
    "app/guardrails.py:PromptInjectionDetected.__init__": "exception constructors",
    "app/guardrails.py:OutputDecision.sendable": "dataclass property",
    "app/retrieval/retrieval.py:UnstructuredRetriever._tokenize": "hot-path arithmetic: called per chunk per query; callers traced",
    "app/retrieval/retrieval.py:UnstructuredRetriever._cosine": "hot-path arithmetic: O(chunks^2) per MMR pass; callers traced",
    "app/retrieval/retrieval.py:UnstructuredRetriever._clip": "hot-path arithmetic; callers traced",
    "app/retrieval/retrieval.py:UnstructuredRetriever._fresh": "hot-path per-chunk copy; callers traced",
    "app/retrieval/retrieval.py:UnstructuredRetriever._allowed": "hot-path per-chunk filter; callers traced",
    "app/retrieval/retrieval.py:QualityThresholds.for_hashing_embedder": "constant factory",
    "app/retrieval/retrieval.py:OpenAIEmbedder.__init__": "client construction; no data",
    "app/retrieval/retrieval.py:HashingEmbedder.__init__": "constant construction",
    "app/retrieval/hybrid.py:OwnershipMismatch.__init__": "exception constructors",
    "app/mcp_client/gateway.py:MCPUnavailable.__init__": "exception constructors",
    "app/mcp_client/gateway.py:MCPPrimitiveError.__init__": "exception constructors",
    "app/mcp_client/gateway.py:MCPGateway._call_tool": "abstract stub",
    "app/mcp_client/gateway.py:MCPGateway._read_resource": "abstract stub",
    "app/mcp_client/gateway.py:MCPGateway._get_prompt": "abstract stub",
    "app/mcp_client/in_process.py:InProcessMCPGateway._database": "generator context manager (env swap)",
    "app/agents/escalation_agent.py:EscalationUnavailable.__init__": "exception constructors",
    "app/agents/_shared.py:AgentStepError.__init__": "exception constructors",
    "app/intake/gate.py:IntakeRejected.__init__": "exception constructors",
    "mcp_servers/common.py:connection": "generator context manager; get_scoped_connection inside is traced",
    "mcp_servers/common.py:with_request_context": "decorator factory applied outside @trace so ENTER lines carry the request ID",
    "app/models/*": "Pydantic contract classes: validators/properties run inside traced callers",
    "app/agents/llm.py:LLMBackend.structured": "Protocol stub (no body)",
}


def _decorated_with_trace(node) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "trace":
            return True
    return False


def untraced_functions() -> list[str]:
    missing = []
    for path in project_sources():
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(body, prefix):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, f"{prefix}{node.name}.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{rel}:{prefix}{node.name}"
                    exempt = name in EXEMPT or f"{rel}:*" in EXEMPT or any(
                        rel.startswith(k[:-2]) for k in EXEMPT if k.endswith("/*"))
                    if not exempt and not _decorated_with_trace(node):
                        missing.append(name)

        visit(tree.body, "")
    return missing


def test_every_project_function_is_traced():
    assert untraced_functions() == []


def test_exemptions_refer_to_real_functions():
    names = set()
    for path in project_sources():
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                names |= {f"{rel}:{node.name}.{n.name}" for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(f"{rel}:{node.name}")
    stale = [k for k in EXEMPT if not k.endswith("*") and k not in names]
    assert stale == []
