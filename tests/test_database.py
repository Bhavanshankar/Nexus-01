"""database.py seeding, fixture consistency, and structural SQLite isolation (app/db_access.py)."""

import sqlite3
from pathlib import Path

import pytest

import database
from app import db_access
from app.guardrails import is_valid_identifier
from mcp_servers.common import connection

TABLES = ["customers", "accounts", "transactions", "policies", "claims", "portfolios", "investment_products",
          "interaction_log", "escalation_flags", "audit_log"]


def _rows(path, sql, params=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_seed_creates_every_table_with_25_rows_in_wal_mode(tmp_path):
    path = tmp_path / "seeded.db"
    database.seed_all(path)
    for table in TABLES:
        assert _rows(path, f"SELECT COUNT(*) FROM {table}")[0][0] == 25, table
    assert _rows(path, "PRAGMA journal_mode")[0][0] == "wal"


def test_seed_backs_up_existing_database(tmp_path, monkeypatch, capsys):
    path = tmp_path / "meridian.db"
    database.seed_all(path)
    monkeypatch.setattr("sys.argv", ["database.py", "--seed", "--db-path", str(path)])
    database.main()
    backups = list((tmp_path / "backups").glob("meridian-*.db"))
    assert len(backups) == 1 and "Backed up existing database" in capsys.readouterr().out


def test_showcase_and_conflict_fixtures_match_the_assignment(db_path):
    assert _rows(db_path, "SELECT customer_id, account_type, status, auto_debit_allowed FROM accounts WHERE account_id='ACC-20077'") == [
        ("CUS-20077", "savings", "active", 1)]
    assert _rows(db_path, "SELECT customer_id, policy_type, status, auto_debit_permitted FROM policies WHERE policy_id='POL-IN-30091'") == [
        ("CUS-20077", "general", "active", 1)]
    assert "savings" in _rows(db_path, "SELECT summary FROM interaction_log WHERE request_id='REQ-ARJ-00001'")[0][0]
    assert _rows(db_path, "SELECT status, auto_debit_allowed FROM accounts WHERE account_id='ACC-30142'") == [("restricted", 0)]
    assert _rows(db_path, "SELECT auto_debit_permitted FROM policies WHERE policy_id='POL-GE-30250'") == [(1,)]


@pytest.mark.parametrize("table,column,kind", [
    ("customers", "customer_id", "customer_id"), ("accounts", "account_id", "account_id"), ("policies", "policy_id", "policy_id"),
    ("portfolios", "portfolio_id", "portfolio_id"), ("investment_products", "product_id", "product_id"), ("claims", "claim_id", "claim_id"),
])
def test_all_seeded_identifiers_use_the_exact_formats(db_path, table, column, kind):
    values = [r[0] for r in _rows(db_path, f"SELECT {column} FROM {table}")]
    assert values and all(is_valid_identifier(kind, v) for v in values)


def test_database_path_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("NEXUS_DB_PATH", raising=False)
    assert db_access.database_path() == db_access.MERIDIAN_DB
    monkeypatch.setenv("NEXUS_DB_PATH", str(tmp_path / "x.db"))
    assert db_access.database_path() == tmp_path / "x.db"
    assert db_access.database_path(tmp_path / "y.db") == tmp_path / "y.db"


# ------------------------------------------------------------------------------ isolation

FOREIGN_READS = {
    "banking": ["customers", "policies", "claims", "portfolios", "investment_products", "audit_log"],
    "insurance": ["customers", "accounts", "transactions", "portfolios", "investment_products"],
    "wealth": ["customers", "accounts", "transactions", "policies", "claims"],
    "concierge_ops": ["accounts", "transactions", "policies", "claims", "portfolios", "investment_products"],
}


@pytest.mark.parametrize("server", list(FOREIGN_READS))
def test_each_server_is_structurally_unable_to_read_foreign_tables(db_path, server):
    scope = db_access.DOMAIN_SCOPES[server]
    conn = db_access.get_scoped_connection(scope["read_tables"], scope["write_tables"], db_path)
    try:
        for table in FOREIGN_READS[server]:
            with pytest.raises(sqlite3.DatabaseError, match="prohibited|not authorized"):
                conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchall()
        for table in scope["read_tables"]:
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    finally:
        conn.close()


def test_joins_through_foreign_tables_are_denied(db_path):
    scope = db_access.DOMAIN_SCOPES["insurance"]
    conn = db_access.get_scoped_connection(scope["read_tables"], scope["write_tables"], db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT p.policy_id, a.balance_cents FROM policies p JOIN accounts a ON a.customer_id = p.customer_id").fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("server", ["banking", "insurance", "wealth"])
def test_read_only_servers_cannot_write(db_path, server):
    scope = db_access.DOMAIN_SCOPES[server]
    conn = db_access.get_scoped_connection(scope["read_tables"], scope["write_tables"], db_path)
    try:
        table = sorted(scope["read_tables"])[0]
        for statement in (f"DELETE FROM {table}", "INSERT INTO audit_log (timestamp, action_type, performed_by, outcome, detail) VALUES ('t','a','b','c','d')"):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(statement)
    finally:
        conn.close()


def test_concierge_writes_only_its_write_tables_and_never_customers(db_path):
    scope = db_access.DOMAIN_SCOPES["concierge_ops"]
    assert scope["write_tables"] == {"interaction_log", "audit_log"}
    conn = db_access.get_scoped_connection(scope["read_tables"], scope["write_tables"], db_path)
    try:
        conn.execute("INSERT INTO audit_log (timestamp, action_type, performed_by, outcome, detail) VALUES ('t','a','b','c','d')")
        for statement in ("UPDATE customers SET kyc_status='verified'", "DELETE FROM escalation_flags", "DROP TABLE audit_log",
                          "CREATE TABLE x (a)", "ATTACH DATABASE 'other.db' AS other", "PRAGMA query_only = OFF"):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(statement)
    finally:
        conn.close()


def test_server_connection_helper_refuses_scope_beyond_ownership(db_path):
    with pytest.raises(PermissionError):
        with connection("insurance", read_tables={"accounts"}):
            pass
    with pytest.raises(PermissionError):
        with connection("banking", read_tables={"accounts"}, write_tables={"accounts"}):
            pass
    with pytest.raises(ValueError):
        with connection("unknown", read_tables={"accounts"}):
            pass


def test_missing_database_is_a_safe_failure(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        db_access.get_scoped_connection({"accounts"}, set(), tmp_path / "missing.db")
    assert str(tmp_path) not in str(exc.value)
