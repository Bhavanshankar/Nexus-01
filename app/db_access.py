"""Least-privilege SQLite access for the Nexus MCP servers.

Only the MCP server processes use this module. Isolation is structural: every connection gets a
SQLite authorizer that denies reads of tables outside the server's scope, denies all writes except
to its own write tables, and denies schema changes, ATTACH and PRAGMA. Read-only servers also run
with ``PRAGMA query_only = ON``. This holds whether the four servers share one SQLite file (the
default) or point at separate files via ``NEXUS_DB_PATH``.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from app.logging_config import get_logger, trace

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MERIDIAN_DB = DATA_DIR / "meridian.db"

SAFE_FUNCTIONS = {
    "lower", "upper", "length", "substr", "coalesce",
    "abs", "round", "min", "max", "count",
}

DENIED_ACTIONS = {
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ANALYZE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_PRAGMA,
}

# Table ownership follows PDF section 3 exactly. Only concierge_ops may read customers (PII), and
# only concierge_ops may write; it writes interaction_log and audit_log via the two write tools.
DOMAIN_SCOPES: dict[str, dict[str, frozenset[str]]] = {
    "banking": {
        "read_tables": frozenset({"accounts", "transactions"}),
        "write_tables": frozenset(),
    },
    "insurance": {
        "read_tables": frozenset({"policies", "claims"}),
        "write_tables": frozenset(),
    },
    "wealth": {
        "read_tables": frozenset({"portfolios", "investment_products"}),
        "write_tables": frozenset(),
    },
    "concierge_ops": {
        "read_tables": frozenset({"customers", "interaction_log", "escalation_flags", "audit_log"}),
        "write_tables": frozenset({"interaction_log", "audit_log"}),
    },
}


@trace(logger)
def database_path(db_path: str | Path | None = None) -> Path:
    """Resolve the database file: explicit argument, then ``NEXUS_DB_PATH``, then data/meridian.db."""
    if db_path:
        return Path(db_path)
    override = os.getenv("NEXUS_DB_PATH")
    return Path(override) if override else MERIDIAN_DB


def _redact_initialize_arguments(arguments):
    # Not traced: a redactor's own ENTER line would log the very data it exists to hide.
    return {
        "db_path": arguments.get("db_path"),
        "schema_sql": "<schema omitted>",
        "seed_rows": "<seed data omitted>",
    }


@trace(logger)
def _authorizer_factory(read_tables, write_tables):
    read_tables = set(read_tables)
    write_tables = set(write_tables)

    # Not traced: SQLite invokes this for every column access while preparing a statement.
    def authorizer(action, arg1, arg2, db_name, source):

        if action in DENIED_ACTIONS:
            return sqlite3.SQLITE_DENY

        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK

        if action == sqlite3.SQLITE_READ:
            return (
                sqlite3.SQLITE_OK
                if arg1 in read_tables or arg1 in write_tables
                else sqlite3.SQLITE_DENY
            )

        if action in (
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
        ):
            return (
                sqlite3.SQLITE_OK
                if arg1 in write_tables
                else sqlite3.SQLITE_DENY
            )

        if action == sqlite3.SQLITE_FUNCTION:
            return (
                sqlite3.SQLITE_OK
                if (arg2 or "").lower() in SAFE_FUNCTIONS
                else sqlite3.SQLITE_DENY
            )

        if action == sqlite3.SQLITE_TRANSACTION:
            return sqlite3.SQLITE_OK

        return sqlite3.SQLITE_DENY

    return authorizer


@trace(logger)
def get_scoped_connection(read_tables, write_tables, db_path=None):
    if not read_tables and not write_tables:
        raise ValueError("At least one read or write table is required")

    path = database_path(db_path)

    if not path.exists():
        raise FileNotFoundError("Database file is not available")

    conn = sqlite3.connect(path, timeout=5, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute(f"PRAGMA query_only = {'OFF' if write_tables else 'ON'}")

    conn.set_authorizer(
        _authorizer_factory(
            read_tables=read_tables,
            write_tables=write_tables,
        )
    )

    return conn


@trace(logger)
def get_domain_scope(domain):
    if domain not in DOMAIN_SCOPES:
        raise ValueError(f"Unknown domain: {domain}")
    return DOMAIN_SCOPES[domain]


@trace(logger)
def get_domain_connection(domain, db_path=None):
    scope = get_domain_scope(domain)
    return get_scoped_connection(
        read_tables=scope["read_tables"],
        write_tables=scope["write_tables"],
        db_path=db_path,
    )


@trace(logger, redact=_redact_initialize_arguments)
def initialize_database(db_path, schema_sql, seed_rows):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(schema_sql)

        for table, columns, rows in seed_rows:
            placeholders = ",".join("?" for _ in columns)
            conn.executemany(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                rows,
            )
        conn.commit()
    finally:
        conn.close()
