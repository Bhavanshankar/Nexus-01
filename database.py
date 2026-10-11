"""Build and seed the Meridian SQLite database.

    python database.py --seed                      # (re)create data/meridian.db, backing up any existing file
    python database.py --seed --db-path PATH       # seed a different file (tests use temporary paths)
    python database.py --seed --no-backup          # skip the backup copy

Seeding replaces the target file's contents, so an existing database is first copied to
``data/backups/`` unless ``--no-backup`` is given.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.db_access import MERIDIAN_DB, database_path, initialize_database
from app.logging_config import get_logger, trace
from db.seed_data import (
    CUSTOMERS, ACCOUNTS, TRANSACTIONS, POLICIES, CLAIMS,
    PORTFOLIOS, INVESTMENT_PRODUCTS, INTERACTIONS, ESCALATIONS, AUDITS,
)

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent

TABLE_COLUMNS = [
    ("customers", ["customer_id", "full_name", "email", "phone", "customer_tier", "kyc_status", "kyc_last_checked_at", "created_at"], CUSTOMERS),
    ("accounts", ["account_id", "customer_id", "account_type", "balance_cents", "currency", "status", "auto_debit_allowed", "opened_at"], ACCOUNTS),
    ("transactions", ["transaction_id", "account_id", "amount_cents", "direction", "description", "occurred_at"], TRANSACTIONS),
    ("policies", ["policy_id", "customer_id", "policy_type", "premium_cents", "payment_frequency", "payment_terms", "auto_debit_permitted", "status", "coverage_clauses_json", "started_at"], POLICIES),
    ("claims", ["claim_id", "policy_id", "status", "filed_at", "last_updated_at"], CLAIMS),
    ("portfolios", ["portfolio_id", "customer_id", "risk_profile", "total_value_cents", "status", "opened_at"], PORTFOLIOS),
    ("investment_products", ["product_id", "name", "product_type", "min_risk_profile", "description"], INVESTMENT_PRODUCTS),
    ("interaction_log", ["customer_id", "request_id", "channel", "summary", "created_at"], INTERACTIONS),
    ("escalation_flags", ["customer_id", "request_id", "escalation_category", "draft_confidence", "reason", "resolved", "created_at"], ESCALATIONS),
    ("audit_log", ["timestamp", "action_type", "performed_by", "customer_id", "outcome", "detail"], AUDITS),
]


@trace(logger)
def _schema():
    return (BASE_DIR / "db" / "schema.sql").read_text(encoding="utf-8")


@trace(logger)
def backup_database(db_path) -> Path | None:
    """Copy an existing database (and WAL sidecars) to data/backups/ before it is replaced."""
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = backup_dir / f"{db_path.stem}-{stamp}{db_path.suffix}"
    shutil.copy2(db_path, target)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, Path(str(target) + suffix))
    return target


@trace(logger)
def seed_all(db_path=None):
    initialize_database(database_path(db_path), _schema(), TABLE_COLUMNS)


@trace(logger)
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create and seed the Meridian SQLite database.")
    parser.add_argument("--seed", action="store_true", help="(re)create and seed the database")
    parser.add_argument("--db-path", default=None, help=f"database file (default: NEXUS_DB_PATH or {MERIDIAN_DB})")
    parser.add_argument("--no-backup", action="store_true", help="do not back up an existing database first")
    args = parser.parse_args()

    if args.seed:
        db_path = database_path(args.db_path)
        if not args.no_backup:
            backup = backup_database(db_path)
            if backup:
                print(f"Backed up existing database to: {backup}")
        seed_all(db_path)
        print(f"Seeded database: {db_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
