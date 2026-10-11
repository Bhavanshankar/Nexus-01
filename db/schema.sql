PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT NOT NULL,
    customer_tier TEXT NOT NULL CHECK(customer_tier IN ('standard','premium','private')),
    kyc_status TEXT NOT NULL CHECK(kyc_status IN ('verified','pending','review')),
    kyc_last_checked_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    account_type TEXT NOT NULL CHECK(account_type IN ('checking','savings','current','salary')),
    balance_cents INTEGER NOT NULL CHECK(balance_cents >= 0),
    currency TEXT NOT NULL DEFAULT 'USD',
    status TEXT NOT NULL CHECK(status IN ('active','restricted','dormant','closed')),
    auto_debit_allowed INTEGER NOT NULL CHECK(auto_debit_allowed IN (0,1)),
    opened_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    amount_cents INTEGER NOT NULL CHECK(amount_cents <> 0),
    direction TEXT NOT NULL CHECK(direction IN ('credit','debit')),
    description TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    policy_type TEXT NOT NULL CHECK(policy_type IN ('life','health','auto','home','general')),
    premium_cents INTEGER NOT NULL CHECK(premium_cents > 0),
    payment_frequency TEXT NOT NULL CHECK(payment_frequency IN ('monthly','quarterly','annual')),
    payment_terms TEXT NOT NULL,
    auto_debit_permitted INTEGER NOT NULL CHECK(auto_debit_permitted IN (0,1)),
    status TEXT NOT NULL CHECK(status IN ('active','lapsed','pending','cancelled')),
    coverage_clauses_json TEXT NOT NULL,
    started_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL REFERENCES policies(policy_id),
    status TEXT NOT NULL CHECK(status IN ('submitted','in_review','approved','denied','closed')),
    filed_at TEXT NOT NULL,
    last_updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    risk_profile TEXT NOT NULL CHECK(risk_profile IN ('conservative','moderate','aggressive')),
    total_value_cents INTEGER NOT NULL CHECK(total_value_cents >= 0),
    status TEXT NOT NULL CHECK(status IN ('active','frozen','closed')),
    opened_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investment_products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    product_type TEXT NOT NULL CHECK(product_type IN ('mutual_fund','etf','bond','annuity','fixed_deposit')),
    min_risk_profile TEXT NOT NULL CHECK(min_risk_profile IN ('conservative','moderate','aggressive')),
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interaction_log (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    request_id TEXT NOT NULL UNIQUE,
    channel TEXT NOT NULL CHECK(channel IN ('chat','phone','email','sms','branch')),
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalation_flags (
    escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    request_id TEXT NOT NULL,
    escalation_category TEXT NOT NULL,
    draft_confidence REAL NOT NULL CHECK(draft_confidence >= 0 AND draft_confidence <= 1),
    reason TEXT NOT NULL,
    resolved INTEGER NOT NULL CHECK(resolved IN (0,1)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action_type TEXT NOT NULL,
    performed_by TEXT NOT NULL,
    customer_id TEXT,
    outcome TEXT NOT NULL,
    detail TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_accounts_customer ON accounts(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_policies_customer ON policies(customer_id);
CREATE INDEX IF NOT EXISTS idx_claims_policy ON claims(policy_id);
CREATE INDEX IF NOT EXISTS idx_portfolios_customer ON portfolios(customer_id);
CREATE INDEX IF NOT EXISTS idx_interaction_customer ON interaction_log(customer_id);
CREATE INDEX IF NOT EXISTS idx_escalation_customer ON escalation_flags(customer_id);
CREATE INDEX IF NOT EXISTS idx_audit_customer ON audit_log(customer_id);
