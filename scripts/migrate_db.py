#!/usr/bin/env python3
"""Migrate legacy DB tables to standard market-data skill schema.

One-time script: reads from old tables, inserts into standard schema tables,
then verifies the migration.
"""
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "spx_lookback_pricer.db"

MARKET_DDL = """
-- Daily time series (spot, close, div yield, volume, etc.)
CREATE TABLE IF NOT EXISTS time_series (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    field       TEXT NOT NULL,
    value       REAL NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, date, field)
);

-- Term structures (yield curves, forward curves, VIX futures, etc.)
CREATE TABLE IF NOT EXISTS term_structures (
    curve_id    TEXT NOT NULL,
    date        TEXT NOT NULL,
    tenor       TEXT NOT NULL,
    value       REAL NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (curve_id, date, tenor)
);

-- Vol surfaces (strike x expiry grid per symbol per date)
CREATE TABLE IF NOT EXISTS vol_surfaces (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    expiry      TEXT NOT NULL,
    strike      REAL NOT NULL,
    iv          REAL NOT NULL,
    delta       REAL,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, date, expiry, strike)
);

-- Option chains (individual contracts)
CREATE TABLE IF NOT EXISTS option_chains (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    expiry      TEXT NOT NULL,
    strike      REAL NOT NULL,
    option_type TEXT NOT NULL,
    bid         REAL,
    ask         REAL,
    mid         REAL,
    volume      INTEGER,
    open_interest INTEGER,
    iv          REAL,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, date, expiry, strike, option_type)
);

-- Intraday data
CREATE TABLE IF NOT EXISTS intraday (
    symbol      TEXT NOT NULL,
    datetime    TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, datetime)
);

-- Tracks what has been downloaded to avoid re-fetching
CREATE TABLE IF NOT EXISTS fetch_log (
    source      TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    last_date   TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (source, dataset, symbol)
);

CREATE INDEX IF NOT EXISTS idx_ts_symbol_date ON time_series(symbol, date);
CREATE INDEX IF NOT EXISTS idx_vol_symbol_date ON vol_surfaces(symbol, date);
CREATE INDEX IF NOT EXISTS idx_opt_symbol_date ON option_chains(symbol, date);
CREATE INDEX IF NOT EXISTS idx_term_curve_date ON term_structures(curve_id, date);
CREATE INDEX IF NOT EXISTS idx_intraday_symbol ON intraday(symbol, datetime);
"""

# Project-specific tables for SSVI parameters (extends standard schema)
SPX_EXTRA_DDL = """
CREATE TABLE IF NOT EXISTS ssvi_parameters (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    theta       REAL NOT NULL,
    rho         REAL NOT NULL,
    beta        REAL NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, date)
);
"""


def migrate(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)

    # Step 1: Create standard schema
    conn.executescript(MARKET_DDL)
    conn.executescript(SPX_EXTRA_DDL)
    conn.commit()
    print("Created standard schema tables.")

    # Step 2: Migrate spx_spot_prices -> time_series
    rows = conn.execute("SELECT date, last, daily_return FROM spx_spot_prices").fetchall()
    for date, last, daily_return in rows:
        conn.execute(
            "INSERT OR IGNORE INTO time_series (symbol, date, field, value, source) VALUES (?, ?, ?, ?, ?)",
            ("SPX", date, "spot", last, "psc"),
        )
        if daily_return is not None:
            conn.execute(
                "INSERT OR IGNORE INTO time_series (symbol, date, field, value, source) VALUES (?, ?, ?, ?, ?)",
                ("SPX", date, "daily_return", daily_return, "psc"),
            )
    conn.commit()
    print(f"  Migrated {len(rows)} spot price rows -> time_series")

    # Step 3: Migrate spx_dividend_yield -> time_series
    rows = conn.execute("SELECT date, dividend_yield FROM spx_dividend_yield").fetchall()
    for date, div_yield in rows:
        conn.execute(
            "INSERT OR IGNORE INTO time_series (symbol, date, field, value, source) VALUES (?, ?, ?, ?, ?)",
            ("SPX", date, "dividend_yield", div_yield, "psc"),
        )
    conn.commit()
    print(f"  Migrated {len(rows)} dividend yield rows -> time_series")

    # Step 4: Migrate ois_curve -> term_structures
    rows = conn.execute("SELECT date, tenor_years, rate, currency FROM ois_curve").fetchall()
    for date, tenor_years, rate, currency in rows:
        curve_id = f"{currency}_OIS"
        tenor_str = f"{tenor_years}y"
        conn.execute(
            "INSERT OR IGNORE INTO term_structures (curve_id, date, tenor, value, source) VALUES (?, ?, ?, ?, ?)",
            (curve_id, date, tenor_str, rate, "psc"),
        )
    conn.commit()
    print(f"  Migrated {len(rows)} OIS curve rows -> term_structures")

    # Step 5: Migrate spx_vol_surface -> vol_surfaces
    rows = conn.execute("SELECT date, strike, tenor, implied_vol FROM spx_vol_surface").fetchall()
    for date, strike, tenor, iv in rows:
        conn.execute(
            "INSERT OR IGNORE INTO vol_surfaces (symbol, date, expiry, strike, iv, source) VALUES (?, ?, ?, ?, ?, ?)",
            ("SPX", date, tenor, strike, iv, "psc"),
        )
    conn.commit()
    print(f"  Migrated {len(rows)} vol surface rows -> vol_surfaces")

    # Step 6: Migrate spx_ssvi_parameters -> ssvi_parameters
    rows = conn.execute("SELECT date, theta, rho, beta FROM spx_ssvi_parameters").fetchall()
    for date, theta, rho, beta in rows:
        conn.execute(
            "INSERT OR IGNORE INTO ssvi_parameters (symbol, date, theta, rho, beta, source) VALUES (?, ?, ?, ?, ?, ?)",
            ("SPX", date, theta, rho, beta, "psc"),
        )
    conn.commit()
    print(f"  Migrated {len(rows)} SSVI parameter rows -> ssvi_parameters")

    # Step 7: Update fetch_log
    # Find date ranges for each dataset
    for dataset, query in [
        ("spot", "SELECT MAX(date) FROM time_series WHERE symbol='SPX' AND field='spot'"),
        ("vol_surfaces", "SELECT MAX(date) FROM vol_surfaces WHERE symbol='SPX'"),
        ("term_structures", "SELECT MAX(date) FROM term_structures WHERE curve_id='USD_OIS'"),
    ]:
        last_date = conn.execute(query).fetchone()[0]
        if last_date:
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log (source, dataset, symbol, last_date) VALUES (?, ?, ?, ?)",
                ("psc", dataset, "SPX", last_date),
            )
    conn.commit()
    print("  Updated fetch_log")

    # Verify
    print("\nVerification:")
    for table in ["time_series", "term_structures", "vol_surfaces", "ssvi_parameters", "fetch_log"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    migrate()
