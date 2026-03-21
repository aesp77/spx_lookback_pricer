#!/usr/bin/env python3
"""Update market data for SPX Lookback Pricer.

Usage:
    poetry run python spx_lookback_pricer/scripts/update_data.py status
    poetry run python spx_lookback_pricer/scripts/update_data.py update
    poetry run python spx_lookback_pricer/scripts/update_data.py copy <source>
    poetry run python spx_lookback_pricer/scripts/update_data.py init <source>
"""
import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "spx_lookback_pricer.db"
SYMBOLS = ["SPX"]
DATASETS = ["spot", "vol_surfaces", "term_structures"]

# Standard market-data schema DDL
MARKET_DDL = """
CREATE TABLE IF NOT EXISTS time_series (
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,
    field       TEXT NOT NULL,
    value       REAL NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (symbol, date, field)
);
CREATE TABLE IF NOT EXISTS term_structures (
    curve_id    TEXT NOT NULL,
    date        TEXT NOT NULL,
    tenor       TEXT NOT NULL,
    value       REAL NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (curve_id, date, tenor)
);
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
CREATE TABLE IF NOT EXISTS fetch_log (
    source      TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    last_date   TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (source, dataset, symbol)
);
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
CREATE INDEX IF NOT EXISTS idx_ts_symbol_date ON time_series(symbol, date);
CREATE INDEX IF NOT EXISTS idx_vol_symbol_date ON vol_surfaces(symbol, date);
CREATE INDEX IF NOT EXISTS idx_opt_symbol_date ON option_chains(symbol, date);
CREATE INDEX IF NOT EXISTS idx_term_curve_date ON term_structures(curve_id, date);
CREATE INDEX IF NOT EXISTS idx_intraday_symbol ON intraday(symbol, datetime);
"""


def get_project_db(db_path: Path, extra_ddl: str = "") -> sqlite3.Connection:
    """Open (or create) a project database with the standard schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(MARKET_DDL)
    if extra_ddl:
        conn.executescript(extra_ddl)
    conn.commit()
    return conn


def check_data_coverage(db_path: Path, symbol: str | None = None) -> dict:
    """Show data coverage summary."""
    conn = get_project_db(db_path)
    coverage = {}

    tables_with_symbol = {
        "time_series": "date",
        "vol_surfaces": "date",
        "option_chains": "date",
        "intraday": "datetime",
    }
    for table, date_col in tables_with_symbol.items():
        if symbol:
            row = conn.execute(
                f"SELECT MIN({date_col}), MAX({date_col}), COUNT(*) FROM {table} WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT MIN({date_col}), MAX({date_col}), COUNT(*) FROM {table}",
            ).fetchone()
        coverage[table] = {"from": row[0], "to": row[1], "rows": row[2]}

    row = conn.execute(
        "SELECT MIN(date), MAX(date), COUNT(*) FROM term_structures"
    ).fetchone()
    coverage["term_structures"] = {"from": row[0], "to": row[1], "rows": row[2]}

    # SSVI parameters (project-specific)
    row = conn.execute(
        "SELECT MIN(date), MAX(date), COUNT(*) FROM ssvi_parameters"
    ).fetchone()
    coverage["ssvi_parameters"] = {"from": row[0], "to": row[1], "rows": row[2]}

    # Fetch log
    fetch_rows = conn.execute("SELECT source, dataset, symbol, last_date FROM fetch_log").fetchall()

    conn.close()

    label = f" for {symbol}" if symbol else ""
    print(f"\nData coverage{label} ({db_path.name}):")
    for table, info in coverage.items():
        if info["from"]:
            print(f"  {table:20s}: {info['from']} to {info['to']} ({info['rows']} rows)")
        else:
            print(f"  {table:20s}: empty")

    if fetch_rows:
        print(f"\nFetch log:")
        for src, ds, sym, last in fetch_rows:
            print(f"  {src}/{ds}/{sym}: last fetched {last}")

    return coverage


def copy_tables(
    source_db: Path,
    target_db: Path,
    tables: list[str] | None = None,
    symbols: list[str] | None = None,
):
    """Copy market data tables from one project's DB to another."""
    all_tables = [
        "time_series", "term_structures", "vol_surfaces",
        "option_chains", "intraday", "fetch_log", "ssvi_parameters",
    ]
    tables = tables or all_tables

    src = sqlite3.connect(source_db)
    tgt = get_project_db(target_db)

    for table in tables:
        exists = src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            print(f"  {table}: not in source, skipping")
            continue

        query = f"SELECT * FROM {table} WHERE 1=1"
        params: list = []

        if symbols and table not in ("term_structures", "fetch_log"):
            placeholders = ",".join("?" * len(symbols))
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)

        rows = src.execute(query, params).fetchall()
        if not rows:
            print(f"  {table}: no matching data")
            continue

        cols = [desc[0] for desc in src.execute(f"SELECT * FROM {table} LIMIT 1").description]
        placeholders = ",".join("?" * len(cols))

        tgt.executemany(
            f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})",
            rows,
        )
        tgt.commit()
        print(f"  {table}: copied {len(rows)} rows")

    src.close()
    tgt.close()
    print(f"\nDone. Copied from {source_db.name} -> {target_db.name}")


def cmd_status(args):
    check_data_coverage(DB_PATH, symbol="SPX")


def cmd_update(args):
    """Incremental update — delegates to existing update_database.py logic."""
    # Import the existing async updater
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from spx_lookback_pricer.data.market_data import SPXDataLoader, DatabaseConfig
    import asyncio

    config = DatabaseConfig(db_type="sqlite", db_path=str(DB_PATH))
    loader = SPXDataLoader(config)

    async def _update():
        stats = await loader.update_recent_data(days_back=args.days)
        return stats

    try:
        stats = asyncio.run(_update())
        print(f"\nUpdated: {stats}")
    except Exception as e:
        print(f"Update failed: {e}")
        print("(PSC modules may not be available — run from work environment)")

    check_data_coverage(DB_PATH, symbol="SPX")


def cmd_copy(args):
    source_db = Path(args.source).expanduser().resolve()
    if not source_db.exists():
        print(f"Source DB not found: {source_db}")
        sys.exit(1)
    copy_tables(source_db, DB_PATH, symbols=SYMBOLS)
    check_data_coverage(DB_PATH, symbol="SPX")


def cmd_init(args):
    """Copy from another project, then update to fill gaps."""
    cmd_copy(args)
    cmd_update(args)


def main():
    parser = argparse.ArgumentParser(description="Update market data for SPX Lookback Pricer")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show data coverage")

    p_update = sub.add_parser("update", help="Fetch new data incrementally")
    p_update.add_argument("--days", type=int, default=5, help="Days to look back (default: 5)")

    p_copy = sub.add_parser("copy", help="Copy data from another project's DB")
    p_copy.add_argument("source", help="Path to source database")

    p_init = sub.add_parser("init", help="Copy from source DB then update")
    p_init.add_argument("source", help="Path to source database")
    p_init.add_argument("--days", type=int, default=5, help="Days to look back")

    args = parser.parse_args()
    {"status": cmd_status, "update": cmd_update, "copy": cmd_copy, "init": cmd_init}[args.command](args)


if __name__ == "__main__":
    main()
