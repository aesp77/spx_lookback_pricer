import os

os.environ["KERAS_BACKEND"] = "torch"

import pytest
import sqlite3
from pathlib import Path


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database with standard schema.
    Automatically cleaned up after the test."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    # Standard market-data schema
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS time_series (
            symbol TEXT NOT NULL, date TEXT NOT NULL, field TEXT NOT NULL,
            value REAL NOT NULL, source TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (symbol, date, field)
        );
        CREATE TABLE IF NOT EXISTS vol_surfaces (
            symbol TEXT NOT NULL, date TEXT NOT NULL, expiry TEXT NOT NULL,
            strike REAL NOT NULL, iv REAL NOT NULL, delta REAL,
            source TEXT NOT NULL, updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (symbol, date, expiry, strike)
        );
        CREATE TABLE IF NOT EXISTS term_structures (
            curve_id TEXT NOT NULL, date TEXT NOT NULL, tenor TEXT NOT NULL,
            value REAL NOT NULL, source TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (curve_id, date, tenor)
        );
        CREATE TABLE IF NOT EXISTS ssvi_parameters (
            symbol TEXT NOT NULL, date TEXT NOT NULL,
            theta REAL NOT NULL, rho REAL NOT NULL, beta REAL NOT NULL,
            source TEXT NOT NULL, updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (symbol, date)
        );
        CREATE TABLE IF NOT EXISTS fetch_log (
            source TEXT NOT NULL, dataset TEXT NOT NULL, symbol TEXT NOT NULL,
            last_date TEXT NOT NULL, updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (source, dataset, symbol)
        );
    """)

    # Insert sample data for testing
    conn.execute(
        "INSERT INTO time_series VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("SPX", "2025-10-09", "spot", 6735.11, "test"),
    )
    conn.execute(
        "INSERT INTO time_series VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("SPX", "2025-10-09", "dividend_yield", 0.01234, "test"),
    )
    conn.execute(
        "INSERT INTO ssvi_parameters VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        ("SPX", "2025-10-09", 0.0263, -0.597, 32.43, "test"),
    )
    conn.commit()
    yield db_path
    conn.close()


@pytest.fixture
def sample_spot():
    return 5700.0


@pytest.fixture
def sample_vol():
    return 0.18


@pytest.fixture
def sample_rate():
    return 0.045


@pytest.fixture
def sample_div_yield():
    return 0.013
