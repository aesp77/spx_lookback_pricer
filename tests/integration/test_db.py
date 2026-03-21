"""Integration tests for database operations."""
import sqlite3


def test_standard_schema_roundtrip(tmp_db):
    """Insert and read back from standard schema tables."""
    conn = sqlite3.connect(tmp_db)

    # Read back spot price
    row = conn.execute(
        "SELECT value FROM time_series WHERE symbol = ? AND field = ?",
        ("SPX", "spot"),
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 6735.11) < 0.01

    # Read back SSVI params
    row = conn.execute(
        "SELECT theta, rho, beta FROM ssvi_parameters WHERE symbol = ?",
        ("SPX",),
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 0.0263) < 0.001

    conn.close()


def test_data_loader_with_standard_schema(tmp_db):
    """SPXDataLoader reads from standard schema tables."""
    from spx_lookback_pricer.data.market_data import SPXDataLoader, DatabaseConfig

    config = DatabaseConfig(db_type="sqlite", db_path=str(tmp_db))
    loader = SPXDataLoader(config)
    data = loader.get_latest_data()

    assert data["spot"]["price"] is not None
    assert abs(data["spot"]["price"] - 6735.11) < 0.01
    assert data["ssvi"]["params"] is not None
