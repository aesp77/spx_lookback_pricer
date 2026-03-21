"""Smoke tests for lookback instruments."""
import numpy as np


def test_percentage_lookback_put_runs():
    """PercentageLookbackPut should run without crashing."""
    from spx_lookback_pricer.instruments.lookback import PercentageLookbackPut

    option = PercentageLookbackPut(
        expiry=1.0,
        protection_level=0.90,
    )
    result = option.monte_carlo_price(
        spot=5700, vol=0.18, rate=0.045, div_yield=0.013,
        n_paths=10000, notional=10_000_000,
    )
    assert "price" in result
    assert result["price"] >= 0


def test_percentage_lookback_call_runs():
    """PercentageLookbackCall should run without crashing."""
    from spx_lookback_pricer.instruments.lookback import PercentageLookbackCall

    option = PercentageLookbackCall(
        expiry=1.0,
        participation_level=1.10,
    )
    result = option.monte_carlo_price(
        spot=5700, vol=0.18, rate=0.045, div_yield=0.013,
        n_paths=10000, notional=10_000_000,
    )
    assert "price" in result
    assert result["price"] >= 0


def test_lookback_put_price_finite():
    """Lookback prices should be finite (no NaN/Inf)."""
    from spx_lookback_pricer.instruments.lookback import PercentageLookbackPut

    option = PercentageLookbackPut(
        expiry=0.5,
        protection_level=0.95,
    )
    result = option.monte_carlo_price(
        spot=5700, vol=0.18, rate=0.045, div_yield=0.013,
        n_paths=10000, notional=10_000_000,
    )
    assert np.isfinite(result["price"])


def test_ratcheting_lookback_put_runs():
    """RatchetingLookbackPut should run without crashing."""
    from spx_lookback_pricer.instruments.lookback import RatchetingLookbackPut

    option = RatchetingLookbackPut(
        protection_level=0.95,
        expiry=1.0,
    )
    result = option.monte_carlo_price(
        spot=5700, vol=0.18, rate=0.045, div_yield=0.013,
        n_paths=10000,
    )
    assert "price" in result
    assert result["price"] >= 0
