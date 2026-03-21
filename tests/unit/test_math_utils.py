"""Smoke tests for math_utils — BS pricing and Greeks."""
import numpy as np


def test_bs_call_put_parity():
    """Call - Put = S*exp(-qT) - K*exp(-rT)."""
    from spx_lookback_pricer.utils.math_utils import black_scholes_price

    S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0
    call = black_scholes_price(S, K, T, r, q, sigma, is_call=True)
    put = black_scholes_price(S, K, T, r, q, sigma, is_call=False)
    parity = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(call - put - parity) < 1e-8


def test_bs_price_positive():
    """BS price should be positive for reasonable inputs."""
    from spx_lookback_pricer.utils.math_utils import black_scholes_price

    price = black_scholes_price(5700, 5700, 0.5, 0.045, 0.013, 0.18, is_call=True)
    assert price > 0


def test_bs_delta_bounded():
    """Call delta must be in [0, 1]."""
    from spx_lookback_pricer.utils.math_utils import black_scholes_delta

    delta = black_scholes_delta(5700, 5700, 0.5, 0.045, 0.013, 0.18, is_call=True)
    assert 0 <= delta <= 1


def test_bs_gamma_positive():
    """Gamma should always be positive."""
    from spx_lookback_pricer.utils.math_utils import black_scholes_gamma

    gamma = black_scholes_gamma(5700, 5700, 0.5, 0.045, 0.013, 0.18)
    assert gamma > 0


def test_normal_cdf():
    """Normal CDF should return ~0.5 at 0."""
    from spx_lookback_pricer.utils.math_utils import normal_cdf

    assert abs(normal_cdf(0) - 0.5) < 1e-10
