# ================================
# file: lookback.py (desk-friendly payoffs)
# ================================
"""
Lookback options with desk-friendly payoff names and CRN-based Greeks.

Payoff naming (matches trading desk language)
---------------------------------------------
- floating_ratio  = Fixed Units / Floating Notional (dollar payoff)
    Put:  payoff = Notional * max(0,  α * max(S) - S_T)
    Call: payoff = Notional * max(0,  S_T - α * min(S))
    → Behaves like a vanilla put/call; Delta has usual sign (put < 0, call > 0).

- fixed_ratio     = Fixed Notional / Floating Units (scale-invariant %)
    Put:  payoff = Notional * max(0, (α * max(S) - S_T) / max(S))
    Call: payoff = Notional * max(0, (S_T - α * min(S)) / min(S))
    → Scale-invariant under GBM; elasticity (percent-delta) ~ 0.

Implementation notes
--------------------
- Exact GBM stepping; CRN (+antithetics) for stable bump Greeks.
- Keep Gamma sign (no abs). Smaller spot bump helps when elasticity ~ 0.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
from abc import ABC, abstractmethod


# -------------------------
# Utilities
# -------------------------

def _simulate_paths_gbm_crn(
    spot: float,
    vol: float,
    rate: float,
    div_yield: float,
    expiry: float,
    Z: np.ndarray,
) -> np.ndarray:
    """Simulate GBM paths using exact discretization with provided shocks Z (CRN).

    Parameters
    ----------
    Z : ndarray, shape (n_paths, n_steps)
        Standard Normal shocks reused across scenarios (CRN). Use -Z for antithetics.
    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps+1)
    """
    n_paths, n_steps = Z.shape
    dt = expiry / n_steps
    drift = (rate - div_yield - 0.5 * vol ** 2) * dt
    diffusion = vol * np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
    return paths


def _discount(rate: float, T: float, x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-rate * T) * x


# -------------------------
# Base
# -------------------------
class LookbackOption(ABC):
    """Base class for all lookback options"""

    @abstractmethod
    def monte_carlo_price(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        n_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        pass


# -------------------------
# Percentage Lookback Put (desk naming)
# -------------------------
class PercentageLookbackPut(LookbackOption):
    """
    Lookback Put with desk-friendly payoff types:

    1) floating_ratio  (Fixed Units / Floating Notional; dollar payoff)
       payoff = Notional * max(0,  α * max(S) - S_T)
       → behaves like a put (negative delta)

    2) fixed_ratio     (Fixed Notional / Floating Units; scale-invariant %)
       payoff = Notional * max(0, (α * max(S) - S_T) / max(S))
       → elasticity ~ 0 under GBM (scale-invariant)
    """

    def __init__(self, expiry: float, payoff_type: str = "floating_ratio", protection_level: float = 0.95):
        assert payoff_type in {"floating_ratio", "fixed_ratio"}
        self.expiry = expiry
        self.payoff_type = payoff_type
        self.protection_level = protection_level

    def monte_carlo_price(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        n_steps: Optional[int] = None,
        notional: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))

        paths = _simulate_paths_gbm_crn(spot, vol, rate, div_yield, self.expiry, Z)
        final = paths[:, -1]
        maxp = paths.max(axis=1)
        alpha = self.protection_level
        strike = alpha * maxp

        if self.payoff_type == "floating_ratio":
            # Dollar payoff (fixed units)
            payoffs = notional * np.maximum(0.0, strike - final)
        elif self.payoff_type == "fixed_ratio":
            # % payoff, normalized by max
            payoffs = notional * np.maximum(0.0, (strike - final) / maxp)
        else:
            raise ValueError("payoff_type must be 'floating_ratio' or 'fixed_ratio'")

        disc_payoff = _discount(rate, self.expiry, payoffs)
        price = disc_payoff.mean()
        std_error = disc_payoff.std(ddof=1) / np.sqrt(n_paths)
        return {
            "price": float(price),
            "std_error": float(std_error),
            "prob_payoff": float((payoffs > 0).mean()),
            "payoff_type": self.payoff_type,
        }

    def calculate_greeks(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        base_notional: float = 10_000_000,
        n_steps: Optional[int] = None,
        bump_pct: float = 0.005,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        n = max(n_paths, 100_000)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, n_steps))
        Zanti = -Z
        alpha = self.protection_level

        def price_given(spot_val: float, vol_val: float, rate_val: float, Zmat: np.ndarray) -> float:
            paths = _simulate_paths_gbm_crn(spot_val, vol_val, rate_val, div_yield, self.expiry, Zmat)
            fin = paths[:, -1]
            maxp = paths.max(axis=1)
            strike = alpha * maxp
            if self.payoff_type == "floating_ratio":
                payoff = np.maximum(0.0, strike - fin)
            else:  # fixed_ratio
                payoff = np.maximum(0.0, (strike - fin) / maxp)
            return float(_discount(rate_val, self.expiry, payoff).mean())

        def price_crn(spot_val: float, vol_val: float, rate_val: float) -> float:
            return 0.5 * (price_given(spot_val, vol_val, rate_val, Z) + price_given(spot_val, vol_val, rate_val, Zanti))

        P0 = price_crn(spot, vol, rate)
        Pup = price_crn(spot * (1 + bump_pct), vol, rate)
        Pdn = price_crn(spot * (1 - bump_pct), vol, rate)

        if P0 > 0:
            delta_pct = (Pup - Pdn) / (2 * bump_pct * P0) * 100.0  # elasticity in %
            gamma_pct = (Pup - 2 * P0 + Pdn) / (bump_pct ** 2 * P0)
        else:
            delta_pct = 0.0
            gamma_pct = 0.0

        # Vega (CRN)
        vol_bump = 0.01
        Pv = price_crn(spot, vol + vol_bump, rate)
        vega_per_unit = Pv - P0
        vega_position = vega_per_unit * base_notional

        # Rho (per bp)
        rate_bump = 0.0001
        Prho = price_crn(spot, vol, rate + rate_bump)
        rho_per_bp = (Prho - P0) * 100.0
        rho_position = rho_per_bp * base_notional

        # Theta (carry proxy)
        theta_daily_per_unit = -P0 * rate / 365.25
        theta_position = theta_daily_per_unit * base_notional

        return {
            "delta": float(delta_pct),
            "gamma": float(gamma_pct),
            "vega": float(vega_position),
            "theta": float(theta_position),
            "rho": float(rho_position),
            "base_price": float(P0 * base_notional),
            "base_price_pct": float(P0 * 100.0),
            "diagnostics": {"paths_used": n, "spot_bump_pct": bump_pct},
        }


# -------------------------
# Ratcheting Lookback Put (dollar payoff)
# -------------------------
class RatchetingLookbackPut(LookbackOption):
    """Strike resets to protection_level * running_max; payoff = max(0, Strike - S_T)."""

    def __init__(self, protection_level: float, expiry: float, initial_spot: Optional[float] = None):
        self.protection_level = protection_level
        self.expiry = expiry
        self.initial_spot = initial_spot
        self.current_max = initial_spot

    def monte_carlo_price(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        if self.current_max is None:
            self.current_max = spot
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))
        paths = _simulate_paths_gbm_crn(spot, vol, rate, div_yield, self.expiry, Z)
        # running max per path
        path_max = np.maximum.accumulate(paths, axis=1)[:, -1]
        strikes = self.protection_level * path_max
        final_spots = paths[:, -1]
        payoffs = np.maximum(strikes - final_spots, 0.0)
        disc = _discount(rate, self.expiry, payoffs)
        return {
            "price": float(disc.mean()),
            "std_error": float(disc.std(ddof=1) / np.sqrt(n_paths)),
            "avg_final_strike": float(strikes.mean()),
            "prob_in_money": float((payoffs > 0).mean()),
        }

    def calculate_greeks(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        base_notional: float = 1.0,
        n_steps: Optional[int] = None,
        bump_pct: float = 0.01,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        n = max(n_paths, 100_000)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, n_steps))
        Zanti = -Z

        def price_given(spot_val: float, vol_val: float, rate_val: float, Zmat: np.ndarray) -> float:
            paths = _simulate_paths_gbm_crn(spot_val, vol_val, rate_val, div_yield, self.expiry, Zmat)
            path_max = np.maximum.accumulate(paths, axis=1)[:, -1]
            strikes = self.protection_level * path_max
            final_spots = paths[:, -1]
            payoffs = np.maximum(strikes - final_spots, 0.0)
            return float(_discount(rate_val, self.expiry, payoffs).mean())

        def price_crn(spot_val: float, vol_val: float, rate_val: float) -> float:
            return 0.5 * (price_given(spot_val, vol_val, rate_val, Z) + price_given(spot_val, vol_val, rate_val, Zanti))

        P0 = price_crn(spot, vol, rate)
        bump = bump_pct * spot
        Pup = price_crn(spot + bump, vol, rate)
        Pdn = price_crn(spot - bump, vol, rate)

        delta_dollar = (Pup - Pdn) / (2 * bump)
        delta_pct = 0.0 if P0 <= 0 else delta_dollar * spot / P0 * 100.0
        gamma_dollar = (Pup - 2 * P0 + Pdn) / (bump ** 2)
        gamma_pct = 0.0 if P0 <= 0 else gamma_dollar * spot * spot / P0

        vol_bump = 0.01
        Pv = price_crn(spot, vol + vol_bump, rate)
        vega_dollar = Pv - P0

        theta_dollar = -P0 * rate / 365.25

        rate_bump = 0.0001
        Pr = price_crn(spot, vol, rate + rate_bump)
        rho_dollar_per_bp = (Pr - P0) * 100.0

        return {
            "delta": float(delta_pct),
            "gamma": float(gamma_pct),
            "vega": float(vega_dollar),
            "theta": float(theta_dollar),
            "rho": float(rho_dollar_per_bp),
            "base_price": float(P0),
            "diagnostics": {"paths_used": n, "spot_bump": float(bump)},
        }


# -------------------------
# Standard Fixed Strike Lookbacks (dollar payoff)
# -------------------------
class StandardFixedStrikeLookbackPut(LookbackOption):
    def __init__(self, strike: float, expiry: float):
        self.strike = strike
        self.expiry = expiry

    def analytical_price(self, spot: float, vol: float, rate: float, div_yield: float) -> Optional[float]:
        S, K, T, r, q, sigma = spot, self.strike, self.expiry, rate, div_yield, vol
        if T <= 0:
            return max(K - S, 0.0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1), 0.0)

    def monte_carlo_price(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))
        paths = _simulate_paths_gbm_crn(spot, vol, rate, div_yield, self.expiry, Z)
        min_prices = paths.min(axis=1)
        payoffs = np.maximum(self.strike - min_prices, 0.0)
        disc = _discount(rate, self.expiry, payoffs)
        return {
            "price": float(disc.mean()),
            "std_error": float(disc.std(ddof=1) / np.sqrt(n_paths)),
            "prob_in_money": float((payoffs > 0).mean()),
        }


class StandardFixedStrikeLookbackCall(LookbackOption):
    def __init__(self, strike: float, expiry: float):
        self.strike = strike
        self.expiry = expiry

    def analytical_price(self, spot: float, vol: float, rate: float, div_yield: float) -> Optional[float]:
        S, K, T, r, q, sigma = spot, self.strike, self.expiry, rate, div_yield, vol
        if T <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0.0)

    def monte_carlo_price(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        n_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))
        paths = _simulate_paths_gbm_crn(spot, vol, rate, div_yield, self.expiry, Z)
        max_prices = paths.max(axis=1)
        payoffs = np.maximum(max_prices - self.strike, 0.0)
        disc = _discount(rate, self.expiry, payoffs)
        return {
            "price": float(disc.mean()),
            "std_error": float(disc.std(ddof=1) / np.sqrt(n_paths)),
            "prob_in_money": float((payoffs > 0).mean()),
        }


# -------------------------
# Percentage Lookback Call (desk naming)
# -------------------------
class PercentageLookbackCall(LookbackOption):
    """
    Lookback Call with desk-friendly payoff types:

    1) floating_ratio  (Fixed Units / Floating Notional; dollar payoff)
       payoff = Notional * max(0,  S_T - α * min(S))

    2) fixed_ratio     (Fixed Notional / Floating Units; scale-invariant %)
       payoff = Notional * max(0, (S_T - α * min(S)) / min(S))
    """

    def __init__(self, expiry: float, payoff_type: str = "floating_ratio", participation_level: float = 1.05):
        assert payoff_type in {"floating_ratio", "fixed_ratio"}
        self.expiry = expiry
        self.payoff_type = payoff_type
        self.participation_level = participation_level

    def monte_carlo_price(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        n_steps: Optional[int] = None,
        notional: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, n_steps))

        paths = _simulate_paths_gbm_crn(spot, vol, rate, div_yield, self.expiry, Z)
        final = paths[:, -1]
        minp = paths.min(axis=1)
        alpha = self.participation_level
        strike = alpha * minp

        if self.payoff_type == "floating_ratio":
            payoffs = notional * np.maximum(0.0, final - strike)           # dollar payoff
        elif self.payoff_type == "fixed_ratio":
            payoffs = np.where(minp > 0.0, notional * np.maximum(0.0, (final - strike) / minp), 0.0)
        else:
            raise ValueError("payoff_type must be 'floating_ratio' or 'fixed_ratio'")

        disc = _discount(rate, self.expiry, payoffs)
        return {
            "price": float(disc.mean()),
            "std_error": float(disc.std(ddof=1) / np.sqrt(n_paths)),
            "prob_payoff": float((payoffs > 0).mean()),
            "payoff_type": self.payoff_type,
        }

    def calculate_greeks(
        self,
        spot: float,
        vol: float,
        rate: float,
        div_yield: float,
        n_paths: int = 100_000,
        base_notional: float = 10_000_000,
        n_steps: Optional[int] = None,
        bump_pct: float = 0.005,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if n_steps is None:
            n_steps = max(int(self.expiry * 252), 1)
        n = max(n_paths, 100_000)
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, n_steps))
        Zanti = -Z
        alpha = self.participation_level

        def price_given(spot_val: float, vol_val: float, rate_val: float, Zmat: np.ndarray) -> float:
            paths = _simulate_paths_gbm_crn(spot_val, vol_val, rate_val, div_yield, self.expiry, Zmat)
            fin = paths[:, -1]
            minp = paths.min(axis=1)
            strike = alpha * minp
            if self.payoff_type == "floating_ratio":
                payoff = np.maximum(0.0, fin - strike)
            else:  # fixed_ratio
                payoff = np.where(minp > 0.0, np.maximum(0.0, (fin - strike) / minp), 0.0)
            return float(_discount(rate_val, self.expiry, payoff).mean())

        def price_crn(spot_val: float, vol_val: float, rate_val: float) -> float:
            return 0.5 * (price_given(spot_val, vol_val, rate_val, Z) + price_given(spot_val, vol_val, rate_val, Zanti))

        P0 = price_crn(spot, vol, rate)
        Pup = price_crn(spot * (1 + bump_pct), vol, rate)
        Pdn = price_crn(spot * (1 - bump_pct), vol, rate)

        if P0 > 0:
            delta_pct = (Pup - Pdn) / (2 * bump_pct * P0) * 100.0
            gamma_pct = (Pup - 2 * P0 + Pdn) / (bump_pct ** 2 * P0)
        else:
            delta_pct = 0.0
            gamma_pct = 0.0

        vol_bump = 0.01
        Pv = price_crn(spot, vol + vol_bump, rate)
        vega_per_unit = Pv - P0
        vega_position = vega_per_unit * base_notional

        rate_bump = 0.0001
        Pr = price_crn(spot, vol, rate + rate_bump)
        rho_per_bp = (Pr - P0) * 100.0
        rho_position = rho_per_bp * base_notional

        theta_daily_per_unit = -P0 * rate / 365.25
        theta_position = theta_daily_per_unit * base_notional

        return {
            "delta": float(delta_pct),
            "gamma": float(gamma_pct),
            "vega": float(vega_position),
            "theta": float(theta_position),
            "rho": float(rho_position),
            "base_price": float(P0 * base_notional),
            "base_price_pct": float(P0 * 100.0),
            "diagnostics": {"paths_used": n, "spot_bump_pct": bump_pct},
        }
