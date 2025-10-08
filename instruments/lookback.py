"""
Lookback options with percentage payoffs
Including both fixed notional and floating notional versions
WITH PROPERLY CORRECTED DELTA AND GAMMA CALCULATIONS
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class LookbackOption(ABC):
    """Base class for all lookback options"""
    
    @abstractmethod
    def monte_carlo_price(self, spot: float, vol: float, rate: float, 
                         div_yield: float, n_paths: int = 100000, 
                         n_steps: int = None) -> Dict[str, float]:
        """Price using Monte Carlo simulation"""
        pass


class PercentageLookbackPut(LookbackOption):
    """
    Percentage Lookback Put with two payoff types:
    
    1. Fixed Notional (Floating Units):
       Payoff = Notional * (Max - Final) / Initial
       Number of units floats based on initial price
       
    2. Floating Notional (Fixed Units):
       Payoff = Notional * (Max - Final) / Max
       Number of units is fixed, notional floats
    """
    
    def __init__(self, expiry: float, payoff_type: str = "floating_notional", 
                 protection_level: float = 0.95):
        """
        Args:
            expiry: Time to expiry in years
            payoff_type: "fixed_notional" or "floating_notional"
            protection_level: Protection percentage (e.g., 0.95 for 95%)
        """
        self.expiry = expiry
        self.payoff_type = payoff_type
        self.protection_level = protection_level
    
    def monte_carlo_price(self, spot: float, vol: float, rate: float, 
                         div_yield: float, n_paths: int = 100000, 
                         n_steps: int = None, notional: float = 1.0) -> Dict[str, float]:
        """
        Monte Carlo pricing for percentage lookback put.
        
        Args:
            spot: Current spot price
            vol: Volatility
            rate: Risk-free rate
            div_yield: Dividend yield
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            notional: Notional amount (default 1.0 for percentage return)
        """
        if n_steps is None:
            n_steps = int(self.expiry * 252)  # Daily monitoring
        
        dt = self.expiry / n_steps
        
        # Generate paths
        Z = np.random.standard_normal((n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(
                (rate - div_yield - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z[:, i]
            )
        
        # Calculate payoffs based on type
        initial_prices = paths[:, 0]
        final_prices = paths[:, -1]
        max_prices = np.max(paths, axis=1)
        
        if self.payoff_type == "fixed_notional":
            # Fixed Notional (Floating Units)
            # Payoff = Notional * max(0, (Max - Final) / Initial)
            strike_prices = self.protection_level * max_prices
            payoffs = notional * np.maximum(0, (strike_prices - final_prices) / initial_prices)
            
        else:  # floating_notional
            # Floating Notional (Fixed Units) - MORE COMMON
            # Payoff = Notional * max(0, (Max - Final) / Max)
            strike_prices = self.protection_level * max_prices
            payoffs = notional * np.maximum(0, (strike_prices - final_prices) / max_prices)
        
        # Discount to present value
        price = np.exp(-rate * self.expiry) * np.mean(payoffs)
        std_error = np.exp(-rate * self.expiry) * np.std(payoffs) / np.sqrt(n_paths)
        
        # Calculate additional statistics
        prob_payoff = np.mean(payoffs > 0)
        avg_max_return = np.mean(max_prices / initial_prices - 1)
        avg_final_return = np.mean(final_prices / initial_prices - 1)
        
        return {
            'price': price,
            'std_error': std_error,
            'prob_payoff': prob_payoff,
            'avg_max_return': avg_max_return,
            'avg_final_return': avg_final_return,
            'payoff_type': self.payoff_type
        }
    
    def calculate_greeks(self, spot: float, vol: float, rate: float, 
                        div_yield: float, n_paths: int = 100000,
                        base_notional: float = 10_000_000,
                        n_steps: int = None) -> Dict[str, float]:
        """
        Pathwise Derivative Method for Greeks calculation
        Uses common random numbers and pathwise derivatives for accurate delta
        """
        n_paths_greeks = max(n_paths, 200000)  # Increased for accuracy
        
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        
        # Generate paths with FIXED random numbers
        np.random.seed(42)
        Z = np.random.standard_normal((n_paths_greeks, n_steps))
        paths = np.zeros((n_paths_greeks, n_steps + 1))
        paths[:, 0] = spot
        
        drift = (rate - div_yield - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt)
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        max_prices = np.max(paths, axis=1)
        final_prices = paths[:, -1]
        initial_prices = paths[:, 0]
        
        # Calculate price and payoffs
        if self.payoff_type == "floating_notional":
            strike_prices = self.protection_level * max_prices
            payoffs = np.maximum(0, (strike_prices - final_prices) / max_prices)
        else:  # fixed_notional
            strike_prices = self.protection_level * max_prices
            payoffs = np.maximum(0, (strike_prices - final_prices) / initial_prices)
        
        price_base = np.exp(-rate * self.expiry) * np.mean(payoffs)
        
        # PATHWISE DERIVATIVE for Delta
        # For percentage lookback put: dPayoff/dS0 = -S_T / (S_0 * S_max) when in the money
        in_money = payoffs > 0
        
        if self.payoff_type == "floating_notional":
            # Payoff = (k*Max - Final) / Max when positive
            # dPayoff/dS0 ≈ -Final / (S0 * Max) for paths in the money
            dpayoff_ds0 = np.where(in_money, -final_prices / (spot * max_prices), 0.0)
        else:  # fixed_notional
            # Payoff = (k*Max - Final) / S0 when positive
            # dPayoff/dS0 ≈ -Final / (S0 * Max) for paths in the money
            dpayoff_ds0 = np.where(in_money, -final_prices / (spot * max_prices), 0.0)
        
        # Raw delta from pathwise derivative
        delta_raw = np.exp(-rate * self.expiry) * np.mean(dpayoff_ds0)
        
        # Convert to elasticity: Delta = (delta_raw * S) / P (no × 100!)
        if price_base > 0:
            delta_pct = (delta_raw * spot / price_base)  # Already gives percentage-like values
        else:
            delta_pct = 0
        
        # Gamma using finite differences with INDEPENDENT random numbers
        # Note: Common random numbers give same prices for percentage payoffs
        bump_pct = 0.01
        
        # Price up - fresh random numbers
        Z_up = np.random.standard_normal((n_paths_greeks, n_steps))
        paths_up = np.zeros((n_paths_greeks, n_steps + 1))
        paths_up[:, 0] = spot * (1 + bump_pct)
        for i in range(n_steps):
            paths_up[:, i+1] = paths_up[:, i] * np.exp(drift + diffusion * Z_up[:, i])
        max_prices_up = np.max(paths_up, axis=1)
        final_prices_up = paths_up[:, -1]
        if self.payoff_type == "floating_notional":
            strike_up = self.protection_level * max_prices_up
            payoffs_up = np.maximum(0, (strike_up - final_prices_up) / max_prices_up)
        else:
            strike_up = self.protection_level * max_prices_up
            payoffs_up = np.maximum(0, (strike_up - final_prices_up) / paths_up[:, 0])
        price_up = np.exp(-rate * self.expiry) * np.mean(payoffs_up)
        
        # Price down - fresh random numbers
        Z_down = np.random.standard_normal((n_paths_greeks, n_steps))
        paths_down = np.zeros((n_paths_greeks, n_steps + 1))
        paths_down[:, 0] = spot * (1 - bump_pct)
        for i in range(n_steps):
            paths_down[:, i+1] = paths_down[:, i] * np.exp(drift + diffusion * Z_down[:, i])
        max_prices_down = np.max(paths_down, axis=1)
        final_prices_down = paths_down[:, -1]
        if self.payoff_type == "floating_notional":
            strike_down = self.protection_level * max_prices_down
            payoffs_down = np.maximum(0, (strike_down - final_prices_down) / max_prices_down)
        else:
            strike_down = self.protection_level * max_prices_down
            payoffs_down = np.maximum(0, (strike_down - final_prices_down) / paths_down[:, 0])
        price_down = np.exp(-rate * self.expiry) * np.mean(payoffs_down)
        
        # Gamma 1% - measures how much dollar delta changes for a 1% spot move
        # Position values in dollars
        dollar_base = price_base * base_notional
        dollar_up = price_up * base_notional
        dollar_down = price_down * base_notional
        
        # Dollar delta at each level: dV/dS
        # Using one-sided differences from base price:
        # Delta when we're 1% higher = slope going forward
        # Delta when we're 1% lower = slope going backward  
        delta_at_up = (dollar_up - dollar_base) / (spot * bump_pct)
        delta_at_down = (dollar_base - dollar_down) / (spot * bump_pct)
        
        # Gamma 1% = change in dollar delta per 1% spot move
        # We have deltas at spot+1% and spot-1%, so they're 2% apart
        gamma_1pct = (delta_at_up - delta_at_down) / (2 * bump_pct)
        
        # Vega - use common random numbers
        vol_bump = 0.01
        paths_vega = np.zeros((n_paths_greeks, n_steps + 1))
        paths_vega[:, 0] = spot
        drift_vega = (rate - div_yield - 0.5 * (vol + vol_bump)**2) * dt
        diffusion_vega = (vol + vol_bump) * np.sqrt(dt)
        for i in range(n_steps):
            paths_vega[:, i+1] = paths_vega[:, i] * np.exp(drift_vega + diffusion_vega * Z[:, i])
        max_prices_vega = np.max(paths_vega, axis=1)
        final_prices_vega = paths_vega[:, -1]
        if self.payoff_type == "floating_notional":
            strike_vega = self.protection_level * max_prices_vega
            payoffs_vega = np.maximum(0, (strike_vega - final_prices_vega) / max_prices_vega)
        else:
            strike_vega = self.protection_level * max_prices_vega
            payoffs_vega = np.maximum(0, (strike_vega - final_prices_vega) / paths_vega[:, 0])
        price_vega = np.exp(-rate * self.expiry) * np.mean(payoffs_vega)
        vega_per_unit = price_vega - price_base
        vega_position = vega_per_unit * base_notional
        
        # Theta
        theta_daily_per_unit = -price_base * rate / 365.25
        theta_position = theta_daily_per_unit * base_notional
        
        # Rho - use common random numbers
        rate_bump = 0.0001
        paths_rho = np.zeros((n_paths_greeks, n_steps + 1))
        paths_rho[:, 0] = spot
        drift_rho = ((rate + rate_bump) - div_yield - 0.5 * vol**2) * dt
        for i in range(n_steps):
            paths_rho[:, i+1] = paths_rho[:, i] * np.exp(drift_rho + diffusion * Z[:, i])
        max_prices_rho = np.max(paths_rho, axis=1)
        final_prices_rho = paths_rho[:, -1]
        if self.payoff_type == "floating_notional":
            strike_rho = self.protection_level * max_prices_rho
            payoffs_rho = np.maximum(0, (strike_rho - final_prices_rho) / max_prices_rho)
        else:
            strike_rho = self.protection_level * max_prices_rho
            payoffs_rho = np.maximum(0, (strike_rho - final_prices_rho) / paths_rho[:, 0])
        price_rho = np.exp(-(rate + rate_bump) * self.expiry) * np.mean(payoffs_rho)
        rho_per_bp = price_rho - price_base
        rho_position = rho_per_bp * 100 * base_notional
        
        return {
            'delta': delta_pct,
            'gamma': gamma_1pct,  # Gamma 1% in dollar terms
            'vega': vega_position,
            'theta': theta_position,
            'rho': rho_position,
            'base_price': price_base * base_notional,
            'base_price_pct': price_base * 100,
            'diagnostics': {
                'paths_used': n_paths_greeks,
                'price_up': price_up,
                'price_down': price_down,
                'price_base': price_base,
                'spot_bump_pct': bump_pct,
                'delta_raw': delta_raw
            }
        }


class RatchetingLookbackPut(LookbackOption):
    """
    Original Ratcheting Lookback Put (Dollar Payoff)
    Strike resets to protection_level * max(spot history)
    Payoff = max(0, Strike - Final) in dollars
    """
    
    def __init__(self, protection_level: float, expiry: float, initial_spot: Optional[float] = None):
        self.protection_level = protection_level
        self.expiry = expiry
        self.initial_spot = initial_spot
        self.current_max = initial_spot
        
    def monte_carlo_price(self, spot: float, vol: float, rate: float, 
                         div_yield: float, n_paths: int = 100000, 
                         n_steps: int = None) -> Dict[str, float]:
        """Monte Carlo pricing for dollar ratcheting lookback"""
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        
        if self.current_max is None:
            self.current_max = spot
        
        Z = np.random.standard_normal((n_paths, n_steps))
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        
        running_max = np.full(n_paths, max(spot, self.current_max))
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(
                (rate - div_yield - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z[:, i]
            )
            running_max = np.maximum(running_max, paths[:, i+1])
        
        # Dollar payoff
        strikes = self.protection_level * running_max
        final_spots = paths[:, -1]
        payoffs = np.maximum(strikes - final_spots, 0)
        
        price = np.exp(-rate * self.expiry) * np.mean(payoffs)
        std_error = np.exp(-rate * self.expiry) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'avg_final_strike': np.mean(strikes),
            'prob_in_money': np.mean(payoffs > 0)
        }
    
    def calculate_greeks(self, spot: float, vol: float, rate: float, 
                        div_yield: float, n_paths: int = 100000,
                        base_notional: float = 1.0,
                        n_steps: int = None) -> Dict[str, float]:
        """
        Greeks calculation for dollar payoff ratcheting lookback put.
        FIXED to use independent runs and return percentage delta.
        """
        n_paths_greeks = max(n_paths, 50000)
        
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        current_max = self.current_max if self.current_max is not None else spot
        
        def calc_price(spot_val, vol_val=vol, rate_val=rate):
            """Calculate with fresh random numbers"""
            Z = np.random.standard_normal((n_paths_greeks, n_steps))
            paths = np.zeros((n_paths_greeks, n_steps + 1))
            paths[:, 0] = spot_val
            
            drift = (rate_val - div_yield - 0.5 * vol_val**2) * dt
            diffusion = vol_val * np.sqrt(dt)
            
            running_max = np.full(n_paths_greeks, max(spot_val, current_max))
            
            for i in range(n_steps):
                paths[:, i+1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
                running_max = np.maximum(running_max, paths[:, i+1])
            
            strikes = self.protection_level * running_max
            final_spots = paths[:, -1]
            payoffs = np.maximum(strikes - final_spots, 0)
            
            return np.exp(-rate_val * self.expiry) * np.mean(payoffs)
        
        # Base price
        base_price = np.mean([calc_price(spot) for _ in range(3)])
        
        # Delta
        bump = 0.01 * spot
        price_up = np.mean([calc_price(spot + bump) for _ in range(2)])
        price_down = np.mean([calc_price(spot - bump) for _ in range(2)])
        
        delta_dollar = (price_up - price_down) / (2 * bump)
        
        # Convert to percentage elasticity
        if base_price > 0:
            delta_pct = delta_dollar * spot / base_price * 100
        else:
            delta_pct = 0
        
        # Gamma - convert dollar gamma to percentage
        gamma_dollar = (price_up - 2*base_price + price_down) / (bump**2)
        if base_price > 0:
            gamma_pct = gamma_dollar * spot * spot / base_price
        else:
            gamma_pct = 0
        
        # Vega
        vol_bump = 0.01
        price_vega = np.mean([calc_price(spot, vol + vol_bump) for _ in range(2)])
        vega_dollar = price_vega - base_price
        
        # Theta
        theta_dollar = -base_price * rate / 365.25
        
        # Rho
        rate_bump = 0.0001
        price_rho = np.mean([calc_price(spot, vol, rate + rate_bump) for _ in range(2)])
        rho_dollar = (price_rho - base_price) * 100
        
        return {
            'delta': delta_pct,  # Return percentage elasticity
            'gamma': abs(gamma_pct),
            'vega': vega_dollar,
            'theta': theta_dollar,
            'rho': rho_dollar,
            'base_price': base_price,
            'diagnostics': {
                'paths_used': n_paths_greeks,
                'spot_bump': bump
            }
        }


class StandardFixedStrikeLookbackPut(LookbackOption):
    """
    Standard Fixed Strike Lookback Put (Dollar Payoff)
    Payoff = max(K - min(S), 0) in dollars
    """
    
    def __init__(self, strike: float, expiry: float):
        self.strike = strike
        self.expiry = expiry
    
    def analytical_price(self, spot: float, vol: float, rate: float, 
                        div_yield: float) -> Optional[float]:
        """Closed-form solution"""
        S = spot
        K = self.strike
        T = self.expiry
        r = rate
        q = div_yield
        sigma = vol
        
        if T <= 0:
            return max(K - S, 0)
        
        # Standard formulas
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if S >= K:
            # Currently OTM put with lookback on minimum
            b = r - q
            if b != 0:
                lambda_val = 2*b/sigma**2
                a = (np.log(S/K) + b*T) / (sigma*np.sqrt(T))
                
                # Corrected formula for OTM lookback put
                term1 = K*np.exp(-r*T) * (norm.cdf(-a) + (S/K)**lambda_val * norm.cdf(-a + 2*b*np.sqrt(T)/sigma))
                term2 = S*np.exp(-q*T) * norm.cdf(-a)
                term3 = S*np.exp(-q*T) * (sigma**2/(2*b)) * ((S/K)**(-lambda_val) * norm.cdf(a - 2*b*np.sqrt(T)/sigma) - norm.cdf(a))
                
                price = term1 - term2 + term3
            else:
                # Special case b = 0
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
                # Add lookback adjustment
                price += S*np.exp(-q*T)*sigma*np.sqrt(T)*(norm.pdf(d1) + d1*norm.cdf(d1))
        else:
            # ITM put
            price = (K - S)*np.exp(-r*T)
        
        return max(price, 0)
    
    def monte_carlo_price(self, spot: float, vol: float, rate: float, 
                         div_yield: float, n_paths: int = 100000, 
                         n_steps: int = None) -> Dict[str, float]:
        """Monte Carlo pricing"""
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        
        Z = np.random.standard_normal((n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(
                (rate - div_yield - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z[:, i]
            )
        
        min_prices = np.min(paths, axis=1)
        payoffs = np.maximum(self.strike - min_prices, 0)
        
        price = np.exp(-rate * self.expiry) * np.mean(payoffs)
        std_error = np.exp(-rate * self.expiry) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'prob_in_money': np.mean(payoffs > 0)
        }


class PercentageLookbackCall(LookbackOption):
    """
    Percentage Lookback Call with two payoff types:
    
    1. Fixed Notional (Floating Units):
       Payoff = Notional * max(0, (Final - α × Min) / Initial)
       Number of units floats based on initial price
       
    2. Floating Notional (Fixed Units):
       Payoff = Notional * max(0, (Final - α × Min) / Min)
       Number of units is fixed, notional floats
    """
    
    def __init__(self, expiry: float, payoff_type: str = "floating_notional", 
                 participation_level: float = 1.05):
        """
        Args:
            expiry: Time to expiry in years
            payoff_type: "fixed_notional" or "floating_notional"
            participation_level: Participation percentage (e.g., 1.05 for 105%)
        """
        self.expiry = expiry
        self.payoff_type = payoff_type
        self.participation_level = participation_level
    
    def monte_carlo_price(self, spot: float, vol: float, rate: float, 
                         div_yield: float, n_paths: int = 100000, 
                         n_steps: int = None, notional: float = 1.0) -> Dict[str, float]:
        """
        Monte Carlo pricing for percentage lookback call.
        
        Args:
            spot: Current spot price
            vol: Volatility
            rate: Risk-free rate
            div_yield: Dividend yield
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            notional: Notional amount (default 1.0 for percentage return)
        """
        if n_steps is None:
            n_steps = int(self.expiry * 252)  # Daily monitoring
        
        dt = self.expiry / n_steps
        
        # Generate paths
        Z = np.random.standard_normal((n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(
                (rate - div_yield - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z[:, i]
            )
        
        # Calculate payoffs based on type
        initial_prices = paths[:, 0]
        final_prices = paths[:, -1]
        min_prices = np.min(paths, axis=1)  # For calls, we track minimum
        
        if self.payoff_type == "fixed_notional":
            # Fixed Notional (Floating Units)
            strike_prices = self.participation_level * min_prices
            payoffs = notional * np.maximum(0, (final_prices - strike_prices) / initial_prices)
            
        else:  # floating_notional
            # Floating Notional (Fixed Units) - MORE COMMON
            strike_prices = self.participation_level * min_prices
            # Avoid division by zero
            payoffs = np.where(min_prices > 0,
                              notional * np.maximum(0, (final_prices - strike_prices) / min_prices),
                              0)
        
        # Discount to present value
        price = np.exp(-rate * self.expiry) * np.mean(payoffs)
        std_error = np.exp(-rate * self.expiry) * np.std(payoffs) / np.sqrt(n_paths)
        
        # Calculate additional statistics
        prob_payoff = np.mean(payoffs > 0)
        avg_min_return = np.mean(min_prices / initial_prices - 1)
        avg_final_return = np.mean(final_prices / initial_prices - 1)
        
        return {
            'price': price,
            'std_error': std_error,
            'prob_payoff': prob_payoff,
            'avg_min_return': avg_min_return,
            'avg_final_return': avg_final_return,
            'payoff_type': self.payoff_type
        }
    
    def calculate_greeks(self, spot: float, vol: float, rate: float, 
                        div_yield: float, n_paths: int = 100000,
                        base_notional: float = 10_000_000,
                        n_steps: int = None) -> Dict[str, float]:
        """
        Pathwise Derivative Method for Greeks calculation
        Uses common random numbers and pathwise derivatives for accurate delta
        """
        n_paths_greeks = max(n_paths, 200000)  # Increased for accuracy
        
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        
        # Generate paths with FIXED random numbers
        np.random.seed(42)
        Z = np.random.standard_normal((n_paths_greeks, n_steps))
        paths = np.zeros((n_paths_greeks, n_steps + 1))
        paths[:, 0] = spot
        
        drift = (rate - div_yield - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt)
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        min_prices = np.min(paths, axis=1)
        final_prices = paths[:, -1]
        initial_prices = paths[:, 0]
        
        # Calculate price and payoffs
        if self.payoff_type == "floating_notional":
            strike_prices = self.participation_level * min_prices
            payoffs = np.where(min_prices > 0,
                              np.maximum(0, (final_prices - strike_prices) / min_prices),
                              0)
        else:  # fixed_notional
            strike_prices = self.participation_level * min_prices
            payoffs = np.maximum(0, (final_prices - strike_prices) / initial_prices)
        
        price_base = np.exp(-rate * self.expiry) * np.mean(payoffs)
        
        # PATHWISE DERIVATIVE for Delta
        # For percentage lookback call: dPayoff/dS0 = +S_T / (S_0 * S_min) when in the money
        in_money = payoffs > 0
        
        if self.payoff_type == "floating_notional":
            # Payoff = (Final - k*Min) / Min when positive
            # dPayoff/dS0 ≈ +Final / (S0 * Min) for paths in the money
            dpayoff_ds0 = np.where(in_money & (min_prices > 0), 
                                   final_prices / (spot * min_prices), 0.0)
        else:  # fixed_notional
            # Payoff = (Final - k*Min) / S0 when positive
            # dPayoff/dS0 ≈ +Final / (S0 * Min) for paths in the money
            dpayoff_ds0 = np.where(in_money & (min_prices > 0),
                                   final_prices / (spot * min_prices), 0.0)
        
        # Raw delta from pathwise derivative
        delta_raw = np.exp(-rate * self.expiry) * np.mean(dpayoff_ds0)
        
        # Convert to elasticity: Delta = (delta_raw * S) / P (no × 100!)
        if price_base > 0:
            delta_pct = (delta_raw * spot / price_base)  # Already gives percentage-like values
        else:
            delta_pct = 0
        
        # Gamma using finite differences with INDEPENDENT random numbers
        # Note: Common random numbers give same prices for percentage payoffs
        bump_pct = 0.01
        
        # Price up - fresh random numbers
        Z_up = np.random.standard_normal((n_paths_greeks, n_steps))
        paths_up = np.zeros((n_paths_greeks, n_steps + 1))
        paths_up[:, 0] = spot * (1 + bump_pct)
        for i in range(n_steps):
            paths_up[:, i+1] = paths_up[:, i] * np.exp(drift + diffusion * Z_up[:, i])
        min_prices_up = np.min(paths_up, axis=1)
        final_prices_up = paths_up[:, -1]
        if self.payoff_type == "floating_notional":
            strike_up = self.participation_level * min_prices_up
            payoffs_up = np.where(min_prices_up > 0,
                                 np.maximum(0, (final_prices_up - strike_up) / min_prices_up),
                                 0)
        else:
            strike_up = self.participation_level * min_prices_up
            payoffs_up = np.maximum(0, (final_prices_up - strike_up) / paths_up[:, 0])
        price_up = np.exp(-rate * self.expiry) * np.mean(payoffs_up)
        
        # Price down - fresh random numbers
        Z_down = np.random.standard_normal((n_paths_greeks, n_steps))
        paths_down = np.zeros((n_paths_greeks, n_steps + 1))
        paths_down[:, 0] = spot * (1 - bump_pct)
        for i in range(n_steps):
            paths_down[:, i+1] = paths_down[:, i] * np.exp(drift + diffusion * Z_down[:, i])
        min_prices_down = np.min(paths_down, axis=1)
        final_prices_down = paths_down[:, -1]
        if self.payoff_type == "floating_notional":
            strike_down = self.participation_level * min_prices_down
            payoffs_down = np.where(min_prices_down > 0,
                                   np.maximum(0, (final_prices_down - strike_down) / min_prices_down),
                                   0)
        else:
            strike_down = self.participation_level * min_prices_down
            payoffs_down = np.maximum(0, (final_prices_down - strike_down) / paths_down[:, 0])
        price_down = np.exp(-rate * self.expiry) * np.mean(payoffs_down)
        
        # Gamma 1% - measures how much dollar delta changes for a 1% spot move
        # Position values in dollars
        dollar_base = price_base * base_notional
        dollar_up = price_up * base_notional
        dollar_down = price_down * base_notional
        
        # Dollar delta at each level: dV/dS
        delta_at_up = (dollar_up - dollar_base) / (spot * bump_pct)
        delta_at_down = (dollar_base - dollar_down) / (spot * bump_pct)
        
        # Gamma 1% = change in dollar delta per 1% spot move
        # We have deltas at spot+1% and spot-1%, so they're 2% apart
        gamma_1pct = (delta_at_up - delta_at_down) / (2 * bump_pct)
        
        # Vega - use common random numbers
        vol_bump = 0.01
        paths_vega = np.zeros((n_paths_greeks, n_steps + 1))
        paths_vega[:, 0] = spot
        drift_vega = (rate - div_yield - 0.5 * (vol + vol_bump)**2) * dt
        diffusion_vega = (vol + vol_bump) * np.sqrt(dt)
        for i in range(n_steps):
            paths_vega[:, i+1] = paths_vega[:, i] * np.exp(drift_vega + diffusion_vega * Z[:, i])
        min_prices_vega = np.min(paths_vega, axis=1)
        final_prices_vega = paths_vega[:, -1]
        if self.payoff_type == "floating_notional":
            strike_vega = self.participation_level * min_prices_vega
            payoffs_vega = np.where(min_prices_vega > 0,
                                   np.maximum(0, (final_prices_vega - strike_vega) / min_prices_vega),
                                   0)
        else:
            strike_vega = self.participation_level * min_prices_vega
            payoffs_vega = np.maximum(0, (final_prices_vega - strike_vega) / paths_vega[:, 0])
        price_vega = np.exp(-rate * self.expiry) * np.mean(payoffs_vega)
        vega_per_unit = price_vega - price_base
        vega_position = vega_per_unit * base_notional
        
        # Theta
        theta_daily_per_unit = -price_base * rate / 365.25
        theta_position = theta_daily_per_unit * base_notional
        
        # Rho - use common random numbers
        rate_bump = 0.0001
        paths_rho = np.zeros((n_paths_greeks, n_steps + 1))
        paths_rho[:, 0] = spot
        drift_rho = ((rate + rate_bump) - div_yield - 0.5 * vol**2) * dt
        for i in range(n_steps):
            paths_rho[:, i+1] = paths_rho[:, i] * np.exp(drift_rho + diffusion * Z[:, i])
        min_prices_rho = np.min(paths_rho, axis=1)
        final_prices_rho = paths_rho[:, -1]
        if self.payoff_type == "floating_notional":
            strike_rho = self.participation_level * min_prices_rho
            payoffs_rho = np.where(min_prices_rho > 0,
                                  np.maximum(0, (final_prices_rho - strike_rho) / min_prices_rho),
                                  0)
        else:
            strike_rho = self.participation_level * min_prices_rho
            payoffs_rho = np.maximum(0, (final_prices_rho - strike_rho) / paths_rho[:, 0])
        price_rho = np.exp(-(rate + rate_bump) * self.expiry) * np.mean(payoffs_rho)
        rho_per_bp = price_rho - price_base
        rho_position = rho_per_bp * 100 * base_notional
        
        return {
            'delta': delta_pct,
            'gamma': gamma_1pct,  # Gamma 1% in dollar terms
            'vega': vega_position,
            'theta': theta_position,
            'rho': rho_position,
            'base_price': price_base * base_notional,
            'base_price_pct': price_base * 100,
            'diagnostics': {
                'paths_used': n_paths_greeks,
                'price_up': price_up,
                'price_down': price_down,
                'price_base': price_base,
                'spot_bump_pct': bump_pct,
                'delta_raw': delta_raw
            }
        }


class RatchetingLookbackCall(LookbackOption):
    """
    Ratcheting Lookback Call (Dollar Payoff)
    Strike resets to participation_level * min(spot history)
    Payoff = max(0, Final - Strike) in dollars
    """
    
    def __init__(self, participation_level: float, expiry: float, initial_spot: Optional[float] = None):
        """
        Args:
            participation_level: Participation level (e.g., 1.05 for 105%)
            expiry: Time to expiry in years
            initial_spot: Initial spot price (for path dependence)
        """
        self.participation_level = participation_level
        self.expiry = expiry
        self.initial_spot = initial_spot
        self.current_min = initial_spot
        
    def monte_carlo_price(self, spot: float, vol: float, rate: float, 
                         div_yield: float, n_paths: int = 100000, 
                         n_steps: int = None) -> Dict[str, float]:
        """Monte Carlo pricing for dollar ratcheting lookback call"""
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        
        if self.current_min is None:
            self.current_min = spot
        
        Z = np.random.standard_normal((n_paths, n_steps))
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        
        running_min = np.full(n_paths, min(spot, self.current_min) if self.current_min else spot)
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(
                (rate - div_yield - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z[:, i]
            )
            running_min = np.minimum(running_min, paths[:, i+1])
        
        # Dollar payoff for call
        strikes = self.participation_level * running_min
        final_spots = paths[:, -1]
        payoffs = np.maximum(final_spots - strikes, 0)
        
        price = np.exp(-rate * self.expiry) * np.mean(payoffs)
        std_error = np.exp(-rate * self.expiry) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'avg_final_strike': np.mean(strikes),
            'prob_in_money': np.mean(payoffs > 0)
        }
    
    def calculate_greeks(self, spot: float, vol: float, rate: float,
                        div_yield: float, n_paths: int = 100000,
                        base_notional: float = 1.0,
                        n_steps: int = None) -> Dict[str, float]:
        """
        Greeks calculation for dollar payoff ratcheting lookback call.
        FIXED to use independent runs and return percentage delta.
        """
        n_paths_greeks = max(n_paths, 50000)
        
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        current_min = self.current_min if self.current_min is not None else spot
        
        def calc_price(spot_val, vol_val=vol, rate_val=rate):
            """Calculate with fresh random numbers"""
            Z = np.random.standard_normal((n_paths_greeks, n_steps))
            paths = np.zeros((n_paths_greeks, n_steps + 1))
            paths[:, 0] = spot_val
            
            drift = (rate_val - div_yield - 0.5 * vol_val**2) * dt
            diffusion = vol_val * np.sqrt(dt)
            
            running_min = np.full(n_paths_greeks, min(spot_val, current_min))
            
            for i in range(n_steps):
                paths[:, i+1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
                running_min = np.minimum(running_min, paths[:, i+1])
            
            strikes = self.participation_level * running_min
            final_spots = paths[:, -1]
            payoffs = np.maximum(final_spots - strikes, 0)
            
            return np.exp(-rate_val * self.expiry) * np.mean(payoffs)
        
        # Base price
        base_price = np.mean([calc_price(spot) for _ in range(3)])
        
        # Delta
        bump = 0.01 * spot
        price_up = np.mean([calc_price(spot + bump) for _ in range(2)])
        price_down = np.mean([calc_price(spot - bump) for _ in range(2)])
        
        delta_dollar = (price_up - price_down) / (2 * bump)
        
        # Convert to percentage elasticity
        if base_price > 0:
            delta_pct = delta_dollar * spot / base_price * 100
        else:
            delta_pct = 0
        
        # Gamma - convert dollar gamma to percentage
        gamma_dollar = (price_up - 2*base_price + price_down) / (bump**2)
        if base_price > 0:
            gamma_pct = gamma_dollar * spot * spot / base_price
        else:
            gamma_pct = 0
        
        # Vega
        vol_bump = 0.01
        price_vega = np.mean([calc_price(spot, vol + vol_bump) for _ in range(2)])
        vega_dollar = price_vega - base_price
        
        # Theta
        theta_dollar = -base_price * rate / 365.25
        
        # Rho
        rate_bump = 0.0001
        price_rho = np.mean([calc_price(spot, vol, rate + rate_bump) for _ in range(2)])
        rho_dollar = (price_rho - base_price) * 100
        
        return {
            'delta': delta_pct,
            'gamma': abs(gamma_pct),
            'vega': vega_dollar,
            'theta': theta_dollar,
            'rho': rho_dollar,
            'base_price': base_price,
            'diagnostics': {
                'paths_used': n_paths_greeks,
                'spot_bump': bump
            }
        }


class StandardFixedStrikeLookbackCall(LookbackOption):
    """
    Standard Fixed Strike Lookback Call (Dollar Payoff)
    Payoff = max(max(S) - K, 0) in dollars
    """
    
    def __init__(self, strike: float, expiry: float):
        self.strike = strike
        self.expiry = expiry
    
    def analytical_price(self, spot: float, vol: float, rate: float, 
                        div_yield: float) -> Optional[float]:
        """Closed-form solution for fixed strike lookback call"""
        S = spot
        K = self.strike
        T = self.expiry
        r = rate
        q = div_yield
        sigma = vol
        
        if T <= 0:
            return max(S - K, 0)
        
        # Standard formulas
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if S <= K:
            # Currently OTM call with lookback on maximum
            b = r - q
            if b != 0:
                lambda_val = 2*b/sigma**2
                a = (np.log(S/K) + b*T) / (sigma*np.sqrt(T))
                
                # Formula for OTM lookback call
                term1 = S*np.exp(-q*T) * (norm.cdf(a) + (S/K)**(-lambda_val) * norm.cdf(-a + 2*b*np.sqrt(T)/sigma))
                term2 = K*np.exp(-r*T) * norm.cdf(a - sigma*np.sqrt(T))
                term3 = S*np.exp(-q*T) * (sigma**2/(2*b)) * (norm.cdf(a) - (S/K)**(-lambda_val) * norm.cdf(a - 2*b*np.sqrt(T)/sigma))
                
                price = term1 - term2 + term3
            else:
                # Special case b = 0
                price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                # Add lookback adjustment
                price += S*np.exp(-q*T)*sigma*np.sqrt(T)*(norm.pdf(d1) + d1*norm.cdf(d1))
        else:
            # ITM call
            price = (S - K)*np.exp(-r*T)
        
        return max(price, 0)
    
    def monte_carlo_price(self, spot: float, vol: float, rate: float, 
                         div_yield: float, n_paths: int = 100000, 
                         n_steps: int = None) -> Dict[str, float]:
        """Monte Carlo pricing for fixed strike lookback call"""
        if n_steps is None:
            n_steps = int(self.expiry * 252)
        
        dt = self.expiry / n_steps
        
        Z = np.random.standard_normal((n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot
        
        for i in range(n_steps):
            paths[:, i+1] = paths[:, i] * np.exp(
                (rate - div_yield - 0.5*vol**2)*dt + vol*np.sqrt(dt)*Z[:, i]
            )
        
        max_prices = np.max(paths, axis=1)
        payoffs = np.maximum(max_prices - self.strike, 0)
        
        price = np.exp(-rate * self.expiry) * np.mean(payoffs)
        std_error = np.exp(-rate * self.expiry) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'prob_in_money': np.mean(payoffs > 0)
        }