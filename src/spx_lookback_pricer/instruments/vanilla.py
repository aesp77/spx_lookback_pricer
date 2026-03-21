"""
Vanilla Options
European and American vanilla options for calibration and control variates
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


class VanillaOption:
    """Base class for vanilla options"""
    
    def __init__(
        self,
        strike: float,
        expiry: float,
        is_call: bool = True
    ):
        """
        Initialize vanilla option
        
        Args:
            strike: Strike price
            expiry: Time to expiry in years
            is_call: True for call, False for put
        """
        self.strike = strike
        self.expiry = expiry
        self.is_call = is_call
        
    def payoff(self, paths: np.ndarray, market_data=None) -> np.ndarray:
        """
        Calculate payoff for simulated paths
        
        Args:
            paths: Array of shape (num_paths, num_steps)
            market_data: Not used
            
        Returns:
            Array of payoffs (based on final price)
        """
        final_prices = paths[:, -1] if paths.ndim > 1 else paths
        
        if self.is_call:
            return np.maximum(final_prices - self.strike, 0)
        else:
            return np.maximum(self.strike - final_prices, 0)
            
    def __repr__(self) -> str:
        option_type = "Call" if self.is_call else "Put"
        return f"VanillaOption{option_type}(K={self.strike}, T={self.expiry})"


class EuropeanOption(VanillaOption):
    """
    European vanilla option with analytical Black-Scholes pricing
    """
    
    def analytical_price(self, market_data, volatility: float) -> float:
        """
        Black-Scholes analytical price
        
        Args:
            market_data: Market data
            volatility: Implied volatility
            
        Returns:
            Option price
        """
        from scipy.stats import norm
        
        S = market_data.spot
        K = self.strike
        T = self.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        sigma = volatility
        
        # Black-Scholes formula
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if self.is_call:
            price = (
                S * np.exp(-q * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2)
            )
        else:
            price = (
                K * np.exp(-r * T) * norm.cdf(-d2)
                - S * np.exp(-q * T) * norm.cdf(-d1)
            )
            
        return price
        
    def implied_volatility(
        self,
        market_price: float,
        market_data,
        initial_guess: float = 0.2
    ) -> float:
        """
        Calculate implied volatility from market price
        
        Args:
            market_price: Observed market price
            market_data: Market data
            initial_guess: Initial volatility guess
            
        Returns:
            Implied volatility
        """
        from scipy.optimize import brentq
        
        def price_diff(vol):
            return self.analytical_price(market_data, vol) - market_price
            
        try:
            # Use Brent's method for root finding
            iv = brentq(price_diff, 0.001, 5.0)
            return iv
        except:
            # Fall back to initial guess if optimization fails
            return initial_guess
            
    def delta(self, market_data, volatility: float) -> float:
        """Calculate option delta"""
        from scipy.stats import norm
        
        S = market_data.spot
        K = self.strike
        T = self.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        sigma = volatility
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if self.is_call:
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)
            
    def gamma(self, market_data, volatility: float) -> float:
        """Calculate option gamma"""
        from scipy.stats import norm
        
        S = market_data.spot
        K = self.strike
        T = self.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        sigma = volatility
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
    def vega(self, market_data, volatility: float) -> float:
        """Calculate option vega"""
        from scipy.stats import norm
        
        S = market_data.spot
        K = self.strike
        T = self.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        sigma = volatility
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
    def theta(self, market_data, volatility: float) -> float:
        """Calculate option theta"""
        from scipy.stats import norm
        
        S = market_data.spot
        K = self.strike
        T = self.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        sigma = volatility
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if self.is_call:
            theta = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)
            )
        else:
            theta = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)
            )
            
        return theta / 365  # Convert to daily theta
        
    def rho(self, market_data, volatility: float) -> float:
        """Calculate option rho"""
        from scipy.stats import norm
        
        S = market_data.spot
        K = self.strike
        T = self.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        sigma = volatility
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if self.is_call:
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
    def __repr__(self) -> str:
        option_type = "Call" if self.is_call else "Put"
        return f"EuropeanOption{option_type}(K={self.strike}, T={self.expiry})"


class AmericanOption(VanillaOption):
    """
    American vanilla option
    
    Requires numerical methods (binomial tree, finite differences, or LSM)
    """
    
    def price_binomial(
        self,
        market_data,
        volatility: float,
        num_steps: int = 100
    ) -> float:
        """
        Price using binomial tree
        
        Args:
            market_data: Market data
            volatility: Volatility
            num_steps: Number of time steps
            
        Returns:
            Option price
        """
        S = market_data.spot
        K = self.strike
        T = self.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        sigma = volatility
        
        dt = T / num_steps
        
        # CRR parameters
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Initialize asset prices at maturity
        prices = np.zeros(num_steps + 1)
        for i in range(num_steps + 1):
            prices[i] = S * (u ** (num_steps - i)) * (d ** i)
            
        # Initialize option values at maturity
        values = np.zeros(num_steps + 1)
        if self.is_call:
            values = np.maximum(prices - K, 0)
        else:
            values = np.maximum(K - prices, 0)
            
        # Backward induction
        for step in range(num_steps - 1, -1, -1):
            for i in range(step + 1):
                # European continuation value
                continuation = discount * (p * values[i] + (1 - p) * values[i + 1])
                
                # Early exercise value
                S_current = S * (u ** (step - i)) * (d ** i)
                if self.is_call:
                    exercise = max(S_current - K, 0)
                else:
                    exercise = max(K - S_current, 0)
                    
                # American option value
                values[i] = max(continuation, exercise)
                
        return values[0]
        
    def analytical_price(self, market_data, volatility: float) -> float:
        """
        Approximate analytical price using binomial method
        
        Note: True analytical solution doesn't exist for American options
        """
        return self.price_binomial(market_data, volatility)
        
    def __repr__(self) -> str:
        option_type = "Call" if self.is_call else "Put"
        return f"AmericanOption{option_type}(K={self.strike}, T={self.expiry})"
