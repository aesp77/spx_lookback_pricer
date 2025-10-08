"""
Mathematical Utilities
Black-Scholes formulas and other mathematical functions
"""

import numpy as np
from scipy.stats import norm
from typing import Union


def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal cumulative distribution function"""
    return norm.cdf(x)


def normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard normal probability density function"""
    return norm.pdf(x)


def black_scholes_d1(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Calculate d1 parameter for Black-Scholes
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        
    Returns:
        d1 value
    """
    if T <= 0 or sigma <= 0:
        return 0.0
        
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def black_scholes_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Calculate d2 parameter for Black-Scholes
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        
    Returns:
        d2 value
    """
    return black_scholes_d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Black-Scholes option price
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        is_call: True for call, False for put
        
    Returns:
        Option price
    """
    if T <= 0:
        if is_call:
            return max(S - K, 0)
        else:
            return max(K - S, 0)
            
    d1 = black_scholes_d1(S, K, T, r, q, sigma)
    d2 = black_scholes_d2(S, K, T, r, q, sigma)
    
    if is_call:
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
    return max(price, 0.0)


def black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Black-Scholes delta
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        is_call: True for call, False for put
        
    Returns:
        Delta
    """
    if T <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
            
    d1 = black_scholes_d1(S, K, T, r, q, sigma)
    
    if is_call:
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1)


def black_scholes_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Black-Scholes gamma (same for calls and puts)
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        
    Returns:
        Gamma
    """
    if T <= 0 or sigma <= 0:
        return 0.0
        
    d1 = black_scholes_d1(S, K, T, r, q, sigma)
    
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float
) -> float:
    """
    Black-Scholes vega (same for calls and puts)
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        
    Returns:
        Vega (per 1% vol change)
    """
    if T <= 0:
        return 0.0
        
    d1 = black_scholes_d1(S, K, T, r, q, sigma)
    
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def black_scholes_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Black-Scholes theta
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        is_call: True for call, False for put
        
    Returns:
        Theta (per day)
    """
    if T <= 0:
        return 0.0
        
    d1 = black_scholes_d1(S, K, T, r, q, sigma)
    d2 = black_scholes_d2(S, K, T, r, q, sigma)
    
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    
    if is_call:
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        
    return (term1 + term2 + term3) / 365.0  # Daily theta


def black_scholes_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Black-Scholes rho
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        is_call: True for call, False for put
        
    Returns:
        Rho (per 1% rate change)
    """
    if T <= 0:
        return 0.0
        
    d2 = black_scholes_d2(S, K, T, r, q, sigma)
    
    if is_call:
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def implied_volatility_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool = True,
    initial_guess: float = 0.2,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method
    
    Args:
        market_price: Observed market price
        S: Spot price
        K: Strike price
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        is_call: True for call, False for put
        initial_guess: Initial volatility guess
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Implied volatility
    """
    sigma = initial_guess
    
    for _ in range(max_iterations):
        # Calculate price and vega
        price = black_scholes_price(S, K, T, r, q, sigma, is_call)
        vega = black_scholes_vega(S, K, T, r, q, sigma) * 100  # Convert back
        
        # Check convergence
        diff = price - market_price
        if abs(diff) < tolerance:
            return sigma
            
        # Newton-Raphson update
        if vega > 1e-10:
            sigma = sigma - diff / vega
            sigma = max(sigma, 0.001)  # Keep sigma positive
        else:
            break
            
    return sigma
