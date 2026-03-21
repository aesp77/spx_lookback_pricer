"""
Greek Calculator
Numerical calculation of option Greeks using finite differences
"""

import numpy as np
from typing import Callable, Dict, Optional
from dataclasses import dataclass


@dataclass
class Greeks:
    """Container for all Greeks"""
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    vanna: Optional[float] = None  # d(delta)/d(vol)
    volga: Optional[float] = None  # d(vega)/d(vol)
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho,
            'vanna': self.vanna,
            'volga': self.volga
        }


class GreekCalculator:
    """
    Calculate option Greeks using finite differences
    
    Supports first and second-order Greeks
    """
    
    def __init__(
        self,
        bump_spot: float = 0.01,
        bump_vol: float = 0.01,
        bump_rate: float = 0.0001,
        bump_time: float = 1/365
    ):
        """
        Initialize Greek calculator
        
        Args:
            bump_spot: Spot bump size (relative)
            bump_vol: Volatility bump size (absolute)
            bump_rate: Rate bump size (absolute)
            bump_time: Time bump size (in years, default 1 day)
        """
        self.bump_spot = bump_spot
        self.bump_vol = bump_vol
        self.bump_rate = bump_rate
        self.bump_time = bump_time
        
    def calculate_delta(
        self,
        pricing_func: Callable,
        spot: float,
        method: str = 'central'
    ) -> float:
        """
        Calculate delta using finite differences
        
        Args:
            pricing_func: Function that takes spot as input and returns price
            spot: Current spot price
            method: 'forward', 'backward', or 'central'
            
        Returns:
            Delta
        """
        bump = spot * self.bump_spot
        
        if method == 'central':
            price_up = pricing_func(spot + bump)
            price_down = pricing_func(spot - bump)
            delta = (price_up - price_down) / (2 * bump)
        elif method == 'forward':
            price = pricing_func(spot)
            price_up = pricing_func(spot + bump)
            delta = (price_up - price) / bump
        elif method == 'backward':
            price = pricing_func(spot)
            price_down = pricing_func(spot - bump)
            delta = (price - price_down) / bump
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return delta
        
    def calculate_gamma(
        self,
        pricing_func: Callable,
        spot: float
    ) -> float:
        """
        Calculate gamma using finite differences
        
        Args:
            pricing_func: Function that takes spot as input and returns price
            spot: Current spot price
            
        Returns:
            Gamma
        """
        bump = spot * self.bump_spot
        
        price = pricing_func(spot)
        price_up = pricing_func(spot + bump)
        price_down = pricing_func(spot - bump)
        
        gamma = (price_up - 2 * price + price_down) / (bump ** 2)
        
        return gamma
        
    def calculate_vega(
        self,
        pricing_func: Callable,
        volatility: float
    ) -> float:
        """
        Calculate vega using finite differences
        
        Args:
            pricing_func: Function that takes volatility as input and returns price
            volatility: Current volatility
            
        Returns:
            Vega (per 1% vol change)
        """
        price_up = pricing_func(volatility + self.bump_vol)
        price_down = pricing_func(volatility - self.bump_vol)
        
        vega = (price_up - price_down) / (2 * self.bump_vol)
        
        return vega / 100.0  # Per 1% change
        
    def calculate_theta(
        self,
        pricing_func: Callable,
        expiry: float
    ) -> float:
        """
        Calculate theta using finite differences
        
        Args:
            pricing_func: Function that takes expiry as input and returns price
            expiry: Current time to expiry
            
        Returns:
            Theta (per day)
        """
        if expiry <= self.bump_time:
            return 0.0
            
        price = pricing_func(expiry)
        price_down = pricing_func(expiry - self.bump_time)
        
        theta = (price_down - price) / self.bump_time
        
        return theta / 365.0  # Per day
        
    def calculate_rho(
        self,
        pricing_func: Callable,
        rate: float
    ) -> float:
        """
        Calculate rho using finite differences
        
        Args:
            pricing_func: Function that takes rate as input and returns price
            rate: Current risk-free rate
            
        Returns:
            Rho (per 1% rate change)
        """
        price_up = pricing_func(rate + self.bump_rate)
        price_down = pricing_func(rate - self.bump_rate)
        
        rho = (price_up - price_down) / (2 * self.bump_rate)
        
        return rho / 100.0  # Per 1% change
        
    def calculate_vanna(
        self,
        pricing_func: Callable,
        spot: float,
        volatility: float
    ) -> float:
        """
        Calculate vanna (d(delta)/d(vol))
        
        Args:
            pricing_func: Function that takes (spot, vol) and returns price
            spot: Current spot price
            volatility: Current volatility
            
        Returns:
            Vanna
        """
        bump_s = spot * self.bump_spot
        
        # Calculate delta at vol + bump
        def delta_up(s):
            price_up = pricing_func(s + bump_s, volatility + self.bump_vol)
            price_down = pricing_func(s - bump_s, volatility + self.bump_vol)
            return (price_up - price_down) / (2 * bump_s)
            
        # Calculate delta at vol - bump
        def delta_down(s):
            price_up = pricing_func(s + bump_s, volatility - self.bump_vol)
            price_down = pricing_func(s - bump_s, volatility - self.bump_vol)
            return (price_up - price_down) / (2 * bump_s)
            
        d_up = delta_up(spot)
        d_down = delta_down(spot)
        
        vanna = (d_up - d_down) / (2 * self.bump_vol)
        
        return vanna
        
    def calculate_volga(
        self,
        pricing_func: Callable,
        volatility: float
    ) -> float:
        """
        Calculate volga (d(vega)/d(vol))
        
        Args:
            pricing_func: Function that takes volatility and returns price
            volatility: Current volatility
            
        Returns:
            Volga
        """
        price = pricing_func(volatility)
        price_up = pricing_func(volatility + self.bump_vol)
        price_down = pricing_func(volatility - self.bump_vol)
        
        volga = (price_up - 2 * price + price_down) / (self.bump_vol ** 2)
        
        return volga
        
    def calculate_all_greeks(
        self,
        pricing_func: Callable,
        spot: float,
        volatility: float,
        rate: float,
        expiry: float,
        calculate_second_order: bool = False
    ) -> Greeks:
        """
        Calculate all Greeks
        
        Args:
            pricing_func: Function that takes (spot, vol, rate, expiry) and returns price
            spot: Current spot price
            volatility: Current volatility
            rate: Current risk-free rate
            expiry: Time to expiry
            calculate_second_order: Whether to calculate second-order Greeks
            
        Returns:
            Greeks object
        """
        # First-order Greeks
        delta = self.calculate_delta(
            lambda s: pricing_func(s, volatility, rate, expiry),
            spot
        )
        
        gamma = self.calculate_gamma(
            lambda s: pricing_func(s, volatility, rate, expiry),
            spot
        )
        
        vega = self.calculate_vega(
            lambda v: pricing_func(spot, v, rate, expiry),
            volatility
        )
        
        theta = self.calculate_theta(
            lambda t: pricing_func(spot, volatility, rate, t),
            expiry
        )
        
        rho = self.calculate_rho(
            lambda r: pricing_func(spot, volatility, r, expiry),
            rate
        )
        
        # Second-order Greeks
        vanna = volga = None
        if calculate_second_order:
            vanna = self.calculate_vanna(
                lambda s, v: pricing_func(s, v, rate, expiry),
                spot,
                volatility
            )
            
            volga = self.calculate_volga(
                lambda v: pricing_func(spot, v, rate, expiry),
                volatility
            )
            
        return Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            vanna=vanna,
            volga=volga
        )
