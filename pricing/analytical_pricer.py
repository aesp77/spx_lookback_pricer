"""
Analytical Pricer
Uses closed-form formulas where available
"""

import numpy as np
from typing import Optional, Dict, Any
from ..models.base_model import PricingResult
import time


class AnalyticalPricer:
    """
    Analytical pricing engine using closed-form formulas
    
    Supports:
    - Vanilla European options (Black-Scholes)
    - Fixed strike lookback options
    - Floating strike lookback options
    """
    
    def __init__(self, name: str = "Analytical Pricer"):
        """Initialize analytical pricer"""
        self.name = name
        
    def price(
        self,
        instrument,
        market_data,
        vol_surface=None,
        volatility: Optional[float] = None,
        calculate_greeks: bool = True
    ) -> PricingResult:
        """
        Price an instrument using analytical formulas
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            vol_surface: Volatility surface (optional)
            volatility: Constant volatility (if no vol_surface)
            calculate_greeks: Whether to calculate Greeks
            
        Returns:
            PricingResult
        """
        start_time = time.time()
        
        # Get volatility
        if vol_surface is not None:
            if hasattr(instrument, 'strike'):
                vol = vol_surface.get_vol(instrument.strike, instrument.expiry)
            else:
                vol = vol_surface.get_atm_vol(instrument.expiry)
        elif volatility is not None:
            vol = volatility
        else:
            raise ValueError("Must provide either vol_surface or volatility")
            
        # Price using instrument's analytical formula
        if hasattr(instrument, 'analytical_price'):
            price = instrument.analytical_price(market_data, vol)
        else:
            raise ValueError(
                f"No analytical formula available for {type(instrument).__name__}. "
                "Use Monte Carlo pricing instead."
            )
            
        # Calculate Greeks if requested and available
        delta = gamma = vega = theta = rho = None
        
        if calculate_greeks:
            if hasattr(instrument, 'delta'):
                delta = instrument.delta(market_data, vol)
            if hasattr(instrument, 'gamma'):
                gamma = instrument.gamma(market_data, vol)
            if hasattr(instrument, 'vega'):
                vega = instrument.vega(market_data, vol)
            if hasattr(instrument, 'theta'):
                theta = instrument.theta(market_data, vol)
            if hasattr(instrument, 'rho'):
                rho = instrument.rho(market_data, vol)
                
        computation_time = time.time() - start_time
        
        return PricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            computation_time=computation_time
        )
        
    def price_portfolio(
        self,
        instruments: list,
        market_data,
        vol_surface=None,
        volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Price a portfolio of instruments
        
        Args:
            instruments: List of instruments
            market_data: Market data
            vol_surface: Volatility surface
            volatility: Constant volatility
            
        Returns:
            Dictionary with portfolio analytics
        """
        results = []
        total_value = 0
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        
        for inst in instruments:
            result = self.price(
                inst,
                market_data,
                vol_surface=vol_surface,
                volatility=volatility
            )
            
            results.append({
                'instrument': str(inst),
                'price': result.price,
                'delta': result.delta,
                'gamma': result.gamma,
                'vega': result.vega
            })
            
            total_value += result.price
            if result.delta is not None:
                total_delta += result.delta
            if result.gamma is not None:
                total_gamma += result.gamma
            if result.vega is not None:
                total_vega += result.vega
                
        return {
            'instruments': results,
            'portfolio_value': total_value,
            'portfolio_delta': total_delta,
            'portfolio_gamma': total_gamma,
            'portfolio_vega': total_vega
        }
        
    def implied_volatility(
        self,
        instrument,
        market_data,
        market_price: float,
        initial_guess: float = 0.2
    ) -> float:
        """
        Calculate implied volatility from market price
        
        Args:
            instrument: Instrument
            market_data: Market data
            market_price: Observed market price
            initial_guess: Initial volatility guess
            
        Returns:
            Implied volatility
        """
        from scipy.optimize import brentq
        
        def price_diff(vol):
            if not hasattr(instrument, 'analytical_price'):
                raise ValueError("Instrument does not support analytical pricing")
            return instrument.analytical_price(market_data, vol) - market_price
            
        try:
            iv = brentq(price_diff, 0.001, 5.0)
            return iv
        except:
            return initial_guess
            
    def __repr__(self) -> str:
        return f"AnalyticalPricer(name='{self.name}')"
