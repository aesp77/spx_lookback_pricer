"""
Black-Scholes Model
Classical Black-Scholes dynamics with extensions for local/stochastic volatility
"""

import numpy as np
from typing import Optional
from .base_model import BasePricingModel, PricingResult
import time


class BlackScholesModel(BasePricingModel):
    """
    Black-Scholes pricing model with optional extensions
    
    Supports:
    - Classical constant volatility
    - Time-dependent volatility
    - Local volatility (from vol surface)
    - Stochastic volatility (Heston-like)
    """
    
    def __init__(
        self,
        model_type: str = 'constant',
        name: str = "Black-Scholes"
    ):
        """
        Initialize Black-Scholes model
        
        Args:
            model_type: Type of model ('constant', 'local', 'stochastic')
            name: Model name
        """
        super().__init__(name)
        self.model_type = model_type
        
        # Stochastic volatility parameters (Heston)
        if model_type == 'stochastic':
            self.set_parameter('kappa', 2.0)      # Mean reversion speed
            self.set_parameter('theta', 0.04)     # Long-run variance
            self.set_parameter('sigma_v', 0.3)    # Vol of vol
            self.set_parameter('rho', -0.7)       # Correlation
            self.set_parameter('v0', 0.04)        # Initial variance
            
    def price(self, instrument, market_data, **kwargs) -> PricingResult:
        """
        Price an instrument using Black-Scholes
        
        Args:
            instrument: Instrument to price (must have analytical formula)
            market_data: Market data
            **kwargs: Additional options (vol_surface, num_simulations, etc.)
            
        Returns:
            PricingResult
        """
        start_time = time.time()
        
        self.validate_inputs(instrument, market_data)
        
        # Get volatility
        vol_surface = kwargs.get('vol_surface', None)
        if vol_surface is not None:
            # Use vol from surface
            vol = vol_surface.get_vol(instrument.strike, instrument.expiry)
        else:
            # Use constant vol from instrument or kwargs
            vol = kwargs.get('volatility', instrument.volatility if hasattr(instrument, 'volatility') else 0.2)
            
        # Price using analytical formula if available
        if hasattr(instrument, 'analytical_price'):
            price = instrument.analytical_price(market_data, vol)
        else:
            # Fall back to generic Black-Scholes
            from ..utils.math_utils import black_scholes_price
            price = black_scholes_price(
                S=market_data.spot,
                K=instrument.strike,
                T=instrument.expiry,
                r=market_data.get_rate(instrument.expiry),
                q=market_data.get_dividend_yield(instrument.expiry),
                sigma=vol,
                is_call=instrument.is_call
            )
            
        # Calculate Greeks if requested
        greeks = kwargs.get('calculate_greeks', True)
        delta = gamma = vega = theta = rho = None
        
        if greeks:
            delta, gamma, vega, theta, rho = self._calculate_greeks(
                instrument, market_data, vol, **kwargs
            )
            
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
        
    def _calculate_greeks(self, instrument, market_data, vol, **kwargs):
        """Calculate analytical Greeks for Black-Scholes"""
        from ..utils.math_utils import (
            black_scholes_delta,
            black_scholes_gamma,
            black_scholes_vega,
            black_scholes_theta,
            black_scholes_rho
        )
        
        S = market_data.spot
        K = instrument.strike
        T = instrument.expiry
        r = market_data.get_rate(T)
        q = market_data.get_dividend_yield(T)
        is_call = instrument.is_call
        
        delta = black_scholes_delta(S, K, T, r, q, vol, is_call)
        gamma = black_scholes_gamma(S, K, T, r, q, vol)
        vega = black_scholes_vega(S, K, T, r, q, vol)
        theta = black_scholes_theta(S, K, T, r, q, vol, is_call)
        rho = black_scholes_rho(S, K, T, r, q, vol, is_call)
        
        return delta, gamma, vega, theta, rho
        
    def calibrate(self, market_prices, instruments, market_data, **kwargs):
        """
        Calibrate Black-Scholes model to market prices
        
        For constant vol: calibrate single volatility
        For local vol: calibrate volatility surface
        For stochastic vol: calibrate Heston parameters
        
        Args:
            market_prices: Array of market prices
            instruments: List of instruments
            market_data: Market data
            **kwargs: Calibration options
            
        Returns:
            Calibration results dictionary
        """
        from scipy.optimize import minimize
        
        if self.model_type == 'constant':
            return self._calibrate_constant_vol(market_prices, instruments, market_data)
        elif self.model_type == 'local':
            return self._calibrate_local_vol(market_prices, instruments, market_data, **kwargs)
        elif self.model_type == 'stochastic':
            return self._calibrate_stochastic_vol(market_prices, instruments, market_data, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def _calibrate_constant_vol(self, market_prices, instruments, market_data):
        """Calibrate single constant volatility"""
        from scipy.optimize import minimize_scalar
        
        def objective(vol):
            """Sum of squared pricing errors"""
            total_error = 0
            for instrument, market_price in zip(instruments, market_prices):
                model_price = self.price(
                    instrument,
                    market_data,
                    volatility=vol,
                    calculate_greeks=False
                ).price
                total_error += (model_price - market_price) ** 2
            return total_error
            
        result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
        
        calibrated_vol = result.x
        self.set_parameter('volatility', calibrated_vol)
        
        return {
            'volatility': calibrated_vol,
            'rmse': np.sqrt(result.fun / len(market_prices)),
            'success': result.success
        }
        
    def _calibrate_local_vol(self, market_prices, instruments, market_data, **kwargs):
        """Calibrate local volatility surface"""
        # This is a simplified implementation
        # In practice, you'd use Dupire's formula or other techniques
        
        vol_surface = kwargs.get('vol_surface')
        if vol_surface is None:
            raise ValueError("vol_surface required for local vol calibration")
            
        # Implied volatility calculation for each instrument
        from scipy.optimize import brentq
        
        calibrated_vols = []
        for instrument, market_price in zip(instruments, market_prices):
            def price_diff(vol):
                model_price = self.price(
                    instrument,
                    market_data,
                    volatility=vol,
                    calculate_greeks=False
                ).price
                return model_price - market_price
                
            try:
                implied_vol = brentq(price_diff, 0.01, 2.0)
                calibrated_vols.append(implied_vol)
                
                # Add to vol surface
                vol_surface.add_point(
                    strike=instrument.strike,
                    expiry=instrument.expiry,
                    vol=implied_vol
                )
            except:
                calibrated_vols.append(np.nan)
                
        # Fit the vol surface
        vol_surface.fit()
        
        return {
            'implied_vols': np.array(calibrated_vols),
            'vol_surface': vol_surface,
            'success': True
        }
        
    def _calibrate_stochastic_vol(self, market_prices, instruments, market_data, **kwargs):
        """Calibrate Heston stochastic volatility parameters"""
        from scipy.optimize import minimize
        
        # Initial guess for Heston parameters
        initial_params = np.array([
            self.get_parameter('kappa', 2.0),
            self.get_parameter('theta', 0.04),
            self.get_parameter('sigma_v', 0.3),
            self.get_parameter('rho', -0.7),
            self.get_parameter('v0', 0.04)
        ])
        
        def objective(params):
            """Sum of squared pricing errors"""
            kappa, theta, sigma_v, rho, v0 = params
            
            # Update parameters
            self.set_parameter('kappa', kappa)
            self.set_parameter('theta', theta)
            self.set_parameter('sigma_v', sigma_v)
            self.set_parameter('rho', rho)
            self.set_parameter('v0', v0)
            
            total_error = 0
            for instrument, market_price in zip(instruments, market_prices):
                # Price using Heston (would need implementation)
                model_price = self._heston_price(instrument, market_data)
                total_error += (model_price - market_price) ** 2
                
            return total_error
            
        # Bounds for parameters
        bounds = [
            (0.1, 10.0),    # kappa
            (0.01, 1.0),    # theta
            (0.01, 2.0),    # sigma_v
            (-0.99, 0.99),  # rho
            (0.01, 1.0)     # v0
        ]
        
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Update calibrated parameters
        kappa, theta, sigma_v, rho, v0 = result.x
        self.set_parameter('kappa', kappa)
        self.set_parameter('theta', theta)
        self.set_parameter('sigma_v', sigma_v)
        self.set_parameter('rho', rho)
        self.set_parameter('v0', v0)
        
        return {
            'parameters': {
                'kappa': kappa,
                'theta': theta,
                'sigma_v': sigma_v,
                'rho': rho,
                'v0': v0
            },
            'rmse': np.sqrt(result.fun / len(market_prices)),
            'success': result.success
        }
        
    def _heston_price(self, instrument, market_data):
        """
        Price using Heston model (semi-analytical)
        This is a placeholder - full implementation would use characteristic function
        """
        # For now, fall back to Black-Scholes with adjusted vol
        vol = np.sqrt(self.get_parameter('v0'))
        return self.price(
            instrument,
            market_data,
            volatility=vol,
            calculate_greeks=False
        ).price
        
    def simulate_path(
        self,
        market_data,
        expiry: float,
        num_steps: int,
        vol_surface=None,
        **kwargs
    ) -> np.ndarray:
        """
        Simulate a single asset price path
        
        Args:
            market_data: Market data
            expiry: Time to expiry
            num_steps: Number of time steps
            vol_surface: Optional volatility surface for local vol
            **kwargs: Additional parameters
            
        Returns:
            Array of prices [S0, S1, ..., ST]
        """
        dt = expiry / num_steps
        times = np.linspace(0, expiry, num_steps + 1)
        
        S = np.zeros(num_steps + 1)
        S[0] = market_data.spot
        
        if self.model_type == 'stochastic':
            # Heston dynamics
            V = np.zeros(num_steps + 1)
            V[0] = self.get_parameter('v0')
            
            kappa = self.get_parameter('kappa')
            theta = self.get_parameter('theta')
            sigma_v = self.get_parameter('sigma_v')
            rho = self.get_parameter('rho')
            
            for i in range(num_steps):
                t = times[i]
                r = market_data.get_rate(t)
                q = market_data.get_dividend_yield(t)
                
                # Correlated random numbers
                z1 = np.random.standard_normal()
                z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal()
                
                # Update variance (with Feller condition)
                V[i+1] = np.maximum(
                    V[i] + kappa * (theta - V[i]) * dt + sigma_v * np.sqrt(V[i]) * np.sqrt(dt) * z2,
                    0.0
                )
                
                # Update price
                S[i+1] = S[i] * np.exp(
                    (r - q - 0.5 * V[i]) * dt + np.sqrt(V[i]) * np.sqrt(dt) * z1
                )
        else:
            # Standard or local vol
            for i in range(num_steps):
                t = times[i]
                r = market_data.get_rate(t)
                q = market_data.get_dividend_yield(t)
                
                # Get volatility
                if vol_surface is not None:
                    sigma = vol_surface.get_vol(S[i], expiry - t)
                else:
                    sigma = kwargs.get('volatility', 0.2)
                    
                z = np.random.standard_normal()
                S[i+1] = S[i] * np.exp(
                    (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
                )
                
        return S
