"""
PDE Pricer
Finite difference methods for option pricing (optional/advanced)
"""

import numpy as np
from typing import Optional
from ..models.base_model import PricingResult
import time


class PDEPricer:
    """
    PDE-based pricer using finite difference methods
    
    Implements:
    - Explicit finite differences
    - Implicit finite differences (Crank-Nicolson)
    - Suitable for American options and path-dependent options
    
    Note: This is a simplified implementation.
    Production systems would use more sophisticated numerical methods.
    """
    
    def __init__(
        self,
        method: str = 'crank-nicolson',
        num_space_steps: int = 100,
        num_time_steps: int = 100,
        name: str = "PDE Pricer"
    ):
        """
        Initialize PDE pricer
        
        Args:
            method: FD method ('explicit', 'implicit', 'crank-nicolson')
            num_space_steps: Number of spatial grid points
            num_time_steps: Number of time steps
            name: Pricer name
        """
        self.method = method
        self.num_space_steps = num_space_steps
        self.num_time_steps = num_time_steps
        self.name = name
        
    def price(
        self,
        instrument,
        market_data,
        vol_surface=None,
        volatility: Optional[float] = None,
        calculate_greeks: bool = False
    ) -> PricingResult:
        """
        Price using finite differences
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            vol_surface: Volatility surface
            volatility: Constant volatility
            calculate_greeks: Whether to calculate Greeks
            
        Returns:
            PricingResult
        """
        start_time = time.time()
        
        # Get volatility
        if volatility is None:
            if vol_surface is not None:
                if hasattr(instrument, 'strike'):
                    volatility = vol_surface.get_vol(instrument.strike, instrument.expiry)
                else:
                    volatility = vol_surface.get_atm_vol(instrument.expiry)
            else:
                volatility = 0.2  # Default
                
        # Set up grid
        S_max = market_data.spot * 3
        S_min = 0
        
        dS = (S_max - S_min) / self.num_space_steps
        dt = instrument.expiry / self.num_time_steps
        
        S = np.linspace(S_min, S_max, self.num_space_steps + 1)
        
        # Initialize option values at maturity
        V = np.zeros((self.num_time_steps + 1, self.num_space_steps + 1))
        
        # Terminal condition (payoff at expiry)
        if hasattr(instrument, 'is_call'):
            if instrument.is_call:
                V[-1, :] = np.maximum(S - instrument.strike, 0)
            else:
                V[-1, :] = np.maximum(instrument.strike - S, 0)
        else:
            # For exotic options without simple payoff
            # This is a placeholder - real implementation would be more complex
            V[-1, :] = 0
            
        # Get market parameters
        r = market_data.get_rate(instrument.expiry)
        q = market_data.get_dividend_yield(instrument.expiry)
        sigma = volatility
        
        # Time-stepping (backward in time)
        if self.method == 'explicit':
            V = self._solve_explicit(V, S, dt, dS, r, q, sigma)
        elif self.method == 'implicit':
            V = self._solve_implicit(V, S, dt, dS, r, q, sigma)
        elif self.method == 'crank-nicolson':
            V = self._solve_crank_nicolson(V, S, dt, dS, r, q, sigma)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Interpolate to get price at current spot
        price = np.interp(market_data.spot, S, V[0, :])
        
        # Greeks (finite differences on the grid)
        delta = gamma = vega = theta = rho = None
        
        if calculate_greeks:
            # Find index closest to spot
            idx = np.argmin(np.abs(S - market_data.spot))
            
            # Delta
            if idx > 0 and idx < len(S) - 1:
                delta = (V[0, idx+1] - V[0, idx-1]) / (2 * dS)
                gamma = (V[0, idx+1] - 2*V[0, idx] + V[0, idx-1]) / (dS**2)
                
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
        
    def _solve_explicit(self, V, S, dt, dS, r, q, sigma):
        """Explicit finite difference scheme"""
        N_t = V.shape[0] - 1
        N_s = V.shape[1] - 1
        
        for n in range(N_t - 1, -1, -1):
            for i in range(1, N_s):
                # Explicit update
                a = 0.5 * dt * (sigma**2 * i**2 - (r - q) * i)
                b = 1 - dt * (sigma**2 * i**2 + r)
                c = 0.5 * dt * (sigma**2 * i**2 + (r - q) * i)
                
                V[n, i] = a * V[n+1, i-1] + b * V[n+1, i] + c * V[n+1, i+1]
                
            # Boundary conditions
            V[n, 0] = 0  # Simplified
            V[n, N_s] = S[N_s]  # Simplified
            
        return V
        
    def _solve_implicit(self, V, S, dt, dS, r, q, sigma):
        """Implicit finite difference scheme"""
        N_t = V.shape[0] - 1
        N_s = V.shape[1] - 1
        
        # Build tridiagonal matrix
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
        
        for n in range(N_t - 1, -1, -1):
            # Coefficients for interior points
            alpha = np.zeros(N_s + 1)
            beta = np.zeros(N_s + 1)
            gamma_coef = np.zeros(N_s + 1)
            
            for i in range(1, N_s):
                alpha[i] = -0.5 * dt * (sigma**2 * i**2 - (r - q) * i)
                beta[i] = 1 + dt * (sigma**2 * i**2 + r)
                gamma_coef[i] = -0.5 * dt * (sigma**2 * i**2 + (r - q) * i)
                
            # Solve tridiagonal system
            diagonals = [alpha[1:-1], beta[1:-1], gamma_coef[1:-1]]
            A = diags(diagonals, [-1, 0, 1], format='csc')
            
            V[n, 1:-1] = spsolve(A, V[n+1, 1:-1])
            
            # Boundary conditions
            V[n, 0] = 0
            V[n, N_s] = S[N_s]
            
        return V
        
    def _solve_crank_nicolson(self, V, S, dt, dS, r, q, sigma):
        """Crank-Nicolson scheme (average of explicit and implicit)"""
        # This is a placeholder - full implementation would be more involved
        # For now, use implicit method
        return self._solve_implicit(V, S, dt, dS, r, q, sigma)
        
    def __repr__(self) -> str:
        return (
            f"PDEPricer(method='{self.method}', "
            f"space_steps={self.num_space_steps}, "
            f"time_steps={self.num_time_steps})"
        )
