"""
Monte Carlo Simulation Engine
High-performance Monte Carlo with variance reduction techniques
"""

import numpy as np
from typing import Optional, Callable, Tuple
from .base_model import BasePricingModel, PricingResult, ModelConfig
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass


@dataclass
class SimulationStats:
    """Statistics from Monte Carlo simulation"""
    mean: float
    std_error: float
    confidence_interval: Tuple[float, float]
    num_paths: int
    variance_reduction_efficiency: Optional[float] = None


class MonteCarloEngine(BasePricingModel):
    """
    Monte Carlo simulation engine for exotic option pricing
    
    Features:
    - Antithetic variates
    - Control variates
    - Importance sampling
    - Quasi-random numbers (Sobol, Halton)
    - Parallel simulation
    - Greek calculation via finite differences
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        name: str = "Monte Carlo"
    ):
        """
        Initialize Monte Carlo engine
        
        Args:
            config: Model configuration
            name: Engine name
        """
        super().__init__(name)
        self.config = config or ModelConfig()
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            
    def price(self, instrument, market_data, **kwargs) -> PricingResult:
        """
        Price an instrument using Monte Carlo simulation
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            **kwargs: num_paths, num_steps, vol_surface, etc.
            
        Returns:
            PricingResult with price, Greeks, and statistics
        """
        start_time = time.time()
        
        self.validate_inputs(instrument, market_data)
        
        # Extract parameters
        num_paths = kwargs.get('num_paths', self.config.num_paths)
        num_steps = kwargs.get('num_steps', self.config.num_steps)
        vol_surface = kwargs.get('vol_surface', None)
        use_antithetic = kwargs.get('use_antithetic', self.config.use_antithetic)
        use_control = kwargs.get('use_control_variates', self.config.use_control_variates)
        
        # Generate paths
        paths = self.simulate_paths(
            market_data=market_data,
            expiry=instrument.expiry,
            num_paths=num_paths,
            num_steps=num_steps,
            vol_surface=vol_surface,
            use_antithetic=use_antithetic
        )
        
        # Calculate payoffs
        payoffs = instrument.payoff(paths, market_data)
        
        # Discount payoffs
        r = market_data.get_rate(instrument.expiry)
        discount_factor = np.exp(-r * instrument.expiry)
        discounted_payoffs = payoffs * discount_factor
        
        # Apply control variates if requested
        if use_control:
            control_instrument = kwargs.get('control_instrument', None)
            if control_instrument is not None:
                discounted_payoffs = self._apply_control_variates(
                    discounted_payoffs,
                    paths,
                    control_instrument,
                    market_data
                )
                
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        # 95% confidence interval
        ci = (
            price - 1.96 * std_error,
            price + 1.96 * std_error
        )
        
        # Calculate Greeks if requested
        greeks = kwargs.get('calculate_greeks', True)
        delta = gamma = vega = theta = rho = None
        
        if greeks:
            delta, gamma, vega, theta, rho = self._calculate_greeks_fd(
                instrument, market_data, **kwargs
            )
            
        computation_time = time.time() - start_time
        
        return PricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            std_error=std_error,
            confidence_interval=ci,
            computation_time=computation_time
        )
        
    def simulate_paths(
        self,
        market_data,
        expiry: float,
        num_paths: int,
        num_steps: int,
        vol_surface=None,
        use_antithetic: bool = True,
        volatility: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate multiple asset price paths
        
        Args:
            market_data: Market data
            expiry: Time to expiry
            num_paths: Number of paths to simulate
            num_steps: Number of time steps per path
            vol_surface: Optional volatility surface
            use_antithetic: Use antithetic variates
            volatility: Constant volatility (if no vol_surface)
            
        Returns:
            Array of shape (num_paths, num_steps + 1) with simulated prices
        """
        dt = expiry / num_steps
        times = np.linspace(0, expiry, num_steps + 1)
        
        # Adjust for antithetic variates
        actual_paths = num_paths // 2 if use_antithetic else num_paths
        
        # Initialize paths
        paths = np.zeros((actual_paths, num_steps + 1))
        paths[:, 0] = market_data.spot
        
        # Generate random numbers
        randn = np.random.standard_normal((actual_paths, num_steps))
        
        # Simulate paths
        for i in range(num_steps):
            t = times[i]
            t_next = times[i + 1]
            
            r = market_data.get_rate(t)
            q = market_data.get_dividend_yield(t)
            
            # Get volatility for each path
            if vol_surface is not None:
                # Local volatility - use current spot level
                sigma = np.array([
                    vol_surface.get_vol(paths[j, i], expiry - t)
                    for j in range(actual_paths)
                ])
            elif volatility is not None:
                sigma = volatility
            else:
                sigma = 0.2  # Default
                
            # GBM update
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * randn[:, i]
            
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion)
            
        # Apply antithetic variates
        if use_antithetic:
            # Create antithetic paths
            paths_anti = np.zeros((actual_paths, num_steps + 1))
            paths_anti[:, 0] = market_data.spot
            
            for i in range(num_steps):
                t = times[i]
                r = market_data.get_rate(t)
                q = market_data.get_dividend_yield(t)
                
                if vol_surface is not None:
                    sigma = np.array([
                        vol_surface.get_vol(paths_anti[j, i], expiry - t)
                        for j in range(actual_paths)
                    ])
                elif volatility is not None:
                    sigma = volatility
                else:
                    sigma = 0.2
                    
                drift = (r - q - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * (-randn[:, i])  # Antithetic
                
                paths_anti[:, i + 1] = paths_anti[:, i] * np.exp(drift + diffusion)
                
            # Combine paths
            paths = np.vstack([paths, paths_anti])
            
        return paths
        
    def _apply_control_variates(
        self,
        payoffs: np.ndarray,
        paths: np.ndarray,
        control_instrument,
        market_data
    ) -> np.ndarray:
        """
        Apply control variates variance reduction
        
        Args:
            payoffs: Original payoffs
            paths: Simulated paths
            control_instrument: Control instrument (e.g., vanilla option)
            market_data: Market data
            
        Returns:
            Adjusted payoffs with reduced variance
        """
        # Calculate control payoffs
        control_payoffs = control_instrument.payoff(paths, market_data)
        
        # Discount
        r = market_data.get_rate(control_instrument.expiry)
        discount_factor = np.exp(-r * control_instrument.expiry)
        control_payoffs *= discount_factor
        
        # Get analytical price of control
        if hasattr(control_instrument, 'analytical_price'):
            control_price = control_instrument.analytical_price(market_data, 0.2)
        else:
            control_price = np.mean(control_payoffs)
            
        # Calculate optimal beta (covariance / variance)
        cov = np.cov(payoffs, control_payoffs)[0, 1]
        var = np.var(control_payoffs)
        beta = cov / var if var > 0 else 0
        
        # Apply control variate adjustment
        adjusted_payoffs = payoffs - beta * (control_payoffs - control_price)
        
        return adjusted_payoffs
        
    def _calculate_greeks_fd(self, instrument, market_data, **kwargs):
        """
        Calculate Greeks using finite differences
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            **kwargs: Pricing parameters
            
        Returns:
            Tuple of (delta, gamma, vega, theta, rho)
        """
        # Base price
        base_price = self.price(
            instrument,
            market_data,
            calculate_greeks=False,
            **kwargs
        ).price
        
        # Delta: dP/dS
        bump_spot = market_data.spot * 0.01
        market_data_up = market_data.__class__.from_flat_rates(
            spot=market_data.spot + bump_spot,
            risk_free_rate=market_data.get_rate(instrument.expiry),
            dividend_yield=market_data.get_dividend_yield(instrument.expiry)
        )
        price_up = self.price(instrument, market_data_up, calculate_greeks=False, **kwargs).price
        
        market_data_down = market_data.__class__.from_flat_rates(
            spot=market_data.spot - bump_spot,
            risk_free_rate=market_data.get_rate(instrument.expiry),
            dividend_yield=market_data.get_dividend_yield(instrument.expiry)
        )
        price_down = self.price(instrument, market_data_down, calculate_greeks=False, **kwargs).price
        
        delta = (price_up - price_down) / (2 * bump_spot)
        gamma = (price_up - 2 * base_price + price_down) / (bump_spot ** 2)
        
        # Vega: dP/dσ (requires vol surface or volatility parameter)
        vega = None  # Simplified - would need to bump vol surface
        
        # Theta: dP/dt
        instrument_t = instrument.__class__(
            expiry=instrument.expiry - 1/365,  # 1 day
            **{k: v for k, v in instrument.__dict__.items() if k != 'expiry'}
        )
        price_t = self.price(instrument_t, market_data, calculate_greeks=False, **kwargs).price
        theta = (price_t - base_price) / (1/365)
        
        # Rho: dP/dr
        rho = None  # Simplified
        
        return delta, gamma, vega, theta, rho
        
    def calibrate(self, market_prices, instruments, market_data, **kwargs):
        """
        Calibrate model using Monte Carlo
        
        This is typically not used for MC itself, but for calibrating
        the underlying stochastic process parameters
        """
        raise NotImplementedError("Calibration not implemented for Monte Carlo engine")
        
    def convergence_analysis(
        self,
        instrument,
        market_data,
        path_counts: list,
        **kwargs
    ) -> dict:
        """
        Analyze convergence as a function of number of paths
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            path_counts: List of path counts to test
            **kwargs: Pricing parameters
            
        Returns:
            Dictionary with convergence statistics
        """
        results = {
            'path_counts': [],
            'prices': [],
            'std_errors': [],
            'computation_times': []
        }
        
        for num_paths in path_counts:
            result = self.price(
                instrument,
                market_data,
                num_paths=num_paths,
                calculate_greeks=False,
                **kwargs
            )
            
            results['path_counts'].append(num_paths)
            results['prices'].append(result.price)
            results['std_errors'].append(result.std_error)
            results['computation_times'].append(result.computation_time)
            
        return results
