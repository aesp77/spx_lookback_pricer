"""
Monte Carlo Pricer
High-level interface for Monte Carlo pricing
"""

import numpy as np
from typing import Optional, Dict, Any
from ..models.monte_carlo import MonteCarloEngine, ModelConfig
from ..models.base_model import PricingResult


class MonteCarloPricer:
    """
    Monte Carlo pricing engine wrapper
    
    Provides simplified interface to Monte Carlo engine with
    common variance reduction techniques and Greeks calculation
    """
    
    def __init__(
        self,
        num_paths: int = 10000,
        num_steps: int = 252,
        use_antithetic: bool = True,
        use_control_variates: bool = False,
        random_seed: Optional[int] = None,
        name: str = "MC Pricer"
    ):
        """
        Initialize Monte Carlo pricer
        
        Args:
            num_paths: Number of simulation paths
            num_steps: Number of time steps per path
            use_antithetic: Use antithetic variates
            use_control_variates: Use control variates
            random_seed: Random seed for reproducibility
            name: Pricer name
        """
        self.name = name
        
        # Create config
        config = ModelConfig()
        config.num_paths = num_paths
        config.num_steps = num_steps
        config.use_antithetic = use_antithetic
        config.use_control_variates = use_control_variates
        config.random_seed = random_seed
        
        # Create engine
        self.engine = MonteCarloEngine(config=config)
        
    def price(
        self,
        instrument,
        market_data,
        vol_surface=None,
        volatility: Optional[float] = None,
        calculate_greeks: bool = True,
        control_instrument=None
    ) -> PricingResult:
        """
        Price an instrument using Monte Carlo
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            vol_surface: Volatility surface
            volatility: Constant volatility
            calculate_greeks: Calculate Greeks via finite differences
            control_instrument: Control variate instrument
            
        Returns:
            PricingResult
        """
        return self.engine.price(
            instrument=instrument,
            market_data=market_data,
            vol_surface=vol_surface,
            volatility=volatility,
            calculate_greeks=calculate_greeks,
            control_instrument=control_instrument
        )
        
    def price_with_confidence(
        self,
        instrument,
        market_data,
        confidence_level: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Price with confidence interval
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            confidence_level: Confidence level (default 95%)
            **kwargs: Additional pricing parameters
            
        Returns:
            Dictionary with price, CI, and statistics
        """
        result = self.price(
            instrument,
            market_data,
            calculate_greeks=False,
            **kwargs
        )
        
        # Calculate confidence interval
        z_score = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }.get(confidence_level, 1.96)
        
        ci_lower = result.price - z_score * result.std_error
        ci_upper = result.price + z_score * result.std_error
        
        return {
            'price': result.price,
            'std_error': result.std_error,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence_level,
            'computation_time': result.computation_time
        }
        
    def convergence_study(
        self,
        instrument,
        market_data,
        path_counts: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Study convergence as function of number of paths
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            path_counts: List of path counts to test
            **kwargs: Additional pricing parameters
            
        Returns:
            Dictionary with convergence data
        """
        if path_counts is None:
            path_counts = [1000, 5000, 10000, 50000, 100000]
            
        return self.engine.convergence_analysis(
            instrument=instrument,
            market_data=market_data,
            path_counts=path_counts,
            **kwargs
        )
        
    def sensitivity_analysis(
        self,
        instrument,
        market_data,
        parameter: str,
        values: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run sensitivity analysis for a parameter
        
        Args:
            instrument: Instrument to price
            market_data: Market data
            parameter: Parameter to vary ('spot', 'volatility', 'rate')
            values: Array of parameter values
            **kwargs: Additional pricing parameters
            
        Returns:
            Dictionary with sensitivity results
        """
        results = {
            'parameter': parameter,
            'values': values.tolist(),
            'prices': [],
            'std_errors': []
        }
        
        for value in values:
            # Create modified market data or kwargs
            if parameter == 'spot':
                md = market_data.__class__.from_flat_rates(
                    spot=value,
                    risk_free_rate=market_data.get_rate(instrument.expiry),
                    dividend_yield=market_data.get_dividend_yield(instrument.expiry)
                )
            elif parameter == 'rate':
                md = market_data.__class__.from_flat_rates(
                    spot=market_data.spot,
                    risk_free_rate=value,
                    dividend_yield=market_data.get_dividend_yield(instrument.expiry)
                )
            elif parameter == 'volatility':
                md = market_data
                kwargs['volatility'] = value
            else:
                raise ValueError(f"Unknown parameter: {parameter}")
                
            result = self.price(instrument, md, calculate_greeks=False, **kwargs)
            results['prices'].append(result.price)
            results['std_errors'].append(result.std_error)
            
        return results
        
    def __repr__(self) -> str:
        return (
            f"MonteCarloPricer(paths={self.engine.config.num_paths}, "
            f"steps={self.engine.config.num_steps})"
        )
