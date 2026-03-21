"""
Base Pricing Model
Abstract base class for all pricing models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PricingResult:
    """Container for pricing results"""
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    
    # Additional metrics
    std_error: Optional[float] = None  # For MC methods
    confidence_interval: Optional[tuple] = None
    computation_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'price': self.price,
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho,
            'std_error': self.std_error,
            'confidence_interval': self.confidence_interval,
            'computation_time': self.computation_time
        }
        
    def __repr__(self) -> str:
        greeks = []
        if self.delta is not None:
            greeks.append(f"Δ={self.delta:.4f}")
        if self.gamma is not None:
            greeks.append(f"Γ={self.gamma:.4f}")
        if self.vega is not None:
            greeks.append(f"ν={self.vega:.4f}")
            
        greek_str = ", ".join(greeks) if greeks else "No Greeks"
        return f"PricingResult(price={self.price:.4f}, {greek_str})"


class BasePricingModel(ABC):
    """
    Abstract base class for pricing models
    
    All pricing models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str = "BaseModel"):
        """
        Initialize the pricing model
        
        Args:
            name: Model name for identification
        """
        self.name = name
        self._parameters = {}
        
    @abstractmethod
    def price(self, instrument, market_data, **kwargs) -> PricingResult:
        """
        Price an instrument given market data
        
        Args:
            instrument: The instrument to price
            market_data: Market data container
            **kwargs: Additional pricing parameters
            
        Returns:
            PricingResult with price and Greeks
        """
        pass
        
    @abstractmethod
    def calibrate(self, market_prices, instruments, market_data, **kwargs):
        """
        Calibrate model parameters to market prices
        
        Args:
            market_prices: Observed market prices
            instruments: Corresponding instruments
            market_data: Market data container
            **kwargs: Calibration options
            
        Returns:
            Calibration results
        """
        pass
        
    def set_parameter(self, name: str, value: Any):
        """Set a model parameter"""
        self._parameters[name] = value
        
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a model parameter"""
        return self._parameters.get(name, default)
        
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return self._parameters.copy()
        
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self._parameters = {}
        
    def validate_inputs(self, instrument, market_data) -> bool:
        """
        Validate inputs before pricing
        
        Args:
            instrument: Instrument to validate
            market_data: Market data to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if market_data.spot <= 0:
            raise ValueError("Spot price must be positive")
            
        if hasattr(instrument, 'expiry') and instrument.expiry <= 0:
            raise ValueError("Expiry must be positive")
            
        return True
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ModelConfig:
    """Configuration container for pricing models"""
    
    def __init__(self):
        self.random_seed: Optional[int] = None
        self.num_paths: int = 10000
        self.num_steps: int = 252
        self.use_antithetic: bool = True
        self.use_control_variates: bool = False
        self.parallel: bool = False
        self.num_workers: Optional[int] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'random_seed': self.random_seed,
            'num_paths': self.num_paths,
            'num_steps': self.num_steps,
            'use_antithetic': self.use_antithetic,
            'use_control_variates': self.use_control_variates,
            'parallel': self.parallel,
            'num_workers': self.num_workers
        }
