# data/vol_surface.py
"""
Volatility surface management and interpolation for SPX options.
Handles both raw implied vol data and SSVI parameterization.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
from typing import Optional, Tuple, Dict, Any, Union
from datetime import datetime, timedelta
import sqlite3

class VolatilitySurface:
    """
    Manages SPX implied volatility surface with multiple interpolation methods.
    Supports both discrete vol points and SSVI parameterization.
    """
    
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            from pathlib import Path
            db_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "db" / "spx_lookback_pricer.db")
        self.db_path = db_path
        self.current_date = None
        self.vol_data = None
        self.ssvi_params = None
        self.interpolator = None

    def _has_table(self, conn, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def load_surface(self, valuation_date: str) -> bool:
        """
        Load volatility surface for a specific date from database.
        Reads from standard schema (vol_surfaces) with fallback to legacy (spx_vol_surface).

        Args:
            valuation_date: Date in 'YYYY-MM-DD' format

        Returns:
            True if data loaded successfully
        """
        self.current_date = valuation_date

        with sqlite3.connect(self.db_path) as conn:
            use_standard = self._has_table(conn, "vol_surfaces")

            # Load vol surface points
            if use_standard:
                self.vol_data = pd.read_sql_query(
                    "SELECT strike, expiry AS tenor, iv AS implied_vol "
                    "FROM vol_surfaces WHERE symbol = 'SPX' AND date = ?",
                    conn,
                    params=(valuation_date,),
                )
            else:
                self.vol_data = pd.read_sql_query(
                    "SELECT strike, tenor, implied_vol "
                    "FROM spx_vol_surface WHERE date = ?",
                    conn,
                    params=(valuation_date,),
                )

            # Load SSVI parameters
            if use_standard and self._has_table(conn, "ssvi_parameters"):
                ssvi_df = pd.read_sql_query(
                    "SELECT theta, rho, beta FROM ssvi_parameters "
                    "WHERE symbol = 'SPX' AND date = ?",
                    conn,
                    params=(valuation_date,),
                )
            else:
                ssvi_df = pd.read_sql_query(
                    "SELECT theta, rho, beta FROM spx_ssvi_parameters WHERE date = ?",
                    conn,
                    params=(valuation_date,),
                )
            
            if not ssvi_df.empty:
                self.ssvi_params = ssvi_df.iloc[0].to_dict()
            
        # Convert tenor strings to years
        if not self.vol_data.empty:
            self.vol_data['tenor_years'] = self.vol_data['tenor'].apply(self._tenor_to_years)
            self._build_interpolator()
            return True
        
        return False
    
    def get_vol(self, strike: float, tenor: Union[float, str], 
                spot: float = None, method: str = 'linear') -> float:
        """
        Get implied volatility for given strike and tenor.
        
        Args:
            strike: Absolute strike price or relative strike (if spot provided)
            tenor: Time to maturity in years or tenor string ('1m', '3m', etc.)
            spot: Current spot price (optional, for relative strikes)
            method: Interpolation method ('linear', 'cubic', 'ssvi', 'rbf')
            
        Returns:
            Implied volatility
        """
        # Convert tenor to years if string
        if isinstance(tenor, str):
            tenor = self._tenor_to_years(tenor)
        
        # Convert to relative strike if spot provided
        if spot is not None:
            strike = strike / spot
        
        if method == 'ssvi' and self.ssvi_params:
            return self._get_vol_ssvi(strike, tenor, spot)
        elif method == 'rbf':
            return self._get_vol_rbf(strike, tenor)
        elif method == 'cubic':
            return self._get_vol_cubic(strike, tenor)
        else:  # linear
            return self._get_vol_linear(strike, tenor)
    
    def get_forward_vol(self, strike: float, t1: float, t2: float, spot: float) -> float:
        """
        Calculate forward volatility between two dates.
        Used for pricing options that depend on future volatility.
        
        Args:
            strike: Strike price
            t1: Start time (years)
            t2: End time (years)
            spot: Current spot price
            
        Returns:
            Forward implied volatility
        """
        # Get total variances
        var1 = self.get_vol(strike, t1, spot) ** 2 * t1
        var2 = self.get_vol(strike, t2, spot) ** 2 * t2
        
        # Forward variance
        if t2 > t1:
            forward_var = (var2 - var1) / (t2 - t1)
            return np.sqrt(max(forward_var, 0.0001))  # Floor at 1% vol
        else:
            return self.get_vol(strike, t1, spot)
    
    def get_local_vol(self, strike: float, tenor: float, spot: float) -> float:
        """
        Calculate local volatility using Dupire formula.
        Important for path-dependent option pricing.
        
        Args:
            strike: Strike price
            tenor: Time to maturity
            spot: Current spot price
            
        Returns:
            Local volatility
        """
        # Finite differences for derivatives
        dk = 0.01 * strike
        dt = 1/365  # 1 day
        
        # Get implied vols
        sigma = self.get_vol(strike, tenor, spot)
        sigma_up = self.get_vol(strike + dk, tenor, spot)
        sigma_down = self.get_vol(strike - dk, tenor, spot)
        sigma_t_up = self.get_vol(strike, tenor + dt, spot)
        
        # Calculate derivatives
        dsigma_dk = (sigma_up - sigma_down) / (2 * dk)
        d2sigma_dk2 = (sigma_up - 2*sigma + sigma_down) / (dk**2)
        dsigma_dt = (sigma_t_up - sigma) / dt
        
        # Dupire formula (simplified)
        d1 = (np.log(spot/strike) + 0.5 * sigma**2 * tenor) / (sigma * np.sqrt(tenor))
        
        numerator = sigma**2 + 2*sigma*tenor*(dsigma_dt + 0.05*sigma)  # Including risk-free rate
        denominator = 1 + 2*d1*strike*np.sqrt(tenor)*dsigma_dk + (strike**2 * tenor / sigma) * d2sigma_dk2
        
        local_var = numerator / denominator
        return np.sqrt(max(local_var, 0.0001))
    
    def _get_vol_ssvi(self, relative_strike: float, tenor: float, spot: float) -> float:
        """
        Get implied vol using SSVI parameterization.
        SSVI: w(k,t) = θ_t/2 * (1 + ρ*φ(θ_t)*k + sqrt((φ(θ_t)*k + ρ)^2 + (1-ρ^2)))
        where φ(θ) = 1/sqrt(θ) * (1 - (1-exp(-θ))/θ)
        """
        if not self.ssvi_params:
            return self._get_vol_linear(relative_strike, tenor)
        
        theta = self.ssvi_params['theta']
        rho = self.ssvi_params['rho']
        beta = self.ssvi_params['beta']
        
        # Log-moneyness
        k = np.log(relative_strike)
        
        # SSVI phi function
        if theta > 1e-6:
            phi = (1/np.sqrt(theta)) * (1 - (1 - np.exp(-theta))/theta)
        else:
            phi = 1
        
        # Total implied variance
        total_var = (theta/2) * (1 + rho*phi*k + np.sqrt((phi*k + rho)**2 + (1 - rho**2)))
        
        # Scale by time
        total_var *= tenor
        
        return np.sqrt(max(total_var/tenor, 0.0001))
    
    def _get_vol_linear(self, relative_strike: float, tenor: float) -> float:
        """Linear interpolation in strike and time dimensions"""
        if self.interpolator is None:
            return 0.20  # Default vol
        
        # Clip to bounds
        strike_min = self.vol_data['strike'].min()
        strike_max = self.vol_data['strike'].max()
        tenor_min = self.vol_data['tenor_years'].min()
        tenor_max = self.vol_data['tenor_years'].max()
        
        relative_strike = np.clip(relative_strike, strike_min, strike_max)
        tenor = np.clip(tenor, tenor_min, tenor_max)
        
        # Correct call to RegularGridInterpolator
        return float(self.interpolator((relative_strike, tenor)))
    
    def _get_vol_cubic(self, relative_strike: float, tenor: float) -> float:
        """Cubic spline interpolation"""
        if self.vol_data is None or self.vol_data.empty:
            return 0.20
        
        # Create 2D cubic interpolator if not exists
        if not hasattr(self, 'cubic_interpolator'):
            pivot = self.vol_data.pivot_table(
                index='strike',
                columns='tenor_years',
                values='implied_vol'
            )
            
            self.cubic_interpolator = interpolate.interp2d(
                pivot.columns,  # tenor
                pivot.index,    # strike
                pivot.values,
                kind='cubic',
                bounds_error=False,
                fill_value=None
            )
        
        vol = self.cubic_interpolator(tenor, relative_strike)[0]
        return float(max(vol, 0.01))
    
    def _get_vol_rbf(self, relative_strike: float, tenor: float) -> float:
        """Radial Basis Function interpolation - smooth and arbitrage-free"""
        if self.vol_data is None or self.vol_data.empty:
            return 0.20
        
        # Create RBF interpolator if not exists
        if not hasattr(self, 'rbf_interpolator'):
            points = self.vol_data[['strike', 'tenor_years']].values
            values = self.vol_data['implied_vol'].values
            
            self.rbf_interpolator = interpolate.RBFInterpolator(
                points,
                values,
                kernel='thin_plate_spline',
                smoothing=0.001
            )
        
        vol = self.rbf_interpolator([[relative_strike, tenor]])[0]
        return float(max(vol, 0.01))
    
    def _build_interpolator(self):
        """Build default interpolator from loaded data"""
        if self.vol_data is not None and not self.vol_data.empty:
            # Create grid interpolator
            pivot = self.vol_data.pivot_table(
                index='strike',
                columns='tenor_years',
                values='implied_vol'
            ).ffill().bfill()  # Fixed deprecation warning
            
            strikes = pivot.index.values
            tenors = pivot.columns.values
            vols = pivot.values
            
            self.interpolator = interpolate.RegularGridInterpolator(
                (strikes, tenors),
                vols,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
    
    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor string to years"""
        if isinstance(tenor, (int, float)):
            return float(tenor)
        
        tenor = tenor.lower()
        if tenor.endswith('d'):
            return int(tenor[:-1]) / 365.25
        elif tenor.endswith('w'):
            return int(tenor[:-1]) / 52
        elif tenor.endswith('m'):
            return int(tenor[:-1]) / 12
        elif tenor.endswith('y'):
            return int(tenor[:-1])
        else:
            raise ValueError(f"Unknown tenor format: {tenor}")
    
    def get_smile(self, tenor: Union[float, str], 
                  strikes: Optional[np.ndarray] = None,
                  spot: float = 4500) -> pd.DataFrame:
        """
        Get volatility smile for a given tenor.
        
        Args:
            tenor: Time to maturity
            strikes: Array of strike prices (if None, uses standard range)
            spot: Current spot price
            
        Returns:
            DataFrame with strikes and implied vols
        """
        if isinstance(tenor, str):
            tenor = self._tenor_to_years(tenor)
        
        if strikes is None:
            # Standard range: 60% to 140% of spot
            strikes = np.linspace(0.6 * spot, 1.4 * spot, 50)
        
        vols = [self.get_vol(k, tenor, spot) for k in strikes]
        
        return pd.DataFrame({
            'strike': strikes,
            'relative_strike': strikes / spot,
            'implied_vol': vols,
            'tenor': tenor
        })
    
    def get_term_structure(self, strike: float, 
                          tenors: Optional[np.ndarray] = None,
                          spot: float = 4500) -> pd.DataFrame:
        """
        Get term structure of implied volatility for a given strike.
        
        Args:
            strike: Strike price
            tenors: Array of tenors in years (if None, uses standard tenors)
            spot: Current spot price
            
        Returns:
            DataFrame with tenors and implied vols
        """
        if tenors is None:
            tenors = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1, 1.5, 2, 3, 5])
        
        vols = [self.get_vol(strike, t, spot) for t in tenors]
        
        return pd.DataFrame({
            'tenor': tenors,
            'tenor_label': [self._years_to_tenor_string(t) for t in tenors],
            'implied_vol': vols,
            'strike': strike
        })
    
    def _years_to_tenor_string(self, years: float) -> str:
        """Convert years to tenor string"""
        if years < 1:
            months = int(years * 12)
            return f"{months}m"
        else:
            return f"{int(years)}y" if years == int(years) else f"{years:.1f}y"
    
    def calibrate_to_market(self, market_vols: pd.DataFrame) -> Dict[str, float]:
        """
        Calibrate SSVI parameters to market volatilities.
        
        Args:
            market_vols: DataFrame with columns: strike, tenor, implied_vol
            
        Returns:
            Dictionary with calibrated SSVI parameters
        """
        # This would implement SSVI calibration
        # For now, return stored parameters
        return self.ssvi_params or {'theta': 0.03, 'rho': -0.5, 'beta': 0.2}
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for loaded vol surface"""
        if self.vol_data is None or self.vol_data.empty:
            return {'status': 'No data loaded'}
        
        return {
            'date': self.current_date,
            'num_points': len(self.vol_data),
            'strikes': f"{self.vol_data['strike'].min():.0%} - {self.vol_data['strike'].max():.0%}",
            'tenors': f"{self.vol_data['tenor_years'].min():.2f}y - {self.vol_data['tenor_years'].max():.2f}y",
            'avg_vol': f"{self.vol_data['implied_vol'].mean():.1%}",
            'min_vol': f"{self.vol_data['implied_vol'].min():.1%}",
            'max_vol': f"{self.vol_data['implied_vol'].max():.1%}",
            'has_ssvi': self.ssvi_params is not None
        }


# Example usage
if __name__ == "__main__":
    # Create vol surface manager
    vol_surface = VolatilitySurface()
    
    # Load surface for a specific date
    if vol_surface.load_surface('2023-06-01'):
        # Get summary
        print("Vol Surface Summary:")
        for key, value in vol_surface.get_summary_stats().items():
            print(f"  {key}: {value}")
        
        # Get implied vol for specific strike/tenor
        spot = 4500
        strike = 4500  # ATM
        tenor = 0.25  # 3 months
        
        vol = vol_surface.get_vol(strike, tenor, spot, method='linear')
        print(f"\nATM 3M vol: {vol:.2%}")
        
        # Get smile
        smile = vol_surface.get_smile('3m', spot=spot)
        print(f"\n3M Smile shape: {len(smile)} points")
        print(smile.head())
        
        # Get term structure
        term_structure = vol_surface.get_term_structure(strike, spot=spot)
        print(f"\nATM Term Structure:")
        print(term_structure)