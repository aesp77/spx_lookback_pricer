"""
Interpolation Utilities
Various interpolation methods for volatility surfaces
"""

import numpy as np
from scipy.interpolate import (
    RBFInterpolator,
    RegularGridInterpolator,
    interp1d,
    CubicSpline
)
from typing import Tuple, Optional


def rbf_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    query_points: np.ndarray,
    kernel: str = 'thin_plate_spline',
    smoothing: float = 0.0
) -> np.ndarray:
    """
    Radial Basis Function interpolation
    
    Args:
        points: Input points (N x D array)
        values: Function values at input points (N array)
        query_points: Points to interpolate (M x D array)
        kernel: RBF kernel type
        smoothing: Smoothing parameter
        
    Returns:
        Interpolated values (M array)
    """
    interpolator = RBFInterpolator(
        points,
        values,
        kernel=kernel,
        smoothing=smoothing
    )
    
    return interpolator(query_points)


def spline_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    method: str = 'cubic'
) -> np.ndarray:
    """
    1D spline interpolation
    
    Args:
        x: Input x values
        y: Input y values
        x_new: New x values to interpolate
        method: Interpolation method ('linear', 'cubic')
        
    Returns:
        Interpolated y values
    """
    if method == 'cubic':
        spline = CubicSpline(x, y)
        return spline(x_new)
    else:
        interpolator = interp1d(x, y, kind=method, fill_value='extrapolate')
        return interpolator(x_new)


def bilinear_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_new: float,
    y_new: float
) -> float:
    """
    2D bilinear interpolation
    
    Args:
        x: X grid values (1D array)
        y: Y grid values (1D array)
        z: Z values on grid (2D array, shape len(y) x len(x))
        x_new: New x value
        y_new: New y value
        
    Returns:
        Interpolated value
    """
    # Find surrounding points
    i = np.searchsorted(x, x_new)
    j = np.searchsorted(y, y_new)
    
    # Handle boundaries
    if i == 0:
        i = 1
    if i >= len(x):
        i = len(x) - 1
    if j == 0:
        j = 1
    if j >= len(y):
        j = len(y) - 1
        
    # Get corner values
    x0, x1 = x[i-1], x[i]
    y0, y1 = y[j-1], y[j]
    
    z00 = z[j-1, i-1]
    z01 = z[j-1, i]
    z10 = z[j, i-1]
    z11 = z[j, i]
    
    # Interpolate
    if x1 != x0 and y1 != y0:
        wx = (x_new - x0) / (x1 - x0)
        wy = (y_new - y0) / (y1 - y0)
        
        result = (
            (1 - wx) * (1 - wy) * z00 +
            wx * (1 - wy) * z01 +
            (1 - wx) * wy * z10 +
            wx * wy * z11
        )
    else:
        result = z00
        
    return result


def interpolate_vol_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    vols: np.ndarray,
    strike_new: np.ndarray,
    expiry_new: np.ndarray,
    method: str = 'rbf'
) -> np.ndarray:
    """
    Interpolate volatility surface
    
    Args:
        strikes: Strike prices (N array)
        expiries: Expiries (N array)
        vols: Volatilities (N array)
        strike_new: New strike prices (M array)
        expiry_new: New expiries (M array)
        method: Interpolation method ('rbf', 'bilinear')
        
    Returns:
        Interpolated volatilities (M array)
    """
    # Convert to moneyness
    spot = np.median(strikes)  # Approximate
    moneyness = np.log(strikes / spot)
    moneyness_new = np.log(strike_new / spot)
    
    if method == 'rbf':
        points = np.column_stack([moneyness, expiries])
        query_points = np.column_stack([moneyness_new, expiry_new])
        
        return rbf_interpolate(points, vols, query_points)
    elif method == 'bilinear':
        # Create grid
        unique_m = np.unique(moneyness)
        unique_t = np.unique(expiries)
        
        # Reshape vols onto grid (this is simplified)
        vol_grid = np.zeros((len(unique_t), len(unique_m)))
        
        for i, t in enumerate(unique_t):
            for j, m in enumerate(unique_m):
                mask = (np.abs(expiries - t) < 1e-6) & (np.abs(moneyness - m) < 1e-6)
                if np.any(mask):
                    vol_grid[i, j] = np.mean(vols[mask])
                    
        # Interpolate
        interpolator = RegularGridInterpolator(
            (unique_t, unique_m),
            vol_grid,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        query_points = np.column_stack([expiry_new, moneyness_new])
        return interpolator(query_points)
    else:
        raise ValueError(f"Unknown method: {method}")


def extrapolate_flat(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray
) -> np.ndarray:
    """
    Flat extrapolation beyond data range
    
    Args:
        x: Input x values
        y: Input y values
        x_new: New x values
        
    Returns:
        Extrapolated y values (flat beyond range)
    """
    result = np.zeros_like(x_new)
    
    for i, x_val in enumerate(x_new):
        if x_val < x[0]:
            result[i] = y[0]
        elif x_val > x[-1]:
            result[i] = y[-1]
        else:
            result[i] = np.interp(x_val, x, y)
            
    return result
