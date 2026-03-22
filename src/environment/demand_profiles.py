"""
Demand profiles for supply chain simulations
"""

import numpy as np
from typing import Callable


def normal_demand(rng: np.random.RandomState, mean: float = 50.0, std: float = 15.0) -> Callable:
    """
    Normal demand profile with occasional spikes.
    
    Args:
        rng: Random number generator
        mean: Mean daily demand
        std: Standard deviation
        
    Returns:
        Function that generates demand values
    """
    def _demand_fn(t: int) -> float:
        base = rng.normal(mean, std)
        # 10% chance of 2.5x spike
        if rng.random() < 0.1:
            base *= 2.5
        return max(0, base)
    
    return _demand_fn


def seasonal_demand(rng: np.random.RandomState, mean: float = 50.0, amplitude: float = 20.0) -> Callable:
    """
    Seasonal demand with trend.
    
    Args:
        rng: Random number generator
        mean: Mean daily demand
        amplitude: Seasonal amplitude
        
    Returns:
        Function that generates demand values
    """
    def _demand_fn(t: int) -> float:
        # Annual seasonality
        seasonal = amplitude * np.sin(2 * np.pi * t / 365)
        base = mean + seasonal + rng.normal(0, mean * 0.2)
        return max(0, base)
    
    return _demand_fn


def trending_demand(rng: np.random.RandomState, initial_mean: float = 50.0, trend_rate: float = 0.05) -> Callable:
    """
    Demand with upward trend.
    
    Args:
        rng: Random number generator
        initial_mean: Starting mean demand
        trend_rate: Daily increase rate
        
    Returns:
        Function that generates demand values
    """
    def _demand_fn(t: int) -> float:
        mean_at_t = initial_mean * (1 + trend_rate) ** (t / 365)
        base = rng.normal(mean_at_t, mean_at_t * 0.3)
        return max(0, base)
    
    return _demand_fn


def volatile_demand(rng: np.random.RandomState, mean: float = 50.0, volatility: float = 0.5) -> Callable:
    """
    High-volatility demand profile.
    
    Args:
        rng: Random number generator
        mean: Mean demand
        volatility: Standard deviation as fraction of mean
        
    Returns:
        Function that generates demand values
    """
    def _demand_fn(t: int) -> float:
        base = rng.normal(mean, mean * volatility)
        return max(0, base)
    
    return _demand_fn
