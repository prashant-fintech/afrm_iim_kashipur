"""
Stock Price Simulation Function using Geometric Brownian Motion
Converted from R to Python for Monte Carlo VaR Analysis
"""

import numpy as np

def stockprice(s0, r, sig, t, n, m):
    """
    Generate stock price paths using Geometric Brownian Motion
    
    Parameters:
    s0 : float - Initial stock price
    r : float - Average return (drift)
    sig : float - Volatility (standard deviation)
    t : float - Time horizon
    n : int - Number of steps in simulation
    m : int - Number of simulations (paths)
    
    Returns:
    numpy.ndarray - Matrix of stock prices (m x n+1)
    """
    del_t = t / n  # Length of one step
    st = np.zeros((m, n + 1))
    st[:, 0] = s0  # Set initial price for all paths
    
    # Generate random normal values for all paths at once (vectorized)
    random_values = np.random.normal(0, 1, (m, n))
    
    # Generate paths using vectorized operations
    for i in range(n):
        st[:, i + 1] = st[:, i] * np.exp(
            (r - sig**2 / 2) * del_t + sig * np.sqrt(del_t) * random_values[:, i]
        )
    
    return st

def stockprice_single_path(s0, r, sig, t, n, m):
    """
    Alternative implementation using loop-based approach (closer to original R code)
    """
    del_t = t / n
    st = np.zeros((m, n + 1))
    st[:, 0] = s0
    
    # Inner loop will generate one path and outer loop will generate m paths
    for j in range(m):
        for i in range(n):
            st[j, i + 1] = st[j, i] * np.exp(
                (r - sig**2 / 2) * del_t + sig * np.sqrt(del_t) * np.random.normal(0, 1)
            )
    
    return st