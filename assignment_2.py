import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import fsolve

# --- CONFIGURATION ---
TICKERS = ['MSFT', 'F', 'PTON'] 
RISK_FREE_RATE = 0.042  # 4.2% (Adjust to current 1-Year Treasury)
T = 1.0                 # Time horizon in years

def merton_solver(variables, E, sigma_E, D, r, T):
    """
    System of equations to solve for Asset Value (V_A) and Asset Volatility (sigma_A)
    """
    V_A, sigma_A = variables
    
    # Avoid negative or zero values during solver iterations
    if V_A <= 0 or sigma_A <= 0:
        return [1e10, 1e10]

    d1 = (np.log(V_A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    
    # Eq 1: Equity Value (Black-Scholes Call Option)
    eq1 = V_A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    
    # Eq 2: Volatility Link
    eq2 = (V_A / E) * norm.cdf(d1) * sigma_A - sigma_E
    
    return [eq1, eq2]

def analyze_firm(ticker):
    print(f"--- Analyzing {ticker} ---")
    
    # 1. GET DATA
    stock = yf.Ticker(ticker)
    
    # Market Cap (E)
    shares = stock.info.get('sharesOutstanding')
    price = stock.history(period='1d')['Close'].iloc[-1]
    E = shares * price
    
    # Equity Volatility (sigma_E) - Historical 1 year
    hist = stock.history(period="1y")
    hist['Returns'] = hist['Close'].pct_change()
    sigma_E = hist['Returns'].std() * np.sqrt(252) # Annualized
    
    # Debt (D) from Balance Sheet (Quarterly)
    bs = stock.quarterly_balance_sheet
    try:
        # Try to get short-term and long-term debt separately
        # Common field names in yfinance balance sheets
        short_term_debt = 0
        long_term_debt = 0
        
        # Try different field names for short-term debt
        if 'Current Debt' in bs.index:
            short_term_debt = bs.loc['Current Debt'].iloc[0]
        elif 'Short Term Debt' in bs.index:
            short_term_debt = bs.loc['Short Term Debt'].iloc[0]
        elif 'Current Liabilities' in bs.index:
            short_term_debt = bs.loc['Current Liabilities'].iloc[0]
            
        # Try different field names for long-term debt
        if 'Long Term Debt' in bs.index:
            long_term_debt = bs.loc['Long Term Debt'].iloc[0]
        elif 'Total Long Term Debt' in bs.index:
            long_term_debt = bs.loc['Total Long Term Debt'].iloc[0]
        elif 'Long Term Debt And Capital Lease Obligation' in bs.index:
            long_term_debt = bs.loc['Long Term Debt And Capital Lease Obligation'].iloc[0]
            
        # Merton model: D = Short-term debt + 0.5 * Long-term debt
        # (Assumes half of LT debt matures within model horizon)
        D = short_term_debt + 0.5 * long_term_debt
        
        # If we couldn't get the breakdown, fall back to Total Debt
        if D == 0 or np.isnan(D):
            if 'Total Debt' in bs.index:
                D = bs.loc['Total Debt'].iloc[0]
                short_term_debt = D  # Assume all short-term if no breakdown
                long_term_debt = 0
                print("Note: Could not break down debt, using Total Debt")
            else:
                D = bs.loc['Total Liab'].iloc[0]
                short_term_debt = D
                long_term_debt = 0
                print("Note: Used Total Liabilities as proxy for Debt")
            
    except Exception as e:
        print(f"Error fetching debt data: {e}")
        return

    # 2. SOLVE FOR V_A and sigma_A
    # Initial guesses: V_A ≈ E + D, sigma_A ≈ sigma_E * (E / (E+D))
    initial_guess = [E + D, sigma_E * 0.5]
    V_A, sigma_A = fsolve(merton_solver, initial_guess, args=(E, sigma_E, D, RISK_FREE_RATE, T))

    # 3. CALCULATE METRICS
    d1 = (np.log(V_A / D) + (RISK_FREE_RATE + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    
    # Delta (N(d1)) - Sensitivity of equity value to asset value
    Delta = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    # Theoretical Equity Value (Black-Scholes Call Option)
    E_est = V_A * norm.cdf(d1) - D * np.exp(-RISK_FREE_RATE * T) * norm.cdf(d2)
    
    # Distance to Default (DD)
    # Note: Risk-Neutral DD is often defined simply as d2
    DD = d2
    
    # Risk Neutral Probability of Default (PD)
    PD = norm.cdf(-DD)
    
    # Market Cap (same as Equity Value E)
    Market_Cap = E
    
    print(f"Market Equity (E):      ${Market_Cap/1e9:.2f} B")
    print(f"Theoretical Equity (C): ${E_est/1e9:.2f} B")
    print(f"Model Error:            ${(E_est - E)/1e9:.4f} B ({(E_est - E)/E * 100:.2f}%)")
    print(f"Equity Volatility:      {sigma_E:.2%}")
    print(f"Short-Term Debt:        ${short_term_debt/1e9:.2f} B")
    print(f"Long-Term Debt:         ${long_term_debt/1e9:.2f} B")
    print(f"Debt Value (D):         ${D/1e9:.2f} B  (ST + 0.5*LT)")
    print("-" * 30)
    print(f"[9] Asset Value (V_A):  ${V_A/1e9:.2f} B")
    print(f"[10] Asset Vol (sigma_A): {sigma_A:.2%}")
    print(f"Delta [N(d1)]:          {Delta:.4f}")
    print(f"d1:                     {d1:.6f}")
    print(f"d2:                     {d2:.6f}")
    print(f"N(d2):                  {N_d2:.6f}")
    print(f"N(-d2) = 1 - N(d2):     {1 - N_d2:.6f}")
    print(f"[12] Dist to Default:     {DD:.4f}")
    print(f"[11] Risk Neutral PD:     {PD:.6%} ({PD:.8f})")
    print("\n")

# Run analysis
for ticker in TICKERS:
    analyze_firm(ticker)