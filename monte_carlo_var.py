"""
Monte Carlo Simulation with Backtesting for VaR Calculation
Converted from R to Python - Adapted for Reliance stock data
1 day ahead VaR calculation with backtesting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stockprice import stockprice

# Set random seed for reproducibility
np.random.seed(42)

def load_stock_data(filename="reliance.csv"):
    """
    Load stock data from CSV file
    Expected columns: Date, Close (or similar price column)
    """
    try:
        # Try different possible column names for stock price
        data = pd.read_csv(filename)
        
        # Print available columns to help user understand the data structure
        print("Available columns in the CSV file:")
        print(data.columns.tolist())
        
        # Try to identify the price column
        price_columns = ['Close', 'close', 'Price', 'price', 'Adj Close', 'Reliance', 'reliance']
        price_col = None
        
        for col in price_columns:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            print(f"Could not find standard price column. Using first numeric column: {data.select_dtypes(include=[np.number]).columns[0]}")
            price_col = data.select_dtypes(include=[np.number]).columns[0]
        
        print(f"Using column '{price_col}' as stock price")
        stock_prices = data[price_col].values
        
        return stock_prices, data
        
    except FileNotFoundError:
        print(f"File {filename} not found. Creating sample Reliance data...")
        return create_sample_data()

def create_sample_data():
    """
    Create sample Reliance stock data for demonstration
    """
    np.random.seed(42)
    n_days = 500
    initial_price = 2500  # Typical Reliance price
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame and save to CSV
    dates = pd.date_range(start='2023-01-01', periods=n_days+1, freq='B')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    sample_data.to_csv('reliance.csv', index=False)
    print("Created sample reliance.csv file with 501 data points")
    
    return np.array(prices), sample_data

def calculate_var_monte_carlo(stock_prices, window_size=250, num_simulations=100000):
    """
    Calculate VaR using Monte Carlo Simulation with backtesting
    
    Parameters:
    stock_prices : array - Historical stock prices
    window_size : int - Rolling window size for parameter estimation
    num_simulations : int - Number of Monte Carlo simulations
    """
    
    # Calculate simple returns
    n = len(stock_prices)
    returns = np.diff(stock_prices) / stock_prices[:-1]
    
    # Save returns to CSV
    returns_df = pd.DataFrame({'Returns': returns})
    returns_df.to_csv('Return.csv', index=False)
    print("Returns saved to Return.csv")
    
    # Remove last observation if needed (equivalent to R's indexing)
    stock_returns = returns[:n-1] if len(returns) == n else returns
    
    # Get the last price for initial simulation
    s0 = stock_prices[-1]
    r = np.mean(stock_returns)
    sig = np.std(stock_returns)
    
    print("\nInitial Analysis:")
    print(f"Last stock price: {s0:.2f}")
    print(f"Average return: {r:.6f}")
    print(f"Volatility: {sig:.6f}")
    
    # Monte Carlo simulation parameters
    t = 1  # Time horizon (1 day)
    n_steps = 1  # Number of steps in simulation
    
    # Run initial Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation with {num_simulations:,} simulations...")
    price_paths = stockprice(s0, r, sig, t, n_steps, num_simulations)
    pl_sim = price_paths[:, 1] - price_paths[:, 0]  # P&L distribution
    
    # Calculate VaR at different confidence levels
    var_95 = np.percentile(pl_sim, 5)
    var_99 = np.percentile(pl_sim, 1)
    var_995 = np.percentile(pl_sim, 0.5)
    
    print("\nValue at Risk (VaR) Results:")
    print(f"95% VaR: {var_95:.2f}")
    print(f"99% VaR: {var_99:.2f}")
    print(f"99.5% VaR: {var_995:.2f}")
    
    # Backtesting
    print("\nStarting backtesting...")
    returns_length = len(stock_returns)
    n1 = returns_length - window_size
    
    if n1 <= 0:
        print(f"Error: Not enough data points. Need at least {window_size + 1} observations.")
        return None
    
    # Initialize arrays for backtesting results
    var_95_mc = np.zeros(n1)
    actual_pl = np.zeros(n1)
    violation = np.zeros(n1)
    excess_cap = np.zeros(n1)
    excess_loss = np.zeros(n1)
    
    print(f"Running backtesting for {n1} observations...")
    
    for i in range(n1):
        # Use rolling window for parameter estimation
        s0_bt = stock_prices[i + window_size]
        r_bt = np.mean(stock_returns[i:i + window_size])
        sig_bt = np.std(stock_returns[i:i + window_size])
        
        # Run Monte Carlo simulation for this observation
        price_paths_bt = stockprice(s0_bt, r_bt, sig_bt, t, n_steps, 1000)  # Use fewer simulations for speed
        pl_sim_bt = price_paths_bt[:, 1] - price_paths_bt[:, 0]
        
        var_95_mc[i] = np.percentile(pl_sim_bt, 5)
        actual_pl[i] = stock_prices[i + window_size + 1] - stock_prices[i + window_size]
        
        # Check for VaR violation
        if actual_pl[i] <= var_95_mc[i]:
            violation[i] = 1
            excess_loss[i] = actual_pl[i] - var_95_mc[i]
            excess_cap[i] = 0
        else:
            violation[i] = 0
            excess_loss[i] = 0
            excess_cap[i] = actual_pl[i] - var_95_mc[i]
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{n1} observations...")
    
    # Calculate backtesting metrics
    failure_rate = np.sum(violation) / n1
    total_excess_cap = np.sum(excess_cap)
    total_excess_loss = np.sum(excess_loss)
    
    print("\nBacktesting Results:")
    print(f"Failure rate: {failure_rate:.4f} ({failure_rate*100:.2f}%)")
    print("Expected failure rate (95% VaR): 0.05 (5%)")
    print(f"Total excess capital: {total_excess_cap:.2f}")
    print(f"Total excess loss: {total_excess_loss:.2f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Violation': violation.astype(int),
        'Excess_Cap': excess_cap,
        'Excess_Loss': excess_loss,
        'VaR_95_MC': var_95_mc,
        'Actual_PL': actual_pl
    })
    
    results_df.to_csv('MCS_Results.csv', index=False)
    print("\nResults saved to MCS_Results.csv")
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'var_995': var_995,
        'failure_rate': failure_rate,
        'total_excess_cap': total_excess_cap,
        'total_excess_loss': total_excess_loss,
        'results_df': results_df,
        'pl_distribution': pl_sim
    }

def plot_results(results):
    """
    Create visualizations of the Monte Carlo VaR results
    """
    if results is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: P&L Distribution
    axes[0, 0].hist(results['pl_distribution'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(results['var_95'], color='red', linestyle='--', label=f"95% VaR: {results['var_95']:.2f}")
    axes[0, 0].axvline(results['var_99'], color='orange', linestyle='--', label=f"99% VaR: {results['var_99']:.2f}")
    axes[0, 0].set_title('Monte Carlo P&L Distribution')
    axes[0, 0].set_xlabel('P&L')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: VaR vs Actual P&L
    df = results['results_df']
    axes[0, 1].plot(df['VaR_95_MC'], label='95% VaR', color='red', alpha=0.7)
    axes[0, 1].plot(df['Actual_PL'], label='Actual P&L', color='blue', alpha=0.7)
    axes[0, 1].set_title('VaR vs Actual P&L (Backtesting)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('P&L')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Violations
    violation_indices = np.where(df['Violation'] == 1)[0]
    axes[1, 0].scatter(violation_indices, df.loc[violation_indices, 'Actual_PL'], 
                      color='red', s=30, alpha=0.7, label=f'Violations ({len(violation_indices)})')
    axes[1, 0].plot(df['Actual_PL'], color='blue', alpha=0.5, label='Actual P&L')
    axes[1, 0].plot(df['VaR_95_MC'], color='red', alpha=0.5, linestyle='--', label='95% VaR')
    axes[1, 0].set_title('VaR Violations')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('P&L')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Excess Capital and Loss
    axes[1, 1].bar(range(len(df)), df['Excess_Cap'], alpha=0.7, color='green', label='Excess Capital')
    axes[1, 1].bar(range(len(df)), df['Excess_Loss'], alpha=0.7, color='red', label='Excess Loss')
    axes[1, 1].set_title('Excess Capital vs Excess Loss')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Amount')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Monte_Carlo_VaR_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the Monte Carlo VaR analysis
    """
    print("Monte Carlo Simulation for VaR Calculation")
    print("=" * 50)
    
    # Load data
    stock_prices, data = load_stock_data("reliance.csv")
    
    print(f"\nLoaded {len(stock_prices)} stock price observations")
    print(f"Price range: {np.min(stock_prices):.2f} - {np.max(stock_prices):.2f}")
    
    # Run Monte Carlo VaR analysis
    results = calculate_var_monte_carlo(stock_prices)
    
    if results is not None:
        # Create visualizations
        plot_results(results)
        
        # Print summary
        print(f"\n{'='*50}")
        print("SUMMARY OF RESULTS")
        print(f"{'='*50}")
        print(f"95% VaR: {results['var_95']:.2f}")
        print(f"99% VaR: {results['var_99']:.2f}")
        print(f"99.5% VaR: {results['var_995']:.2f}")
        print(f"Backtesting failure rate: {results['failure_rate']:.4f} ({results['failure_rate']*100:.2f}%)")
        print(f"Total excess capital: {results['total_excess_cap']:.2f}")
        print(f"Total excess loss: {results['total_excess_loss']:.2f}")
        
        # Interpretation
        print(f"\n{'='*50}")
        print("INTERPRETATION")
        print(f"{'='*50}")
        if 0.03 <= results['failure_rate'] <= 0.07:
            print("✓ Model performance: GOOD - Failure rate is within acceptable range")
        elif results['failure_rate'] < 0.03:
            print("! Model performance: CONSERVATIVE - Too few violations (model may be overestimating risk)")
        else:
            print("✗ Model performance: POOR - Too many violations (model underestimates risk)")

if __name__ == "__main__":
    main()