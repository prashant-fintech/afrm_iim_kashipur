# Monte Carlo VaR Analysis - R to Python Conversion

This project converts the original R Monte Carlo simulation code for Value at Risk (VaR) calculation to Python, adapted to work with Reliance stock data.

## Files Overview

### 1. `stockprice.py`
- **Purpose**: Contains the stock price simulation function using Geometric Brownian Motion
- **Key Function**: `stockprice(s0, r, sig, t, n, m)` - generates multiple stock price paths
- **Conversion Notes**: 
  - Vectorized implementation for better performance
  - Alternative loop-based version available (closer to original R code)
  - Uses NumPy for efficient array operations

### 2. `monte_carlo_var.py`
- **Purpose**: Main Monte Carlo VaR analysis with backtesting
- **Key Features**:
  - Loads Reliance stock data from CSV
  - Calculates rolling window VaR estimates
  - Performs backtesting validation
  - Generates comprehensive visualizations
  - Exports results to CSV files

### 3. `reliance.csv`
- **Purpose**: Input data file containing Reliance stock prices
- **Format**: Expected columns - Date, Close (or similar price column names)
- **Flexibility**: Code auto-detects price column names

## Key Differences from Original R Code

### Data Loading
- **R**: `data=read.csv("nifty.csv", header=T)`
- **Python**: `pd.read_csv()` with automatic column detection

### Return Calculation
- **R**: `ret1=diff(nifty)/lag(nifty)`
- **Python**: `np.diff(stock_prices) / stock_prices[:-1]`

### Monte Carlo Simulation
- **R**: Nested loops for path generation
- **Python**: Vectorized operations using NumPy (with loop alternative)

### Statistical Functions
- **R**: `quantile(pl_sim, 0.05)`
- **Python**: `np.percentile(pl_sim, 5)`

## Usage Instructions

### 1. Setup Environment
```bash
# Activate virtual environment (if using one)
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install required packages
pip install numpy pandas matplotlib
```

### 2. Prepare Data
- Ensure `reliance.csv` exists with stock price data
- File should contain Date and Close price columns
- If file doesn't exist, the code will create sample data

### 3. Run Analysis
```bash
python monte_carlo_var.py
```

### 4. Expected Outputs
- **Console Output**: Real-time progress and results summary
- **Return.csv**: Calculated stock returns
- **MCS_Results.csv**: Detailed backtesting results
- **Monte_Carlo_VaR_Analysis.png**: Comprehensive visualization plots

## Output Files Description

### MCS_Results.csv
Contains backtesting results with columns:
- `Violation`: Binary indicator (1 = VaR violation, 0 = no violation)
- `Excess_Cap`: Excess capital when no violation occurs
- `Excess_Loss`: Excess loss when violation occurs
- `VaR_95_MC`: Monte Carlo estimated 95% VaR
- `Actual_PL`: Actual profit/loss observed

### Return.csv
Contains calculated stock returns for the time series.

## Key Parameters (Configurable)

### In `calculate_var_monte_carlo()`:
- `window_size=250`: Rolling window for parameter estimation (equivalent to R's 250)
- `num_simulations=100000`: Number of Monte Carlo simulations for initial analysis
- `1000`: Number of simulations for backtesting (reduced for speed)

### Monte Carlo Parameters:
- `t=1`: Time horizon (1 day)
- `n_steps=1`: Number of steps in simulation
- Confidence levels: 95%, 99%, 99.5%

## Interpretation of Results

### Failure Rate Analysis
- **Expected**: ~5% for 95% VaR
- **Good Performance**: 3% - 7% range
- **Conservative Model**: < 3% (overestimating risk)
- **Poor Model**: > 7% (underestimating risk)

### Key Metrics
1. **VaR Values**: Worst expected loss at given confidence levels
2. **Failure Rate**: Percentage of times actual loss exceeded VaR
3. **Excess Capital**: Opportunity cost when model is too conservative
4. **Excess Loss**: Additional loss when model fails

## Visualizations

The code generates four plots:
1. **P&L Distribution**: Histogram with VaR thresholds
2. **VaR vs Actual P&L**: Time series comparison
3. **VaR Violations**: Scatter plot highlighting violations
4. **Excess Capital vs Loss**: Bar chart showing model efficiency

## Technical Improvements from R Version

1. **Vectorization**: Faster computation using NumPy arrays
2. **Error Handling**: Automatic data validation and column detection
3. **Visualization**: Comprehensive matplotlib-based plots
4. **Progress Tracking**: Real-time backtesting progress indicators
5. **Flexible Input**: Handles various CSV column naming conventions
6. **Sample Data**: Automatic generation if input file missing

## Troubleshooting

### Common Issues:
1. **Missing CSV file**: Code will create sample data automatically
2. **Column name mismatch**: Code auto-detects common price column names
3. **Insufficient data**: Error message if less than 251 observations
4. **Memory issues**: Reduce `num_simulations` for large datasets

### Performance Tips:
- Use vectorized version of `stockprice()` for better performance
- Reduce backtesting simulations (1000) vs initial analysis (100000)
- Consider parallel processing for very large datasets

## Mathematical Foundation

The code implements:
1. **Geometric Brownian Motion**: `S(t+1) = S(t) * exp((r - σ²/2)Δt + σ√Δt * Z)`
2. **VaR Calculation**: Percentile-based approach using empirical distribution
3. **Rolling Window**: 250-day parameter estimation window
4. **Backtesting**: Out-of-sample validation framework

## Extensions and Modifications

The code can be easily extended for:
- Different confidence levels
- Alternative VaR methodologies (Historical, Parametric)
- Multiple asset portfolios
- Different time horizons
- Custom risk metrics (Expected Shortfall, etc.)

---

## Example Run Output

```
Monte Carlo Simulation for VaR Calculation
==================================================

Available columns in the CSV file:
['Date', 'Close']
Using column 'Close' as stock price

Loaded 501 stock price observations
Price range: 2500.50 - 2680.95

Initial Analysis:
Last stock price: 2680.95
Average return: 0.000267
Volatility: 0.012543

Running Monte Carlo simulation with 100,000 simulations...

Value at Risk (VaR) Results:
95% VaR: -27.45
99% VaR: -41.23
99.5% VaR: -46.87

Starting backtesting...
Running backtesting for 251 observations...
Processed 50/251 observations...
...

Backtesting Results:
Failure rate: 0.0478 (4.78%)
Expected failure rate (95% VaR): 0.05 (5%)
Total excess capital: 1245.67
Total excess loss: -234.89

Results saved to MCS_Results.csv

==================================================
SUMMARY OF RESULTS
==================================================
95% VaR: -27.45
99% VaR: -41.23
99.5% VaR: -46.87
Backtesting failure rate: 0.0478 (4.78%)
Total excess capital: 1245.67
Total excess loss: -234.89

==================================================
INTERPRETATION
==================================================
✓ Model performance: GOOD - Failure rate is within acceptable range
```

This Python implementation provides the same functionality as the original R code while offering better performance, visualization, and user experience.