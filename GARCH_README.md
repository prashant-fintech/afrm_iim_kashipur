# Applied Financial Risk Management - IIM Kashipur
## GARCH Model Implementation and Analysis

This directory contains comprehensive GARCH modeling tools for financial risk management education and practice.

## Quick Start Guide

### 1. Environment Setup

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Analysis

#### Option A: Using the Python Script
```python
from garch_analysis import GARCHAnalysis

# Initialize analysis
garch = GARCHAnalysis(symbol="^NSEI", start_date="2020-01-01")

# Run complete analysis
garch.fetch_data()
garch.calculate_returns()
garch.fit_garch_model()
garch.display_results()
```

#### Option B: Using the Jupyter Notebook
Open `GARCH_Analysis.ipynb` in Jupyter and run all cells for a comprehensive analysis.

### 3. Expected Output Format

When you run the GARCH analysis, you'll get detailed output similar to this:

```
                 Constant Mean - GARCH Model Results
==============================================================================
Dep. Variable:              Adj Close   R-squared:                      -0.001
Mean Model:             Constant Mean   Adj. R-squared:                 -0.001
Vol Model:                      GARCH   Log-Likelihood:               -5141.39
Distribution:                  Normal   AIC:                           10290.8
Method:            Maximum Likelihood   BIC:                           10315.4
                                        No. Observations:                 3520
Date:                Fri, Dec 02 2016   Df Residuals:                     3516
Time:                        22:22:28   Df Model:                            4

                                  Mean Model
==============================================================================
                 coef    std err          t      P>|t|        95.0% Conf. Int.
------------------------------------------------------------------------------
mu             0.0531  1.487e-02      3.569  3.581e-04   [2.392e-02,8.220e-02]

                               Volatility Model
==============================================================================
                 coef    std err          t      P>|t|        95.0% Conf. Int.
------------------------------------------------------------------------------
omega          0.0156  4.932e-03      3.155  1.606e-03   [5.892e-03,2.523e-02]
alpha[1]       0.0879  1.140e-02      7.710  1.260e-14     [6.554e-02,  0.110]
beta[1]        0.9014  1.183e-02     76.163      0.000       [  0.878,  0.925]
==============================================================================

Covariance estimator: robust
```

## Files Description

- **`garch_analysis.py`**: Main Python module with GARCHAnalysis class
- **`GARCH_Analysis.ipynb`**: Comprehensive Jupyter notebook tutorial
- **`requirements.txt`**: Required Python packages
- **`README.md`**: This file

## Key Features

### 1. Data Collection
- Automatic data fetching from Yahoo Finance
- Support for any stock symbol or index
- Flexible date range selection

### 2. GARCH Modeling
- GARCH(p,q) model implementation
- Multiple mean models (Constant, Zero, AR)
- Various error distributions (Normal, t-distribution, GED)
- Model comparison and selection

### 3. Risk Management Applications
- Value at Risk (VaR) calculation
- Volatility forecasting
- Conditional volatility analysis
- Risk metric visualization

### 4. Model Diagnostics
- Residual analysis
- Goodness-of-fit tests
- Parameter significance testing
- Stationarity checks

## Example Usage Scenarios

### Scenario 1: Basic GARCH Analysis
```python
from garch_analysis import example_analysis
example_analysis()
```

### Scenario 2: Custom Analysis
```python
from garch_analysis import GARCHAnalysis

# Analyze specific stock
garch = GARCHAnalysis(symbol="RELIANCE.NS", start_date="2019-01-01")
garch.fetch_data()
garch.calculate_returns()
garch.fit_garch_model(p=1, q=1)
garch.display_results()
garch.plot_volatility()
garch.forecast_volatility(horizon=30)
garch.calculate_var(confidence_level=0.01)
```

### Scenario 3: Model Comparison
```python
# Compare different GARCH specifications
models = [(1,1), (1,2), (2,1), (2,2)]
results = {}

for p, q in models:
    garch = GARCHAnalysis(symbol="^NSEI")
    garch.fetch_data()
    garch.calculate_returns()
    result = garch.fit_garch_model(p=p, q=q)
    results[f"GARCH({p},{q})"] = {
        'AIC': result.aic,
        'BIC': result.bic,
        'LogLikelihood': result.loglikelihood
    }
```

## Understanding GARCH Output

### Key Parameters:
- **μ (mu)**: Mean return
- **ω (omega)**: Long-term variance level
- **α (alpha)**: ARCH effect (short-term volatility)
- **β (beta)**: GARCH effect (volatility persistence)

### Model Statistics:
- **Log-Likelihood**: Higher is better
- **AIC/BIC**: Lower is better for model comparison
- **α + β**: Should be < 1 for stationarity

### Risk Metrics:
- **Conditional Volatility**: Time-varying volatility
- **VaR**: Maximum expected loss at given confidence level
- **Volatility Forecasts**: Future volatility predictions

## Educational Applications

This toolkit is designed for:
- Risk Management courses
- Financial Econometrics classes
- Quantitative Finance programs
- Professional risk analyst training
- Academic research in volatility modeling

## Advanced Features

### Custom Model Specifications
```python
# EGARCH model (asymmetric effects)
from arch import arch_model
model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1)

# GJR-GARCH model (threshold effects)
model = arch_model(returns, vol='GARCH', p=1, o=1, q=1)

# Different error distributions
model = arch_model(returns, dist='t')  # t-distribution
model = arch_model(returns, dist='ged')  # Generalized Error Distribution
```

### Portfolio Risk Analysis
```python
# Multiple asset GARCH analysis
symbols = ["^NSEI", "^BSESN", "RELIANCE.NS"]
portfolio_garch = {}

for symbol in symbols:
    garch = GARCHAnalysis(symbol=symbol)
    garch.fetch_data()
    garch.calculate_returns()
    garch.fit_garch_model()
    portfolio_garch[symbol] = garch
```

## Troubleshooting

### Common Issues:
1. **Data Not Found**: Check symbol format (e.g., "RELIANCE.NS" for NSE)
2. **Model Convergence**: Try different starting values or simpler models
3. **Stationarity Violation**: Check α + β < 1 condition
4. **Package Errors**: Ensure all dependencies are installed correctly

### Performance Tips:
- Use sufficient data (minimum 500 observations)
- Check for outliers in returns data
- Consider different error distributions if normality is violated
- Use information criteria for model selection

## References and Further Reading

1. Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
2. Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity"
3. Nelson, D. B. (1991). "Conditional Heteroskedasticity in Asset Returns"
4. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks"

## Contact and Support

For questions related to this implementation or the Applied Financial Risk Management course at IIM Kashipur, please refer to course materials or contact the course instructor.

---

*This toolkit is designed for educational purposes as part of the Applied Financial Risk Management course at IIM Kashipur. It provides hands-on experience with GARCH modeling for volatility analysis and risk management applications.*