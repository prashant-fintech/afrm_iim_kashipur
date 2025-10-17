"""
GARCH Model Implementation for Financial Risk Management
Applied Financial Risk Management - IIM Kashipur

This module demonstrates GARCH modeling for volatility forecasting
and risk management applications.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GARCHAnalysis:
    """
    A comprehensive class for GARCH modeling and analysis
    """
    
    def __init__(self, symbol="^NSEI", start_date="2020-01-01", end_date=None):
        """
        Initialize the GARCH analysis with stock data
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (default: ^NSEI for Nifty 50)
        start_date : str
            Start date for data retrieval
        end_date : str
            End date for data retrieval (default: today)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.model = None
        self.results = None
        
    def fetch_data(self):
        """Fetch stock price data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"Data fetched successfully for {self.symbol}")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"Total observations: {len(self.data)}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
            
        return self.data
    
    def calculate_returns(self, price_column='Close'):
        """
        Calculate logarithmic returns
        
        Parameters:
        -----------
        price_column : str
            Column name for price data (default: 'Close')
        """
        if self.data is None:
            print("No data available. Please fetch data first.")
            return None
            
        # Calculate log returns
        self.returns = np.log(self.data[price_column] / self.data[price_column].shift(1)) * 100
        
        # Remove NaN values
        self.returns = self.returns.dropna()
        
        print(f"Returns calculated successfully")
        print(f"Return observations: {len(self.returns)}")
        print(f"Mean return: {self.returns.mean():.4f}%")
        print(f"Volatility (std): {self.returns.std():.4f}%")
        
        return self.returns
    
    def plot_returns(self, figsize=(12, 8)):
        """Plot price and returns time series"""
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Price series
        axes[0, 0].plot(self.data.index, self.data['Close'])
        axes[0, 0].set_title(f'{self.symbol} - Price Series')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns series
        axes[0, 1].plot(self.returns.index, self.returns)
        axes[0, 1].set_title(f'{self.symbol} - Returns Series')
        axes[0, 1].set_ylabel('Returns (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1, 0].hist(self.returns, bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns (%)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.returns, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def fit_garch_model(self, p=1, q=1, mean_model='Constant', vol_model='GARCH', 
                       distribution='Normal'):
        """
        Fit GARCH model to returns data
        
        Parameters:
        -----------
        p : int
            Number of lags for GARCH term (default: 1)
        q : int
            Number of lags for ARCH term (default: 1)
        mean_model : str
            Mean model specification (default: 'Constant')
        vol_model : str
            Volatility model specification (default: 'GARCH')
        distribution : str
            Error distribution (default: 'Normal')
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
            
        try:
            # Define the GARCH model
            self.model = arch_model(
                self.returns, 
                mean=mean_model, 
                vol=vol_model, 
                p=p, 
                q=q,
                dist=distribution
            )
            
            # Fit the model
            self.results = self.model.fit(disp='off')
            
            print("GARCH model fitted successfully!")
            print(f"Model: {mean_model} Mean - {vol_model}({p},{q})")
            print(f"Distribution: {distribution}")
            
            return self.results
            
        except Exception as e:
            print(f"Error fitting GARCH model: {e}")
            return None
    
    def display_results(self):
        """Display formatted GARCH model results"""
        if self.results is None:
            print("No model results available. Please fit the model first.")
            return
            
        print(self.results.summary())
        
    def plot_volatility(self, figsize=(12, 8)):
        """Plot conditional volatility"""
        if self.results is None:
            print("No model results available. Please fit the model first.")
            return
            
        # Extract conditional volatility
        conditional_vol = self.results.conditional_volatility
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Returns with volatility
        axes[0].plot(self.returns.index, self.returns, alpha=0.7, label='Returns')
        axes[0].plot(conditional_vol.index, conditional_vol, 'r-', label='Conditional Volatility')
        axes[0].plot(conditional_vol.index, -conditional_vol, 'r-', alpha=0.5)
        axes[0].fill_between(conditional_vol.index, -conditional_vol, conditional_vol, 
                           alpha=0.2, color='red')
        axes[0].set_title(f'{self.symbol} - Returns and Conditional Volatility')
        axes[0].set_ylabel('Returns (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volatility only
        axes[1].plot(conditional_vol.index, conditional_vol, 'r-', linewidth=1.5)
        axes[1].set_title(f'{self.symbol} - Conditional Volatility')
        axes[1].set_ylabel('Volatility (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def forecast_volatility(self, horizon=10):
        """
        Forecast volatility
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon (default: 10 days)
        """
        if self.results is None:
            print("No model results available. Please fit the model first.")
            return None
            
        try:
            # Generate forecasts
            forecasts = self.results.forecast(horizon=horizon)
            
            print(f"\nVolatility Forecasts for next {horizon} periods:")
            print("=" * 50)
            
            forecast_df = pd.DataFrame({
                'Period': range(1, horizon + 1),
                'Volatility_Forecast': forecasts.variance.iloc[-1, :].values ** 0.5
            })
            
            print(forecast_df.to_string(index=False, float_format='%.4f'))
            
            return forecast_df
            
        except Exception as e:
            print(f"Error generating forecasts: {e}")
            return None
    
    def calculate_var(self, confidence_level=0.05):
        """
        Calculate Value at Risk (VaR) using GARCH volatility
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (default: 0.05 for 95% VaR)
        """
        if self.results is None:
            print("No model results available. Please fit the model first.")
            return None
            
        # Get the latest volatility forecast
        latest_vol = self.results.conditional_volatility.iloc[-1]
        
        # Calculate VaR (assuming normal distribution)
        from scipy.stats import norm
        var_multiplier = norm.ppf(confidence_level)
        var = var_multiplier * latest_vol
        
        print(f"\nValue at Risk (VaR) Analysis:")
        print("=" * 40)
        print(f"Confidence Level: {(1-confidence_level)*100:.0f}%")
        print(f"Latest Volatility: {latest_vol:.4f}%")
        print(f"VaR (1-day): {var:.4f}%")
        print(f"VaR (10-day): {var * np.sqrt(10):.4f}%")
        
        return {
            'confidence_level': confidence_level,
            'volatility': latest_vol,
            'var_1d': var,
            'var_10d': var * np.sqrt(10)
        }

def example_analysis():
    """Example analysis demonstrating GARCH modeling"""
    print("GARCH Model Analysis Example")
    print("=" * 50)
    
    # Initialize analysis for Nifty 50
    garch = GARCHAnalysis(symbol="^NSEI", start_date="2020-01-01")
    
    # Fetch data
    data = garch.fetch_data()
    if data is None:
        return
    
    # Calculate returns
    returns = garch.calculate_returns()
    if returns is None:
        return
    
    # Plot data
    garch.plot_returns()
    
    # Fit GARCH(1,1) model
    results = garch.fit_garch_model(p=1, q=1)
    if results is None:
        return
    
    # Display results
    garch.display_results()
    
    # Plot volatility
    garch.plot_volatility()
    
    # Generate forecasts
    forecasts = garch.forecast_volatility(horizon=10)
    
    # Calculate VaR
    var_results = garch.calculate_var(confidence_level=0.05)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    example_analysis()