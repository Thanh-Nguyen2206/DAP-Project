"""
Demo Data Download - For Report Screenshot
Demonstrate data fetching capabilities
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("DATA DOWNLOAD DEMONSTRATION - STOCK MARKET ANALYSIS PROJECT")
print("="*80)

# Demo 1: Real-time data fetching
print("\nDEMO 1: REAL-TIME DATA FETCHING FROM YAHOO FINANCE")
print("-"*80)

symbols = ['AAPL', 'GOOGL', 'MSFT']
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print(f"Fetching data for: {', '.join(symbols)}")
print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print()

for symbol in symbols:
    try:
        print(f"Downloading {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        print(f"[SUCCESS] {symbol}: Successfully fetched {len(df)} days of data")
        print(f"   Latest Price: ${df['Close'].iloc[-1]:.2f}")
        print(f"   Volume: {df['Volume'].iloc[-1]:,.0f}")
        print(f"   Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print()
        
    except Exception as e:
        print(f"[ERROR] {symbol}: Error - {e}")
        print()

# Demo 2: Data structure demonstration
print("\nDEMO 2: DATA STRUCTURE EXAMPLE")
print("-"*80)

# Fetch sample data
sample_ticker = yf.Ticker('AAPL')
sample_data = sample_ticker.history(period='5d')

print("Sample AAPL Stock Data (Last 5 days):")
print(sample_data.to_string())
print()

print("Data Information:")
print(f"  Shape: {sample_data.shape}")
print(f"  Columns: {list(sample_data.columns)}")
print(f"  Memory Usage: {sample_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
print()

# Demo 3: Data quality check
print("\nDEMO 3: DATA QUALITY ASSESSMENT")
print("-"*80)

print("Missing Values:")
print(sample_data.isnull().sum())
print()

print("Data Completeness:")
completeness = (1 - sample_data.isnull().sum() / len(sample_data)) * 100
print(completeness)
print()

print("Descriptive Statistics:")
print(sample_data.describe())
print()

# Demo 4: Synthetic data generation (for demo mode)
print("\nDEMO 4: SYNTHETIC DATA GENERATION (DEMO MODE)")
print("-"*80)

dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
np.random.seed(42)

base_price = 150
returns = np.random.normal(0.001, 0.02, len(dates))
prices = [base_price]

for ret in returns[1:]:
    prices.append(prices[-1] * (1 + ret))

demo_data = pd.DataFrame({
    'Open': prices,
    'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
    'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
    'Close': prices,
    'Volume': np.random.randint(1000000, 10000000, len(dates))
}, index=dates)

print("Generated Synthetic Demo Data:")
print(demo_data.to_string())
print()

print("[SUCCESS] Demo data generation successful!")
print(f"   Generated {len(demo_data)} days of synthetic data")
print(f"   Price range: ${demo_data['Close'].min():.2f} - ${demo_data['Close'].max():.2f}")
print()

print("="*80)
print("ALL DATA DOWNLOAD DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
print("="*80)
