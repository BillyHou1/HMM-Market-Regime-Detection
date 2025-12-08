# Data Pipeline
import os
import pandas as pd
import yfinance as yf
# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
# Function to download data
def download_data(ticker='SPY', start='2010-01-01', end='2025-11-27'):
    print(f"Downloading {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=True)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    # flatten columns if multi-index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"Downloaded {len(data)} trading days")
    #save raw data
    raw_path = os.path.join(DATA_DIR, 'raw', f'{ticker.lower()}_raw.csv')
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    data.to_csv(raw_path)
    print(f"Saved: {raw_path}")
    return data
# Function to validate data
def validate_data(data):
    close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    print(f"\nValidation:")
    print(f"  Days: {len(data)}")
    print(f"  Range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"  Price: ${data[close_col].min():.2f} - ${data[close_col].max():.2f}")
    print(f"  Missing: {data.isnull().sum().sum()}")
    return data.isnull().sum().sum() == 0

# Example usage
if __name__ == "__main__":
    data = download_data()
    validate_data(data)
    print("\nSample:")
    print(data.tail())
