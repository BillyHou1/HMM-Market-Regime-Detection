import os
import pandas as pd
import yfinance as yf

def download_data(ticker='SPY', start='2010-01-01', end='2025-11-27', save_dir='data/raw'):
    print(f"Downloading {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=True, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"Downloaded {len(data)} trading days")
    os.makedirs(save_dir, exist_ok=True)
    raw_path = os.path.join(save_dir, f'{ticker.lower()}_raw.csv')
    data.to_csv(raw_path)
    print(f"Saved: {raw_path}")
    return data, raw_path

def validate_data(data):
    close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    n_missing = data.isnull().sum().sum()
    print(f"Validation: days={len(data)} range={data.index.min().date()}->{data.index.max().date()} "
          f"price=${data[close_col].min():.2f}-${data[close_col].max():.2f} missing={n_missing}")
    return n_missing == 0

if __name__ == "__main__":
    data, _ = download_data()
    validate_data(data)
