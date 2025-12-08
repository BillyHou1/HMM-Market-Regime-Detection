# Feature Engineering
import numpy as np
import pandas as pd
# Constants for feature windows
RETURN_WIN = 5
VOL_WIN = 20
MOM_WIN = 60
STD_WIN = 252
# Function to compute features
def compute_features(close):
    daily_ret = close.pct_change()
    return_5d = close.pct_change(RETURN_WIN)
    volatility_20d = daily_ret.rolling(VOL_WIN).std() * np.sqrt(252)
    ma60 = close.rolling(MOM_WIN).mean()
    momentum_60d = (close - ma60) / ma60
    # downside risk: std of negative returns only
    def downside_std(x):
        neg = x[x < 0]
        return neg.std() if len(neg) >= 2 else 0.0
    downside_risk_20d = daily_ret.rolling(VOL_WIN).apply(downside_std) * np.sqrt(252)
    return pd.DataFrame({
        'return_5d': return_5d,
        'volatility_20d': volatility_20d,
        'momentum_60d': momentum_60d,
        'downside_risk_20d': downside_risk_20d
    }, index=close.index)
# Function to compute rolling z-score
def rolling_zscore(series, window=STD_WIN):
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma
# Main processing function
def process_features(input_path='data/raw/spy_raw.csv', output_path='data/processed/spy_features.csv'):
    print("Loading data...")
    data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    print(f"Loaded {len(data)} rows")
    print("Computing features...")
    features = compute_features(close)
    features['close'] = close
    print("Standardizing...")
    for col in ['return_5d', 'volatility_20d', 'momentum_60d', 'downside_risk_20d']:
        features[f'{col}_z'] = rolling_zscore(features[col]).clip(-3, 3)
    features = features.dropna()
    print(f"Final: {len(features)} rows")
    features.to_csv(output_path)
    print(f"Saved: {output_path}")
    return features

# Example usage
if __name__ == "__main__":
    features = process_features()
    print("\nSample:")
    print(features.tail())
