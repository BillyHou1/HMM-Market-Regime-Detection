import os
import numpy as np
import pandas as pd

RETURN_WIN, VOL_WIN, MOM_WIN, STD_WIN = 5, 20, 60, 252
ANN = np.sqrt(252)

def compute_features(close):
    daily_ret = close.pct_change()
    return_5d = close.pct_change(RETURN_WIN)
    volatility_20d = daily_ret.rolling(VOL_WIN).std() * ANN
    ma60 = close.rolling(MOM_WIN).mean()
    momentum_60d = (close - ma60) / ma60
    def downside_std(x):
        neg = x[x < 0]
        return neg.std() if len(neg) >= 2 else 0.0
    downside_risk_20d = daily_ret.rolling(VOL_WIN).apply(downside_std, raw=False) * ANN
    return pd.DataFrame({
        'return_5d': return_5d,
        'volatility_20d': volatility_20d,
        'momentum_60d': momentum_60d,
        'downside_risk_20d': downside_risk_20d,
    }, index=close.index)

def rolling_zscore(series, window=STD_WIN):
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma

def process_features(input_path, output_path='data/processed/spy_features.csv'):
    print(f"Loading: {input_path}")
    data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    print(f"Loaded {len(data)} rows; computing features...")
    features = compute_features(close)
    features['close'] = close
    for col in ['return_5d', 'volatility_20d', 'momentum_60d', 'downside_risk_20d']:
        features[f'{col}_z'] = rolling_zscore(features[col]).clip(-3, 3)
    features = features.dropna()
    print(f"Final: {len(features)} rows after dropna")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path)
    print(f"Saved: {output_path}")
    return features

if __name__ == "__main__":
    process_features('data/raw/spy_raw.csv')
