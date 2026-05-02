import os
import pickle
import numpy as np
import pandas as pd
from feature_engine import compute_features, rolling_zscore, STD_WIN

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'hmm_model.pkl')

def model_path_for_class(asset_class):
    candidate = os.path.join(MODELS_DIR, f'hmm_{asset_class}.pkl')
    return candidate if os.path.exists(candidate) else DEFAULT_MODEL_PATH

def load_bundle(model_path=DEFAULT_MODEL_PATH):
    with open(model_path, 'rb') as f:
        b = pickle.load(f)
    b.setdefault('feature_means', None)
    b.setdefault('feature_stds', None)
    return b

def _features_zscored(close, bundle):
    f = compute_features(close)
    base = ['return_5d', 'volatility_20d', 'momentum_60d', 'downside_risk_20d']
    for c in base:
        f[f'{c}_z'] = rolling_zscore(f[c]).clip(-3, 3)
    return f.dropna()

def predict_regime_series(close, model_path=DEFAULT_MODEL_PATH):
    b = load_bundle(model_path)
    feats = _features_zscored(close, b)
    X = feats[b['features']].values
    states = b['model'].predict(X)
    out = pd.DataFrame({'state': states,
                        'state_name': [b['names'][s] for s in states]},
                       index=feats.index)
    return out, b

def _smooth_probs(probs, alpha=0.02):
    import numpy as np
    p = np.asarray(probs, dtype=float)
    k = len(p)
    return (p + alpha) / (1.0 + k * alpha)

def predict_regime_latest(close, model_path=DEFAULT_MODEL_PATH, smooth=True, asset_class=None):
    if asset_class and model_path == DEFAULT_MODEL_PATH:
        model_path = model_path_for_class(asset_class)
    b = load_bundle(model_path)
    feats = _features_zscored(close, b)
    if feats.empty:
        raise ValueError("Not enough history to compute features (need ~252 days).")
    X = feats[b['features']].values
    state = int(b['model'].predict(X[-1:])[0])
    probs_raw = b['model'].predict_proba(X[-1:])[0]
    probs = _smooth_probs(probs_raw) if smooth else probs_raw
    try:
        loglik = float(b['model'].score(X[-1:]))
    except Exception:
        loglik = None
    return {
        'state': state,
        'state_name': b['names'][state],
        'probs': {b['names'][s]: float(probs[s]) for s in range(b['n_states'])},
        'probs_raw': {b['names'][s]: float(probs_raw[s]) for s in range(b['n_states'])},
        'log_likelihood': loglik,
        'features_latest': {f: float(feats.iloc[-1][f]) for f in b['features']},
        'as_of': str(feats.index[-1].date()),
        'train_window': f"{b['train_start']} to {b['train_end']}",
        'expert_model': b.get('asset_class', 'spy_legacy'),
        'expert_tickers': b.get('tickers'),
        'model_path': os.path.basename(model_path),
    }

if __name__ == "__main__":
    import yfinance as yf
    df = yf.download('SPY', period='2y', progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    out = predict_regime_latest(close)
    print(out)
