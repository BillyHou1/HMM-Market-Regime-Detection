"""Walk-forward backtest comparing the SPY-only HMM to asset-class-specific HMMs."""
import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engine import compute_features, rolling_zscore
from predict import predict_regime_latest, model_path_for_class, MODELS_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSET_CLASS_MAP = {
    'TLT': 'bond_etf', 'IEF': 'bond_etf', 'AGG': 'bond_etf', 'LQD': 'bond_etf',
    'GLD': 'commodity_etf', 'USO': 'commodity_etf', 'UNG': 'commodity_etf', 'SLV': 'commodity_etf',
    'VNQ': 'reit', 'IYR': 'reit',
    'UUP': 'currency_etf', 'FXE': 'currency_etf',
    'BTC-USD': 'crypto', 'ETH-USD': 'crypto',
    'SPY': 'equity_us', 'QQQ': 'equity_us', 'AAPL': 'equity_us', 'NVDA': 'equity_us',
    'EFA': 'equity_intl', 'EEM': 'equity_intl',
}

FEATURES = ['return_5d_z', 'volatility_20d_z', 'momentum_60d_z', 'downside_risk_20d_z']

def fetch_close(ticker, start='2018-01-01', end='2026-01-01'):
    cache_dir = os.path.join(REPO_ROOT, 'data', 'backtest_raw')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{ticker.replace(".", "_").replace("-", "_")}.csv')
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        df.to_csv(cache_path)
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    return df[col].dropna()

def features_z(close):
    f = compute_features(close)
    base = ['return_5d', 'volatility_20d', 'momentum_60d', 'downside_risk_20d']
    for c in base:
        f[f'{c}_z'] = rolling_zscore(f[c]).clip(-3, 3)
    return f.dropna()

def regime_conditional_var(returns_train, states_train, alpha=0.05):
    """Compute α-percentile of returns within each state — used as forward VaR."""
    out = {}
    for s in np.unique(states_train):
        mask = states_train == s
        if mask.sum() >= 20:
            out[s] = np.percentile(returns_train[mask], alpha * 100)
        else:
            out[s] = np.percentile(returns_train, alpha * 100)
    return out

def kupiec(n, n_viol, alpha=0.05):
    if n_viol == 0 or n_viol == n:
        return None
    p = n_viol / n
    lr = -2 * (n_viol * np.log(alpha/p) + (n - n_viol) * np.log((1-alpha)/(1-p)))
    return float(1 - scipy_stats.chi2.cdf(lr, df=1))

def christoffersen(violations):
    n00 = n01 = n10 = n11 = 0
    v = list(violations)
    for i in range(1, len(v)):
        if not v[i-1] and not v[i]: n00 += 1
        elif not v[i-1] and v[i]:   n01 += 1
        elif v[i-1] and not v[i]:   n10 += 1
        else:                        n11 += 1
    if n00 + n01 == 0 or n10 + n11 == 0:
        return None
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p in (0, 1):
        return None
    eps = 1e-10
    L0 = n00*np.log(1-p+eps) + n01*np.log(p+eps) + n10*np.log(1-p+eps) + n11*np.log(p+eps)
    L1 = n00*np.log(1-p01+eps) + n01*np.log(p01+eps) + n10*np.log(1-p11+eps) + n11*np.log(p11+eps)
    lr = -2 * (L0 - L1)
    return float(1 - scipy_stats.chi2.cdf(lr, df=1))

def evaluate_one(ticker, model_path, alpha=0.05):
    """Walk-forward eval of one HMM on one ticker."""
    close = fetch_close(ticker)
    if close is None or len(close) < 600:
        return None
    f = features_z(close)
    if f.empty:
        return None
    daily_ret = close.pct_change().reindex(f.index)

    with open(model_path, 'rb') as fh:
        bundle = pickle.load(fh)
    model = bundle['model']
    n_states = bundle['n_states']

    # Need enough training history before first eval year
    eval_start_idx = max(252, int(0.6 * len(f)))
    train_X = f[FEATURES].values[:eval_start_idx]
    train_ret = daily_ret.values[:eval_start_idx]
    train_states = model.predict(train_X)
    var_per_state = regime_conditional_var(train_ret, train_states, alpha)

    eval_X = f[FEATURES].values[eval_start_idx:]
    eval_ret = daily_ret.values[eval_start_idx:]
    eval_states = model.predict(eval_X)
    eval_var = np.array([var_per_state.get(int(s),
                          np.percentile(train_ret, alpha * 100)) for s in eval_states])
    violations = eval_ret < eval_var
    n = len(violations)
    n_viol = int(violations.sum())
    state_dist = {bundle['names'].get(int(s), str(s)): int((eval_states == s).sum())
                  for s in np.unique(eval_states)}
    return {
        'ticker': ticker, 'n': n, 'n_viol': n_viol,
        'viol_rate': n_viol / n if n else 0,
        'expected_rate': alpha,
        'kupiec_p': kupiec(n, n_viol, alpha),
        'chris_p': christoffersen(violations),
        'state_distribution': state_dist,
        'states_used': sorted(set(int(s) for s in eval_states)),
        'n_states_total': n_states,
    }

def main():
    p = argparse.ArgumentParser(description='Walk-forward backtest of HMMs')
    p.add_argument('--tickers', default='TLT,USO,GLD,VNQ,UUP,BTC-USD,EFA,EEM,SPY,AAPL',
                   help='comma-sep list')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--out', default='outputs/backtest_summary.json')
    args = p.parse_args()
    tickers = args.tickers.split(',')

    legacy_path = os.path.join(MODELS_DIR, 'hmm_model.pkl')
    if not os.path.exists(legacy_path):
        print(f"⚠  legacy model not found: {legacy_path}; train it first")
        return
    print(f"\n{'='*72}\n  Walk-Forward Backtest — Legacy SPY-only vs Asset-Class Experts\n"
          f"  alpha={args.alpha}  tickers={tickers}\n{'='*72}\n")

    rows = []
    for t in tickers:
        cls = ASSET_CLASS_MAP.get(t)
        expert_path = model_path_for_class(cls) if cls else legacy_path
        if expert_path == legacy_path:
            print(f"  ⚠  no expert model for {t} ({cls}); using legacy for both")

        res_legacy = evaluate_one(t, legacy_path, args.alpha)
        res_expert = evaluate_one(t, expert_path, args.alpha)
        if res_legacy is None or res_expert is None:
            print(f"  {t}: no data, skip")
            continue

        def fmt(r):
            kp = f"{r['kupiec_p']:.3f}" if r['kupiec_p'] is not None else 'n/a'
            cp = f"{r['chris_p']:.3f}" if r['chris_p'] is not None else 'n/a'
            return f"viol={r['viol_rate']:.2%}  kupiec={kp}  chris={cp}  states_used={r['states_used']}"
        print(f"\n  {t}  ({cls or 'legacy'})")
        print(f"    legacy: {fmt(res_legacy)}")
        print(f"    expert: {fmt(res_expert)}")
        rows.append({
            'ticker': t, 'asset_class': cls,
            'legacy': res_legacy, 'expert': res_expert,
            'expert_path': os.path.basename(expert_path),
        })

    # Aggregate: how often does expert deliver a closer-to-alpha violation rate?
    legacy_dist = [abs(r['legacy']['viol_rate'] - args.alpha) for r in rows]
    expert_dist = [abs(r['expert']['viol_rate'] - args.alpha) for r in rows]
    print(f"\n{'='*72}\n  Aggregate")
    print(f"  legacy mean |viol-alpha|: {np.mean(legacy_dist):.4f}")
    print(f"  expert mean |viol-alpha|: {np.mean(expert_dist):.4f}")
    expert_better = sum(1 for l, e in zip(legacy_dist, expert_dist) if e < l)
    print(f"  expert closer to alpha on {expert_better}/{len(rows)} tickers")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import json
    with open(args.out, 'w') as f:
        json.dump({'rows': rows, 'alpha': args.alpha,
                    'mean_legacy_dist': float(np.mean(legacy_dist)),
                    'mean_expert_dist': float(np.mean(expert_dist)),
                    'expert_better_count': expert_better,
                    'n_tickers': len(rows)}, f, indent=2, default=str)
    print(f"  Saved: {args.out}")

if __name__ == "__main__":
    main()
