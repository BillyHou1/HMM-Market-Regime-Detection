"""Train one HMM per asset class on a basket of liquid tickers."""
import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from feature_engine import compute_features, rolling_zscore
from hmm_model import compute_bic, labels_for
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSET_CLASS_BASKETS = {
    'equity_us':     ['SPY', 'QQQ', 'DIA', 'IWM'],
    'equity_intl':   ['EFA', 'EEM', 'VEA', 'VWO'],
    'bond_etf':      ['TLT', 'IEF', 'AGG', 'LQD', 'HYG'],
    'commodity_etf': ['GLD', 'USO', 'UNG', 'DBC', 'SLV'],
    'reit':          ['VNQ', 'IYR', 'XLRE', 'SCHH'],
    'currency_etf':  ['UUP', 'FXE', 'FXY'],
    'crypto':        ['BTC-USD', 'ETH-USD'],
}

FEATURES = ['return_5d_z', 'volatility_20d_z', 'momentum_60d_z', 'downside_risk_20d_z']

def fetch_basket(tickers, start='2014-01-01', end='2025-11-26', save_dir='data/raw_multi'):
    os.makedirs(save_dir, exist_ok=True)
    out = {}
    for t in tickers:
        path = os.path.join(save_dir, f'{t.replace(".", "_").replace("-", "_")}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            print(f"  fetching {t}...")
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                print(f"  ⚠  {t}: no data, skip")
                continue
            df.to_csv(path)
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        out[t] = df[col].dropna()
        print(f"  {t}: {len(out[t])} rows")
    return out

def features_from_close(close):
    f = compute_features(close)
    base = ['return_5d', 'volatility_20d', 'momentum_60d', 'downside_risk_20d']
    for c in base:
        f[f'{c}_z'] = rolling_zscore(f[c]).clip(-3, 3)
    return f.dropna()

def stack_basket_features(closes):
    parts = []
    for t, c in closes.items():
        f = features_from_close(c)
        if not f.empty:
            f['_source'] = t
            parts.append(f)
    if not parts:
        return None, None
    stacked = pd.concat(parts, axis=0).sort_index()
    X = stacked[FEATURES].values
    return X, stacked

def select_model(X, min_states=2, max_states=5, n_seeds=5, cov_type='diag'):
    print(f"  Selecting HMM ({min_states}..{max_states} states, cov={cov_type})...")
    results = []
    for n in range(min_states, max_states + 1):
        best_model, best_score = None, -np.inf
        for seed in range(n_seeds):
            m = hmm.GaussianHMM(n_components=n, covariance_type=cov_type,
                                n_iter=300, random_state=42 + seed,
                                tol=1e-4, init_params='stmc', params='stmc')
            try:
                m.fit(X)
                s = m.score(X)
            except Exception as e:
                continue
            if s > best_score:
                best_score, best_model = s, m
        if best_model is None:
            continue
        bic = compute_bic(best_model, X)
        results.append({'n': n, 'logL': best_score, 'bic': bic, 'model': best_model})
        print(f"    n={n}: logL={best_score:10.1f}  BIC={bic:10.1f}")
    if not results:
        return None
    best = min(results, key=lambda x: x['bic'])
    print(f"  Selected: {best['n']} states  (BIC={best['bic']:.1f})")
    return best['model']

def name_states(model, X, stacked, asset_class='equity_us'):
    states = model.predict(X)
    k = model.n_components
    stats = {}
    for s in range(k):
        mask = states == s
        if mask.sum() == 0:
            stats[s] = {'ret': 0.0, 'vol': 1.0, 'count': 0}
            continue
        sub = stacked.iloc[mask]
        stats[s] = {'ret': sub['return_5d'].mean(),
                    'vol': sub['volatility_20d'].mean(),
                    'count': int(mask.sum())}
    score = {s: stats[s]['ret'] - 0.5 * stats[s]['vol'] for s in range(k)}
    order = sorted(range(k), key=lambda s: -score[s])
    bank = labels_for(asset_class, k)
    names = {s: bank[i] for i, s in enumerate(order)}
    print("  States:")
    for s in range(k):
        r, v, n = stats[s]['ret'], stats[s]['vol'], stats[s]['count']
        print(f"    s={s} {names[s]:<16s} ret={r:+.4f} vol={v:.3f}  n={n}")
    return states, names, stats

def train_one_class(asset_class, tickers, models_dir='models', skip_download=False,
                    min_states=2, max_states=5, cov_type='diag'):
    print(f"\n{'='*60}\n  Training HMM for {asset_class}: {tickers}\n{'='*60}")
    closes = fetch_basket(tickers,
                          start='2014-01-01' if not skip_download else '2014-01-01',
                          end='2025-11-26')
    if not closes:
        print(f"  ⚠  no data for {asset_class}, skip")
        return None
    X, stacked = stack_basket_features(closes)
    if X is None or len(X) < 1000:
        print(f"  ⚠  insufficient features ({len(X) if X is not None else 0} rows), skip")
        return None
    print(f"  Features: {len(X)} rows across {len(closes)} tickers")
    model = select_model(X, min_states, max_states, cov_type=cov_type)
    if model is None:
        print(f"  ⚠  HMM fit failed for {asset_class}")
        return None
    states, names, stats = name_states(model, X, stacked, asset_class=asset_class)
    feat_means = pd.DataFrame(X, columns=FEATURES).mean().to_dict()
    feat_stds = pd.DataFrame(X, columns=FEATURES).std().to_dict()
    bundle = {
        'model': model,
        'names': names,
        'features': FEATURES,
        'feature_means': feat_means,
        'feature_stds': feat_stds,
        'asset_class': asset_class,
        'tickers': tickers,
        'train_start': str(min(c.index.min() for c in closes.values()).date()),
        'train_end': str(max(c.index.max() for c in closes.values()).date()),
        'n_states': model.n_components,
        'cov_type': cov_type,
        'n_train_rows': len(X),
    }
    out_path = os.path.join(REPO_ROOT, models_dir, f'hmm_{asset_class}.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  ✓  Saved: {out_path}")
    return bundle

def main():
    p = argparse.ArgumentParser(description='Train per-asset-class HMMs')
    p.add_argument('--classes', default='all',
                   help='comma-separated subset of: ' + ','.join(ASSET_CLASS_BASKETS))
    p.add_argument('--skip-download', action='store_true')
    p.add_argument('--min-states', type=int, default=2)
    p.add_argument('--max-states', type=int, default=5)
    p.add_argument('--cov', default='diag', choices=['diag', 'full', 'spherical', 'tied'])
    args = p.parse_args()
    classes = (list(ASSET_CLASS_BASKETS.keys()) if args.classes == 'all'
               else args.classes.split(','))
    summary = {}
    for c in classes:
        if c not in ASSET_CLASS_BASKETS:
            print(f"  ⚠  unknown class {c}, skip"); continue
        b = train_one_class(c, ASSET_CLASS_BASKETS[c],
                             skip_download=args.skip_download,
                             min_states=args.min_states, max_states=args.max_states,
                             cov_type=args.cov)
        if b:
            summary[c] = {'n_states': b['n_states'], 'rows': b['n_train_rows'],
                           'tickers': b['tickers']}
    print(f"\n{'='*60}\n  Summary: trained {len(summary)}/{len(classes)} classes\n{'='*60}")
    for c, info in summary.items():
        print(f"  {c:14s} states={info['n_states']}  rows={info['rows']}  tickers={info['tickers']}")

if __name__ == "__main__":
    main()
