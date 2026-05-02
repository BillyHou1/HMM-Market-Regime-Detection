import os
import pickle
import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm
warnings.filterwarnings('ignore')

FEATURES = ['return_5d_z', 'volatility_20d_z', 'momentum_60d_z', 'downside_risk_20d_z']

def compute_bic(model, X):
    n, d = X.shape
    k = model.n_components
    n_params = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2
    return -2 * model.score(X) + n_params * np.log(n)

def select_model(X, min_states=2, max_states=5, n_seeds=3):
    print(f"Selecting HMM ({min_states}..{max_states} states)...")
    results = []
    for n in range(min_states, max_states + 1):
        best_model, best_score = None, -np.inf
        for seed in range(n_seeds):
            m = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=200, random_state=42+seed)
            try:
                m.fit(X)
                s = m.score(X)
            except Exception:
                continue
            if s > best_score:
                best_score, best_model = s, m
        bic = compute_bic(best_model, X)
        results.append({'n': n, 'bic': bic, 'model': best_model})
        print(f"  n={n}: logL={best_score:.1f} BIC={bic:.1f}")
    best = min(results, key=lambda x: x['bic'])
    print(f"Selected: {best['n']} states (lowest BIC)")
    return best['model']

LABEL_BANK = {
    2: ["Bull", "Bear"],
    3: ["Bull", "Sideways", "Bear"],
    4: ["Steady Bull", "Volatile Rally", "Quiet Decline", "Panic"],
    5: ["Steady Bull", "Volatile Rally", "Quiet Decline", "Correction", "Crisis"],
    6: ["Steady Bull", "Volatile Rally", "Sideways", "Quiet Decline", "Correction", "Crisis"],
    7: ["Steady Bull", "Volatile Rally", "Sideways", "Quiet Decline", "Correction", "High-Vol", "Crisis"],
}

ASSET_CLASS_LABEL_BANKS = {
    'equity_us': LABEL_BANK,
    'equity_intl': LABEL_BANK,
    'bond_etf': {
        2: ["Stable Yields", "Vol Spike"],
        3: ["Falling Yields", "Stable", "Rising Yields"],
        4: ["Falling Yields", "Stable", "Drifting Higher", "Yield Shock"],
        5: ["Falling Yields", "Stable", "Drifting", "Rising Yields", "Yield Shock"],
    },
    'commodity_etf': {
        2: ["Trending Up", "Trending Down"],
        3: ["Up Trend", "Range", "Down Trend"],
        4: ["Bull Run", "Range Bound", "Choppy", "Bear / Crash"],
        5: ["Bull Run", "Drifting Up", "Range", "Drifting Down", "Crash"],
    },
    'currency_etf': {
        2: ["Stable", "Stress"],
        3: ["Strengthening", "Range", "Weakening"],
        4: ["Strengthening", "Range", "Drifting Weak", "FX Stress"],
        5: ["Strengthening", "Stable", "Range", "Weakening", "FX Stress"],
    },
    'reit': {
        2: ["Healthy", "Distressed"],
        3: ["Healthy Growth", "Range", "Distressed"],
        4: ["Healthy Growth", "Cap-Rate Stable", "Cap-Rate Rising", "Real-Estate Stress"],
        5: ["Healthy Growth", "Stable", "Drifting", "Cap-Rate Rising", "Real-Estate Stress"],
    },
    'crypto': {
        2: ["Bull", "Bear/Crash"],
        3: ["Bull", "Range", "Bear/Crash"],
        4: ["Bull Run", "Quiet Bull", "Distribution", "Capitulation"],
        5: ["Bull Run", "Quiet Bull", "Range", "Distribution", "Capitulation"],
    },
    'equity_hk': LABEL_BANK,
    'equity_cn': LABEL_BANK,
    'index': LABEL_BANK,
}

def labels_for(asset_class, k):
    bank = ASSET_CLASS_LABEL_BANKS.get(asset_class, LABEL_BANK)
    return bank.get(k, [f"State {i}" for i in range(k)])

def name_states(model, X, data):
    states = model.predict(X)
    k = model.n_components
    stats = {s: {'ret': data.loc[states == s, 'return_5d'].mean(),
                 'vol': data.loc[states == s, 'volatility_20d'].mean(),
                 'count': int((states == s).sum())} for s in range(k)}
    score = {s: stats[s]['ret'] - 0.5 * stats[s]['vol'] for s in range(k)}
    order = sorted(range(k), key=lambda s: -score[s])
    bank = LABEL_BANK.get(k, [f"State {i}" for i in range(k)])
    names = {s: bank[i] for i, s in enumerate(order)}
    print("State analysis:")
    for s in range(k):
        r, v, n = stats[s]['ret'], stats[s]['vol'], stats[s]['count']
        print(f"  s={s} {names[s]:<16s} ret={r:+.4f} vol={v:.3f} {n/len(states)*100:.1f}%")
    return states, names

def train_hmm(data_path='data/processed/spy_features.csv',
              model_path='models/hmm_model.pkl',
              results_path='outputs/hmm_results.csv',
              min_states=2, max_states=5):
    print(f"Loading: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    X = data[FEATURES].values
    print(f"Data: {len(data)} rows, {len(FEATURES)} features")
    model = select_model(X, min_states, max_states)
    states, names = name_states(model, X, data)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {
        'model': model,
        'names': names,
        'features': FEATURES,
        'feature_means': data[FEATURES].mean().to_dict(),
        'feature_stds': data[FEATURES].std().to_dict(),
        'train_start': str(data.index[0].date()),
        'train_end': str(data.index[-1].date()),
        'n_states': model.n_components,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Model saved: {model_path}")
    data['state'] = states
    data['state_name'] = [names[s] for s in states]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    data.to_csv(results_path)
    print(f"Results saved: {results_path}")
    return model, data, names

if __name__ == "__main__":
    train_hmm()
