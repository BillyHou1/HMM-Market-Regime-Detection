import os
import pickle
import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.special import logsumexp
warnings.filterwarnings('ignore')

FEATURES = ['return_5d_z', 'volatility_20d_z', 'momentum_60d_z', 'downside_risk_20d_z']

def compute_bic(model, X, lengths=None):
    n, d = X.shape
    k = model.n_components
    cov = {'full': k * d * (d + 1) // 2, 'diag': k * d, 'spherical': k, 'tied': d * (d + 1) // 2}[model.covariance_type]
    n_params = (k - 1) + k * (k - 1) + k * d + cov
    return -2 * model.score(X, lengths) + n_params * np.log(n)

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

VOL_LABELS = {
    2: ["Low vol", "High vol"],
    3: ["Low vol", "Mid vol", "High vol"],
    4: ["Low vol", "Mid-low vol", "Mid-high vol", "High vol"],
    5: ["Very low vol", "Low vol", "Mid vol", "High vol", "Very high vol"],
}
def labels_for(k):
    return VOL_LABELS.get(k, [f"State {i}" for i in range(k)])
def filter_probs(model, X):
    # forward filtering: the state probabilities at t only use observations up to t
    log_b = model._compute_log_likelihood(X)
    log_a = np.log(model.transmat_ + 1e-300)
    a = np.log(model.startprob_ + 1e-300) + log_b[0]
    a -= logsumexp(a)
    out = np.empty_like(log_b)
    out[0] = a
    for t in range(1, len(X)):
        a = logsumexp(a[:, None] + log_a, axis=0) + log_b[t]
        a -= logsumexp(a)
        out[t] = a
    return np.exp(out)
def filter_states(model, X):
    return filter_probs(model, X).argmax(axis=1)
def name_states(states, data, k):
    stats = {s: {'ret': data.loc[states == s, 'return_5d'].mean(),
                 'vol': data.loc[states == s, 'volatility_20d'].mean(),
                 'count': int((states == s).sum())} for s in range(k)}
    order = sorted(range(k), key=lambda s: stats[s]['vol'])
    bank = labels_for(k)
    names = {s: bank[i] for i, s in enumerate(order)}
    print("State analysis (training period):")
    for s in order:
        r, v, n = stats[s]['ret'], stats[s]['vol'], stats[s]['count']
        print(f"  s={s} {names[s]:<14s} ret5d={r:+.4f} vol={v:.3f} {n/len(states)*100:.1f}%")
    return names
def train_hmm(data_path='data/processed/spy_features.csv',
              model_path='models/hmm_model.pkl',
              results_path='outputs/hmm_results.csv',
              min_states=2, max_states=5, train_end='2023-01-01'):
    print(f"Loading: {data_path}")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    train = data[data.index < pd.Timestamp(train_end)]
    X_train = train[FEATURES].values
    print(f"Data: {len(data)} rows, training on {len(train)} rows before {train_end}")
    model = select_model(X_train, min_states, max_states)
    names = name_states(model.predict(X_train), train, model.n_components)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {
        'model': model,
        'names': names,
        'features': FEATURES,
        'train_start': str(train.index[0].date()),
        'train_end': str(train.index[-1].date()),
        'n_states': model.n_components,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Model saved: {model_path}")
    states = filter_states(model, data[FEATURES].values)
    data['state'] = states
    data['state_name'] = [names[s] for s in states]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    data.to_csv(results_path)
    print(f"Results saved: {results_path}")
    return model, data, names
if __name__ == "__main__":
    train_hmm()