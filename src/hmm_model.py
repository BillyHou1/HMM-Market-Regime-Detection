# HMM Model

import numpy as np
import pandas as pd
import pickle
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')
# Selected features
FEATURES = ['return_5d_z', 'volatility_20d_z', 'momentum_60d_z', 'downside_risk_20d_z']
# Function to compute BIC
def compute_bic(model, X):
    n, d = X.shape
    k = model.n_components
    # params: init + transitions + means + covariances
    n_params = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2
    logL = model.score(X)
    return -2 * logL + n_params * np.log(n)

# Function to select the best HMM model based on BIC
def select_model(X, min_states=2, max_states=5):
    print("Selecting model...")
    results = []
    for n in range(min_states, max_states + 1):
        best_model, best_score = None, -np.inf
        for seed in range(3):  # try a few random inits
            model = hmm.GaussianHMM(n_components=n, covariance_type='full',
                                     n_iter=100, random_state=42+seed)
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        bic = compute_bic(best_model, X)
        results.append({'n': n, 'bic': bic, 'model': best_model})
        print(f"  {n} states: BIC = {bic:.1f}")
    best = min(results, key=lambda x: x['bic'])
    print(f"Selected: {best['n']} states")
    return best['model']
# Function to analyze and name states
def analyze_states(model, X, data):
    states = model.predict(X)
    n_states = model.n_components
    # calc stats per state
    stats = {}
    for s in range(n_states):
        mask = states == s
        stats[s] = {
            'ret': data.loc[mask, 'return_5d'].mean(),
            'vol': data.loc[mask, 'volatility_20d'].mean(),
            'count': mask.sum()
        }
    ret_med = np.median([stats[s]['ret'] for s in range(n_states)])
    vol_med = np.median([stats[s]['vol'] for s in range(n_states)])
    # assign names
    names = {}
    print("\nState Analysis:")
    for s in range(n_states):
        r, v = stats[s]['ret'], stats[s]['vol']
        if r > ret_med and v < vol_med:
            names[s] = "Steady Bull"
        elif r > ret_med:
            names[s] = "Volatile Rally"
        elif v < vol_med:
            names[s] = "Quiet Decline"
        else:
            names[s] = "Panic"
        pct = stats[s]['count'] / len(states) * 100
        print(f"  State {s} ({names[s]}): {pct:.1f}%")
    return states, names
# Main training function
def train_hmm(data_path='data/processed/spy_features.csv'):
    print("Loading Data...")
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    X = data[FEATURES].values
    print(f"Data: {len(data)} rows, {len(FEATURES)} features")
    model = select_model(X)
    states, names = analyze_states(model, X, data)
    # save
    with open('models/hmm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved: models/hmm_model.pkl")
    data['state'] = states
    data['state_name'] = [names[s] for s in states]
    data.to_csv('outputs/hmm_results.csv')
    print("Results saved: outputs/hmm_results.csv")
    return model, data

# Example usage
if __name__ == "__main__":
    model, results = train_hmm()
