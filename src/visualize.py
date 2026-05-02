import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PALETTE = ['#2E7D32', '#1976D2', '#FBC02D', '#E64A19', '#6A1B9A', '#00838F', '#5D4037']

def _color_map(names):
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted(names.keys()))}

def plot_price_states(data, names, out_path):
    cmap = _color_map(names)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(data.index, data['close'], color='black', lw=0.7, alpha=0.7)
    for s, nm in names.items():
        mask = data['state'] == s
        ax.fill_between(data.index, data['close'].min(), data['close'].max(),
                        where=mask, color=cmap[s], alpha=0.18, label=f"{s}: {nm}")
    ax.set_title('SPY Price with HMM Regime Overlay')
    ax.set_ylabel('Adj Close')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_state_timeline(data, names, out_path):
    cmap = _color_map(names)
    fig, ax = plt.subplots(figsize=(13, 2.4))
    for s in names:
        mask = data['state'] == s
        ax.scatter(data.index[mask], np.full(mask.sum(), s), color=cmap[s], s=4, label=names[s])
    ax.set_yticks(sorted(names.keys()))
    ax.set_yticklabels([names[s] for s in sorted(names.keys())])
    ax.set_title('Regime Timeline')
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_feature_distributions(data, names, out_path):
    feats = ['return_5d', 'volatility_20d', 'momentum_60d', 'downside_risk_20d']
    cmap = _color_map(names)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, f in zip(axes.flat, feats):
        for s, nm in names.items():
            vals = data.loc[data['state'] == s, f].dropna()
            if len(vals): ax.hist(vals, bins=40, color=cmap[s], alpha=0.45, label=nm, density=True)
        ax.set_title(f); ax.legend(fontsize=7)
    fig.suptitle('Feature distribution by regime')
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_transition_matrix(model, names, out_path):
    A = model.transmat_
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(A, cmap='Blues', vmin=0, vmax=1)
    labels = [names[s] for s in range(len(names))]
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right'); ax.set_yticklabels(labels)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}", ha='center', va='center',
                    color='white' if A[i,j] > 0.5 else 'black', fontsize=9)
    ax.set_title('Transition matrix P(s_t+1 | s_t)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def plot_state_durations(data, names, out_path):
    runs = []
    cur, run = data['state'].iloc[0], 1
    for s in data['state'].iloc[1:]:
        if s == cur: run += 1
        else: runs.append((cur, run)); cur, run = s, 1
    runs.append((cur, run))
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = _color_map(names)
    by_state = {s: [r for st, r in runs if st == s] for s in names}
    ax.boxplot([by_state[s] for s in sorted(names.keys())],
               labels=[names[s] for s in sorted(names.keys())], showfliers=False)
    ax.set_ylabel('Run length (days)'); ax.set_title('Regime persistence')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"Saved: {out_path}")

def make_all(data, model, names, out_dir='outputs/figures'):
    os.makedirs(out_dir, exist_ok=True)
    plot_price_states(data, names, os.path.join(out_dir, 'price_states.png'))
    plot_state_timeline(data, names, os.path.join(out_dir, 'state_timeline.png'))
    plot_feature_distributions(data, names, os.path.join(out_dir, 'feature_dist.png'))
    plot_transition_matrix(model, names, os.path.join(out_dir, 'transition_matrix.png'))
    plot_state_durations(data, names, os.path.join(out_dir, 'state_durations.png'))

if __name__ == "__main__":
    import pickle
    data = pd.read_csv('outputs/hmm_results.csv', index_col=0, parse_dates=True)
    with open('models/hmm_model.pkl', 'rb') as f:
        ck = pickle.load(f)
    make_all(data, ck['model'], ck['names'])
