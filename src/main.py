import os
import argparse
import pickle
from data_pipeline import download_data, validate_data
from feature_engine import process_features
from hmm_model import train_hmm
from visualize import make_all

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', default='SPY')
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-11-27')
    p.add_argument('--min-states', type=int, default=2)
    p.add_argument('--max-states', type=int, default=5)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--model-dir', default='models')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--skip-download', action='store_true')
    p.add_argument('--skip-plots', action='store_true')
    args = p.parse_args()

    raw_dir = os.path.join(args.data_dir, 'raw')
    proc_path = os.path.join(args.data_dir, 'processed', f'{args.ticker.lower()}_features.csv')
    raw_path = os.path.join(raw_dir, f'{args.ticker.lower()}_raw.csv')
    model_path = os.path.join(args.model_dir, 'hmm_model.pkl')
    results_path = os.path.join(args.out_dir, 'hmm_results.csv')
    fig_dir = os.path.join(args.out_dir, 'figures')

    if args.skip_download and os.path.exists(raw_path):
        print(f"Reusing: {raw_path}")
    else:
        data, raw_path = download_data(args.ticker, args.start, args.end, raw_dir)
        validate_data(data)

    process_features(raw_path, proc_path)
    model, results, names = train_hmm(proc_path, model_path, results_path,
                                       args.min_states, args.max_states)
    if not args.skip_plots:
        make_all(results, model, names, fig_dir)

if __name__ == "__main__":
    main()
