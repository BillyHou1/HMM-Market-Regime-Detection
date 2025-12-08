# Main script to run the full pipeline
from data_pipeline import download_data
from feature_engine import process_features
from hmm_model import train_hmm

print("Download data")
data = download_data()

print("\nCompute features")
features = process_features()

print("\nTrain HMM")
model, results = train_hmm()

print("\nDone")
print(f"Results saved to outputs/hmm_results.csv")
