import pandas as pd

# 1. Raw
raw = pd.read_csv("data/raw/AAPL_all_time_data.csv", parse_dates=["Date"])
print("RAW HEAD:\n", raw.head(), "\n")

# 2. Processed
feat = pd.read_csv("data/processed/AAPL_features.csv", parse_dates=["Date"])
print("PROCESSED HEAD:\n", feat.head())