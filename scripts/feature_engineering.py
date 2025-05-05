# feature_engineering.py

import pandas as pd
import numpy as np
import os

def create_features():
    # 1. Read raw data
    raw_csv_path = os.path.join("data", "raw", "AAPL_all_time_data.csv")
    df = pd.read_csv(raw_csv_path)
    
    # 2. Convert Date to datetime, set index, sort
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # 3. Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log1p(df['Return'])
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Momentum'] = df['MA_20'] - df['MA_50']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # 4. Drop rows with NaNs
    df.dropna(inplace=True)
    
    # 5. Save processed data
    os.makedirs("data/processed", exist_ok=True)
    processed_csv_path = os.path.join("data", "processed", "AAPL_features.csv")
    df.to_csv(processed_csv_path)
    
    print("✅ Features created and saved successfully!")
    print(df.head())

if __name__ == "__main__":
    create_features()