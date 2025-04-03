# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def exploratory_data_analysis():
    # 1. Read raw data
    csv_path = os.path.join("data", "raw", "AAPL_all_time_data.csv")
    df = pd.read_csv(csv_path)
    
    # 2. Convert Date to datetime, set index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # 3. Basic checks
    print(df.info())
    print(df.describe())
    print("Missing values:", df.isnull().sum())
    
    # 4. Plot Closing Price
    plt.plot(df.index, df['Close'])
    plt.title("Apple Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()

if __name__ == "__main__":
    exploratory_data_analysis()
