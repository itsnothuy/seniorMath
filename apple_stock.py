# apple_stock.py

import yfinance as yf
import pandas as pd

def fetch_apple_stock_data():
    # Create a Ticker object for Apple
    apple = yf.Ticker("AAPL")
    
    # Fetch all available historical data
    df = apple.history(period="max")
    
    # Reset index so 'Date' becomes a column
    df = df.reset_index()
    
    # Save to CSV
    df.to_csv("AAPL_all_time_data.csv", index=False)
    
    # Display first few rows
    print("✅ Apple stock data fetched successfully!")
    print(df.head())

if __name__ == "__main__":
    fetch_apple_stock_data()
