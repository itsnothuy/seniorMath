import yfinance as yf, pandas as pd, numpy as np
TICKER = "AAPL"
raw = yf.download(TICKER, start="1980-01-02", progress=False)

# Feature engineering
df = raw.assign(
    LogReturn = np.log(raw["Adj Close"]).diff(),
    MA_20     = raw["Adj Close"].rolling(20).mean(),
    MA_50     = raw["Adj Close"].rolling(50).mean(),
    Momentum  = raw["Adj Close"].diff(10),
    Volume_Change = raw["Volume"].pct_change()
).dropna()

# Train / test split (chronological)
train_frac = 0.8
split = int(len(df)*train_frac)
df.to_csv("AAPL_features.csv")
df.iloc[:split].to_csv("AAPL_train.csv")
df.iloc[split:].to_csv("AAPL_test.csv")
print(df.tail())
