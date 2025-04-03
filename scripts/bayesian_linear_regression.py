# bayesian_linear_regression.py

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

def run_bayesian_linear_regression():
    # 1. Load processed data
    processed_csv_path = os.path.join("data", "processed", "AAPL_features.csv")
    df = pd.read_csv(processed_csv_path, index_col='Date', parse_dates=True)

    # 2. Define features & target
    features = ['MA_20', 'MA_50', 'Momentum', 'Volume_Change']
    target = 'LogReturn'

    X = df[features].values
    y = df[target].values

    # 3. Train/test split (80/20)
    train_size = int(len(df)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. Bayesian Linear Model
    with pm.Model() as blr_model:
        beta_0 = pm.Normal("beta_0", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X_train.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)

        mu = beta_0 + pm.math.dot(X_train, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        trace_blr = pm.sample(
            draws=2000,
            tune=1000,
            chains=2,
            cores=2,
            target_accept=0.9
        )

    # 5. Diagnostics & plots
    az.plot_trace(trace_blr, var_names=["beta_0", "beta", "sigma"])
    plt.show()

    summary = az.summary(trace_blr, var_names=["beta_0", "beta", "sigma"])
    print(summary)

    # You can also save the trace if desired:
    # az.to_netcdf(trace_blr, "blr_trace.nc")

if __name__ == "__main__":
    run_bayesian_linear_regression()
