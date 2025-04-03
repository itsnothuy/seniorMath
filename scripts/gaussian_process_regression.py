# gaussian_process_regression.py

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

def run_gaussian_process_regression():
    # 1. Load processed data
    processed_csv_path = os.path.join("data", "processed", "AAPL_features.csv")
    df = pd.read_csv(processed_csv_path, index_col='Date', parse_dates=True)

    # 2. Define features & target
    features = ['MA_20', 'MA_50', 'Momentum', 'Volume_Change']
    target = 'LogReturn'

    X = df[features].values
    y = df[target].values

    # 3. Train/test split
    train_size = int(len(df)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. GP Model
    with pm.Model() as gpr_model:
        length_scale = pm.Exponential("length_scale", 1.0)
        eta = pm.HalfCauchy("eta", 2.5)
        cov_func = eta**2 * pm.gp.cov.ExpQuad(X_train.shape[1], length_scale)

        gp = pm.gp.Marginal(cov_func=cov_func)
        sigma = pm.HalfCauchy("sigma", beta=2)

        y_obs = gp.marginal_likelihood("y_obs", X=X_train, y=y_train, noise=sigma)

        trace_gpr = pm.sample(
            draws=2000,
            tune=1000,
            chains=2,
            cores=2,
            target_accept=0.9
        )

    # 5. Diagnostics
    az.plot_trace(trace_gpr, var_names=["length_scale", "eta", "sigma"])
    plt.show()

    summary = az.summary(trace_gpr, var_names=["length_scale", "eta", "sigma"])
    print(summary)

if __name__ == "__main__":
    run_gaussian_process_regression()
