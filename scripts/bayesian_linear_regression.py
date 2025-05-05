# bayesian_linear_regression.py

import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def run_bayesian_linear_regression():
    # 1. Load processed data
    processed_csv_path = os.path.join("data", "processed", "AAPL_features.csv")
    df = pd.read_csv(processed_csv_path, index_col='Date', parse_dates=True)

    # 2. Define features & target
    features = ['MA_20', 'MA_50', 'Momentum', 'Volume_Change']
    target = 'LogReturn'
    
    # 3. Clean the data: Replace infinities with NaN and drop rows with missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    print("Missing values before cleaning:")
    print(df[features + [target]].isna().sum())
    
    df_clean = df.dropna(subset=features + [target])
    print("Data shape before cleaning:", df.shape)
    print("Data shape after cleaning:", df_clean.shape)

    # 4. Extract features and target from the cleaned DataFrame
    X = df_clean[features].values
    y = df_clean[target].values

    # 5. Train/test split (80/20)
    train_size = int(len(df_clean) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 6. Verify that training data contains only finite values
    if not np.all(np.isfinite(X_train)):
        print("Warning: X_train still contains non-finite values!")
    if not np.all(np.isfinite(y_train)):
        print("Warning: y_train still contains non-finite values!")

    # 7. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print statistics for debugging
    print("X_train_scaled mean (per feature):", np.mean(X_train_scaled, axis=0))
    print("X_train_scaled std (per feature):", np.std(X_train_scaled, axis=0))

    # 8. Define the Bayesian Linear Model using standardized features
    with pm.Model() as blr_model:
        beta_0 = pm.Normal("beta_0", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X_train_scaled.shape[1])
        sigma = pm.HalfCauchy("sigma", beta=2)

        mu = beta_0 + pm.math.dot(X_train_scaled, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        # 9. Compute a MAP estimate to use as an initial point
        init_vals = pm.find_MAP(model=blr_model)
        print("\nRaw MAP initial values:", init_vals)
        
        # The MAP dict includes both 'sigma' and 'sigma_log__'; the compiled logp function expects
        # only the free variable names, which for sigma is 'sigma_log__'. Filter out 'sigma'.
        init_vals_filtered = {k: v for k, v in init_vals.items() if k != "sigma"}
        print("Filtered MAP initial values:", init_vals_filtered)

        # Compile a log probability function and evaluate the MAP point.
        logp_fn = blr_model.compile_logp()
        print("Initial log probability (MAP):", logp_fn(init_vals_filtered))

        # 10. Run the sampler with error handling to output debug information if needed.
        try:
            trace_blr = pm.sample(
                draws=2000,
                tune=1000,
                chains=2,
                cores=2,
                target_accept=0.9
            )
        except pm.exceptions.SamplingError as e:
            print("SamplingError encountered during pm.sample()!")
            print("Debugging information:")
            blr_model.debug()  # Prints detailed info about model evaluation
            raise e

    # 11. Diagnostics & plots
    az.plot_trace(trace_blr, var_names=["beta_0", "beta", "sigma"])
    plt.show()

    summary = az.summary(trace_blr, var_names=["beta_0", "beta", "sigma"])
    print(summary)

if __name__ == "__main__":
    run_bayesian_linear_regression()
