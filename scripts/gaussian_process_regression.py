# import os
# import pandas as pd
# import numpy as np
# import pymc as pm
# import arviz as az
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler


# def run_gaussian_process_regression():
#     """
#     Loads processed Apple stock data and runs a Gaussian Process regression
#     with basic sampler settings, mirroring the first Bayesian LR script.
#     """

#     # 1. Load processed data
#     processed_csv_path = os.path.join("data", "processed", "AAPL_features.csv")
#     df = pd.read_csv(processed_csv_path, index_col='Date', parse_dates=True)

#     # 2. Define features & target
#     features = ['MA_20', 'MA_50', 'Momentum', 'Volume_Change']
#     target = 'LogReturn'

#     # 3. Clean the data: Replace infinities with NaN and drop rows with missing values
#     df = df.replace([np.inf, -np.inf], np.nan)
#     print("Missing values before cleaning:")
#     print(df[features + [target]].isna().sum())

#     df_clean = df.dropna(subset=features + [target])
#     print("Data shape before cleaning:", df.shape)
#     print("Data shape after cleaning:", df_clean.shape)

#     # 4. Extract features and target
#     X = df_clean[features].values
#     y = df_clean[target].values

#     # 5. Train/test split (80/20)
#     train_size = int(len(df_clean) * 0.8)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]

#     # 6. Standardize features (important for GP kernels)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     print("X_train_scaled mean (per feature):", np.mean(X_train_scaled, axis=0))
#     print("X_train_scaled std (per feature):", np.std(X_train_scaled, axis=0))

#     # 7. Define Gaussian Process model
#     with pm.Model() as gpr_model:
#         # Hyperpriors
#         length_scale = pm.Exponential("length_scale", 1.0)
#         eta = pm.HalfCauchy("eta", beta=2.5)
#         # Covariance function
#         cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=X_train_scaled.shape[1], ls=length_scale)

#         # GP marginal
#         gp = pm.gp.Marginal(cov_func=cov)
#         # Noise term
#         sigma = pm.HalfCauchy("sigma", beta=2)

#         # Marginal likelihood
#         y_obs = gp.marginal_likelihood("y_obs", X=X_train_scaled, y=y_train, noise=sigma)

#         # 8. Compute a MAP estimate for initialization
#         init_vals = pm.find_MAP(model=gpr_model)
#         print("Raw MAP initial values:", init_vals)
#         # Filter out direct 'sigma'
#         init_vals_filtered = {k: v for k, v in init_vals.items() if k != "sigma"}
#         print("Filtered MAP initial values:", init_vals_filtered)

#         # Compile log probability at MAP
#         logp_fn = gpr_model.compile_logp()
#         print("Initial log probability (MAP):", logp_fn(init_vals_filtered))

#         # 9. Run sampler with basic settings
#         try:
#             trace_gpr = pm.sample(
#                 draws=2000,
#                 tune=1000,
#                 chains=2,
#                 cores=2,
#                 target_accept=0.9
#             )
#         except pm.exceptions.SamplingError as e:
#             print("SamplingError encountered during pm.sample()!")
#             print("Debugging information:")
#             gpr_model.debug()
#             raise e

#     # 10. Diagnostics & plots
#     az.plot_trace(trace_gpr, var_names=["length_scale", "eta", "sigma"])
#     plt.show()

#     summary = az.summary(trace_gpr, var_names=["length_scale", "eta", "sigma"])
#     print(summary)


# if __name__ == "__main__":
#     run_gaussian_process_regression()
import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def run_gaussian_process_regression():
    """
    Loads processed Apple stock data and runs a Gaussian Process regression
    with basic sampler settings, mirroring the first Bayesian LR script.
    """

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

    # 4. Extract features and target
    X = df_clean[features].values
    y = df_clean[target].values

    # 5. Train/test split (80/20)
    train_size = int(len(df_clean) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 6. Standardize features (important for GP kernels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("X_train_scaled mean (per feature):", np.mean(X_train_scaled, axis=0))
    print("X_train_scaled std (per feature):", np.std(X_train_scaled, axis=0))

    # 7. Define Gaussian Process model
    with pm.Model() as gpr_model:
        # Hyperpriors
        length_scale = pm.Exponential("length_scale", 1.0)
        eta = pm.HalfCauchy("eta", beta=2.5)
        # Covariance function
        cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=X_train_scaled.shape[1], ls=length_scale)

        # GP marginal
        gp = pm.gp.Marginal(cov_func=cov)
        # Noise term
        sigma = pm.HalfCauchy("sigma", beta=2)

        # Marginal likelihood
        y_obs = gp.marginal_likelihood("y_obs", X=X_train_scaled, y=y_train, noise=sigma)

        # 8. Compute MAP and filter for unconstrained variables
        init_vals = pm.find_MAP(model=gpr_model)
        print("MAP initial values:", init_vals)
        # Only keep unconstrained parameters (free_RVs)
        free_names = [rv.name for rv in gpr_model.free_RVs]
        init_vals_filtered = {name: init_vals[name] for name in free_names}
        print("Filtered MAP initial values (unconstrained space):", init_vals_filtered)

        # Evaluate log probability at MAP on unconstrained space
        logp_fn = gpr_model.compile_logp()
        lp = logp_fn(**init_vals_filtered)
        print("Initial log probability (MAP):", lp)

        # 9. Run sampler with basic settings
        try:
            trace_gpr = pm.sample(
                draws=2000,
                tune=1000,
                chains=2,
                cores=2,
                target_accept=0.9,
                start=init_vals_filtered
            )
        except pm.exceptions.SamplingError as e:
            print("SamplingError encountered during pm.sample()!")
            print("Debugging information:")
            gpr_model.debug()
            raise e

    # 10. Diagnostics & plots
    az.plot_trace(trace_gpr, var_names=["length_scale", "eta", "sigma"])
    plt.show()

    summary = az.summary(trace_gpr, var_names=["length_scale", "eta", "sigma"])
    print(summary)


if __name__ == "__main__":
    run_gaussian_process_regression()
