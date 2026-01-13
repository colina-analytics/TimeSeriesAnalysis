#%% Imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "..", "TimeSeriesAnalysis-main", "TimeSeriesAnalysis-main")
    )
)

from myproject_utils import load_project_df, data_cleanup
from tsa_lth.analysis import naive_pred

# pip install lightgbm
from lightgbm import LGBMRegressor


#%% Settings
k = 7
season_k = 24

windows = [
    ("Modeling", 1500, 2508),
    ("Validation", 2508, 3012),
    ("Test 1", 3012, 3180),
    ("Test 2", 3555, 3755),
]
name_to_win = {w[0]: (w[1], w[2]) for w in windows}
model_start, model_end = name_to_win["Modeling"]
eval_names = ["Validation", "Test 1", "Test 2"]

# feature lags (hours)
y_lags  = [1, 24, 168]
x_lags  = [0, 1, 24]


#%% Load data
df = data_cleanup(load_project_df())

y  = df["power_MJ_s"].to_numpy()
x1 = df["ambient_temp_C"].to_numpy()
x2 = df["supply_temp_C"].to_numpy()
N = len(y)

d = pd.DataFrame({"y": y, "x1": x1, "x2": x2})


#%% Build lag features
def add_lags(d, col, lags):
    for L in lags:
        d[f"{col}_lag{L}"] = d[col].shift(L)
    return d

d_feat = d.copy()
d_feat = add_lags(d_feat, "y",  y_lags)
d_feat = add_lags(d_feat, "x1", x_lags)
d_feat = add_lags(d_feat, "x2", x_lags)

max_lag = max(y_lags + x_lags)
d_feat = d_feat.iloc[max_lag:].copy()
d_feat["t"] = np.arange(max_lag, N)  # original index


#%% Train on Modeling window (1-step model)
train_mask = (d_feat["t"] >= model_start) & (d_feat["t"] < model_end)
train_df = d_feat.loc[train_mask].copy()

feature_cols = [c for c in train_df.columns if c not in ["y", "t"]]
X_train = train_df[feature_cols].to_numpy()
y_train = train_df["y"].to_numpy()

model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=0,
)
model.fit(X_train, y_train)


#%% k-step forecasting via recursion
def forecast_k_window(start, end, k):
    """
    Produce k-step-ahead predictions aligned like your other code:
    compare y_true[k:] vs yhat[:-k]
    """
    assert end > start
    yhat_1step = []

    # local copy for recursive y-lags (use true history up to start-1)
    y_hist = d["y"].to_numpy().copy()

    for t in range(start, end):
        row = {}

        # y lags (recursive uses predicted values once we step into window)
        for L in y_lags:
            row[f"y_lag{L}"] = y_hist[t - L]

        # x lags (known, not recursive)
        for L in x_lags:
            row[f"x1_lag{L}"] = d["x1"].to_numpy()[t - L] if L > 0 else d["x1"].to_numpy()[t]
            row[f"x2_lag{L}"] = d["x2"].to_numpy()[t - L] if L > 0 else d["x2"].to_numpy()[t]

        X = np.array([row[c] for c in feature_cols], dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        yhat_1step.append(pred)

        # update history for future lags inside window
        y_hist[t] = pred

    yhat_1step = np.array(yhat_1step)
    ytrue = y[start:end]

    # align as k-step (same convention you used)
    yhat_k = yhat_1step[:-k]
    ytrue_k = ytrue[k:]
    assert len(yhat_k) == len(ytrue_k) and len(yhat_k) > 0

    return ytrue_k, yhat_k


#%% Evaluate windows
for nm in eval_names:
    s, e = name_to_win[nm]
    if s - max_lag <= 0:
        raise ValueError("Window starts too early for chosen lags.")

    ytrue_k, yhat_k = forecast_k_window(s, e, k)
    mse_ml = np.mean((ytrue_k - yhat_k) ** 2)

    y_naive, _, _ = naive_pred(
        data=y,
        test_data_ind=range(s, e),
        k=k,
        season_k=season_k,
    )
    y_naive_k = y_naive[k:]
    mse_nv = np.mean((ytrue_k - y_naive_k) ** 2)

    print(f"\n=== {nm} | LightGBM ARX | k={k} ===")
    print(f"ML     MSE: {mse_ml:.3f}")
    print(f"Naive  MSE: {mse_nv:.3f}")
    print(f"Improvement: {(1 - mse_ml/mse_nv) * 100:.2f}%")

    L = min(500, len(ytrue_k))
    plt.figure(figsize=(10,4))
    plt.plot(ytrue_k[-L:], label="Real")
    plt.plot(yhat_k[-L:],  label="LightGBM")
    plt.plot(y_naive_k[-L:], label="Naive", alpha=0.4)
    plt.title(f"{nm}: LightGBM vs Naive ({k}-step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
