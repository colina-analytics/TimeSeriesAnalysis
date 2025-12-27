#%% Imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nixtla import NixtlaClient

sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "..", "TimeSeriesAnalysis-main", "TimeSeriesAnalysis-main")
    )
)

from myproject_utils import load_project_df, data_cleanup
from tsa_lth.analysis import naive_pred


#%% Load & prepare data
df = data_cleanup(load_project_df())

y  = df["power_MJ_s"].values
x1 = df["ambient_temp_C"].values
x2 = df["supply_temp_C"].values

N = len(y)
assert N == len(x1) == len(x2)

dates = pd.date_range(
    start="2018-01-01",
    periods=N,
    freq="h"
)

df_tg = pd.DataFrame({
    "unique_id": "series_1",
    "ds": dates,
    "y": y,
    "x1": x1,
    "x2": x2
})


#%% FAIR split (same as Kalman)
k = 7
buffer = 200

val_start = 2000
val_end   = val_start + 2 * 168
train_end = val_start - buffer

assert train_end > 0
assert val_end <= N
assert k < (val_end - val_start)

train_df = df_tg.iloc[:train_end]


#%% TimeGPT client
client = NixtlaClient(
    api_key="nixak-AGpLwPTIu7cnYSHdWESjpJlskqdfac3d7MK0zC5PO9P4mgVevuymTX0liGiZsKu1u7NDHCtmy8G89i94"
)


#%% IN-SAMPLE CHECK (API-safe)
# TimeGPT does NOT allow h=0 → use h=1 + add_history=True

# Future exog for the 1-step ahead forecast
X_future_insample = df_tg.loc[
    train_end : train_end,
    ["unique_id", "ds", "x1", "x2"]
]

forecast_insample = client.forecast(
    df=train_df,
    h=1,                     # must be > 0
    freq="h",
    target_col="y",
    time_col="ds",
    id_col="unique_id",
    X_df=X_future_insample,
    hist_exog_list=["x1", "x2"],  # Use exogenous inputs for fair comparison
    add_history=True,
    model="timegpt-1"
)

yhat_insample_full = forecast_insample["TimeGPT"].values

# Drop the final forecasted step (the h=1 future prediction)
yhat_insample = yhat_insample_full[:-1]

# TimeGPT needs warmup context, so it returns fewer predictions than input rows
# Align y_train to match (take the last N values)
n_preds = len(yhat_insample)
y_train = y[train_end - n_preds : train_end]

print(f"TimeGPT returned {n_preds} in-sample predictions (warmup: {train_end - n_preds} points)")
assert len(yhat_insample) == len(y_train), f"Length mismatch: {len(yhat_insample)} vs {len(y_train)}"

# TimeGPT MSE
mse_insample = np.mean((y_train - yhat_insample) ** 2)

# Naive predictor for comparison (1-step ahead, seasonal k=24)
# For in-sample, naive predicts y[t] = y[t-24]
insample_start = train_end - n_preds
insample_end = train_end
insample_indices = range(insample_start, insample_end)

y_naive_insample, _, _ = naive_pred(
    data=y,
    test_data_ind=insample_indices,
    k=1,
    season_k=24
)

mse_naive_insample = np.mean((y_train - y_naive_insample) ** 2)

print(f"\n{'='*50}")
print(f"IN-SAMPLE (1-step) COMPARISON:")
print(f"{'='*50}")
print(f"TimeGPT MSE: {mse_insample:.3f}")
print(f"Naive   MSE: {mse_naive_insample:.3f}")
print(f"Improvement: {(1 - mse_insample/mse_naive_insample) * 100:.2f}%")
print(f"{'='*50}")

plt.figure(figsize=(12,5))
plt.plot(y_train[-500:], label="Real (train)", alpha=0.7)
plt.plot(yhat_insample[-500:], label="TimeGPT (in-sample)", alpha=0.7)
plt.plot(y_naive_insample[-500:], label="Naive (seasonal)", alpha=0.5)
plt.title("In-sample 1-step predictions (last 500 points)")
plt.legend()
plt.grid(True)
plt.show()


#%% OUT-OF-SAMPLE forecast (fair)
h = val_end - train_end
assert h > k

X_future = df_tg.loc[
    train_end : train_end + h - 1,
    ["unique_id", "ds", "x1", "x2"]
]

assert len(X_future) == h

forecast = client.forecast(
    df=train_df,
    h=h,
    freq="h",
    target_col="y",
    time_col="ds",
    id_col="unique_id",
    X_df=X_future,
    hist_exog_list=["x1", "x2"],  # Use exogenous inputs for fair comparison
    model="timegpt-1"
)

yhat_full = forecast["TimeGPT"].values
assert len(yhat_full) == h


#%% ALIGNMENT (CRITICAL)
forecast_idx = np.arange(train_end, train_end + h)

mask = (forecast_idx >= val_start) & (forecast_idx < val_end)

yhat_test = yhat_full[mask]
y_test = y[val_start:val_end]

assert len(yhat_test) == len(y_test)

# k-step shift
yhat_k = yhat_test[:-k]
y_test = y_test[k:]

assert len(yhat_k) == len(y_test)
assert len(yhat_k) > 0


#%% MSE comparison
mse_tg = np.mean((y_test - yhat_k) ** 2)

y_naive, _, _ = naive_pred(
    data=y,
    test_data_ind=range(val_start, val_end),
    k=k,
    season_k=24
)

y_naive = y_naive[k:]
assert len(y_naive) == len(y_test)

mse_naive = np.mean((y_test - y_naive) ** 2)

print(f"\nTimeGPT MSE (k={k}): {mse_tg:.3f}")
print(f"Naive   MSE (k={k}): {mse_naive:.3f}")
print(f"Improvement: {(1 - mse_tg/mse_naive) * 100:.2f}%")


#%% Plot (out-of-sample)
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Real", alpha=0.7)
plt.plot(yhat_k, label="TimeGPT", alpha=0.7)
plt.title(f"TimeGPT {k}-step prediction (fair split)")
plt.legend()
plt.grid(True)
plt.show()
