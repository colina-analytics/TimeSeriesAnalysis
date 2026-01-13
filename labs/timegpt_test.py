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


#%% Settings
k = 7                  # <- set k here (1 or 7 etc)
season_k = 24          # seasonal naive baseline

# MUST already exist (example):
windows = [
    ("Modeling", 1500, 2508),
    ("Validation", 2508, 3012),
    ("Test 1", 3012, 3180),
    ("Test 2", 3555, 3755),
]
assert "windows" in globals() and len(windows) == 4

name_to_win = {w[0]: (w[1], w[2]) for w in windows}
model_start, model_end = name_to_win["Modeling"]
eval_names = ["Validation", "Test 1", "Test 2"]


#%% Load & prepare data
df = data_cleanup(load_project_df())

y  = df["power_MJ_s"].to_numpy()
x1 = df["ambient_temp_C"].to_numpy()
x2 = df["supply_temp_C"].to_numpy()

N = len(y)
assert N == len(x1) == len(x2)

dates = pd.date_range(start="2018-01-01", periods=N, freq="h")

df_tg = pd.DataFrame({
    "unique_id": "series_1",
    "ds": dates,
    "y": y,
    "x1": x1,
    "x2": x2,
})

train_df = df_tg.iloc[model_start:model_end].copy()
assert len(train_df) == (model_end - model_start)


#%% TimeGPT client (DO NOT hardcode key)
client = NixtlaClient(
    api_key="nixak-AGpLwPTIu7cnYSHdWESjpJlskqdfac3d7MK0zC5PO9P4mgVevuymTX0liGiZsKu1u7NDHCtmy8G89i94"
)
#%% Forecast ONCE (from end of Modeling to the max window end)
max_end = max(e for _, _, e in windows)
H = max_end - model_end
assert H > 0

X_future = df_tg.loc[model_end : max_end - 1, ["unique_id", "ds", "x1", "x2"]].copy()
assert len(X_future) == H

forecast = client.forecast(
    df=train_df,
    h=H,
    freq="h",
    target_col="y",
    time_col="ds",
    id_col="unique_id",
    X_df=X_future,
    hist_exog_list=["x1", "x2"],
    model="timegpt-1",
)
yhat_all = forecast["TimeGPT"].to_numpy()
assert len(yhat_all) == H


#%% Evaluate per window (TimeGPT vs naive)
def eval_window(win_name: str, start: int, end: int, k: int):
    # TimeGPT slice aligned to global index
    a = start - model_end
    b = end - model_end
    yhat = yhat_all[a:b]
    ytrue = y[start:end]

    # k-step alignment (compare y[t] vs yhat[t-k])
    yhat_k = yhat[:-k]
    ytrue_k = ytrue[k:]
    assert len(yhat_k) == len(ytrue_k) and len(yhat_k) > 0

    mse_tg = np.mean((ytrue_k - yhat_k) ** 2)

    # Naive
    y_naive, _, _ = naive_pred(
        data=y,
        test_data_ind=range(start, end),
        k=k,
        season_k=season_k,
    )
    y_naive_k = y_naive[k:]
    assert len(y_naive_k) == len(ytrue_k)

    mse_nv = np.mean((ytrue_k - y_naive_k) ** 2)

    return mse_tg, mse_nv, ytrue_k, yhat_k, y_naive_k


results = {}
for nm in eval_names:
    s, e = name_to_win[nm]
    mse_tg, mse_nv, ytrue_k, yhat_k, ynaive_k = eval_window(nm, s, e, k)
    results[nm] = (mse_tg, mse_nv)

    print(f"\n=== {nm} (idx {s}:{e}) | k={k} ===")
    print(f"TimeGPT MSE: {mse_tg:.3f}")
    print(f"Naive   MSE: {mse_nv:.3f}")
    print(f"Improvement: {(1 - mse_tg/mse_nv) * 100:.2f}%")

    # Plot last 500 points of the aligned series
    L = min(500, len(ytrue_k))
    plt.figure(figsize=(10, 4))
    plt.plot(ytrue_k[-L:], label="Real", alpha=0.8)
    plt.plot(yhat_k[-L:],  label="TimeGPT", alpha=0.8)
    plt.plot(ynaive_k[-L:], label="Naive", alpha=0.4)
    plt.title(f"{nm}: TimeGPT vs Naive ({k}-step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
#%% DIAGNOSIS PLOT

plt.figure(figsize=(10,4))
plt.plot(y[model_start:model_end], label="Modeling")
plt.plot(y[model_end:max_end], label="Post-model", alpha=0.8)
plt.legend()
plt.title("Level shift between training and evaluation")
plt.grid(True)
plt.show()



#%% NEW METHOD


def eval_window_timegpt_expanding(name, start, end, k):
    # --- train ---
    train_df = df_tg.iloc[model_start:start].copy()
    assert len(train_df) > 0

    # --- horizon ---
    h = end - start
    assert h > k

    # --- future exog ---
    X_future = df_tg.loc[start:end-1, ["unique_id", "ds", "x1", "x2"]]

    # --- forecast ---
    fc = client.forecast(
        df=train_df,
        h=h,
        freq="h",
        target_col="y",
        time_col="ds",
        id_col="unique_id",
        X_df=X_future,
        hist_exog_list=["x1", "x2"],
        model="timegpt-1",
    )

    yhat = fc["TimeGPT"].to_numpy()
    ytrue = y[start:end]

    # --- k-step alignment ---
    yhat_k = yhat[:-k]
    ytrue_k = ytrue[k:]
    assert len(yhat_k) == len(ytrue_k)

    mse_tg = np.mean((ytrue_k - yhat_k) ** 2)

    # --- naive ---
    y_naive, _, _ = naive_pred(
        data=y,
        test_data_ind=range(start, end),
        k=k,
        season_k=season_k,
    )
    y_naive_k = y_naive[k:]
    mse_nv = np.mean((ytrue_k - y_naive_k) ** 2)

    return mse_tg, mse_nv, ytrue_k, yhat_k, y_naive_k


for name in ["Validation", "Test 1", "Test 2"]:
    s, e = name_to_win[name]

    mse_tg, mse_nv, ytrue_k, yhat_k, ynaive_k = \
        eval_window_timegpt_expanding(name, s, e, k)

    print(f"\n=== {name} | expanding | k={k} ===")
    print(f"TimeGPT MSE: {mse_tg:.3f}")
    print(f"Naive   MSE: {mse_nv:.3f}")
    print(f"Improvement: {(1 - mse_tg/mse_nv) * 100:.2f}%")

    L = min(500, len(ytrue_k))
    plt.figure(figsize=(10,4))
    plt.plot(ytrue_k[-L:], label="Real")
    plt.plot(yhat_k[-L:], label="TimeGPT")
    plt.plot(ynaive_k[-L:], label="Naive", alpha=0.4)
    plt.title(f"{name}: TimeGPT vs Naive (expanding, {k}-step)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


