#%% Imports
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "..", "TimeSeriesAnalysis-main", "TimeSeriesAnalysis-main")
    )
)
from myproject_utils import load_project_df, data_cleanup
from tsa_lth.analysis import naive_pred


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


#%% Load & prepare data
df = data_cleanup(load_project_df())

y  = df["power_MJ_s"].to_numpy()
x1 = df["ambient_temp_C"].to_numpy()
x2 = df["supply_temp_C"].to_numpy()

dates = pd.date_range(start="2018-01-01", periods=len(y), freq="h")

df_prophet = pd.DataFrame({
    "ds": dates,
    "y": y,
    "x1": x1,
    "x2": x2,
})


#%% Train on Modeling window (ONLY)
train_df = df_prophet.iloc[model_start:model_end].copy()


#%% Prophet model
m = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
)

m.add_regressor("x1")
m.add_regressor("x2")
m.fit(train_df)


#%% Forecast once (from end of Modeling to max end), then slice per window
max_end = max(e for _, _, e in windows)
H = max_end - model_end
assert H > k

future = df_prophet.iloc[model_end:max_end].copy()
fc = m.predict(future)
yhat_all = fc["yhat"].to_numpy()
assert len(yhat_all) == H


#%% Evaluate per window
def eval_window(name, start, end, k):
    # prophet slice aligned to global index
    a = start - model_end
    b = end - model_end
    yhat = yhat_all[a:b]
    ytrue = y[start:end]
    dates = df[start:end-7]['date']

    # k-step alignment (same convention as your other scripts)
    yhat_k = yhat[:-k]
    ytrue_k = ytrue[k:]
    assert len(yhat_k) == len(ytrue_k) and len(yhat_k) > 0

    mse_prophet = np.mean((ytrue_k - yhat_k) ** 2)

    y_naive, _, _ = naive_pred(
        data=y,
        test_data_ind=range(start, end),
        k=k,
        season_k=season_k,
    )
    y_naive_k = y_naive[k:]
    
    if True:
        ytrue0  = ytrue_k - ytrue_k.mean()
        
        yhat0   = yhat_k  - yhat_k.mean()       # Prophet gets its own intercept
        ynaive0 = y_naive_k - y_naive_k.mean()  # Naive gets its own intercept
        
        mse_prophet = np.mean((ytrue0 - yhat0)**2)
        mse_naive   = np.mean((ytrue0 - ynaive0)**2)
        
        return mse_prophet, mse_naive, ytrue0, yhat0, ynaive0, dates
    
    else:
        mse_naive = np.mean((ytrue_k - y_naive_k) ** 2)
        return mse_prophet, mse_naive, ytrue_k, yhat_k, y_naive_k, dates

        


for nm in eval_names:
    s, e = name_to_win[nm]
    mse_p, mse_n, ytrue_k, yhat_k, ynaive_k, dates = eval_window(nm, s, e, k)

    print(f"\n=== {nm} | Prophet+X | k={k} ===")
    print(f"Prophet MSE: {mse_p:.3f}")
    print(f"Naive   MSE: {mse_n:.3f}")
    print(f"Improvement: {(1 - mse_p/mse_n) * 100:.2f}%")

    L = min(500, len(ytrue_k))
    plt.figure(figsize=(10,4))
    plt.plot(dates, ytrue_k[-L:], label="Real")
    plt.plot(dates, yhat_k[-L:],  label="Prophet+X")
    plt.plot(dates, ynaive_k[-L:], label="Naive", alpha=0.4)
    plt.title(f"{nm}: Prophet+X ({k}-step)")
    plt.ylabel('Power (MJ/s)')
    plt.grid(True)
    plt.xlabel
    plt.legend()
    plt.tight_layout()
    plt.show()
