import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Paths
sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "..", "TimeSeriesAnalysis-main", "TimeSeriesAnalysis-main")
    )
)

from myproject_utils import load_project_df, data_cleanup
from tsa_lth.analysis import naive_pred


# ======================
# Load & prepare data
# ======================
df = data_cleanup(load_project_df())

y  = df["power_MJ_s"].values
x1 = df["ambient_temp_C"].values
x2 = df["supply_temp_C"].values

dates = pd.date_range(
    start="2018-01-01",
    periods=len(y),
    freq="h"   # <-- lowercase h (fix warning)
)

df_prophet = pd.DataFrame({
    "ds": dates,
    "y": y,
    "x1": x1,
    "x2": x2
})


# ======================
# FAIR split (same as Kalman)
# ======================
k = 7
buffer = 200
val_start = 2000
val_end   = val_start + 2*168

train_end = val_start - buffer

train_df = df_prophet.iloc[:train_end]
test_df  = df_prophet.iloc[val_start:val_end]


# ======================
# Prophet model
# ======================
m = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)

m.add_regressor("x1")
m.add_regressor("x2")

m.fit(train_df)


# ======================
# Forecast
# ======================
future = df_prophet.iloc[:val_end].copy()
forecast = m.predict(future)

yhat = forecast["yhat"].values


# ======================
# k-step alignment
# ======================
yhat_prophet_full = forecast["yhat"].values
yhat_k = yhat_prophet_full[val_start - train_end : val_end - train_end]
y_test = y[val_start:val_end]


# ======================
# MSE comparison
# ======================
mse_prophet = np.mean((y_test - yhat_k)**2)

y_naive, _, _ = naive_pred(
    data=y,
    test_data_ind=range(val_start, val_end),
    k=k,
    season_k=24
)

mse_naive = np.mean((y_test - y_naive)**2)

print(f"Prophet+X MSE (k={k}): {mse_prophet:.3f}")
print(f"Naive       MSE (k={k}): {mse_naive:.3f}")
print(f"Improvement: {(1 - mse_prophet/mse_naive)*100:.2f}%")


# ======================
# Plot
# ======================
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Real")
plt.plot(yhat_k, label="Prophet + regressors")
plt.title(f"Prophet + regressors ({k}-step)")
plt.legend()
plt.grid(True)
plt.show()
