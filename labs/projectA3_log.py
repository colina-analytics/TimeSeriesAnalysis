# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
import importlib
import scipy.io as sio

# Add path to tsa_lth library
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.getcwd(), "..", "TimeSeriesAnalysis-main", "TimeSeriesAnalysis-main"
        )
    )
)

# Import and reload to get the latest changes
from myproject_utils import (
    data_cleanup,
    get_modeling_dataset,
    load_project_df,
    plot_ccf,
    simulate_data,
    test_model,
)
import tsa_lth.analysis
import tsa_lth.modelling
import tsa_lth.tests

importlib.reload(tsa_lth.analysis)
importlib.reload(tsa_lth.modelling)
importlib.reload(tsa_lth.tests)

from tsa_lth.analysis import box_cox, plotACFnPACF, normplot, xcorr, pzmap, kovarians, naive_pred
from tsa_lth.modelling import estimateARMA, estimateBJ, polydiv
from tsa_lth.modelling import filter as tsa_filter
from tsa_lth.tests import whiteness_test, check_if_normal


import pandas as pd
import scipy.io as sio


# %% SELECTING DATA

start_model = 1500
weeks_model = 6
h = 168

df = load_project_df()
df = data_cleanup(df)

x, y = get_modeling_dataset(df=df, start=start_model, n_weeks=weeks_model, plot=True)

# Quick sanity plots (modeling window)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(x, linewidth=1)
ax.set_title("Input x (modeling window)")
ax.set_xlabel("Sample")
ax.set_ylabel("x")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(y, linewidth=1)
ax.set_title("Output y (modeling window)")
ax.set_xlabel("Sample")
ax.set_ylabel("y")
plt.tight_layout()
plt.show()

# %% SEE VALIDATION / TEST PERIODS

def clamp(a, lo, hi):
    return max(lo, min(a, hi))

n = len(df)
windows = [
    ("Modeling",    start_model,                 start_model + weeks_model*h),
    ("Validation",  start_model + weeks_model*h, start_model + (weeks_model+3)*h),
    ("Test 1",      start_model + (weeks_model+3)*h, start_model + (weeks_model+4)*h),
    ("Test 2",      n - 200,                     n),
]

colors = {
    "Modeling":   "tab:blue",
    "Validation": "tab:orange",
    "Test 1":     "tab:green",
    "Test 2":     "tab:red",
}

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["power_MJ_s"], linewidth=1)

ymin, ymax = ax.get_ylim()
for name, s, e in windows:
    s = clamp(s, 0, n - 1)
    e = clamp(e, 1, n)
    if e <= s:
        continue

    x0 = df["date"].iloc[s]
    x1 = df["date"].iloc[e - 1]

    ax.axvspan(x0, x1, alpha=0.25, color=colors[name], label=name)
    ax.text(x0 + (x1 - x0) / 2, ymax - 0.08 * (ymax - ymin),
            name, ha="center", va="top", fontsize=10)

ax.set_title("Power usage with modeling / validation / test splits")
# ax.set_xlabel("Date")
ax.set_ylabel("Power (MJ/s)")
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.show()


# %% TRANSFORM

lambda_max, offsetValue = box_cox(y, plotIt=True, titleStr='Box-Cox normality plot', transform=False)
print(f'lambda_max(y) = {lambda_max:.4f}')

lambda_max, offsetValue = box_cox(x, plotIt=True, titleStr='Box-Cox normality plot', transform=False)
print(f'lambda_max(x) = {lambda_max:.4f}')


# %% NORM-PLOT OUTPUT

normplot(y, titleStr='y')
normplot(np.log(y), titleStr='log(y))')
normplot(np.sqrt(y), titleStr='sqrt(y))')
# sqtr power is better, but original was OK anyways

log_y = np.log(y)

# %% NORM-PLOT INPUT

normplot(x, titleStr='x')
normplot(np.log(x), titleStr='log(x)')
normplot(np.sqrt(x), titleStr='sqrt(x)')

# Keep x as-is

# %% MODEL FOR INPUT

noLags = 200
plotACFnPACF(x, titleStr='Input Model', noLags=noLags)


#%% MODELLING INPUT WITH ARMA

noLags = 200


plotACFnPACF(x, noLags, "Input", includeZeroLag=True)

input_model = estimateARMA(
    x,
    A=24,
    A_free=[1, 1, 1, 0, *np.zeros(7), 0, 0, *np.zeros(10), 1, 1],
    C=24,
    C_free=[1, 0, 1, 0,  *np.zeros(6), 0, 0, 0, *np.zeros(9), 0, 0, 0],
    diff=0,
    titleStr="Input Model",
    noLags=noLags,
)


# %% CREATE W_T AND EPS_T


A = input_model.A
C = input_model.C

# Create pre-whitened series
w_t = tsa_filter(A, C, x, remove=True)
eps_t = tsa_filter(A, C, log_y, remove=True)

# Check if w_t is reasonably white (visual check)
plotACFnPACF(w_t, noLags=100, titleStr="w_t - Pre-whitened Input")
whiteness_test(w_t)

# Now plot the CCF
cxy, lags = plot_ccf(w_t, eps_t, noLags=60)


d, r, s = 0, 1, 1

# Not clear, seems like d = 0 and ringing, but we can check model orders later

#%% CHECK INPUT CONTRIBUTION

d, r, s = 1, 1, 1


B_free = np.array([0] * d + [1] + [1] * s) * 0.3
A2_free = np.array([1] + [1] * r) * 0.3
C1_free = [1]
A1_free = [1]

bjModel = estimateBJ(
    y=log_y,
    x=x,
    B=len(B_free) - 1,
    A2=len(A2_free) - 1,
    C1=len(C1_free) - 1,
    A1=len(A1_free) - 1,
    B_free=B_free,
    A2_free=A2_free,
    C1_free=C1_free,
    A1_free=A1_free
    )


xfilt = signal.lfilter(bjModel.B, bjModel.F, x)
rmv = bjModel.model._samps_to_remove()
y_cut = log_y[rmv:]
xfilt_cut = xfilt[rmv:]


fig, ax = plt.subplots()
ax.plot(y_cut, label='Output y', alpha=0.7 )
ax.plot(xfilt_cut, label='Filtered input (B/A2)x', alpha=0.7)
ax.legend()
ax.grid(True)
plt.show()


# %% CREATE ARMA for eps_t

noLags = 170

plotACFnPACF(eps_t, titleStr='eps_t', noLags=noLags)

A_free = [1, 1, 0, 1, 0, *np.zeros(19), 1]
C_free = [1, 0, 1, *np.zeros(22), 1]

eps_t_model = estimateARMA(
    eps_t,
    A=len(A_free) - 1,
    C=len(C_free) - 1,
    diff=0,
    A_free=A_free,
    C_free=C_free,
    noLags=noLags
)




# %% CREATE BJ-MODEL

d, r, s = 1, 1, 1


# From input
B_free = np.array([0] * d + [1] + [1] * s) * 0.8
A2_free = np.array([1] + [1] * r) * 0.8

# From eps_t
A1_free = [1, 1, 0, 0, 0, *np.zeros(19), 1, *np.zeros(8), 0]
C1_free = [1, 1, 0, 0, *np.zeros(8), 1, 0, 0, *np.zeros(0), 0]

# Trial
A1_free = [1, 1, 0, *np.zeros(21), 1, 1, 1]
C1_free = [1, 1, 1, 0, *np.zeros(8), 1, 1, 0, *np.zeros(8), 1]

# Trial
A1_free = [1, 1, *np.zeros(10), 1, 1, *np.zeros(10), 1, 0, 1]
A1_free = A1_free + list(np.zeros(168 - len(A1_free))) + [1]
C1_free = [1, 1, 1, 0, 1, *np.zeros(6), 1, 1]

bjModel = estimateBJ(
    y=y,
    x=x,
    d=0,
    diff=0,
    B=len(B_free) - 1,
    A2=len(A2_free) - 1,
    A1=len(A1_free) - 1,
    C1=len(C1_free) - 1,
    B_free=B_free,
    A2_free=A2_free,
    A1_free=A1_free,
    C1_free=C1_free,
    titleStr="BJ Model",
    noLags=170
)


#%% Input contribution
y_x = tsa_filter(bjModel.B, bjModel.F, x)
rmv = bjModel.model._samps_to_remove()
y_x = y_x[rmv:]
y_obs = log_y[rmv:]

plt.figure()
plt.plot(y_obs, label="Observed y", alpha=0.7)
plt.plot(y_x, label="Input contribution B/F x", linewidth=2)
plt.legend()
plt.grid(True)
plt.title("BJ input contribution to output")
plt.show()

res_no_input = y_obs - y_x
var_y = np.var(y_obs)
var_res = np.var(res_no_input)

print("Frac explained by input:", 1 - var_res/var_y)


# %% VALIDATE  MODEL

v_start = start_model + weeks_model * h
start_indexes = [v_start, v_start + 3*h, len(df) - 200]
n_weeks = [3, 1, 1]

for k in [1, 7]:
    for start_index, n in zip(start_indexes, n_weeks):
        test_model(df, k=k, start_index=start_index, n_weeks=n,
                   BJmodel=bjModel, buffer=200, plot=True, transform="log")



#%% CODE FOR SERVER

import numpy as np
from scipy import signal
from tsa_lth.modelling import polydiv

def solutionA(payload):
    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])

    # === BJ model for log(y) ===
    B = np.array([-0.0106, 0.0081])
    F = np.array([1.0, -0.9606])

    # C order 12 (need indices up to 12)
    C = np.zeros(13)
    C[0]  = 1.0
    C[1]  = 0.3536
    C[2]  = 0.1283
    C[4]  = 0.1070
    C[11] = 0.1470
    C[12] = -0.0613

    # D order 168 (need indices up to 168)
    D = np.zeros(169)
    D[0]   = 1.0
    D[1]   = -0.8059
    D[12]  = -0.1850
    D[13]  = 0.1308
    D[24]  = -0.2125
    D[26]  = 0.1457
    D[168] = -0.0742

    y_raw = data[:, 1]
    x = data[:, 2]

    # log-space output
    if np.any(y_raw <= 0):
        raise ValueError("power_MJ_s has non-positive values; log undefined.")
    y = np.log(y_raw)

    test_idx = np.arange(start_idx, end_idx)

    # Equivalent polynomials
    A_eq = np.convolve(F, D)
    B_eq = np.convolve(D, B)
    C_eq = np.convolve(F, C)

    # k-step predictor (log-space)
    F_k, G_k = polydiv(C_eq, A_eq, k)
    Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)

    yhat_log = (
        signal.lfilter(Fhat, [1], x) +
        signal.lfilter(Ghat, C_eq, x) +
        signal.lfilter(G_k, C_eq, y)
    )

    # return in original space
    yhat = np.exp(yhat_log[test_idx])
    return yhat.tolist()


def server_style_test_with_naive(start, end, k):
    df = load_project_df()

    payload = {
        "data": df.values,
        "k_steps": k,
        "start_idx": start,
        "end_idx": end,
    }

    yhat = np.array(solutionA(payload))                 # original space
    y = df["power_MJ_s"].values[start-1:end]           # original space

    # naive in original space (to match professor check)
    season = None if k == 1 else 24
    y_naive, _, _ = naive_pred(
        data=df["power_MJ_s"].values,
        test_data_ind=range(start-1, end),
        k=k,
        season_k=season
    )

    mse_model = np.mean((y - yhat) ** 2)
    mse_naive = np.mean((y - y_naive) ** 2)
    return mse_model, mse_naive


df_raw = load_project_df()

fig, ax = plt.subplots()
ax.plot(df_raw['power_MJ_s'].values)

for k in [1, 7]:
    print(f"\n==== k={k} ====")
    for s, e in [(1000,1168), (2900,3068),(4700,4868),]:
        m, n = server_style_test_with_naive(s, e, k)
        print(f"[{s},{e}]  model={m:.3f}  naive={n:.3f}")
        ax.axvline(s, color='red', linestyle='--')
        ax.axvline(e, color='red', linestyle='--')
        
        
