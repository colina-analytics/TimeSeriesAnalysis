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

lambda_max, offsetValue = box_cox(y, plotIt=True, titleStr='Output', transform=False)
print(f'lambda_max(y) = {lambda_max:.4f}')

lambda_max, offsetValue = box_cox(x, plotIt=True, titleStr='Input', transform=False)
print(f'lambda_max(x) = {lambda_max:.4f}')


# %% NORM-PLOT OUTPUT

normplot(y, titleStr='y')
normplot(np.log(y), titleStr='log(y)')
normplot(np.sqrt(y), titleStr='sqrt(y)')

log_y = np.log(y)

# %% NORM-PLOT INPUT

normplot(x, titleStr='x')
normplot(np.log(x), titleStr='log(x))')
normplot(np.sqrt(x), titleStr='sqrt(x))')

# Keep x as-is

# %% MODEL FOR INPUT

noLags = 200
plotACFnPACF(x, titleStr='Input Model', noLags=noLags)


#%% TESTING STATIONARITY
from myproject_utils import test_stationarity

# Test original temperature
p_original = test_stationarity(x, "Original Temperature")

# Test differenced temperature (non-seasonal)
x_diff1 = tsa_filter([1, -1], [1], x, remove=True)
p_diff1 = test_stationarity(x_diff1, "Differenced (1) Temperature")

# Test seasonally differenced temperature (period=24)
x_diff24 = tsa_filter([1, *np.zeros(23), -1], [1], x, remove=True)
p_diff24 = test_stationarity(x_diff24, "Seasonally Differenced (24) Temperature")

# Test both regular and seasonal differencing
x_diff_both = np.diff(x_diff24, n=1)  # First seasonal, then regular
p_diff_both = test_stationarity(x_diff_both, "Differenced (1,24) Temperature")




#%% MODELLING INPUT WITH ARMA - NO DIFF

noLags = 200


plotACFnPACF(x, noLags, "Input", includeZeroLag=True)


input_model = estimateARMA(
    x,
    A=24,
    A_free=[1, 1, 1, 0, *np.zeros(7), 0, 0, *np.zeros(10), 1, 1],
    C=24,
    C_free=[1, 0, 0, 0,  *np.zeros(6), 0, 0, 0, *np.zeros(9), 1, 0, 0],
    diff=0,
    titleStr="Input Model",
    noLags=noLags,
)



# %% CREATE W_T AND EPS_T


A = input_model.A
C = input_model.C

# Create pre-whitened series
w_t = tsa_filter(A, C, x, remove=True)
eps_t = tsa_filter(A, C, y, remove=True)

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
    y=y,
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
y_cut = y[rmv:]
xfilt_cut = xfilt[rmv:]


fig, ax = plt.subplots()
ax.plot(y_cut, label='Output y', alpha=0.7 )
ax.plot(xfilt_cut, label='Filtered input (B/A2)x', alpha=0.7)
ax.set_xlabel('Index')
ax.set_ylabel('Power (MJ/s)')
ax.legend()
ax.grid(True)
plt.show()


# %% CREATE ARMA for eps_t

noLags = 170

plotACFnPACF(eps_t, titleStr='eps_t', noLags=noLags)

A_free = [1, 1, 0, 1, 0, *np.zeros(19), 1, *np.zeros(8), 0]
C_free = [1, 1, 0, 0, *np.zeros(6),1, 0, 1, 0, 0, *np.zeros(0), 0]

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


# A1_free = np.array([1, 1, 1, 0,0,0,0,1, *np.zeros(15), 0, 1, 1]) * 0.3
# C1_free = np.array([1, 1, 0, 0, 1, *np.zeros(23), 0])       * 0.3    # white MA


# A_free = [1, 1, 0, 0, 0, *np.zeros(19), 1, *np.zeros(8), 0]
# C_free = [1, 1, 0, 0, *np.zeros(8), 1, 0, 0, *np.zeros(0), 0]


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

# A1_free = A1_free + list(np.zeros(168 - len(A1_free))) + [1]
# C1_free = C1_free + list(np.zeros(167 - len(C1_free))) + [1, 1]


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
    titleStr="BJ minimal baseline",
    noLags=170
)


#%%

# Input contribution
y_x = tsa_filter(bjModel.B, bjModel.F, x)
rmv = bjModel.model._samps_to_remove()
y_x = y_x[rmv:]
y_obs = y[rmv:]

plt.figure()
plt.plot(y_obs, label="Observed y", alpha=0.7)
# plt.plot(y_x, label="Input contribution B/F x", linewidth=2)
y_x_level = y_x + (y_obs.mean() - y_x.mean())
plt.plot(y_x_level, label="Input contribution (B/F)x", alpha=0.7)

plt.legend()
plt.grid(True)
plt.title("BJ input contribution to output")
plt.xlabel('Index')
plt.ylabel('Power (MJ/s)')
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
    print(f"------------ Prediction for K = {k} -------------")
    for start_index, n in zip(start_indexes, n_weeks):
        test_model(df, k=k, start_index=start_index, n_weeks=n,
                   BJmodel=bjModel, inputModel=input_model, buffer=200, plot=True)



#%% CODE FOR SERVER



def solutionA(payload):
    import numpy as np
    from scipy import signal
    from tsa_lth.modelling import polydiv

    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])

    # --- fixed BJ model ---
    B = np.array([-1.2241, 1.0474])
    F = np.array([1.0, -0.9589])

    C = np.zeros(24)
    C[0]  = 1.0
    C[1]  = 0.3946
    C[2]  = 0.1956
    C[12] = 0.0514
    C[13] = 0.0610
    C[23] = 0.2649

    D = np.zeros(27)
    D[0]  = 1.0
    D[1]  = -0.7937
    D[24] = -0.3655
    D[25] = 0.0596
    D[26] = 0.0990

    y = data[:, 1]
    x = data[:, 2]
    test_idx = np.arange(start_idx, end_idx)

    # --- input ARMA model for x: A(z)x = C(z)e ---
    Ax = np.zeros(25)
    Ax[0]  = 1.0
    Ax[1]  = -1.5098
    Ax[2]  = 0.5622
    Ax[23] = -0.1898
    Ax[24] = 0.1389

    Cx = np.zeros(25)
    Cx[0]  = 1.0
    Cx[22] = 0.0772

    # k-step predicted input series (same length as x)
    Fx_k, Gx_k = polydiv(Cx, Ax, k)
    xhatk = signal.lfilter(Gx_k, Cx, x)

    # --- BJ k-step predictor ---
    A_eq = np.convolve(F, D)
    B_eq = np.convolve(D, B)
    C_eq = np.convolve(F, C)

    F_k, G_k = polydiv(C_eq, A_eq, k)
    Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)

    yhat = (
        signal.lfilter(Fhat, [1], xhatk) +
        signal.lfilter(Ghat, C_eq, x) +
        signal.lfilter(G_k, C_eq, y)
    )

    return yhat[test_idx].tolist()




def server_style_test_with_naive(start, end, k):
    df = load_project_df()

    payload = {
        "data": df.values,
        "k_steps": k,
        "start_idx": start,
        "end_idx": end
    }

    yhat = np.array(solutionA(payload))
    y = df['power_MJ_s'].values[start-1:end]

    season = None if k == 1 else 24
    y_naive, _, _ = naive_pred(
        data=df['power_MJ_s'].values,
        test_data_ind=range(start-1, end),
        k=k,
        season_k=season
    )

    # ⬅️ DO NOT SLICE y_naive AGAIN

    mse_model = np.mean((y - yhat)**2)
    mse_naive = np.mean((y - y_naive)**2)

    return mse_model, mse_naive

df_raw = load_project_df()

fig, ax = plt.subplots()
ax.plot(df_raw['power_MJ_s'].values)

for k in [1, 7]:
    print(f"\n==== k={k} ====")
    for s, e in [(2900,3068),(4700,4868),(1000,1168)]:
        m, n = server_style_test_with_naive(s, e, k)
        print(f"[{s},{e}]  model={m:.3f}  naive={n:.3f}")
        ax.axvline(s, color='red', linestyle='--')
        ax.axvline(e, color='red', linestyle='--')
        
        
