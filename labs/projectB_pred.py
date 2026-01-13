# Project B - Dual Input BJ Model
# Using both ambient air temperature (x1) and supply water temperature (x2) as inputs

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
from tsa_lth.modelling import MultiInputPEM, estimateARMA, estimateBJ, polydiv
from tsa_lth.modelling import filter as tsa_filter
from tsa_lth.tests import whiteness_test, check_if_normal

import pandas as pd


from projectB_gridsearch import (
    load_grid_search_results,
    get_model_config,
    build_model_from_config,
    print_model_config
)

# %% LOAD AND CLEAN DATA

df = load_project_df()
df = data_cleanup(df)

# %% SHOW MODEL / VALIDATION / TEST SPLITS (PART B)

start_model = 1500 - 4*168          # your Part B start (0)
weeks_model = 10     # your Part B weeks (10)
h = 168

n_total = len(df)

windows = [
    ("Modeling",    start_model,                 start_model + weeks_model*h),
    ("Validation",  start_model + weeks_model*h, start_model + (weeks_model+3)*h),
    ("Test 1",      start_model + (weeks_model+3)*h, start_model + (weeks_model+4)*h),
    ("Test 2",      3555,               3755),
]

colors = {
    "Modeling":   "tab:blue",
    "Validation": "tab:orange",
    "Test 1":     "tab:green",
    "Test 2":     "tab:red",
}

def clamp(a, lo, hi):
    return max(lo, min(a, hi))

# --- full-series overview with shaded windows ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["power_MJ_s"], linewidth=1)

ymin, ymax = ax.get_ylim()
for name, s, e in windows:
    s = clamp(s, 0, n_total - 1)
    e = clamp(e, 1, n_total)
    if e <= s:
        continue

    x0 = df["date"].iloc[s]
    x1 = df["date"].iloc[e - 1]

    ax.axvspan(x0, x1, alpha=0.25, color=colors[name], label=name)
    ax.text(
        x0 + (x1 - x0) / 2,
        ymax - 0.08 * (ymax - ymin),
        name,
        ha="center",
        va="top",
        fontsize=10,
    )

ax.set_title("Part B: Power usage with modeling / validation / test splits")
ax.set_ylabel("Power (MJ/s)")
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="upper left", frameon=True)
plt.tight_layout()
plt.show()


#%%
# --- quick sanity plots for the modeling window (x1, x2, y) ---
s0 = start_model
e0 = start_model + weeks_model*h

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df["date"].iloc[s0:e0], df["ambient_temp_C"].iloc[s0:e0], linewidth=1)
ax.set_title("Part B: Input x1 (ambient temp) — modeling window")
ax.set_ylabel("°C")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df["date"].iloc[s0:e0], df["supply_temp_C"].iloc[s0:e0], linewidth=1)
ax.set_title("Part B: Input x2 (supply temp) — modeling window")
ax.set_ylabel("°C")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df["date"].iloc[s0:e0], df["power_MJ_s"].iloc[s0:e0], linewidth=1)
ax.set_title("Part B: Output y (power) — modeling window")
ax.set_ylabel("Power (MJ/s)")
plt.tight_layout()
plt.show()



#%% SELECTING DATA

# Extract all three signals for Part B

# s0 = 0
# e0 = -2
x1 = df['ambient_temp_C'][s0:e0+1].values    # Input 1: Ambient temperature (same as Part A)
x2 = df['supply_temp_C'][s0:e0+1].values  # Input 2: Supply water temperature (NEW)
y = df['power_MJ_s'][s0:e0+1].values 


# %% PLOT ALL THREE SIGNALS

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(x1)
axes[0].set_ylabel('Temperature (C)')
axes[0].set_title('Input 1: Ambient Air Temperature (x1)')
axes[0].grid(True)

axes[1].plot(x2, color='orange')
axes[1].set_ylabel('Temperature (C)')
axes[1].set_title('Input 2: Supply Water Temperature (x2)')
axes[1].grid(True)

axes[2].plot(y, color='green')
axes[2].set_ylabel('Power (MJ/s)')
axes[2].set_title('Output: Power Load (y)')
axes[2].set_xlabel('Time (hours)')
axes[2].grid(True)

plt.tight_layout()
plt.show()


# %% PLOT INPUT-OUTPUT RELATIONSHIPS

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# x1 vs y
axes[0].scatter(x1, y, alpha=0.3, s=5)
axes[0].set_xlabel('Ambient Temperature (C)')
axes[0].set_ylabel('Power (MJ/s)')
axes[0].set_title('Power vs Ambient Temperature')
axes[0].grid(True)

# x2 vs y
axes[1].scatter(x2, y, alpha=0.3, s=5, color='orange')
axes[1].set_xlabel('Supply Water Temperature (C)')
axes[1].set_ylabel('Power (MJ/s)')
axes[1].set_title('Power vs Supply Water Temperature')
axes[1].grid(True)

plt.tight_layout()
plt.show()



# %% ========== INPUT 1 MODEL (Ambient Temperature) ==========

noLags = 200
plotACFnPACF(x1, noLags=noLags, titleStr='Input 1: Ambient Temperature (x1)')


# %% ARMA MODEL FOR INPUT 1

A_free = np.array([1, 1, 1, 1, *np.zeros(8), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) * 0.3
C_free = np.array([1, 1, 1, 1, *np.zeros(18), 0, 0, 0, 0]) * 0.3

inputModel1 = estimateARMA(
    x1,
    A=len(A_free) - 1,
    C=len(C_free) - 1,
    diff=1,
    A_free=A_free,
    C_free=C_free,
    titleStr='Input 1 Model: Ambient Temperature',
    noLags=noLags
)



# Sanity check Predicting and plotting
k = 7
print(f'Predictions for x1 with k={k}')
A1_full = np.convolve([1, -1], inputModel1.A)

Fx1, Gx1 = polydiv(inputModel1.C, A1_full, k)
xhatk1 = signal.lfilter(Gx1, inputModel1.C, x1)

rmv = max(len(A1_full), len(inputModel1.C))  # Proper burn-in

fig, ax = plt.subplots()
ax.plot(x1[rmv:], label='data')
ax.plot(xhatk1[rmv:], label='prediction')
ax.legend()
ax.set_title(f'{k}-step prediction of x1')
plt.show()

# Model prediction residual
res_model_x1 = x1[rmv:] - xhatk1[rmv:]
mse_model_x1 = np.mean(res_model_x1**2)

# Naive prediction
naive_pred_x1 = x1[rmv-k:-k]
res_naive_x1 = x1[rmv:] - naive_pred_x1
mse_naive_x1 = np.mean(res_naive_x1**2)

print(f'Naive MSE: {mse_naive_x1:.4f}')
print(f'Model MSE: {mse_model_x1:.4f}')
print(f'Improvement: {100*(1 - mse_model_x1/mse_naive_x1):.2f}%')
print('SUCCESS!' if mse_model_x1 < mse_naive_x1 else 'FAIL!')


print('WARNING! Setting input_model.A as convolved with diff!')
inputModel1.A = np.convolve([1, -1], inputModel1.A)


#%% PREDICTING INPUT 1

# Get full data
x1_full = df['ambient_temp_C'].values

# Predict Input
k = 1
print(f'k = {k}')
Fx1, Gx1 = polydiv(inputModel1.C, inputModel1.A, k)
x1hatk = signal.lfilter(Gx1, inputModel1.C, x1_full)
x1_resid = x1_full - x1hatk

# Naive prediction
season = None if k == 1 else 24
x1_naive, x1_var_naive, x1_ehat_naive = naive_pred(data=x1_full, test_data_ind=np.arange(0, len(x1_full)), k=k, season_k=season)


# Window
window = windows[3]
name, start_index, end_index = window
x1hatk = x1hatk[start_index:end_index]
x1_naive = x1_naive[start_index:end_index]
dates = df[start_index:end_index]['date']

x1_resid = x1_resid[start_index:end_index]
x1_ehat_naive = x1_ehat_naive[start_index:end_index]


# Plotting
fig, ax = plt.subplots(figsize=[10, 4])
ax.set_ylabel('Temperature (C)')
ax.plot(dates, x1_full[start_index:end_index], label='Data', alpha=0.7)
ax.plot(dates, x1hatk, label='Prediction', alpha=0.7)
# ax.plot(dates, x1_naive, label='Naive Prediction')
ax.legend()


# Comaprison
print('Window:', name)
model_mse = np.mean(x1_resid**2)
naive_mse = np.mean(x1_ehat_naive**2)
print('model_mse:', round(model_mse, 3))
print('naive_mse:', round(naive_mse, 3))


# Whiteness
if k == 1 :
    plotACFnPACF(x1_resid, titleStr=f'{name} {k}-step Prediction Residual for x_1', noLags=100)
    whiteness_test(x1_resid)

# %% ========== INPUT 2 MODEL (Supply Water Temperature) ==========
# This is NEW for Part B - we need to analyze and model x2

plotACFnPACF(x2, noLags=noLags, titleStr='Input 2: Supply Water Temperature (x2)')


# %%  ARMA MODEL FOR INPUT 2

A_free = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, *np.zeros(14), 1]) * 0.3
C_free = np.array([1, 1, 1, 0, 0, 0, 0]) * 0.3

inputModel2 = estimateARMA(
    x2,
    A=len(A_free) - 1,
    C=len(C_free) - 1,
    diff=0,
    A_free=A_free,
    C_free=C_free,
    titleStr='Input 2 Model: Supply Water Temperature',
    noLags=60
)

#%% PREDICTING INPUT 2

# Get full data
x2_full = df['supply_temp_C'].values

# Predict Input
k = 7
print(f'k = {k}')
Fx2, Gx2 = polydiv(inputModel2.C, inputModel2.A, k)
x2hatk = signal.lfilter(Gx2, inputModel2.C, x2_full)
x2_resid = x2_full - x2hatk

# Naive prediction
season = None if k == 1 else 24
x2_naive, x2_var_naive, x2_ehat_naive = naive_pred(data=x2_full, test_data_ind=np.arange(0, len(x2_full)), k=k, season_k=season)


# Window
window = windows[3]
name, start_index, end_index = window
x2hatk = x2hatk[start_index:end_index]
x2_naive = x2_naive[start_index:end_index]
dates = df[start_index:end_index]['date']

x2_resid = x2_resid[start_index:end_index]
x2_ehat_naive = x2_ehat_naive[start_index:end_index]


# Plotting
fig, ax = plt.subplots(figsize=[10, 4])
ax.set_ylabel('Temperature (C)')
ax.plot(dates, x2_full[start_index:end_index], label='Data', alpha=0.7)
ax.plot(dates, x2hatk, label='Prediction', alpha=0.7)
ax.plot(dates, x2_naive, label='Naive Prediction', alpha=0.2)
ax.legend()


# Comaprison
print('Window:', name)
model_mse = np.mean(x2_resid**2)
naive_mse = np.mean(x2_ehat_naive**2)
print('model_mse:', round(model_mse, 3))
print('naive_mse:', round(naive_mse, 3))


# Whiteness
if k == 1 :
    plotACFnPACF(x2_resid, titleStr=f'{name} {k}-step Prediction Residual for x_2', noLags=100)
    whiteness_test(x2_resid)
    pass


# %% ========== CCF ANALYSIS FOR INPUT 1 (Ambient Temperature) ==========


# Pre-whiten input 1 and output
A1 = inputModel1.A
C1 = inputModel1.C
w1_t = tsa_filter(A1, C1, x1, remove=True)
eps1_t = tsa_filter(A1, C1, y, remove=True)

# Check that w1_t is reasonably white
plotACFnPACF(w1_t, noLags=50, titleStr='w1_t - Pre-whitened Input 1 (Ambient Temp)')
whiteness_test(w1_t)

# Plot CCF between pre-whitened x1 and y
print("\n" + "="*60)
print("CCF: Pre-whitened x1 (ambient temp) vs Pre-whitened y")
print("="*60)
cxy1, lags1 = plot_ccf(w1_t, eps1_t, noLags=60, titleStr='Crosscorrelation between x1 and y')


# %% ========== CCF ANALYSIS FOR INPUT 2 (Supply Water Temperature) ==========



# Pre-whiten input 2 and output
A2 = inputModel2.A
C2 = inputModel2.C
w2_t = tsa_filter(A2, C2, x2, remove=True)  # Pre-whitened x2
eps2_t = tsa_filter(A2, C2, y, remove=True)  # Pre-whitened y (using x2's model)

# Check that w2_t is reasonably white
plotACFnPACF(w2_t, noLags=50, titleStr='w2_t - Pre-whitened Input 2 (Supply Water Temp)')
whiteness_test(w2_t)

# Plot CCF between pre-whitened x2 and y
print("\n" + "="*60)
print("CCF: Pre-whitened x2 (supply water temp) vs Pre-whitened y")
print("="*60)
cxy2, lags2 = plot_ccf(w2_t, eps2_t, noLags=60)


# %% ========== DUAL-INPUT BJ MODEL (MultiInputPEM) ==========


B = [[0, 1, 1], [0, 1]]
A2 = [[1, 0.5], [1.0]]

# MA(2)+MA(24): C has terms at lags 1, 2, 24
C1_0 = [1.0, 0.5, 0.3, *np.zeros(21), 0.3]  # length 25

# AR(1)+AR(24): D has terms at lags 1, 24
D1_0 = [1.0, -0.5, *np.zeros(22), 0.3]  # length 25

x_multi = np.column_stack([x1, x2])
model = MultiInputPEM(y=y, x=x_multi, A=1, B=B, F=A2, C=C1_0, D=D1_0, nk=[0, 0])
model.set_free_params(
    B_free=None,
    F_free=None,
    C_free=[False, True, True, *[False]*21, True],  # Estimate lags 1, 2, 24
    D_free=[False, True, *[False]*22, True]          # Estimate lags 1, 24
)

foundModel = model.fit(method="LS", verbose=0)
foundModel.summary()

res = foundModel.resid
plotACFnPACF(res, titleStr="Multi-Input BJ (Model 80)", noLags=100)
whiteness_test(res)


# Plotting
rmv = 100
x1_contribution = tsa_filter(B[0], A2[0], x1)
x2_contribution = tsa_filter(B[1], A2[1], x2)
fig, ax = plt.subplots()

ax.plot(x1_contribution[rmv:], label='x1', alpha=0.4)
ax.plot(x2_contribution[rmv:], label='x2', alpha=0.4)
ax.plot(y[rmv:], label='y_diff', alpha=0.4)


#%% LOADING MODEL FROM GRIDSERACH

# 1. Load the saved grid search results
data = load_grid_search_results('grid_search_results.json')
configs = data['configs']
results_df = pd.DataFrame(data['results'])

# 2. Find the model you want (e.g., best by FitPercent)
best_id = results_df.sort_values('FitPercent', ascending=False).iloc[0]['model_id']
best_id = 169
best_id = 191
best_id = 178

print(f"Best model by FitPercent: {best_id}")

# 3. Get the configuration and build the |model
config = get_model_config(best_id, configs)
# print_model_config(config)

# 4. Fit the model
x_multi = np.column_stack([x1, x2])
foundModel = build_model_from_config(config, y, x_multi)
foundModel.summary()


    # %% VALIDATION / TEST RUNS (PART B) — driven by "windows"

from myproject_utils import predict_model


buffer = 200
k = 7
h = 168

def slice_block(df, s, e, buffer):
    s0 = max(0, s - buffer)
    e0 = min(len(df) - 1, e + buffer - 1)
    x1 = df["ambient_temp_C"].iloc[s0:e0+1].to_numpy()
    x2 = df["supply_temp_C"].iloc[s0:e0+1].to_numpy()
    y  = df["power_MJ_s"].iloc[s0:e0+1].to_numpy()
    return x1, x2, y

for name, s, e in windows:
    if name == "Modeling":
        continue

    print(f"{'='*30} {name}  (k={k}) {'='*30}")
    indexes = [s, e]
    val_x1, val_x2, val_y = slice_block(df, s, e, buffer)
    _ = predict_model(foundModel, inputModel1, inputModel2, val_x1, val_x2, val_y, indexes=indexes, k=k)



#%%

from myproject_utils import test_double_input_model


for k in [1, 7]:
    print(f'------ K = {k} ------')
    for name, start_index, end_index in windows:
        if name == "Modeling":
            continue
    
        y_real, yhatk, dates= test_double_input_model(
            df,
            foundModel,
            inputModel1,
            inputModel2,
            k,
            start_index,
            end_index,
            buffer=200,
        )
        print('--' * 20)
        
        fig, ax = plt.subplots(figsize=[10,4])
        ax.set_title(name)
        ax.set_ylabel('Power (MJ/s)')
        ax.plot(dates, y_real, label='Data')
        ax.plot(dates, yhatk, label=f'{k}-step Prediction')
        plt.xticks(rotation=20)
        ax.legend()
        
        
        if k == 1:
            ehat = y_real - yhatk
            plotACFnPACF(ehat, noLags=100, titleStr=name)
            whiteness_test(ehat)
        
    



#%% PACKAGE INTO SOLUTION B




def solutionB(payload):
    
    import numpy as np
    from scipy import signal
    from tsa_lth.modelling import polydiv

    
    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])
    
    # ================= Fixed BJ model (Part B) =================
    # Input 1 (ambient air temperature)
    B1 = np.array([-0.6263, -0.0032, 0.0703])
    F1 = np.array([1.0])
    
    # Input 2 (supply water temperature)
    B2 = np.array([1.8581, -0.0549])
    F2 = np.array([1.0])
    
    # Noise model
    C = np.zeros(25)
    C[0]  = 1.0
    C[1]  = 0.5348
    C[2]  = 0.2798
    C[24] = 0.1094
    
    D = np.zeros(25)
    D[0]  = 1.0
    D[1]  = -0.8009
    D[24] = -0.1979


    # ==========================================================
    y  = data[:, 1]
    x1 = data[:, 2]
    x2 = data[:, 3]

    test_idx = np.arange(start_idx, end_idx)
    
    
    # Predict input 1
    nabla = np.array([1.0, -1.0])    
    C1 = np.array([1.0, -1.5755, 0.3283, 0.3343])    
    A1 = np.zeros(25)
    A1[0]  = 1.0
    A1[1]  = -2.0167
    A1[2]  = 1.1923
    A1[3]  = -0.0915
    A1[13] = 0.0378
    A1[24] = 0.0166    
    A1 = np.convolve(nabla, A1)
    Fx1, Gx1 = polydiv(C1, A1, k)
    xhatk1 = signal.lfilter(Gx1, C1, x1)
    
    
    # Predict input 2
    A2 = np.zeros(25)
    A2[0]  = 1.0
    A2[1]  = -1.8676
    A2[2]  = 1.4929
    A2[3]  = -0.5771
    A2[9]  = -0.0356
    A2[24] = -0.0126    
    C2 = np.array([1.0, -0.9854, 0.5295])
    
    Fx2, Gx2 = polydiv(C2, A2, k)
    xhatk2 = signal.lfilter(Gx2, C2, x2)
        

    # --- Equivalent polynomials ---
    A_eq = np.convolve(D, np.convolve(F1, F2))
    B1_eq = np.convolve(D, np.convolve(B1, F2))
    B2_eq = np.convolve(D, np.convolve(B2, F1))
    C_eq = np.convolve(C, np.convolve(F1, F2))

    # --- k-step predictor ---
    Fk, Gk = polydiv(C_eq, A_eq, k)
    Fh1, Gh1 = polydiv(np.convolve(Fk, B1_eq), C_eq, k)
    Fh2, Gh2 = polydiv(np.convolve(Fk, B2_eq), C_eq, k)
    

    yhat = (
        signal.lfilter(Fh1, [1], xhatk1) +
        signal.lfilter(Gh1, C_eq, x1) +
        signal.lfilter(Fh2, [1], xhatk2) +
        signal.lfilter(Gh2, C_eq, x2) +
        signal.lfilter(Gk, C_eq, y)
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

    yhat = np.array(solutionB(payload))
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
        
        
        




