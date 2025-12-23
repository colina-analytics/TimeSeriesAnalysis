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

# %% SELECT MODELING DATASET - Same as Part A

start = 0
n_weeks = 10
end = start + 168 * n_weeks - 1
n = end - start + 1

# Get dates for reference
start_date = df.iloc[start]['date']
end_date = df.iloc[end]['date']
print(f'Modeling data from index {start} to {end}, total length {n}')
print(f'Model start_date: {start_date}, end_date: {end_date}')

# Extract all three signals for Part B
x1 = df['ambient_temp_C'][start:end+1].values    # Input 1: Ambient temperature (same as Part A)
x2 = df['supply_temp_C'][start:end+1].values  # Input 2: Supply water temperature (NEW)
# x2 -= np.mean(x2)
y = df['power_MJ_s'][start:end+1].values         # Output: Power

print(f'x1 (ambient temp) shape: {x1.shape}')
print(f'x2 (supply temp) shape: {x2.shape}')
print(f'y (power) shape: {y.shape}')


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


# %% SET PARAMETERS

noLags = 200
pstart = len(y)  # For later prediction split if needed


# %% ========== INPUT 1 MODEL (Ambient Temperature) ==========
# This should be the SAME model as Part A since it's the same data

plotACFnPACF(x1, noLags=noLags, titleStr='Input 1: Ambient Temperature (x1)')


# %% ARMA MODEL FOR INPUT 1
# Using the same structure as Part A - adjust if your final Part A model was different

# From your projectA3.py, the input model was:
# A_free = np.array([1, 1, 1, 1, *np.zeros(8), 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) * 0.3
# C_free = np.array([1, 1, 0, 1, *np.zeros(18), 0, 0, 0, 0]) * 0.3
# with diff=1

# Let's start with this structure - you can adjust based on your final Part A model
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

# Predicting and plotting
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

# Naive k-step prediction: x̂(t+k) = x(t), so x̂(t) = x(t-k)
# For k=1: predict x(t) using x(t-1)
naive_pred_x1 = x1[rmv-k:-k]  # x(rmv-1), x(rmv), ..., x(n-2)
res_naive_x1 = x1[rmv:] - naive_pred_x1  # x(rmv) - x(rmv-1), etc.
mse_naive_x1 = np.mean(res_naive_x1**2)

print(f'Naive MSE: {mse_naive_x1:.4f}')
print(f'Model MSE: {mse_model_x1:.4f}')
print(f'Improvement: {100*(1 - mse_model_x1/mse_naive_x1):.2f}%')

if mse_model_x1 < mse_naive_x1:
    print('SUCCESS!')
else:
    print('FAIL!')

print('WARNING! Setting input_model.A as convolved with diff!')
inputModel1.A = np.convolve([1, -1], inputModel1.A)


# %% ========== INPUT 2 MODEL (Supply Water Temperature) ==========
# This is NEW for Part B - we need to analyze and model x2

plotACFnPACF(x2, noLags=noLags, titleStr='Input 2: Supply Water Temperature (x2)')


# %% TODO: ARMA MODEL FOR INPUT 2
# After analyzing ACF/PACF of x2, determine appropriate model structure
# May need differencing if non-stationary (check for slowly decaying ACF)


plotACFnPACF(x2, noLags=noLags, titleStr='x2')

x2_diff = tsa_filter([1, -1], [1], x2, remove=True)
plt.plot(x2_diff)
plt.title('x2 Diff')

plotACFnPACF(x2_diff, noLags=noLags, titleStr='Differenced x2')

# %% Very simple model - ARIMA(1,1,1) or even simpler

A_free = np.array([1, 0, 1, 1, 0, *np.zeros(67), 1]) * 0.3
C_free = np.array([1, 0, 1, 0, 0]) * 0.3

inputModel2 = estimateARMA(
    x2,
    A=len(A_free) - 1,
    C=len(C_free) - 1,
    diff=1,
    A_free=A_free,
    C_free=C_free,
    titleStr='Input 2 Model: Supply Water Temperature',
    noLags=noLags
)


# Predicting and plotting
k = 1
print(f'Predictions for x1 with k={k}')
A2_full = np.convolve([1, -1], inputModel2.A)

Fx2, Gx2 = polydiv(inputModel2.C, A2_full, k)
xhatk2 = signal.lfilter(Gx2, inputModel2.C, x2)

rmv = max(len(A2_full), len(inputModel2.C))  # Proper burn-in

fig, ax = plt.subplots()
ax.plot(x2[rmv:], label='data')
ax.plot(xhatk2[rmv:], label='prediction')
ax.legend()
ax.set_title(f'{k}-step prediction of x2')
plt.show()

# Model prediction residual
res_model_x2 = x2[rmv:] - xhatk2[rmv:]
mse_model_x2 = np.mean(res_model_x2**2)

# Naive k-step prediction: x̂(t+k) = x(t), so x̂(t) = x(t-k)
# For k=2: predict x(t) using x(t-2)
naive_pred_x2 = x2[rmv-k:-k]  # x(rmv-2), x(rmv), ..., x(n-2)
res_naive_x2 = x2[rmv:] - naive_pred_x2  # x(rmv) - x(rmv-2), etc.
mse_naive_x2 = np.mean(res_naive_x2**2)

print(f'Naive MSE: {mse_naive_x2:.4f}')
print(f'Model MSE: {mse_model_x2:.4f}')
print(f'Improvement: {100*(1 - mse_model_x1/mse_naive_x1):.2f}%')

if mse_model_x1 < mse_naive_x1:
    print('SUCCESS!')
else:
    print('FAIL!')
    
    
print('WARNING! Setting input_model.A as convolved with diff!')
inputModel2.A = np.convolve([1, -1], inputModel2.A)

# %% ========== CCF ANALYSIS FOR INPUT 1 (Ambient Temperature) ==========
# Pre-whiten x1 and y using inputModel1, then compute CCF
# IMPORTANT: Since models were estimated with diff=1, we need to include nabla in A

nabla = np.array([1, -1])

# Get polynomials and include differencing
A1 = np.convolve(inputModel1.A, nabla)  # Full A includes differencing
C1 = inputModel1.C

# Pre-whiten input 1 and output
w1_t = tsa_filter(A1, C1, x1, remove=True)  # Pre-whitened x1
eps1_t = tsa_filter(A1, C1, y, remove=True)  # Pre-whitened y (using x1's model)

# Check that w1_t is reasonably white
plotACFnPACF(w1_t, noLags=50, titleStr='w1_t - Pre-whitened Input 1 (Ambient Temp)')
whiteness_test(w1_t)

# Plot CCF between pre-whitened x1 and y
print("\n" + "="*60)
print("CCF: Pre-whitened x1 (ambient temp) vs Pre-whitened y")
print("="*60)
cxy1, lags1 = plot_ccf(w1_t, eps1_t, noLags=60)


# %% ========== CCF ANALYSIS FOR INPUT 2 (Supply Water Temperature) ==========
# Pre-whiten x2 and y using inputModel2, then compute CCF

# Get polynomials and include differencing
A2 = np.convolve(inputModel2.A, nabla)  # Full A includes differencing
C2 = inputModel2.C

# Pre-whiten input 2 and output
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
best_id = 80
print(f"Best model by FitPercent: {best_id}")

# 3. Get the configuration and build the model
config = get_model_config(best_id, configs)
print_model_config(config)

# 4. Fit the model
x_multi = np.column_stack([x1, x2])
foundModel = build_model_from_config(config, y, x_multi)
foundModel.summary()


# %% PREDICTING on Modelling data


def predict_model(foundModel, inputModel1, inputModel2,  x1, x2, y, k, buffer=200):

    # Get polynomials
    KA = np.convolve(np.convolve(foundModel.D, foundModel.F[0]), foundModel.F[1])
    KB = np.convolve(np.convolve(foundModel.D, foundModel.B[0]), foundModel.F[1])
    KC = np.convolve(np.convolve(foundModel.F[0], foundModel.F[1]), foundModel.C)
    KD = np.convolve(np.convolve(foundModel.D, foundModel.B[1]), foundModel.F[0])
    
    Fy, Gy = polydiv(foundModel.C, foundModel.D, k)
    Fh1, Gh1 = polydiv(np.convolve(Fy, KB), KC, k)
    Fh2, Gh2 = polydiv(np.convolve(Fy, KD), KC, k)
    
    # Predict the input signals.
    Fx1, Gx1 = polydiv(inputModel1.C, inputModel1.A, k)
    xhatk1 = signal.lfilter(Gx1, inputModel1.C, x1)
    
    Fx2, Gx2 = polydiv(inputModel2.C, inputModel2.A, k)
    xhatk2 = signal.lfilter(Gx2, inputModel2.C, x2)
    
    
    # Predict signal
    yhatk = (signal.lfilter(Fh1, 1, xhatk1) + signal.lfilter(Gh1, KC, x1) +
             signal.lfilter(Fh2, 1, xhatk2) + signal.lfilter(Gh2, KC, x2) +
             signal.lfilter(Gy, KC, y))
    
    season = None if k == 1 else 24
    y_naive, var_naive, ehat_naive = naive_pred(data=y, test_data_ind=range(len(y)), k=k, season_k=season)
    

    
    # Align
    # bias = np.mean(y) - np.mean(yhatk)
    # yhatk += bias
    rmv = buffer
    yhatk = yhatk[rmv:]
    y_filt = y[rmv:]
    y_naive = y_naive[rmv:]
    
    
    # Residuals
    ehat_k = y_filt - yhatk
    
    mse_model = np.mean(ehat_k**2)
    mse_naive = np.mean(ehat_naive**2)
    
    print(f'mse_model: {mse_model}')
    print(f'mse_naive: {mse_naive}')
    
    if mse_model < mse_naive:
        print('SUCESS!')
    else:
        print('FAIL!')
        
    # Plot the resulting predictions
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    axes[0].plot(x1[buffer:], label='x_1(t)')
    axes[0].plot(xhatk1[buffer:], label='Predicted data')
    axes[0].set_title(f'{k}-step predictions of x_1(t)')
    axes[0].legend(loc='upper left')
    
    axes[1].plot(x2[buffer:], label='x_2(t)')
    axes[1].plot(xhatk2[buffer:], label='Predicted data')
    axes[1].set_title(f'{k}-step predictions of x_2(t)')
    axes[1].legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_filt, label='y(t)')
    plt.plot(yhatk, label='Predicted data')
    plt.plot(y_naive, label='Naive')
    plt.title(f'{k}-step predictions of y(t)')
    plt.legend(loc='upper left')
    plt.show()
    
    
        
#%% TEST VALUDATION


buffer = 200
k = 7

# Validation 1
val_n_weeks = 2
val_start = end - buffer 
val_end = val_start + 168*val_n_weeks + buffer
val_x1 = df['ambient_temp_C'][val_start:val_end+1].values    # Input 1: Ambient temperature (same as Part A)
val_x2 = df['supply_temp_C'][val_start:val_end+1].values  # Input 2: Supply water temperature (NEW)
val_y = df['power_MJ_s'][val_start:val_end+1].values 
predict_model(foundModel, inputModel1, inputModel2, val_x1, val_x2, val_y, k=k)


print('=' * 100)

# Validation 2
val_n_weeks = 2
val_start = val_end - buffer 
val_end = val_start + 168*val_n_weeks + buffer
val_x1 = df['ambient_temp_C'][val_start:val_end+1].values    # Input 1: Ambient temperature (same as Part A)
val_x2 = df['supply_temp_C'][val_start:val_end+1].values  # Input 2: Supply water temperature (NEW)
val_y = df['power_MJ_s'][val_start:val_end+1].values 
predict_model(foundModel, inputModel1, inputModel2, val_x1, val_x2, val_y, k=k)


print('=' * 100)


# Test 1 
val_n_weeks = 1
# val_start = len(y) - 200
val_start = 1000 - buffer
val_end = val_start + 168*val_n_weeks + buffer
val_x1 = df['ambient_temp_C'][val_start:val_end+1].values    # Input 1: Ambient temperature (same as Part A)
val_x2 = df['supply_temp_C'][val_start:val_end+1].values  # Input 2: Supply water temperature (NEW)
val_y = df['power_MJ_s'][val_start:val_end+1].values 
predict_model(foundModel, inputModel1, inputModel2, val_x1, val_x2, val_y, k=k)



#%% PACKAGE INTO SOLUTION B


import numpy as np
from scipy import signal
from tsa_lth.modelling import polydiv


def solutionB_1(payload):
    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])

    # ================= Fixed BJ model (Part B) =================
    # Input 1 (ambient temp)
    B1 = np.array([-0.6145, -0.3033, 0.0881])
    F1 = np.array([1.0, 0.4795])

    # Input 2 (supply temp)
    B2 = np.array([1.8582, -0.0541])
    F2 = np.array([1.0])

    # Noise model
    C = np.zeros(25)
    C[0]  = 1.0
    C[1]  = 0.5334
    C[2]  = 0.2826
    C[24] = 0.1106

    D = np.zeros(25)
    D[0]  = 1.0
    D[1]  = -0.8008
    D[24] = -0.1981

    # ==========================================================
    y  = data[:, 1]
    x1 = data[:, 2]
    x2 = data[:, 3]

    test_idx = np.arange(start_idx, end_idx)

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
        signal.lfilter(Fh1, [1], x1) +
        signal.lfilter(Gh1, C_eq, x1) +
        signal.lfilter(Fh2, [1], x2) +
        signal.lfilter(Gh2, C_eq, x2) +
        signal.lfilter(Gk, C_eq, y)
    )

    return yhat[test_idx].tolist()


def solutionB(payload):
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
        signal.lfilter(Fh1, [1], x1) +
        signal.lfilter(Gh1, C_eq, x1) +
        signal.lfilter(Fh2, [1], x2) +
        signal.lfilter(Gh2, C_eq, x2) +
        signal.lfilter(Gk, C_eq, y)
    )

    return yhat[test_idx].tolist()





