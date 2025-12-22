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
)
import tsa_lth.analysis
import tsa_lth.modelling
import tsa_lth.tests

importlib.reload(tsa_lth.analysis)
importlib.reload(tsa_lth.modelling)
importlib.reload(tsa_lth.tests)

from tsa_lth.analysis import box_cox, plotACFnPACF, normplot, xcorr, pzmap, kovarians
from tsa_lth.modelling import estimateARMA, estimateBJ, polydiv
from tsa_lth.modelling import filter as tsa_filter
from tsa_lth.tests import whiteness_test, check_if_normal


import pandas as pd
import scipy.io as sio


# %% BASIC VISUALIZATION

df = load_project_df()
idxs = df["obs_num"].values
y = df["power_MJ_s"].values

plt.figure()
plt.plot(idxs, y, label="power data")
plt.xlabel("Observation Number")
plt.ylabel("Power (MJ/s)")
plt.title("Power Data over Observations")
plt.legend()

# %% CLEAN UP DATA
df = data_cleanup(df)

# %% INPUT - OUTPUT VISUALIZATION CHECK

y = df["power_MJ_s"].values
x = df["ambient_temp_C"].values
dates = df["date"]

fig, ax = plt.subplots()
ax.plot(dates, y)
ax2 = ax.twinx()
ax2.plot(dates, x, color="orange")
ax.set_xlabel("Observation Number")
ax.set_ylabel("Power (MJ/s)")
ax2.set_ylabel("Ambient Temperature (C)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
plt.title("Power and Ambient Temperature over Observations")
plt.show()


# %% DECIDE ON MODELLING DATASET

# x, y = simulate_data(n=10_000)
model_data_start = 800
model_n_weeks = 6
model_slice = slice(model_data_start, model_data_start + 168*model_n_weeks - 1)
x, y = get_modeling_dataset(df, start=model_data_start, n_weeks=model_n_weeks, plot=True)
# x = x - np.mean(x)
# y = y - np.mean(y)

fig, ax = plt.subplots()
ax.plot(dates[model_slice], y)
ax2 = ax.twinx()
ax2.plot(dates[model_slice], x, color="orange")
ax.set_xlabel("Observation Number")
ax.set_ylabel("Power (MJ/s)")
ax2.set_ylabel("Ambient Temperature (C)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
plt.title("Power and Ambient Temperature over Observations")
plt.show()


# %% SHOULD WE TRANSFORM THE DATA?

lambda_max = box_cox(x, titleStr="Input (Temp)")
# Not needed for Temp

lambda_max = box_cox(y, titleStr="Output (Power)")
# Max Lambda = 0.6134 so maybe sqrt but could just ignore


# %% CHECK INPUT-OUTPUT CORRELATION

noLags = round(len(y) / 4)
noLags = min(noLags, 400)
plot_ccf(x, y, noLags)


# %% MODEL INPUT (X)

noLags = 200


plotACFnPACF(x, noLags, "Input", includeZeroLag=True)


input_model = estimateARMA(
    x,
    A=24,
    A_free=[1, 1, 1, 0, *np.zeros(8), 0, *np.zeros(10), 1, 1],
    C=24,
    C_free=[1, 1, 0, 1,  *np.zeros(8), 0, *np.zeros(9), 1, 0, 0],
    diff=0,
    titleStr="No diff input model 10",
    noLags=noLags,
)

print(f"Original var: {np.var(x)}")
print(f"Residual var: {np.var(input_model.resid)}")

# %% CHECK CROSS-CORRELATION


A = input_model.A
C = input_model.C

eps_t = tsa_filter(A, C, y, remove=True)
w_t = tsa_filter(A, C, x, remove=True)


plot_ccf(eps_t, w_t, noLags=20)

# max at lag=0 --> d=0
# ringing --> r = 2
# hard to tell width, maybe 3 -->


# %% CHECK INPUT CONTRIBUTION

from tsa_lth.modelling import PEM

d, r, s = (0, 2, 1)
B_init = [0] * d + [1] * (s+1)
A2_init = [1] + [1] * r
C1_init = [1]
A1_init = [1]

input_bj = PEM(y, x, B=B_init, F=A2_init, C=C1_init, D=A1_init)
B_free = [0] * d + [1] * (s+1)
A2_free = [1] + [1] * r
C1_free = None
A1_free = None
input_bj.set_free_params(B_free=B_free, F_free=A2_free, C_free=C1_free, D_free=A1_free)
input_bj_model = input_bj.fit()
etilde = input_bj_model.resid
input_bj_model.summary()


xfilt = signal.lfilter(input_bj_model.B, input_bj_model.F, x)
y_cut = y[len(input_bj_model.F):]
xfilt_cut = xfilt[len(input_bj_model.F):]

fig, ax = plt.subplots()
ax.plot(y_cut, label='Output y', alpha=0.7 )
ax.plot(xfilt_cut, label='Filtered input (B/A2)x', alpha=0.7)
ax.legend()
ax.grid(True)
plt.show()

var_y = np.var(y)
var_etilde = np.var(etilde) 

print(f'var_y: {var_y}')
print(f'var_etilde: {var_etilde}')


# %% MODEL E_TILDE AS ARMA


arma = estimateARMA(etilde, A=0, C=0, noLags=noLags)
# Strong seasonality, but also AR(2)

arma = estimateARMA(etilde, A=2, C=0, noLags=noLags)
# Strong 12 and 24 hours cycles


arma = estimateARMA(
    etilde, 
    A=2,
    A_free=[1, 1, 1], 
    C=24, 
    C_free=[1, *np.zeros(23), 1],
    noLags=noLags
    )
# Still strong AR(12)


arma = estimateARMA(
    etilde, 
    A=12,
    A_free=[1, 1, 1, *np.zeros(9), 1], 
    C=24, 
    C_free=[1, *np.zeros(23), 1],
    noLags=noLags
    )
# Maybe we need to add 23 and 25 to the C\


arma = estimateARMA(
    etilde, 
    A=12,
    A_free=[1, 1, 1, *np.zeros(9), 1], 
    C=25, 
    C_free=[1, *np.zeros(22), 1, 1, 1],
    noLags=noLags
    )

# Close to white! Maybe a 168 for the weekly cycle?


arma = estimateARMA(
    etilde, 
    A=12,
    A_free=[1, 1, 1, *np.zeros(9), 1], 
    C=168, 
    C_free=[1, *np.zeros(22), 1, 1, 1, *np.zeros(168-25-1), 1],
    noLags=noLags
    ) 
# almost white, maybe add that 11 for the A


arma = estimateARMA(
    etilde, 
    A=12,
    A_free=[1, 1, 1, *np.zeros(8), 1, 1], 
    C=168, 
    C_free=[1, *np.zeros(22), 1, 1, 1, *np.zeros(168-25-1), 1],
    noLags=noLags
    ) 
# Completely white! can we remove coeffs?


arma = estimateARMA(
    etilde, 
    A=12,
    A_free=[1, 1, 1, *np.zeros(8), 1, 1], 
    C=25, 
    C_free=[1, *np.zeros(22), 1, 1, 1],
    noLags=noLags
    ) 
# Worked well without the C(168)


arma = estimateARMA(
    etilde, 
    A=12,
    A_free=[1, 1, 1, *np.zeros(8), 1, 1], 
    C=25, 
    C_free=[1, *np.zeros(22), 1, 1, 1],
    noLags=noLags
    ) 
# I think we can be happy here

# %% COMPLETE BJ MODEL


B_init = input_bj_model.B
A2_init = input_bj_model.F
C1_init = arma.C
A1_init = arma.A

# C1_init = 24
# A1_init = 12

B_free = np.array([1 if coeff > 0.001 else 0 for coeff in input_bj_model.B])
A2_free = np.array([1 if coeff > 0.001 else 0 for coeff in input_bj_model.F])

C1_free = np.array([1, *np.zeros(22), 1, 1, 1]) * 0.2
A1_free = np.array([1, 1, 1, *np.zeros(8), 1, 1]) * 0.2
# C1_free = [1, *np.zeros(23), 1]
# A1_free = [1, 1, *np.zeros(10), 1]

model_boxj = PEM(y, x, B=B_init, F=A2_init, C=C1_init, D=A1_init)
model_boxj.set_free_params(B_free=B_free, F_free=A2_free, C_free=C1_free, D_free=A1_free)
MboxJ = model_boxj.fit()
ehat = MboxJ.resid
MboxJ.summary()

plotACFnPACF(ehat, titleStr='Complete BJ Model')

whiteness_test(MboxJ.resid)


# %% CHECKING

# Analyze the model
xfilt = signal.lfilter(MboxJ.B, MboxJ.F, x)
y_cut = y[len(MboxJ.F):]
xfilt_cut = xfilt[len(MboxJ.F):]

fig, ax = plt.subplots()
ax.plot(y_cut, label='Output y', alpha=0.7 )
ax.plot(xfilt_cut, label='Filtered input x', alpha=0.7)
ax.legend()
ax.grid(True)
plt.show()


# Checking variance reduction
var_y = np.var(y)
var_etilde = np.var(etilde) 
var_ehat = np.var(ehat) 

print(f'var_y: {var_y}')
print(f'var_etilde: {var_etilde}')
print(f'var_ehat: {var_ehat}')

# Check for whiteness
whiteness_test(ehat)


plot_ccf(w_t, ehat, noLags=200)


# Maybe we'll have to go back bc we lost too much input


# %% CHECKING WITH VALIDATION DATA


# Create Polys
B  = np.array(MboxJ.B)
F  = np.array(MboxJ.F)   # A2
C  = np.array(MboxJ.C)   # C1
D  = np.array(MboxJ.D)   # A1

A_eq = np.convolve(F, D)
B_eq = np.convolve(D, B)
C_eq = np.convolve(F, C)

k = 1
F_k, G_k = polydiv(C_eq, A_eq, k)
Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)


# Predict over validation data
buffer = 200
val_data_start = model_data_start + 168*model_n_weeks - buffer
val_data_n_weeks = 4
val_data_end = val_data_start + 168*val_data_n_weeks + buffer


val_x = df['ambient_temp_C'][val_data_start:val_data_end].values
val_y = df['power_MJ_s'][val_data_start:val_data_end].values
val_dates = df['date'][val_data_start:val_data_end].values


yhat_k = (
    signal.lfilter(Fhat, [1], val_x) +
    signal.lfilter(Ghat, C_eq, val_x) +
    signal.lfilter(G_k, C_eq, val_y)
)

rmv = max(len(G_k), len(Ghat))
yhat_k = yhat_k[rmv + buffer:]
y_filtered = val_y[rmv + buffer:]
dates = val_dates[rmv + buffer:]


# Residual analysis
ehat_k = y_filtered - yhat_k
plotACFnPACF(ehat_k, noLags=180, titleStr=f'{k}-step pred. residual')
whiteness_test(ehat_k)
print(f'Original variance: {np.var(y_filtered):.2f}')
print(f'Residual variance: {np.var(ehat_k):.2f}')


# Compare with naive predictor (k-step persistence)
y_true = y_filtered[k:]
yhat_naive = y_filtered[:-k]
ehat_naive = y_true - yhat_naive
print(f'Naive residual variance: {np.var(ehat_naive):.2f}')

# --- MSE ---
mse_model = np.mean(ehat_k**2)
mse_naive = np.mean(ehat_naive**2)

print(f'Model MSE: {mse_model:.2f}')
print(f'Naive MSE: {mse_naive:.2f}')
print(f'MSE ratio (model / naive): {mse_model / mse_naive:.3f}')


# Plotting
fig, ax = plt.subplots(figsize=[10, 6]) 
ax.plot(dates[k:], y_filtered[k:], label='Data', alpha=0.7) 
ax.plot(dates[k:], yhat_k[k:], label='Prediction', alpha=0.7) 
ax.plot(dates[k:], yhat_naive, label='Naive', alpha=0.4)
plt.xticks(rotation=30)
ax.legend()


# %% Testing


# Select data range
buffer = 200
test1_data_start = val_data_end - buffer
test1_n_weeks = 1
test1_data_end = test1_data_start + 168*test1_n_weeks + buffer


test_x = df['ambient_temp_C'][test1_data_start:test1_data_end].values
test_y = df['power_MJ_s'][test1_data_start:test1_data_end].values


# Select data range
buffer = 200
test2_data_start = 700 + test1_data_end - buffer
test2_n_weeks = 2
test2_data_end = test2_data_start + 168*test2_n_weeks + buffer


test_x = df['ambient_temp_C'][test2_data_start:test2_data_end].values
test_y = df['power_MJ_s'][test2_data_start:test2_data_end].values
test_dates = df['date'][test2_data_start:test2_data_end].values


# Predict over data
k = 1
F_k, G_k = polydiv(C_eq, A_eq, k)
Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)
yhat_k = (
    signal.lfilter(Fhat, [1], test_x) +
    signal.lfilter(Ghat, C_eq, test_x) +
    signal.lfilter(G_k, C_eq, test_y)
)

rmv = max(len(G_k), len(Ghat))
yhat_k = yhat_k[rmv + buffer:]
y_filtered = test_y[rmv + buffer:]
dates = test_dates[rmv + buffer:]

# Compare with naive predictor (k-step persistence)
y_true = y_filtered[k:]
yhat_naive = y_filtered[:-k]
ehat_naive = y_true - yhat_naive

# Residual analysis
ehat_k = y_filtered - yhat_k
plotACFnPACF(ehat_k, noLags=60, titleStr=f'{k}-step pred. residual')
whiteness_test(ehat_k)
print(f'Original variance: {np.var(y_filtered):.2f}')
print(f'Residual variance: {np.var(ehat_k):.2f}')
print(f'Naive residual variance: {np.var(ehat_naive):.2f}')


# --- MSE ---
mse_model = np.mean(ehat_k**2)
mse_naive = np.mean(ehat_naive**2)

print(f'Model MSE: {mse_model:.2f}')
print(f'Naive MSE: {mse_naive:.2f}')
print(f'MSE ratio (model / naive): {mse_model / mse_naive:.3f}')


# Plotting
fig, ax = plt.subplots(figsize=[10, 6]) 
ax.plot(dates[k:], y_filtered[k:], label='Data', alpha=0.7) 
ax.plot(dates[k:], yhat_k[k:], label='Prediction', alpha=0.7) 
ax.plot(dates[k:], yhat_naive, label='Naive', alpha=0.4)
plt.xticks(rotation=30)
ax.legend()

# %% TESTING PACKAGE


from myproject_utils import test_model


test_model(df, start_index=800, n_weeks=6, k=7, BJmodel=MboxJ, buffer=200, plot=True)
