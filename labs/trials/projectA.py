# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
import importlib
import scipy.io as sio

# Add path to tsa_lth library
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'TimeSeriesAnalysis-main', 'TimeSeriesAnalysis-main')))

# Import and reload to get the latest changes
from myproject_utils import data_cleanup, get_modeling_dataset, load_project_df, simulate_data
import tsa_lth.analysis
import tsa_lth.modelling
import tsa_lth.tests
importlib.reload(tsa_lth.analysis)
importlib.reload(tsa_lth.modelling)
importlib.reload(tsa_lth.tests)

from tsa_lth.analysis import plotACFnPACF, normplot, xcorr, pzmap, kovarians
from tsa_lth.modelling import estimateARMA, estimateBJ, polydiv
from tsa_lth.tests import whiteness_test, check_if_normal


import pandas as pd
import scipy.io as sio


# %% BASIC VISUALIZATION

df = load_project_df()
idxs = df['obs_num'].values
y = df['power_MJ_s'].values

plt.figure()
plt.plot(idxs, y, label='power data')
plt.xlabel('Observation Number')
plt.ylabel('Power (MJ/s)')
plt.title('Power Data over Observations')
plt.legend()

# %% CLEAN UP DATA
df = data_cleanup(df)

# %% INPUT - OUTPUT VISUALIZATION CHECK

y = df['power_MJ_s'].values
x = df['ambient_temp_C'].values
dates = df['date']

fig, ax = plt.subplots()
ax.plot(dates, y)
ax2 = ax.twinx()
ax2.plot(dates, x, color='orange')
ax.set_xlabel('Observation Number')
ax.set_ylabel('Power (MJ/s)')
ax2.set_ylabel('Ambient Temperature (C)', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
plt.title('Power and Ambient Temperature over Observations')
plt.show()


# %% DECIDE ON MODELLING DATASET


x, y = simulate_data(n=10_000)
x, y = get_modeling_dataset(df, start=800, n_weeks=6, plot=True)


# %% MODEL INPUT AS ARMA

from tsa_lth.modelling import estimateARMA
from tsa_lth.modelling import filter as tsa_filter

log_x = np.log(x)

sday = 24
dayPoly = np.concatenate([[1], np.zeros(sday-1), [-1]])
filt_x = tsa_filter(dayPoly, 1, x, remove=True)

noLags = 100
plotACFnPACF(x, noLags=200, titleStr='Data')
plotACFnPACF(filt_x, noLags=200, titleStr='Filt Data')


my_input_arma = estimateARMA(
    filt_x,
    A=2,
    C=0,
    titleStr='ARMA Model 1',
    noLags=noLags
)


my_input_arma = estimateARMA(
    filt_x,
    A=2,
    C=24,
    C_free=[1, *np.zeros(23), 1],
    titleStr='ARMA Model 2',
    noLags=noLags
)

my_input_arma = estimateARMA(
    filt_x,
    A=2,
    C=24,
    C_free=[1, *np.zeros(23), 1],
    titleStr='ARMA Model 2.5',
    noLags=noLags
)


my_input_arma = estimateARMA(
    filt_x,
    A=3,
    C=24,
    C_free=[1, *np.zeros(23), 1],
    titleStr='ARMA Model 3',
    noLags=noLags
)

my_input_arma = estimateARMA(
    filt_x,
    A=3,
    C=24,
    C_free=[1, 0, 0, 1,  *np.zeros(20), 1],
    titleStr='ARMA Model 3',
    noLags=noLags
)


# my_input_arma = estimateARMA(
#     filt_x,
#     A=25,
#     C=24,
#     A_free=[1, 1, 1, 1, *np.zeros(20), 0, 0],
#     C_free=[1, 0, 0, 1, *np.zeros(20), 1],
#     titleStr='ARMA Model 4',
#     noLags=noLags
# )

# my_input_arma = estimateARMA(
#     filt_x,
#     A=25,
#     C=25,
#     A_free=[1, 1, 1, 1, *np.zeros(20), 1, 1],
#     C_free=[1, 0, 0, 1, *np.zeros(18), 1, 1, 1, 1],
#     titleStr='ARMA Model 5',
#     noLags=noLags
# )

# my_input_arma = estimateARMA(
#     filt_x,
#     A=25,
#     C=25,
#     A_free=[1, 1, 1, 1, *np.zeros(9), 1, *np.zeros(10), 1, 1],
#     C_free=[1, 0, 0, 1, *np.zeros(18), 1, 1, 1, 1],
#     titleStr='ARMA Model 6',
#     noLags=noLags
# )

# my_input_arma = estimateARMA(
#     filt_x,
#     A=25,
#     C=25,
#     A_free=[1, 1, 1, 1, *np.zeros(19), 1, 1, 0],
#     C_free=[1, 0, 1, 1, *np.zeros(19), 1, 1, 0],
#     titleStr='ARMA Model 7',
#     noLags=noLags
# )


# my_input_arma = estimateARMA(
#     filt_x,
#     A=25,
#     C=25,
#     A_free=[1, 1, 1, 1, *np.zeros(9), 1,  *np.zeros(9), 1, 1, 0],
#     C_free=[1, 0, 1, 1, *np.zeros(19), 1, 1, 0],
#     titleStr='ARMA Model 7',
#     noLags=noLags
# )


# %% CHECK CROSS CORRELATION W_T EPS_T

filt_x = tsa_filter(dayPoly, 1, x, remove=True)
filt_y = tsa_filter(dayPoly, 1, y, remove=True)

w_t   = signal.lfilter(my_input_arma.A, my_input_arma.C, filt_x)
eps_t = signal.lfilter(my_input_arma.A, my_input_arma.C, filt_y)

rmv = my_input_arma.model._samps_to_remove()
rmv = 100
w_t, eps_t = w_t[rmv:], eps_t[rmv:]

assert len(w_t) == len(eps_t)

plotACFnPACF(w_t, titleStr='w_t')
plotACFnPACF(eps_t, titleStr='eps_t')


def compute_ccf(x, y, maxlag):
    Cxy = np.correlate(y - np.mean(y), x - np.mean(x), mode='full')
    Cxy = Cxy / (np.std(y) * np.std(x) * len(y))
    lags = np.arange(-maxlag, maxlag + 1)
    mid = len(Cxy) // 2
    Cxy = Cxy[mid - maxlag:mid + maxlag + 1]
    return lags, Cxy


nLag = 200
lags, ccf_vals = compute_ccf(w_t, eps_t, maxlag=nLag)
fig, ax = plt.subplots()
ax.stem(lags, ccf_vals, basefmt=' ')
condInt = 2 / np.sqrt(len(eps_t))
ax.axhline(condInt, color='r', linestyle='--', label='95% confidence')
ax.axhline(-condInt, color='r', linestyle='--')
ax.set_xlabel('Lag')
ax.set_ylabel('CCF')
ax.set_title('Cross-correlation between w_t and eps_t')
ax.set_xlim(-nLag, nLag)
plt.grid(True)
plt.show()


#%% SANITY CHECK FUNCTION

u = np.random.randn(1000)
v = np.roll(u, 5)   # v is u delayed by 5
lags, ccf_vals = compute_ccf(u, v, 20)
print(lags[np.argmax(ccf_vals)])

nLag = 20
fig, ax = plt.subplots()
ax.stem(lags, ccf_vals, basefmt=' ')
condInt = 2 / np.sqrt(len(eps_t))
ax.axhline(condInt, color='r', linestyle='--', label='95% confidence')
ax.axhline(-condInt, color='r', linestyle='--')
ax.set_xlabel('Lag')
ax.set_ylabel('CCF')
ax.set_title('Cross-correlation between w_t and eps_t')
ax.set_xlim(-nLag, nLag)
plt.grid(True)
plt.show()


#%% SANITY CHECK DATA


sday = 24
dayPoly = np.r_[1, np.zeros(sday-1), -1]

# keep dates aligned with filters
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
dates_xy = df['date'].iloc[start:end].values

filt_x = tsa_filter(dayPoly, 1, x, remove=True)
filt_y = tsa_filter(dayPoly, 1, y, remove=True)
dates_f = dates_xy[sday:]              # remove=True drops first 24

w_t   = signal.lfilter(my_input_arma.A, my_input_arma.C, filt_x)
v_t   = signal.lfilter(my_input_arma.A, my_input_arma.C, filt_y)

rmv = my_input_arma.model._samps_to_remove()
w_t, v_t = w_t[rmv:], v_t[rmv:]
dates_f  = dates_f[rmv:]               # SAME trim
print(len(w_t), len(v_t), len(dates_f))
print(dates_f[:3], dates_f[-3:])       # sanity: same timebase

# quick sanity: put in a df so you can eyeball
tmp = pd.DataFrame({"date": dates_f, "w_t": w_t, "v_t": v_t})
print(tmp.head(20))


# %% EXAMINE BJ MODEL


noLags = 200

sday = 25
sweek = 168
dayPoly = np.r_[1, np.zeros(sday-1), -1]
weekPoly = np.concatenate([[1], np.zeros(sweek-1), [-1]])

y_filt = y.copy()
y_filt = tsa_filter(dayPoly, 1, y_filt, remove=True)
y_filt = tsa_filter(weekPoly, 1, y_filt, remove=True)

x_filt = x[193:]

bj_model = estimateBJ(
    y_filt,
    x_filt,
    d=1,
    B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
    B_free=[True, True],      # estimate both taps
    A2=[1],                   # s=0
    C1=[1],
    A1=[1],
    titleStr="BJ model 1",
    noLags=noLags,
)



# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
#     B_free=[True, True],      # estimate both taps
#     A2=[1],                   # s=0
#     C1=[1],
#     A1=[1, 1, 1],
#     titleStr="BJ model 2 (d=0, r=1, s=0)",
#     noLags=noLags,
# )


# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
#     B_free=[True, True],      # estimate both taps
#     A2=[1],                   # s=0
#     C1=[1, *np.zeros(23), 1],
#     C1_free=[1, *np.zeros(23), 1],
#     A1=[1, 1, 1],
#     titleStr="BJ model 3 (d=0, r=1, s=0)",
#     noLags=noLags,
# )



# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
#     B_free=[True, True],      # estimate both taps
#     A2=[1],                   # s=0
#     C1=[1, *np.zeros(23), 1],
#     C1_free=[1, *np.zeros(23), 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 1],
#     titleStr="BJ model 4(d=0, r=1, s=0)",
#     noLags=noLags,
# )



# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1, 1],
#     B_free=[True, True],
#     A2=[1],
#     C1=[1, *np.zeros(23), 1],
#     C1_free=[1, *np.zeros(23), 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 1],   # add ONLY weekly AR at lag 7 (minimal next step)
#     A1_free=[1, 1, 1, 0, 0, 0, 0, 1],
#     titleStr="BJ model 5",
#     noLags=noLags,
# )


# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1, 1],
#     B_free=[True, True],
#     A2=[1],
#     C1=[1, *np.zeros(22), 1, 1],
#     C1_free=[1, *np.zeros(22), 1, 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 1, *np.zeros(16), 1],   
#     A1_free=[1, 1, 1, 0, 0, 0, 0, 1, *np.zeros(16), 1],
#     titleStr="BJ model 6",
#     noLags=noLags,
# )


# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=[1],
#     C1=[1, *np.zeros(22), 1, 1],
#     C1_free=[1, *np.zeros(22), 1, 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 1, *np.zeros(16), 1],   
#     A1_free=[1, 1, 1, 0, 0, 0, 0, 1, *np.zeros(16), 1],
#     titleStr="BJ model 7",
#     noLags=noLags,
# )



# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=[1],
#     C1=[1, *np.zeros(22), 1, 1],
#     C1_free=[1, *np.zeros(22), 1, 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 1, *np.zeros(16), 1],   
#     A1_free=[1, 1, 1, 0, 0, 0, 0, 1, *np.zeros(16), 1],
#     titleStr="BJ model 8",
#     diff=168,
#     noLags=noLags,
# )


# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=[1],
#     C1=[1, *np.zeros(22), 1, 1],
#     C1_free=[1, *np.zeros(22), 1, 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 0, *np.zeros(16), 1],   
#     A1_free=[1, 1, 1, 0, 0, 0, 0, 0, *np.zeros(16), 1],
#     titleStr="BJ model 9",
#     diff=168,
#     noLags=noLags,
# )


# bj_model = estimateBJ(
#     y,
#     x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=[1],
#     C1=[1, *np.zeros(22), 0, 1],
#     C1_free=[1, *np.zeros(22), 0, 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 0, *np.zeros(16), 1],   
#     A1_free=[1, 1, 1, 0, 0, 0, 0, 0, *np.zeros(16), 1],
#     titleStr="BJ model 10",
#     diff=168,
#     noLags=noLags,
# )


# bj_model = estimateBJ(
#     y, x,
#     d=6,
#     B=[1, 1],
#     B_free=[True, True],
#     A2=[1, 1],
#     C1=[1, *np.zeros(23), 1],
#     C1_free=[1, *np.zeros(23), 1],
#     A1=[1, 1, 1, 0, 0, 0, 0, 0],
#     A1_free=[1, 1, 1, 0, 0, 0, 0, 0],
#     diff=[168],
#     titleStr="BJ next: add input delay 2",
#     noLags=noLags,
# )


#%% BJ TRIAL 2 -    q=r=s=0


# noLags = 200
# bj_model = estimateBJ(
#     y, x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=[1],
#     C1=[1],
#     C1_free=[1],
#     A1=[1],
#     A1_free=[1],
#     diff=None,
#     titleStr="",
#     noLags=noLags,
# )

# bj_model = estimateBJ(
#     y, x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=[1],
#     C1=[1],
#     C1_free=[1],
#     A1=[1, 1, 1],
#     A1_free=[1, 1, 1],
#     diff=None,
#     titleStr="",
#     noLags=noLags,
# )

# bj_model = estimateBJ(
#     y, x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=1,
#     C1=24,
#     C1_free=[1, *np.zeros(23), 1],
#     A1=2,
#     A1_free=[1, 1, 1],
#     diff=None,
#     titleStr="",
#     noLags=noLags,
# )

# bj_model = estimateBJ(
#     y, x,
#     d=0,
#     B=[1],
#     B_free=[True],
#     A2=1,
#     C1=24,
#     C1_free=[1, *np.zeros(23), 1],
#     A1=7,
#     A1_free=[1, 1, 1, *np.zeros(4), 1],
#     diff=None,
#     titleStr="",
#     noLags=noLags,
# )

bj_model = estimateBJ(
    y, x,
    d=24,
    B=[1, 1, 1],
    B_free=[True, False, True],
    A2=2,
    A2_free=[1, 1, 1],
    C1=25,
    C1_free=[1, *np.zeros(6), 1,  *np.zeros(15), 1, 1, 1],
    A1=7,
    A1_free=[1, 1, 1, *np.zeros(4), 1],
    diff=0,
    titleStr="",
    noLags=noLags,
)







#%% DOING FINAL CHECKS


bj_model.d = 24
bj_model.A2 = bj_model.F


output = y - np.mean(y)
_input = x

# output = tsa_filter(dayPoly, 1 , y)
# output = output[24:]

_input = tsa_filter(bj_model.B, bj_model.A2, x)
_input = _input - np.mean(_input)
# _input = _input[24:]


output = output[100:]
_input = _input[100:]



diff = output - _input


plt.plot(output)
plt.plot(_input)
plt.plot(diff)

print(f'var(output): {np.var(output)}:4f')
print(f'var(output - input): {np.var(diff)}:4f')


#%% TRIAL COPYING CODE 6


noLags = 200

sday = 25
sweek = 168
dayPoly = np.r_[1, np.zeros(sday-1), -1]
weekPoly = np.concatenate([[1], np.zeros(sweek-1), [-1]])

y_filt = y.copy()
y_filt = tsa_filter(dayPoly, 1, y_filt, remove=True)
y_filt = tsa_filter(weekPoly, 1, y_filt, remove=True)

x_filt = x[193:]

bj_model = estimateBJ(
    y_filt,
    x_filt,
    d=1,
    B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
    B_free=[True, True],      # estimate both taps
    A2=[1],                   # s=0
    C1=[1],
    A1=[1],
    titleStr="BJ model 1",
    noLags=noLags,
)


bj_model = estimateBJ(
    y_filt,
    x_filt,
    d=1,
    B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
    B_free=[True, True],      # estimate both taps
    A2=[1],                   # s=0
    C1=[1],
    A1=2,
    A1_free=[1, 1, 1],
    titleStr="BJ model 2",
    noLags=noLags,
)



C1 = np.convolve(dayPoly, weekPoly)
C1[1:] = 0.3 * C1[1:]
C1_free=np.abs(C1) > 1e-10


bj_model = estimateBJ(
    y_filt,
    x_filt,
    d=1,
    B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
    B_free=[True, True],      # estimate both taps
    A2=[1],                   # s=0
    C1=len(C1_free)-1,
    C1_free = C1_free, 
    A1=2,
    A1_free=[1, 1, 1],
    titleStr="BJ model 3",
    noLags=noLags,
)




#%% CHECKS ON MODEL


# %% PREPROCESS (same filtering for y and x)

y_filt = tsa_filter(dayPoly, 1, y, remove=True)
y_filt = tsa_filter(weekPoly, 1, y_filt, remove=True)

x_filt = tsa_filter(dayPoly, 1, x, remove=True)
x_filt = tsa_filter(weekPoly, 1, x_filt, remove=True)


# %% FIT BJ MODEL (already done)

bj_model = estimateBJ(
    y_filt,
    x_filt,
    d=1,
    B=[1, 1],                 # B(z) = b0 + b1 z^-1  (r=1)
    B_free=[True, True],      # estimate both taps
    A2=[1],                   # s=0
    C1=len(C1_free)-1,
    C1_free = C1_free, 
    A1=2,
    A1_free=[1, 1, 1],
    titleStr="BJ model 3",
    noLags=noLags,
)


# y_nd = y.copy()
# x_nd = x.copy()

# bj_model = estimateBJ(
#     y_nd,
#     x_nd,
#     d=1,
#     B=[1, 1],
#     B_free=[1, 1],
#     A2=[1],
#     C1=[1],
#     A1=2,
#     A1_free=[1, 1, 1],
#     titleStr="BJ undifferenced",
#     noLags=200,
# )



# %% 1) INPUT–RESIDUAL INDEPENDENCE CHECK (CORRECT)

e = bj_model.resid
rmv = bj_model.model._samps_to_remove()

e = e[rmv:]
x_chk = x_filt[rmv:]

Cxe = np.correlate(
    x_chk - x_chk.mean(),
    e - e.mean(),
    mode="full"
) / (np.std(x_chk) * np.std(e) * len(e))

lags = np.arange(-noLags, noLags + 1)
mid = len(Cxe) // 2
Cxe = Cxe[mid-noLags:mid+noLags+1]

plt.stem(lags, Cxe)
ci = 2 / np.sqrt(len(e))
plt.axhline(ci, color="r", ls="--")
plt.axhline(-ci, color="r", ls="--")
plt.title("CCF(input, BJ residual)")
plt.tight_layout()
plt.show()


# %% 2) VARIANCE EXPLAINED BY INPUT (CORRECT)

# input contribution u_t = (B / A2) x_t
u = tsa_filter(bj_model.B, bj_model.F, x_filt, remove=False)

# align
y_var = y_filt[rmv:]
u_var = u[rmv:]

# remove means
y_var -= y_var.mean()
u_var -= u_var.mean()

res_no_input = y_var - u_var

print(f"Var(y):            {np.var(y_var):.4f}")
print(f"Var(y - input):    {np.var(res_no_input):.4f}")
print(f"Frac explained x:  {1 - np.var(res_no_input)/np.var(y_var):.4f}")


















