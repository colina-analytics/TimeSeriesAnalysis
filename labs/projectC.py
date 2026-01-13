import os, sys
import numpy as np

# Add path to tsa_lth library
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.getcwd(), "..", "TimeSeriesAnalysis-main", "TimeSeriesAnalysis-main"
        )
    )
)

from myproject_utils import load_project_df, data_cleanup
from projectB_gridsearch import (
    build_model_from_config,
    load_grid_search_results,
    get_model_config,
    print_model_config
)

from projectC_utils import (
    extract_bj_polys,
    init_kalman_state,
    run_kalman,
    evaluate_prediction,
    plot_predictions,   
    plot_parameters,
    plot_parameter_evolution,
    kalman_descriptor
)

#%% Load data
df = data_cleanup(load_project_df())
x1 = df['ambient_temp_C'].values
x2 = df['supply_temp_C'].values
y  = df['power_MJ_s'].values


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


#%% Load Part B model

MODEL_ID = 169
data = load_grid_search_results('grid_search_results.json')
config = get_model_config(MODEL_ID, data['configs'])
print_model_config(config)

foundModel = build_model_from_config(config, y, np.column_stack([x1, x2]))
foundModel.summary()

KA, KB1, KB2, KC = extract_bj_polys(foundModel)
desc = kalman_descriptor(foundModel)


if False:
    # Remove MA lag 23 from Kalman state
    desc = [d for d in desc if not (d[0] == "e" and d[1] == 23)]
    KC[23] = 0.0
    
    # Remove AR(24)
    desc = [d for d in desc if not (d[0] == "y" and d[1] == 24)]
    KA[24] = 0.0
    
    # Remove AR(168)
    desc = [d for d in desc if not (d[0] == "y" and d[1] == 168)]
    KA[168] = 0.0
    
    # Remove AR(169)
    desc = [d for d in desc if not (d[0] == "y" and d[1] == 169)]
    KA[169] = 0.0
    
    # Remove AR(72)
    desc = [d for d in desc if not (d[0] == "y" and d[1] == 72)]
    KA[72] = 0.0
    
    # Remove on input 1
    desc = [d for d in desc if not (d[0] == "x1" and d[1] == 2)]
    KB1[2] = 0.0
    
    # Remove input 1
    desc = [d for d in desc if not (d[0] == "x1" and d[1] == 3)]
    KB1[3] = 0.0
    

if True:
    # Remove AR(169)
    desc = [d for d in desc if not (d[0] == "y" and d[1] == 169)]
    KA[169] = 0.0
    
    # Remove MA lag 23 from Kalman state
    desc = [d for d in desc if not (d[0] == "e" and d[1] == 23)]
    KC[23] = 0.0
    
    # Remove AR(72)
    desc = [d for d in desc if not (d[0] == "y" and d[1] == 72)]
    KA[72] = 0.0
    
    # Remove input 1
    desc = [d for d in desc if not (d[0] == "x1" and d[1] == 3)]
    KB1[3] = 0.0
    
    # Remove input 1
    desc = [d for d in desc if not (d[0] == "x1" and d[1] == 2)]
    KB1[2] = 0.0



eP = np.asarray(foundModel.resid)

#%% Kalman setup

from tsa_lth.analysis import plotACFnPACF
from tsa_lth.tests import whiteness_test

N = len(y)
k = 1
buffer = 200
run_start = max(25, buffer)

xt, Rx_t1 = init_kalman_state(desc, KA, KB1, KB2, KC, run_start, N)

A  = np.eye(len(desc))
Rw = 1
Re = 1e-6 * np.eye(len(desc))

yhat_k, h_et, xStd = run_kalman(
    y, x1, x2,
    desc,
    xt, Rx_t1,
    A, Rw, Re,
    k=k,
    run_start=run_start
)


# Test indexes [2900, 3068; 4700, 4868; 1000, 1168]


import matplotlib.pyplot as plt

window = windows[1]
for window in windows:
    val_start = window[1]
    val_end = window[2]
    # val_start = 2900
    # val_end = 3068
    test_idx = np.arange(val_start, val_end)
    dates = df['date'][test_idx].values
    
    
    # MSE evaluation
    mse_k, mse_naive, _ = evaluate_prediction(
        y,
        yhat_k,
        test_idx,
        k
    )
    
    print(f"k={k}  Kalman (aligned)={mse_k:.3f}  Naive={mse_naive:.3f}")
    
    plt.figure(figsize=(10,5))
    plt.plot(dates, y[test_idx], label='Real')
    plt.plot(dates, yhat_k[test_idx], label='Kalman')
    plt.title('')
    plt.legend()
    # plt.grid(True)
    plt.ylabel('Power (MJ/s)')
    plt.show()
    
    if k == 1:
        ehat = y[test_idx] - yhat_k[test_idx]
        plotACFnPACF(ehat, noLags=100, titleStr=window[0])
        whiteness_test(ehat)

#%%

# Parameter evolution (unchanged)
plot_parameter_evolution(
    xt, xStd,
    baseline_params=None,
    param_names=[f"{v}(t-{l})" for v, l in desc],
    startInd=run_start,
    title=""
)
