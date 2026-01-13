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

#%% Load Part B model
MODEL_ID = 45
data = load_grid_search_results('grid_search_results.json')
config = get_model_config(MODEL_ID, data['configs'])
print_model_config(config)

foundModel = build_model_from_config(config, y, np.column_stack([x1, x2]))
foundModel.summary()

KA, KB1, KB2, KC = extract_bj_polys(foundModel)
desc = kalman_descriptor(foundModel)


if True:
    # Remove MA lag 24 from Kalman state
    desc = [d for d in desc if not (d[0] == "e" and d[1] == 24)]
    KC[24] = 0.0
    
    # Remove AR(24)
    desc = [d for d in desc if not (d[0] == "y" and d[1] == 24)]
    KA[24] = 0.0
    
    # Remove AR(2)
    # desc = [d for d in desc if not (d[0] == "y" and d[1] == 2)]
    # KA[2] = 0.0
    
    # Remove delay on input 1
    desc = [d for d in desc if not (d[0] == "x1" and d[1] == 1)]
    KB1[1] = 0.0
    
    # Remove input 1
    # desc = [d for d in desc if not (d[0] == "x1" and d[1] == 0)]
    # KB1[0] = 0.0
    
    # Remove dealy on input 2
    # desc = [d for d in desc if not (d[0] == "x2" and d[1] == 0)]
    # KB2[1] = 0.0



eP = np.asarray(foundModel.resid)

#%% Kalman setup
N = len(y)
k = 7
buffer = 200
run_start = max(25, buffer)

xt, Rx_t1 = init_kalman_state(desc, KA, KB1, KB2, KC, run_start, N)

A  = np.eye(len(desc))
Rw = np.std(eP)
Re = 1e-6 * np.eye(len(desc))

#%% Run Kalman
yhat_k, h_et, xStd = run_kalman(
    y, x1, x2,
    desc,
    xt, Rx_t1,
    A, Rw, Re,
    k=k,
    run_start=run_start
)

#%% Evaluation (ALIGNED k-step)

val_start = 3200
val_end = val_start + 2*168
test_idx = np.arange(val_start, val_end)


if True:
    m = k  # alignment shift (k-step)
    
    # Align predictions and truth
    yhat_k_aligned = yhat_k[m:]
    y_aligned = y[:-m]
    
    # Align test indices
    test_idx_aligned = test_idx[test_idx < len(y_aligned)]
    
if False:
    y_aligned = y
    yhat_k_aligned = yhat_k
    test_idx_aligned = test_idx

# MSE evaluation
mse_k, mse_naive, _ = evaluate_prediction(
    y_aligned,
    yhat_k_aligned,
    test_idx_aligned,
    k
)

print(f"k={k}  Kalman (aligned)={mse_k:.3f}  Naive={mse_naive:.3f}")

# Plot aligned prediction
plot_predictions(
    y_aligned,
    yhat_k_aligned,
    test_idx_aligned,
    f"{k}-step Kalman prediction (aligned)"
)

# Parameter evolution (unchanged)
plot_parameter_evolution(
    xt, xStd,
    baseline_params=None,
    param_names=[f"{v}(t-{l})" for v, l in desc],
    startInd=run_start,
    title="Recursive Kalman parameters"
)
