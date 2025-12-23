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

from tsa_lth.analysis import box_cox, plotACFnPACF, normplot, xcorr, pzmap, kovarians
from tsa_lth.modelling import MultiInputPEM, estimateARMA, estimateBJ, polydiv
from tsa_lth.modelling import filter as tsa_filter
from tsa_lth.tests import whiteness_test, check_if_normal

import pandas as pd


# %% LOAD AND CLEAN DATA

if __name__ == "__main__":
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



#%% ========== HELPER FUNCTIONS FOR MODEL LOADING ==========
import json
from datetime import datetime
from itertools import product


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


def get_model_config(model_id, configs):
    """Get configuration for a specific model ID."""
    for config in configs:
        if config['model_id'] == model_id:
            return config
    raise ValueError(f"Model ID {model_id} not found")


def build_model_from_config(config, y, x_multi):
    """
    Build and fit a MultiInputPEM model from a saved configuration.

    Parameters:
    -----------
    config : dict
        Model configuration dictionary (from grid search)
    y : array
        Output signal
    x_multi : array
        Stacked input signals [x1, x2]

    Returns:
    --------
    fitted : MultiInputPEMResult
        Fitted model result
    """
    B = [config['B1'], config['B2']]
    F = [config['F1'], config['F2']]

    model = MultiInputPEM(
        y=y,
        x=x_multi,
        A=1,
        B=B,
        F=F,
        C=config['C_init'].copy(),
        D=config['D_init'].copy(),
        nk=[0, 0]
    )

    model.set_free_params(
        B_free=None,
        F_free=None,
        C_free=config['C_free'],
        D_free=config['D_free']
    )

    fitted = model.fit(method='LS', verbose=0)
    return fitted


def load_grid_search_results(filepath='grid_search_results.json'):
    """Load grid search results and configurations from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def print_model_config(config):
    """Pretty print a model configuration."""
    print(f"\n{'='*60}")
    print(f"MODEL {config['model_id']}: {config['noise_desc']}")
    print('='*60)
    print(f"B1 = {config['B1']}")
    print(f"F1 = {config['F1']}")
    print(f"B2 = {config['B2']}")
    print(f"F2 = {config['F2']}")
    print(f"C_init = {config['C_init']}")
    print(f"D_init = {config['D_init']}")
    print(f"C_free = {config['C_free']}")
    print(f"D_free = {config['D_free']}")


#%% ========== GRID SEARCH FOR MULTI-INPUT MODEL ==========

if __name__ == "__main__":

    # Prepare data
    x_multi = np.column_stack([x1, x2])

    # Define parameter grid
    # Transfer functions (based on CCF analysis and Part A results)
    # Target: ~160 configs (double the original 80)

    # B1 options: ambient temperature (Part A had 3 terms: [-0.6145, -0.3033, 0.0881])
    B1_options = [
        [0, 1],              # delay=1, s=0
        [0, 1, 0.5],         # delay=1, s=1
        [0, 1, 0.5, 0.3],    # delay=1, s=2 (like Part A)
    ]

    # F1 options: dynamics for ambient temp (Part A had AR(1): [1.0, 0.4795])
    F1_options = [
        [1],                 # no dynamics
        [1, 0.5],            # AR(1) - like Part A
    ]

    # B2 options: supply water temp (Part A had [1.8582, -0.0541] - immediate with lag)
    B2_options = [
        [1],                 # just gain (immediate)
        [1, 0.5],            # immediate + lag 1 (like Part A)
        [0, 1],              # delay=1
    ]

    # F2 options: keep simple for supply temp (Part A had [1.0])
    F2_options = [
        [1],                 # no dynamics
    ]

    # Noise model variations (main focus)
    # Part A had: MA(2)+MA(24) with AR(1)+AR(24)
    # 3 x 2 x 3 x 9 = 162 configs
    noise_configs = [
        # (C_init, D_init, C_free, D_free, description)
        # Basic models
        ([1], [1], [False], [False], "No noise model"),
        ([1, 0.5], [1, -0.5], [False, True], [False, True], "ARMA(1,1)"),
        ([1, 0.5, 0.3], [1, -0.5], [False, True, True], [False, True], "ARMA(1,2)"),
        ([1, 0.5, 0.3], [1, -0.5, 0.3], [False, True, True], [False, True, True], "ARMA(2,2)"),

        # With lag 24 (seasonality) - matching Part A structure
        ([1, 0.5, *[0]*22, 0.3], [1, -0.5],
         [False, True, *[False]*22, True], [False, True], "MA(1)+MA(24), AR(1)"),
        ([1, 0.5, *[0]*22, 0.3], [1, -0.5, *[0]*22, 0.3],
         [False, True, *[False]*22, True], [False, True, *[False]*22, True], "MA(1)+MA(24), AR(1)+AR(24)"),

        # MA(2) + seasonal (like Part A: C[0,1,2,24])
        ([1, 0.5, 0.3, *[0]*21, 0.2], [1, -0.5],
         [False, True, True, *[False]*21, True], [False, True], "MA(2)+MA(24), AR(1)"),
        ([1, 0.5, 0.3, *[0]*21, 0.2], [1, -0.5, *[0]*22, 0.2],
         [False, True, True, *[False]*21, True], [False, True, *[False]*22, True], "MA(2)+MA(24), AR(1)+AR(24)"),

        # Higher AR orders with seasonality
        ([1, 0.5, 0.3, *[0]*21, 0.2], [1, -0.5, 0.3, *[0]*21, 0.2],
         [False, True, True, *[False]*21, True], [False, True, True, *[False]*21, True], "MA(2)+MA(24), AR(2)+AR(24)"),
    ]

    # Generate all combinations
    all_configs = []
    model_id = 0

    for B1, F1, B2, (C_init, D_init, C_free, D_free, noise_desc) in product(
        B1_options, F1_options, B2_options, noise_configs
    ):
        model_id += 1
        all_configs.append({
            'model_id': model_id,
            'B1': B1,
            'F1': F1,
            'B2': B2,
            'F2': [1],
            'C_init': C_init,
            'D_init': D_init,
            'C_free': C_free,
            'D_free': D_free,
            'noise_desc': noise_desc
        })

    print(f"Total configurations to test: {len(all_configs)}")


    #%% RUN GRID SEARCH

    results = []

    for config in all_configs:
        print(f"\n{'='*60}")
        print(f"Testing Model {config['model_id']}: B1={len(config['B1'])-1}, F1={len(config['F1'])-1}, "
              f"B2={len(config['B2'])-1}, Noise: {config['noise_desc']}")
        print('='*60)

        try:
            # Build model
            B = [config['B1'], config['B2']]
            F = [config['F1'], config['F2']]

            model = MultiInputPEM(
                y=y,
                x=x_multi,
                A=1,
                B=B,
                F=F,
                C=config['C_init'].copy(),
                D=config['D_init'].copy(),
                nk=[0, 0]
            )

            model.set_free_params(
                B_free=None,
                F_free=None,
                C_free=config['C_free'],
                D_free=config['D_free']
            )

            # Fit
            fitted = model.fit(method='LS', verbose=0)

            # Check for explosion (coefficients too large)
            max_coef = max(
                np.max(np.abs(fitted.C)),
                np.max(np.abs(fitted.D)),
                max(np.max(np.abs(b)) for b in fitted.B),
                max(np.max(np.abs(f)) for f in fitted.F)
            )

            # Whiteness test
            resid = fitted.resid
            n = len(resid)
            nlags = min(25, n // 5)

            # Simple Ljung-Box calculation
            acf_vals = np.correlate(resid - np.mean(resid), resid - np.mean(resid), mode='full')
            acf_vals = acf_vals[n-1:] / acf_vals[n-1]
            Q = n * (n + 2) * np.sum(acf_vals[1:nlags+1]**2 / (n - np.arange(1, nlags+1)))
            lb_threshold = 37.65  # chi2(25) at 5%
            lb_pass = Q < lb_threshold

            result = {
                'model_id': config['model_id'],
                'B1': config['B1'],
                'F1': config['F1'],
                'B2': config['B2'],
                'F2': config['F2'],
                'noise_desc': config['noise_desc'],
                'C_order': len(config['C_init']) - 1,
                'D_order': len(config['D_init']) - 1,
                'MSE': float(fitted.MSE),
                'AIC': float(fitted.AIC),
                'BIC': float(fitted.BIC),
                'R2': float(fitted.scores.get('R2', np.nan)),
                'FitPercent': float(fitted.scores.get('FitPercent', np.nan)),
                'LjungBox_Q': float(Q),
                'LjungBox_pass': bool(lb_pass),
                'max_coef': float(max_coef),
                'stable': max_coef < 10,
                'converged': True,
                'error': None
            }

            print(f"  MSE: {result['MSE']:.2f}, AIC: {result['AIC']:.2f}, "
                  f"R2: {result['R2']:.4f}, LB_pass: {result['LjungBox_pass']}, "
                  f"max_coef: {result['max_coef']:.2f}")

        except Exception as e:
            result = {
                'model_id': config['model_id'],
                'B1': config['B1'],
                'F1': config['F1'],
                'B2': config['B2'],
                'F2': config['F2'],
                'noise_desc': config['noise_desc'],
                'C_order': len(config['C_init']) - 1,
                'D_order': len(config['D_init']) - 1,
                'MSE': np.nan,
                'AIC': np.nan,
                'BIC': np.nan,
                'R2': np.nan,
                'FitPercent': np.nan,
                'LjungBox_Q': np.nan,
                'LjungBox_pass': False,
                'max_coef': np.nan,
                'stable': False,
                'converged': False,
                'error': str(e)
            }
            print(f"  FAILED: {e}")

        results.append(result)


    #%% SAVE RESULTS TO JSON (includes full configurations for easy loading)

    output_file = 'grid_search_results.json'

    # Convert all configs and results to JSON-serializable types
    serializable_configs = convert_to_serializable(all_configs)
    serializable_results = convert_to_serializable(results)

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_models': len(results),
            'configs': serializable_configs,  # Save full configurations for model recreation
            'results': serializable_results
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Configs and results saved - use load_grid_search_results() to load")


    #%% SUMMARY TABLE

    df_results = pd.DataFrame(results)

    # Sort by AIC (lower is better)
    df_sorted = df_results[df_results['converged']].sort_values('AIC')

    print("\n" + "="*80)
    print("TOP 10 MODELS BY AIC (converged only)")
    print("="*80)
    print(df_sorted[['model_id', 'noise_desc', 'MSE', 'AIC', 'R2', 'LjungBox_pass', 'stable']].head(10).to_string())

    # Best stable model
    stable_models = df_sorted[df_sorted['stable']]
    if len(stable_models) > 0:
        print("\n" + "="*80)
        print("BEST STABLE MODEL")
        print("="*80)
        best = stable_models.iloc[0]
        print(f"Model ID: {best['model_id']}")
        print(f"Noise: {best['noise_desc']}")
        print(f"B1: {best['B1']}, F1: {best['F1']}")
        print(f"B2: {best['B2']}, F2: {best['F2']}")
        print(f"MSE: {best['MSE']:.2f}, AIC: {best['AIC']:.2f}, R2: {best['R2']:.4f}")
        print(f"Ljung-Box pass: {best['LjungBox_pass']}")

    # Models that pass whiteness test
    white_models = df_sorted[df_sorted['LjungBox_pass'] & df_sorted['stable']]
    if len(white_models) > 0:
        print("\n" + "="*80)
        print(f"MODELS WITH WHITE RESIDUALS ({len(white_models)} found)")
        print("="*80)
        print(white_models[['model_id', 'noise_desc', 'MSE', 'AIC', 'R2']].to_string())
    else:
        print("\n⚠️  No models with white residuals found. May need more complex noise structure.")


    #%% ========== EXAMPLE: LOAD AND TEST A SPECIFIC MODEL ==========
    # This shows how to load a model by ID from the saved results

    def test_model_by_id(model_id, y, x_multi, configs):
        """
        Load, fit, and display results for a specific model ID.

        Usage in another script:
        ------------------------
        from projectB_gridsearch import load_grid_search_results, get_model_config, build_model_from_config

        # Load results
        data = load_grid_search_results('grid_search_results.json')
        configs = data['configs']

        # Get and build model 80
        config = get_model_config(80, configs)
        fitted = build_model_from_config(config, y, x_multi)
        fitted.summary()
        """
        config = get_model_config(model_id, configs)
        print_model_config(config)

        fitted = build_model_from_config(config, y, x_multi)
        fitted.summary()

        # Plot residuals
        plotACFnPACF(fitted.resid, titleStr=f"Model {model_id} Residuals", noLags=100)
        whiteness_test(fitted.resid)

        return fitted, config


    # Example: test model 80 (best by FitPercent)
    # Uncomment to run:
    fitted_model, config = test_model_by_id(80, y, x_multi, all_configs)


#%% ========== QUICK REFERENCE: How to use in projectB.py ==========
"""
# Add this to the top of projectB.py to import the helper functions:

from projectB_gridsearch import (
    load_grid_search_results,
    get_model_config,
    build_model_from_config,
    print_model_config
)

# Then load and use any model like this:

# 1. Load the saved grid search results
data = load_grid_search_results('grid_search_results.json')
configs = data['configs']
results_df = pd.DataFrame(data['results'])

# 2. Find the model you want (e.g., best by FitPercent)
best_id = results_df.sort_values('FitPercent', ascending=False).iloc[0]['model_id']
print(f"Best model by FitPercent: {best_id}")

# 3. Get the configuration and build the model
config = get_model_config(best_id, configs)
print_model_config(config)

# 4. Fit the model
foundModel = build_model_from_config(config, y, x_multi)
foundModel.summary()

# 5. Use for predictions...
"""