#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Project B - Dual Input BJ Model
# Using both ambient air temperature (x1) and supply water temperature (x2) as inputs

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os
import importlib
import pandas as pd
import json
from datetime import datetime
from itertools import product

# Add path to tsa_lth library
sys.path.append(
    os.path.abspath(
        os.path.join(os.getcwd(), "..", "TimeSeriesAnalysis-main", "TimeSeriesAnalysis-main")
    )
)

# Import utils + tsa_lth
from myproject_utils import data_cleanup, load_project_df

import tsa_lth.analysis
import tsa_lth.modelling
import tsa_lth.tests

importlib.reload(tsa_lth.analysis)
importlib.reload(tsa_lth.modelling)
importlib.reload(tsa_lth.tests)

from tsa_lth.analysis import plotACFnPACF
from tsa_lth.modelling import MultiInputPEM
from tsa_lth.tests import whiteness_test, monti_test


#%% ========== HELPERS FOR JSON / MODEL REBUILD ==========

def convert_to_serializable(obj):
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


def load_grid_search_results(filepath='grid_search_results.json'):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_model_config(model_id, configs):
    for config in configs:
        if config['model_id'] == model_id:
            return config
    raise ValueError(f"Model ID {model_id} not found")


def print_model_config(config):
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


def build_model_from_config(config, y, x_multi):
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


#%% ========== MAIN ==========

if __name__ == "__main__":

    df = load_project_df()
    df = data_cleanup(df)

    #%% SELECT MODELING DATASET - Same as Part A
    start = 1500 - 4*168
    n_weeks = 10
    end = start + 168 * n_weeks - 1
    n = end - start + 1

    start_date = df.iloc[start]['date']
    end_date = df.iloc[end]['date']
    print(f'Modeling data from index {start} to {end}, total length {n}')
    print(f'Model start_date: {start_date}, end_date: {end_date}')

    x1 = df['ambient_temp_C'][start:end+1].values
    x2 = df['supply_temp_C'][start:end+1].values
    y  = df['power_MJ_s'][start:end+1].values

    print(f'x1 (ambient temp) shape: {x1.shape}')
    print(f'x2 (supply temp) shape: {x2.shape}')
    print(f'y  (power) shape: {y.shape}')

    # Prepare data
    x_multi = np.column_stack([x1, x2])

    #%% PARAMETER GRID

    B1_options = [
        [0, 1],              # delay=1, s=0
        [0, 1, 0.5],         # delay=1, s=1
        [0, 1, 0.5, 0.3],    # delay=1, s=2
    ]

    F1_options = [
        [1],                 # no dynamics
        [1, 0.5],            # AR(1)
    ]

    B2_options = [
        [1],                 # gain only
        [1, 0.5],            # immediate + lag
        [0, 1],              # delay=1
    ]

    F2_options = [
        [1],
    ]

    # (C_init, D_init, C_free, D_free, description)
    noise_configs = [
        ([1], [1], [False], [False], "No noise model"),
        ([1, 0.5], [1, -0.5], [False, True], [False, True], "ARMA(1,1)"),
        ([1, 0.5, 0.3], [1, -0.5], [False, True, True], [False, True], "ARMA(1,2)"),
        ([1, 0.5, 0.3], [1, -0.5, 0.3], [False, True, True], [False, True, True], "ARMA(2,2)"),

        ([1, 0.5, *[0]*22, 0.3], [1, -0.5],
         [False, True, *[False]*22, True], [False, True], "MA(1)+MA(24), AR(1)"),

        ([1, 0.5, *[0]*22, 0.3], [1, -0.5, *[0]*22, 0.3],
         [False, True, *[False]*22, True], [False, True, *[False]*22, True], "MA(1)+MA(24), AR(1)+AR(24)"),

        ([1, 0.5, 0.3, *[0]*21, 0.2], [1, -0.5],
         [False, True, True, *[False]*21, True], [False, True], "MA(2)+MA(24), AR(1)"),

        ([1, 0.5, 0.3, *[0]*21, 0.2], [1, -0.5, *[0]*22, 0.2],
         [False, True, True, *[False]*21, True], [False, True, *[False]*22, True], "MA(2)+MA(24), AR(1)+AR(24)"),

        ([1, 0.5, 0.3, *[0]*21, 0.2], [1, -0.5, 0.3, *[0]*21, 0.2],
         [False, True, True, *[False]*21, True], [False, True, True, *[False]*21, True], "MA(2)+MA(24), AR(2)+AR(24)"),
        
        
    ]
    
    noise_configs += [

    # AR(1) + AR(72) + AR(168)
    (
        [1],  # C_init
        [1, -0.5, *[0]*70, 0.2, *[0]*95, 0.2],  # D_init (lags 1,72,168)
        [False],  # C_free
        [False, True, *[False]*70, True, *[False]*95, True],  # D_free
        "AR(1)+AR(72)+AR(168)"
    ),

    # MA(1)+MA(24) + AR(1)+AR(72)+AR(168)
    (
        [1, 0.5, *[0]*22, 0.3],  # C_init (lags 1 and 24)
        [1, -0.5, *[0]*70, 0.2, *[0]*95, 0.2],  # D_init
        [False, True, *[False]*22, True],  # C_free
        [False, True, *[False]*70, True, *[False]*95, True],  # D_free
        "MA(1)+MA(24), AR(1)+AR(72)+AR(168)"
    ),

    # MA(2)+MA(24) + AR(2)+AR(72)+AR(168)
    (
        [1, 0.5, 0.3, *[0]*21, 0.2],  # C_init (lags 1,2,24)
        [1, -0.5, 0.3, *[0]*69, 0.2, *[0]*95, 0.2],  # D_init (lags 1,2,72,168)
        [False, True, True, *[False]*21, True],  # C_free
        [False, True, True, *[False]*69, True, *[False]*95, True],  # D_free
        "MA(2)+MA(24), AR(2)+AR(72)+AR(168)"
    ),
]
    
    noise_configs += [
    (
        # C(z) = 1 + c1 z^-1 + c2 z^-2 + c23 z^-23
        [1, 0.4, 0.2, *[0]*20, 0.2],
        # D(z) = 1 - a1 z^-1 - a24 z^-24 - a72 z^-72 - a168 z^-168 + a169 z^-169
        [1,
         -0.6,                    # z^-1
         *[0]*22,
         -0.15,                   # z^-24
         *[0]*47,
         -0.12,                   # z^-72
         *[0]*95,
         -0.27,                   # z^-168
         0.21],                   # z^-169
        # C_free
        [False, True, True, *[False]*20, True],
        # D_free
        [False,
         True,
         *[False]*22,
         True,
         *[False]*47,
         True,
         *[False]*95,
         True,
         True],
        "MA(1,2,23) + AR(1,24,72,168,169)"
    )
]


    # Generate all combinations
    all_configs = []
    model_id = 0

    for B1, F1, B2, F2, (C_init, D_init, C_free, D_free, noise_desc) in product(
        B1_options, F1_options, B2_options, F2_options, noise_configs
    ):
        model_id += 1
        all_configs.append({
            'model_id': model_id,
            'B1': B1,
            'F1': F1,
            'B2': B2,
            'F2': F2,
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
        print(f"Testing Model {config['model_id']}: "
              f"B1={len(config['B1'])-1}, F1={len(config['F1'])-1}, "
              f"B2={len(config['B2'])-1}, F2={len(config['F2'])-1}, "
              f"Noise: {config['noise_desc']}")
        print('='*60)

        try:
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

            # Explosion check
            max_coef = max(
                np.max(np.abs(fitted.C)),
                np.max(np.abs(fitted.D)),
                max(np.max(np.abs(b)) for b in fitted.B),
                max(np.max(np.abs(f)) for f in fitted.F)
            )

            resid = fitted.resid
            n_res = len(resid)
            K = min(25, n_res // 5)

            # Monti test
            monti_pass, monti_Q, monti_chiV = monti_test(
                resid, K=K, alpha=0.05, return_val=True
            )

            mse = float(fitted.MSE)
            rmse = float(np.sqrt(mse))
            aic = float(fitted.AIC)
            fpe = float(getattr(fitted, "FPE", np.nan))

            result = {
                'model_id': config['model_id'],
                'B1': config['B1'],
                'F1': config['F1'],
                'B2': config['B2'],
                'F2': config['F2'],
                'noise_desc': config['noise_desc'],
                'C_order': len(config['C_init']) - 1,
                'D_order': len(config['D_init']) - 1,

                'MSE': mse,
                'RMSE': rmse,
                'AIC': aic,
                'FPE': fpe,

                'Monti_Q': float(monti_Q),
                'Monti_chiV': float(monti_chiV),
                'Monti_pass': bool(monti_pass),

                'R2': float(fitted.scores.get('R2', np.nan)),
                'FitPercent': float(fitted.scores.get('FitPercent', np.nan)),

                'max_coef': float(max_coef),
                'stable': bool(max_coef < 10),
                'converged': True,
                'error': None
            }

            print(f"  MSE: {result['MSE']:.4f}, RMSE: {result['RMSE']:.4f}, "
                  f"AIC: {result['AIC']:.2f}, FPE: {result['FPE']:.4f}, "
                  f"Monti_Q: {result['Monti_Q']:.2f}, Monti_pass: {result['Monti_pass']}, "
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
                'RMSE': np.nan,
                'AIC': np.nan,
                'FPE': np.nan,

                'Monti_Q': np.nan,
                'Monti_chiV': np.nan,
                'Monti_pass': False,

                'R2': np.nan,
                'FitPercent': np.nan,

                'max_coef': np.nan,
                'stable': False,
                'converged': False,
                'error': str(e)
            }
            print(f"  FAILED: {e}")

        results.append(result)

    #%% SAVE RESULTS TO JSON (configs + results)
    output_file = 'grid_search_results.json'
    serializable_configs = convert_to_serializable(all_configs)
    serializable_results = convert_to_serializable(results)

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_models': len(results),
            'configs': serializable_configs,
            'results': serializable_results
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    #%% SUMMARY TABLE (BEST BY LOWEST MSE)
    df_results = pd.DataFrame(results)
    df_sorted = df_results[df_results['converged']].sort_values('MSE')

    print("\n" + "="*80)
    print("TOP 10 MODELS BY MSE (converged only)")
    print("="*80)
    print(df_sorted[['model_id', 'noise_desc', 'MSE', 'RMSE', 'AIC', 'FPE',
                     'Monti_Q', 'Monti_pass', 'stable']].head(10).to_string())

    # Best stable model by MSE
    stable_models = df_sorted[df_sorted['stable']]
    if len(stable_models) > 0:
        print("\n" + "="*80)
        print("BEST STABLE MODEL (by MSE)")
        print("="*80)
        best = stable_models.iloc[0]
        print(f"Model ID: {best['model_id']}")
        print(f"Noise: {best['noise_desc']}")
        print(f"B1: {best['B1']}, F1: {best['F1']}")
        print(f"B2: {best['B2']}, F2: {best['F2']}")
        print(f"MSE: {best['MSE']:.4f}, RMSE: {best['RMSE']:.4f}, AIC: {best['AIC']:.2f}, FPE: {best['FPE']:.4f}")
        print(f"Monti_pass: {best['Monti_pass']} (Q={best['Monti_Q']:.2f}, chiV={best['Monti_chiV']:.2f})")

    # Models with white residuals (Monti) + stable
    white_models = df_sorted[df_sorted['Monti_pass'] & df_sorted['stable']]
    if len(white_models) > 0:
        print("\n" + "="*80)
        print(f"MODELS WITH WHITE RESIDUALS (Monti) AND STABLE ({len(white_models)} found)")
        print("="*80)
        print(white_models[['model_id', 'noise_desc', 'MSE', 'RMSE', 'AIC', 'FPE', 'Monti_Q']].to_string())
    else:
        print("\n⚠️  No stable models with Monti-white residuals found.")

    #%% OPTIONAL: TEST A SPECIFIC MODEL ID (example)
    def test_model_by_id(model_id, y, x_multi, configs):
        config = get_model_config(model_id, configs)
        print_model_config(config)

        fitted = build_model_from_config(config, y, x_multi)
        fitted.summary()

        plotACFnPACF(fitted.resid, titleStr=f"Model {model_id} Residuals", noLags=100)
        whiteness_test(fitted.resid)

        return fitted, config

    # Example: test model 80
    fitted_model, config = test_model_by_id(80, y, x_multi, all_configs)
