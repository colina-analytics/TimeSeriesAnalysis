import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import scipy.io as sio

from tsa_lth.analysis import naive_pred, plotACFnPACF
from tsa_lth.modelling import polydiv

def load_project_df() -> pd.DataFrame:
    mat_data = sio.loadmat('../data/projectData25.mat')
    data = mat_data['data']

    cols = [
        'obs_num', 'power_MJ_s', 'ambient_temp_C', 'supply_temp_C',
        'year', 'month', 'day', 'hour'
    ]

    return pd.DataFrame(data[:, :8], columns=cols)


def data_cleanup(df) -> pd.DataFrame:
     
    # Detect gaps
    y = df['power_MJ_s'].values
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    date_diffs = df['date'].diff().dt.total_seconds()

    gap_mask = date_diffs.ne(3600) & date_diffs.notna()
    gap_rows = df.loc[gap_mask, ['date']].copy()
    gap_rows['delta_seconds'] = date_diffs[gap_mask]
    gap_rows['prev_date'] = df['date'].shift(1)[gap_mask]

    print(gap_rows)

    # Plot with gaps highlighted
    plt.figure(figsize=[10,6])
    plt.plot(df['date'], y, label='Power Usage')

    for _, row in gap_rows.iterrows():
        print(f"Gap: {row['prev_date']} → {row['date']} ({row['delta_seconds']} s)")
        plt.axvline(row['date'], color='red', linestyle='--', alpha=0.6)

    plt.ylabel('Power (MJ/s)')
    plt.xlabel('Date')
    plt.title('Power Data with Detected Time Gaps')
    plt.legend()
    plt.show()

    ## We proceed by only taking before the huge gap (index 3755) for the part A of the project
    df: pd.DataFrame = df.iloc[:3755].reset_index(drop=True)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['time_diffs'] = df['date'].diff().dt.total_seconds()

    # Errors mask
    mask = df['time_diffs'] != 3600
    print(df[mask])

    # Resolve duplicate timestamps
    df = df.sort_values('date')
    df = df.groupby('date', as_index=False).mean(numeric_only=True)
    df['time_diffs'] = df['date'].diff().dt.total_seconds()

    # Errors mask
    mask = df['time_diffs'] != 3600
    print('---' * 50)
    print('Errors after removing duplicates:')
    print(df[mask])
    print('---' * 50)


    # Reindex to strict hourly grid
    df = df.set_index('date')

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h'
    )

    df = df.reindex(full_index)

    # Interpolate missing values in time (index must be DatetimeIndex)
    cols = ["power_MJ_s", "ambient_temp_C", "supply_temp_C"]
    df[cols] = df[cols].interpolate(method="time")

    # Final regularity check
    time_diffs = df.index.to_series().diff().dt.total_seconds()
    assert time_diffs.dropna().eq(3600).all()
    print('Data has been cleaned!')

    # Add usefule columns
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df["date"] = df.index
    df['log_power'] = np.log(df['power_MJ_s'])

    return df


def get_modeling_dataset(df: pd.DataFrame, start, n_weeks, plot=True):
    
    # Choosing period
    start = start
    end = start + 168*n_weeks - 1
    n = end - start + 1
    start_date = df.iloc[start]['date']
    end_date = df.iloc[end]['date']

    # Confirming dates
    print(f'Modeling data from index {start} to {end}, total length {n}')
    print(f'Model start_date: {start_date}, end_date: {end_date}')

    # Freezing input - output data to use
    x = df['ambient_temp_C'][start:end].values
    y = df['power_MJ_s'][start:end].values

    # Plotting
    if plot:
        plt.figure()
        plt.plot(df['date'], df['power_MJ_s'])
        plt.axvline(start_date, linestyle='--', color='red')
        plt.axvline(end_date, linestyle='--', color='red')

        # Plotting input - output
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_ylabel('Power')
        ax2 = ax.twinx()
        ax2.plot(x, color='orange')
        ax2.set_ylabel('Temperature')
        ax.set_title('Modeling dataset')
        fig.show()




    return np.array(x), np.array(y)


def simulate_data(n):
    from tsa_lth.modelling import simulate_model, simulateARMA

    x = simulateARMA(AR=[1, 0.7, 0.3, -0.2], MA=[1, 0.7, -0.3, 0.1], size=n)

    y = simulate_model(
        x=x,
        A=[1, -0.5],
        B=[0, 0.8],
        F=[1, -0.3],
        C=[1],
        D=[1],
        size=n
    )

    return x, y


def plot_ccf(x, y, noLags, titleStr='Crosscorrelation between in- and output'):
    Cxy = np.correlate(y - np.mean(y), x - np.mean(x), mode='full')
    Cxy = Cxy / (np.std(y) * np.std(x) * len(y))
    lags = np.arange(-noLags, noLags + 1)
    mid = len(Cxy) // 2
    Cxy = Cxy[mid - noLags:mid + noLags + 1]

    plt.figure()
    plt.stem(lags, Cxy)
    condInt = 2 / np.sqrt(len(y))
    plt.axhline(condInt, color='r', linestyle='--')
    plt.axhline(-condInt, color='r', linestyle='--')
    plt.xlabel('Lag')
    plt.ylabel('Amplitude')
    plt.title(titleStr)
    plt.tight_layout()
    plt.show()

    return Cxy, lags


def test_model_safe(df:pd.DataFrame, start_index, n_weeks, k, BJmodel, buffer=200, plot=True):
    startInd = start_index - buffer
    startIdn = max(startInd, 0)
    endInd = startInd + n_weeks*168 + buffer

    indexes = np.arange(startIdn, endInd, 1)

    assert 'power_MJ_s' in df.columns, f"y-column not found in df"
    assert 'ambient_temp_C' in df.columns, f"x-column not found in df"

    y = df['power_MJ_s'].iloc[indexes].to_numpy()
    x = df['ambient_temp_C'].iloc[indexes].to_numpy()
    dates = df['date'].iloc[indexes].to_numpy()


    # Create Polys
    B  = np.array(BJmodel.B)
    F  = np.array(BJmodel.F)   # A2
    C  = np.array(BJmodel.C)   # C1
    D  = np.array(BJmodel.D)   # A1

    A_eq = np.convolve(F, D)
    B_eq = np.convolve(D, B)
    C_eq = np.convolve(F, C)

    F_k, G_k = polydiv(C_eq, A_eq, k)
    Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)

    # Predict
    yhat_k = (
        signal.lfilter(Fhat, [1], x) +
        signal.lfilter(Ghat, C_eq, x) +
        signal.lfilter(G_k, C_eq, y)
    )
    rmv = max(len(G_k), len(Ghat))
    season = None if k == 1 else 24
    y_naive, _, _ = naive_pred(data=df['power_MJ_s'], test_data_ind=indexes, k=k, season_k=season)

    # Remove buffer
    yhat_k = yhat_k[buffer:]
    y_filtered = y[buffer:]
    y_naive = y_naive[buffer:]
    dates = dates[buffer:]

    # Variance Residual analysis
    ehat_k = y_filtered - yhat_k
    ehat_naive = y_filtered - y_naive
    print(f'Original variance: {np.var(y_filtered):.2f}')
    print(f'Residual variance: {np.var(ehat_k):.2f}')
    print(f'Naive residual variance: {np.var(ehat_naive):.2f}')

    # MSE
    mse_model = np.mean(ehat_k**2)
    mse_naive = np.mean(ehat_naive**2)
    print('-'*50)
    print(f'Model MSE: {mse_model:.2f}')
    print(f'Naive MSE: {mse_naive:.2f}')
    print(f'MSE ratio (model / naive): {mse_model / mse_naive:.3f}')
    print('-'*50)
    if mse_model < mse_naive:
        print('SUCCESS!!! Model MSE was less than Naive MSE.')
    else:
        print('FAIL!!! Model MSE was greater than naive MSE')


    # Plotting
    if plot:
        _, ax = plt.subplots(figsize=[10, 6]) 
        ax.plot(dates, y_filtered, label='Data', alpha=0.7) 
        ax.plot(dates, yhat_k, label=f'{k}-step Prediction', alpha=0.7) 
        ax.plot(dates, y_naive, label='Naive', alpha=0.4)
        plt.xticks(rotation=30)
        ax.legend()


    return mse_model < mse_naive


def test_model_no_input_pred(
    df: pd.DataFrame,
    start_index,
    n_weeks,
    k,
    BJmodel,
    buffer=200,
    plot=True,
    transform="none",          # "none" | "log"
    invert_for_plot=True,      # only affects plotting
):
    startInd = start_index - buffer
    startIdn = max(startInd, 0)
    endInd = startInd + n_weeks * 168 + buffer
    indexes = np.arange(startIdn, endInd, 1)

    y_raw = df["power_MJ_s"].iloc[indexes].to_numpy()
    x = df["ambient_temp_C"].iloc[indexes].to_numpy()
    dates = df["date"].iloc[indexes].to_numpy()

    # --- choose modeling space ---
    if transform == "log":
        if np.any(y_raw <= 0):
            raise ValueError("transform='log' but y has non-positive values.")
        y = np.log(y_raw)
    elif transform == "none":
        y = y_raw
    else:
        raise ValueError("transform must be 'none' or 'log'")

    # --- BJ k-step prediction (in modeling space) ---
    B = np.array(BJmodel.B)
    F = np.array(BJmodel.F)
    C = np.array(BJmodel.C)
    D = np.array(BJmodel.D)

    A_eq = np.convolve(F, D)
    B_eq = np.convolve(D, B)
    C_eq = np.convolve(F, C)

    F_k, G_k = polydiv(C_eq, A_eq, k)
    Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)

    yhat_k = (
        signal.lfilter(Fhat, [1], x) +
        signal.lfilter(Ghat, C_eq, x) +
        signal.lfilter(G_k, C_eq, y)
    )

    # --- naive baseline (same space, full-series indexing) ---
    season = None if k == 1 else 24

    y_full = df["power_MJ_s"].to_numpy()
    if transform == "log":
        if np.any(y_full <= 0):
            raise ValueError("transform='log' but y has non-positive values.")
        y_full = np.log(y_full)

    y_naive, _, _ = naive_pred(
        data=y_full,
        test_data_ind=indexes,
        k=k,
        season_k=season
    )


    # Remove buffer
    yhat_k = yhat_k[buffer:]
    y_filt = y[buffer:]
    y_naive = y_naive[buffer:]
    dates = dates[buffer:]

    # Metrics (in modeling space)
    ehat_k = y_filt - yhat_k
    ehat_naive = y_filt - y_naive

    # print(f"Original variance: {np.var(y_filt):.2f}")
    # print(f"Residual variance: {np.var(ehat_k):.2f}")
    # print(f"Naive residual variance: {np.var(ehat_naive):.2f}")

    mse_model = np.mean(ehat_k**2)
    mse_naive = np.mean(ehat_naive**2)
    print("-"*50)
    print(f"Model MSE: {mse_model:.4f}")
    print(f"Naive MSE: {mse_naive:.4f}")
    print(f"MSE ratio (model / naive): {mse_model / mse_naive:.4f}")
    # print("-"*50)
    print("SUCCESS!!!" if mse_model < mse_naive else "FAIL!!!")

    # --- Plot (optionally inverted back to original units) ---
    if plot:
        if transform == "log" and invert_for_plot:
            y_plot = np.exp(y_filt)
            yhat_plot = np.exp(yhat_k)
            ynaive_plot = np.exp(y_naive)
            ylab = "Power (MJ/s)"
        else:
            y_plot, yhat_plot, ynaive_plot = y_filt, yhat_k, y_naive
            ylab = "log(Power)" if transform == "log" else "Power (MJ/s)"

        _, ax = plt.subplots(figsize=[10, 6])
        ax.plot(dates, y_plot, label="Data", alpha=0.7)
        ax.plot(dates, yhat_plot, label=f"{k}-step Prediction", alpha=0.7)
        # ax.plot(dates, ynaive_plot, label="Naive", alpha=0.4)
        ax.set_ylabel(ylab)
        plt.xticks(rotation=30)
        ax.legend()

    return mse_model < mse_naive


def test_model(
    df: pd.DataFrame,
    start_index,
    n_weeks,
    k,
    BJmodel,
    inputModel,
    buffer=200,
    plot=True,
    transform="none",          # "none" | "log"
    invert_for_plot=True,      # only affects plotting
):
    startInd = start_index - buffer
    startIdn = max(startInd, 0)
    endInd = startInd + n_weeks * 168 + buffer
    indexes = np.arange(startIdn, endInd, 1)

    y_raw = df["power_MJ_s"].iloc[indexes].to_numpy()
    x = df["ambient_temp_C"].iloc[indexes].to_numpy()
    dates = df["date"].iloc[indexes].to_numpy()

    # --- choose modeling space ---
    if transform == "log":
        if np.any(y_raw <= 0):
            raise ValueError("transform='log' but y has non-positive values.")
        y = np.log(y_raw)
    elif transform == "none":
        y = y_raw
    else:
        raise ValueError("transform must be 'none' or 'log'")
    
    # --- Predicting input ---
    Fx, Gx = polydiv(inputModel.C, inputModel.A, k)
    xhatk = signal.lfilter(Gx, inputModel.C, x)

    # --- BJ k-step prediction (in modeling space) ---
    B = np.array(BJmodel.B)
    F = np.array(BJmodel.F)
    C = np.array(BJmodel.C)
    D = np.array(BJmodel.D)

    A_eq = np.convolve(F, D)
    B_eq = np.convolve(D, B)
    C_eq = np.convolve(F, C)

    F_k, G_k = polydiv(C_eq, A_eq, k)
    Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)

    yhat_k = (
        signal.lfilter(Fhat, [1], xhatk) +
        signal.lfilter(Ghat, C_eq, x) +
        signal.lfilter(G_k, C_eq, y)
    )

    # --- naive baseline (same space, full-series indexing) ---
    season = None if k == 1 else 24

    y_full = df["power_MJ_s"].to_numpy()
    if transform == "log":
        if np.any(y_full <= 0):
            raise ValueError("transform='log' but y has non-positive values.")
        y_full = np.log(y_full)

    y_naive, _, ehat_naive = naive_pred(
        data=y_full,
        test_data_ind=indexes,
        k=k,
        season_k=season
    )


    # Remove buffer
    yhat_k = yhat_k[buffer:]
    y_filt = y[buffer:]
    y_naive = y_naive[buffer:]
    dates = dates[buffer:]

    # Metrics (in modeling space)
    # print(f"Original variance: {np.var(y_filt):.2f}")
    # print(f"Residual variance: {np.var(ehat_k):.2f}")
    # print(f"Naive residual variance: {np.var(ehat_naive):.2f}")

    ehat_k = y_filt - yhat_k
    mse_model = np.mean(ehat_k**2)
    mse_naive = np.mean(ehat_naive**2)
    print("-"*50)
    print(f"Model MSE: {mse_model:.4f}")
    print(f"Naive MSE: {mse_naive:.4f}")
    # print(f"MSE ratio (model / naive): {mse_model / mse_naive:.4f}")
    # print("-"*50)
    print("SUCCESS!!!" if mse_model < mse_naive else "FAIL!!!")

    # --- Plot (optionally inverted back to original units) ---
    if plot:
        if transform == "log" and invert_for_plot:
            y_plot = np.exp(y_filt)
            yhat_plot = np.exp(yhat_k)
            ynaive_plot = np.exp(y_naive)
            ylab = "Power (MJ/s)"
        else:
            y_plot, yhat_plot, ynaive_plot = y_filt, yhat_k, y_naive
            ylab = "log(Power)" if transform == "log" else "Power (MJ/s)"

        _, ax = plt.subplots(figsize=[10, 6])
        ax.plot(dates, y_plot, label="Data", alpha=0.7)
        ax.plot(dates, yhat_plot, label=f"{k}-step Prediction", alpha=0.7)
        ax.plot(dates, ynaive_plot, label="Naive", alpha=0.2)
        ax.set_ylabel(ylab)
        plt.xticks(rotation=30)
        ax.legend()

    return mse_model < mse_naive


from statsmodels.tsa.stattools import adfuller

def test_stationarity(series, name=""):
    result = adfuller(series, autolag='AIC')
    print(f'{name} - ADF Statistic: {result[0]:.4f}')
    print(f'{name} - p-value: {result[1]:.4f}')
    print(f'{name} - Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print(f'    → {name} is STATIONARY (reject null hypothesis)')
    else:
        print(f'    → {name} is NON-STATIONARY (fail to reject null hypothesis)')
    
    return result[1]  # Return p-value


def predict_model(foundModel, inputModel1, inputModel2,  x1, x2, y, indexes, k,  buffer=200):

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

    # ---------------- NAIVE BASELINE (aligned to your buffered local y) ----------------
    season = None if k == 1 else 24
    y_naive, _, ehat_naive = naive_pred(
        data=y,
        test_data_ind=np.arange(0, len(y)),
        k=k,
        season_k=season
    )

    # Align
    # bias = np.mean(y) - np.mean(yhatk)
    # yhatk += bias
    rmv = buffer
    yhatk = yhatk[rmv:]
    y_filt = y[rmv:]
    y_naive = y_naive[rmv:]
    ehat_naive = ehat_naive[rmv:]

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
    # fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # axes[0].plot(x1[buffer:], label='x_1(t)')
    # axes[0].plot(xhatk1[buffer:], label='Predicted data')
    # axes[0].set_title(f'{k}-step predictions of x_1(t)')
    # axes[0].legend(loc='upper left')

    # axes[1].plot(x2[buffer:], label='x_2(t)')
    # axes[1].plot(xhatk2[buffer:], label='Predicted data')
    # axes[1].set_title(f'{k}-step predictions of x_2(t)')
    # axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(y_filt, label='y(t)')
    plt.plot(yhatk, label='Predicted data')
    # plt.plot(y_naive, label='Naive')
    plt.title(f'{k}-step predictions of y(t)')
    plt.legend(loc='upper left')
    plt.show()


def test_double_input_model(
    df: pd.DataFrame,
    foundModel,
    inputModel1,
    inputModel2,
    k,
    start_index,
    end_index,
    buffer=200,
):

    # Extract data from df
    s = start_index - buffer
    e = end_index
    y = df['power_MJ_s'][s:e].values
    x1 = df["ambient_temp_C"][s:e].values
    x2 = df["supply_temp_C"][s:e].values
    dates = df["date"][s:e].values

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

    # Predict naive
    season = None if k == 1 else 24
    y_naive, var_naive, ehat_naive = naive_pred(data=y, test_data_ind=np.arange(0, len(y)), k=k, season_k=season)

    # Remove buffer
    xhatk1 = xhatk1[buffer:]
    xhatk2 = xhatk2[buffer:]
    yhatk = yhatk[buffer:]
    y_naive = y_naive[buffer:]
    ehat_naive = ehat_naive[buffer:]
    y = y[buffer:]

    # Compare MSE
    ehatk = y - yhatk
    naive_mse = np.mean(ehat_naive**2)
    model_mse = np.mean(ehatk**2)
    print('naive_mse:', round(naive_mse, 3))
    print('model_mse:', round(model_mse, 3))
    print('SUCCESS!' if model_mse < naive_mse else 'FAIL!')

    return y, yhatk, dates[buffer:]



def test_single_input_model(
    df: pd.DataFrame,
    foundModel,
    inputModel,
    k,
    start_index,
    end_index,
    buffer=200,
):

    # Extract data from df
    s = start_index - buffer
    e = end_index
    y = df['power_MJ_s'][s:e].values
    x = df["ambient_temp_C"][s:e].values
    dates = df["date"][s:e].values

    # Predict Input
    Fx, Gx = polydiv(inputModel.C, inputModel.A, k)
    xhatk = signal.lfilter(Gx, inputModel.C, x)

    # --- BJ k-step prediction (in modeling space) ---
    B = np.array(foundModel.B)
    F = np.array(foundModel.F)
    C = np.array(foundModel.C)
    D = np.array(foundModel.D)

    A_eq = np.convolve(F, D)
    B_eq = np.convolve(D, B)
    C_eq = np.convolve(F, C)

    F_k, G_k = polydiv(C_eq, A_eq, k)
    Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)

    yhatk = (
        signal.lfilter(Fhat, [1], xhatk) +
        signal.lfilter(Ghat, C_eq, x) +
        signal.lfilter(G_k, C_eq, y)
    )

    # Predict naive
    season = None if k == 1 else 24
    y_naive, var_naive, ehat_naive = naive_pred(data=y, test_data_ind=np.arange(0, len(y)), k=k, season_k=season)

    # Remove buffer
    xhatk = xhatk[buffer:]
    yhatk = yhatk[buffer:]
    y_naive = y_naive[buffer:]
    ehat_naive = ehat_naive[buffer:]
    y = y[buffer:]
    dates = dates[buffer:]

    # Compare MSE
    ehatk = y - yhatk
    naive_mse = np.mean(ehat_naive**2)
    model_mse = np.mean(ehatk**2)
    print('naive_mse:', round(naive_mse, 3))
    print('model_mse:', round(model_mse, 3))
    print('SUCCESS!' if model_mse < naive_mse else 'FAIL!')

    return y, yhatk, dates
