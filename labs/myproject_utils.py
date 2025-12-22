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
    plt.figure()
    plt.plot(df['date'], y, label='power data')

    for _, row in gap_rows.iterrows():
        print(f"Gap: {row['prev_date']} → {row['date']} ({row['delta_seconds']} s)")
        plt.axvline(row['date'], color='red', linestyle='--', alpha=0.6)

    plt.ylabel('Power (MJ/s)')
    plt.xlabel('Date')
    plt.title('Power Data with Detected Time Gaps')
    plt.legend()
    plt.show()


    # %% CLEAN UP DATA CONTINUED

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

    # Interpolate missing values in time
    df[['power_MJ_s', 'ambient_temp_C', 'supply_temp_C']] = (
        df[['power_MJ_s', 'ambient_temp_C', 'supply_temp_C']]
        .interpolate(method='time')
    )

    # Final regularity check
    time_diffs = df.index.to_series().diff().dt.total_seconds()
    assert time_diffs.dropna().eq(3600).all()
    print('Data has been cleaned!')


    # Add usefule columns
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
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



def plot_ccf(x, y, noLags):
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
    plt.title('Crosscorrelation between in- and output')
    plt.tight_layout()
    plt.show()

    return Cxy, lags


def test_model(df:pd.DataFrame, start_index, n_weeks, k, BJmodel, buffer=200, plot=True):
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
        ax.plot(dates, yhat_k, label='Prediction', alpha=0.7) 
        ax.plot(dates, y_naive, label='Naive', alpha=0.4)
        plt.xticks(rotation=30)
        ax.legend()



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












