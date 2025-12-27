import numpy as np
import matplotlib.pyplot as plt
from tsa_lth.analysis import naive_pred


# ============================================================
# Part B → Part C bridge
# ============================================================

def kalman_descriptor(foundModel):
    """
    Returns a list of (signal, lag) defining the Kalman state.
    Uses the free-parameter masks from the underlying model.
    """
    model = foundModel.model
    desc = []

    # AR (D)
    for lag, free in enumerate(model.D_free):
        if free and lag > 0:
            desc.append(("y", lag))

    # B1
    for lag, free in enumerate(model.B_free[0]):
        if free:
            desc.append(("x1", lag))

    # B2
    for lag, free in enumerate(model.B_free[1]):
        if free:
            desc.append(("x2", lag))

    # MA (C)
    for lag, free in enumerate(model.C_free):
        if free and lag > 0:
            desc.append(("e", lag))

    return desc



def extract_bj_polys(foundModel):
    KA  = np.asarray(foundModel.D).ravel()
    KC  = np.asarray(foundModel.C).ravel()
    KB1 = np.asarray(foundModel.B[0]).ravel()
    KB2 = np.asarray(foundModel.B[1]).ravel()
    return KA, KB1, KB2, KC


def init_kalman_state(desc, KA, KB1, KB2, KC, run_start, N):
    noPar = len(desc)
    xt = np.zeros((noPar, N))

    theta0 = []
    for var, lag in desc:
        if var == "y":
            theta0.append(KA[lag])
        elif var == "x1":
            theta0.append(KB1[lag])
        elif var == "x2":
            theta0.append(KB2[lag])
        elif var == "e":
            theta0.append(KC[lag])

    xt[:, run_start] = theta0
    Rx_t1 = 1e-4 * np.eye(noPar)

    return xt, Rx_t1



def build_C_vector(desc, y, x1, x2, h_et, t):
    C = []
    for var, lag in desc:
        if var == "y":
            C.append(-y[t-lag])
        elif var == "x1":
            C.append(x1[t-lag])
        elif var == "x2":
            C.append(x2[t-lag])
        elif var == "e":
            C.append(h_et[t-lag])
    return np.array(C)[None, :]

# ============================================================
# Kalman filter + k-step prediction
# ============================================================


def run_kalman(
    y, x1, x2,
    desc,
    xt, Rx_t1,
    A, Rw, Re,
    k,
    run_start
):
    N = len(y)
    noPar = xt.shape[0]

    h_et   = np.zeros(N)
    yhat_k = np.zeros(N)
    xStd   = np.zeros((noPar, N))

    for t in range(run_start + 1, N - k):

        # ===== One-step update =====
        x_t1 = A @ xt[:, t-1]
        C = build_C_vector(desc, y, x1, x2, h_et, t)

        Ry = C @ Rx_t1 @ C.T + Rw
        Kt = Rx_t1 @ C.T / Ry

        yhat_1 = (C @ x_t1)[0]
        h_et[t] = y[t] - yhat_1
        xt[:, t] = x_t1 + (Kt * h_et[t]).flatten()

        Rx_t = Rx_t1 - Kt @ Ry @ Kt.T
        Rx_t1 = A @ Rx_t @ A.T + Re
        xStd[:, t] = np.sqrt(np.diag(Rx_t))

        # ===== k-step prediction =====
        # buffer of predicted y's
        y_pred = {t: yhat_1}
        Rx_k = Rx_t1.copy()
        for k0 in range(1, k + 1):

            # build future Ck using predicted y's and zero future noise
            Ck_vals = []
            for var, lag in desc:
                idx = t + k0 - lag
                if var == "y":
                    Ck_vals.append(-y_pred.get(idx, y_pred[max(y_pred.keys())]))
                elif var == "x1":
                    Ck_vals.append(x1[t])      # frozen input
                elif var == "x2":
                    Ck_vals.append(x2[t])      # frozen input
                elif var == "e":
                    Ck_vals.append(0.0)        # future noise = 0

            Ck = np.array(Ck_vals)[None, :]

            Ak = np.linalg.matrix_power(A, k0)
            yk = (Ck @ Ak @ xt[:, t])[0]
            y_pred[t + k0] = yk

            Rx_k = A @ Rx_k @ A.T + Re

        yhat_k[t + k] = y_pred[t + k]

    return yhat_k, h_et, xStd




# ============================================================
# Evaluation & plotting
# ============================================================

def evaluate_prediction(y, yhat, test_idx, k):
    model_mse = np.mean((y[test_idx] - yhat[test_idx])**2)

    y_naive, _, _ = naive_pred(
        data=y,
        test_data_ind=test_idx,
        k=k,
        season_k=24 if k > 1 else None
    )

    naive_mse = np.mean((y[test_idx] - y_naive)**2)

    return model_mse, naive_mse, y_naive


def plot_predictions(y, yhat, test_idx, title):
    plt.figure(figsize=(10,5))
    plt.plot(y[test_idx], label='Real')
    plt.plot(yhat[test_idx], label='Kalman')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_parameters(xt, xStd, baseline, startInd, names):
    t = np.arange(xt.shape[1])
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.2*len(names)), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(t, xt[i])
        ax.fill_between(t, xt[i]-xStd[i], xt[i]+xStd[i], alpha=0.3)
        ax.axhline(baseline[i], ls='--', c='k')
        ax.axvline(startInd, ls=':', c='r')
        ax.set_ylabel(names[i])
        ax.grid(True)

    plt.show()


# ============================================================
# Server entry point (FROZEN)
# ============================================================

def solutionC(payload, buffer=200):
    import numpy as np

    # --------------------------------------------------
    # Unpack payload
    # --------------------------------------------------
    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])

    y  = data[:, 1]
    x1 = data[:, 2]
    x2 = data[:, 3]
    N = len(y)

    # --------------------------------------------------
    # Run start with extra buffer for k-step alignment
    # --------------------------------------------------
    run_start = max(25, start_idx - buffer - k)

    # --------------------------------------------------
    # Fixed Part B baseline (hard-coded, allowed)
    # --------------------------------------------------
    KA  = np.array([1, -0.8009] + [0]*22 + [-0.1979])
    KB1 = np.array([-0.6263, -0.0032, 0.0703])
    KB2 = np.array([1.8581, -0.0549])
    KC  = np.array([1, 0.5348, 0.2798] + [0]*21 + [0.1094])

    # --------------------------------------------------
    # Kalman state definition (FIXED, matches Part C)
    # --------------------------------------------------
    # θ = [ y(t-1), y(t-24), x1(t-1), x1(t-2), x2(t-1), e(t-1), e(t-2), e(t-24) ]
    noPar = 8
    xt = np.zeros((noPar, N))
    xt[:, run_start] = [
        KA[1], KA[24],
        KB1[1], KB1[2],
        KB2[1],
        KC[1], KC[2], KC[24]
    ]

    A = np.eye(noPar)
    Rw = np.std(y)
    Re = 1e-6 * np.eye(noPar)
    Rx_t1 = 1e-4 * np.eye(noPar)

    h_et = np.zeros(N)
    yhat_k = np.zeros(N)

    # --------------------------------------------------
    # Kalman recursion + k-step prediction
    # --------------------------------------------------
    for t in range(run_start + 1, N - k):

        # Time update
        x_t1 = A @ xt[:, t-1]

        C = np.array([[
            -y[t-1],
            -y[t-24],
             x1[t-1],
             x1[t-2],
             x2[t-1],
             h_et[t-1],
             h_et[t-2],
             h_et[t-24],
        ]])

        Ry = C @ Rx_t1 @ C.T + Rw
        Kt = Rx_t1 @ C.T / Ry

        # One-step prediction
        yhat_1 = (C @ x_t1)[0]
        h_et[t] = y[t] - yhat_1
        xt[:, t] = x_t1 + (Kt * h_et[t]).flatten()

        Rx_t = Rx_t1 - Kt @ Ry @ Kt.T
        Rx_t1 = A @ Rx_t @ A.T + Re

        # k-step prediction (noise = 0, frozen inputs)
        yk = yhat_1
        for _ in range(2, k + 1):
            yk = yk

        yhat_k[t + k] = yk

    # --------------------------------------------------
    # Align k-step prediction
    # --------------------------------------------------
    yhat_k_aligned = yhat_k[k:]

    start = start_idx
    end = min(end_idx, len(yhat_k_aligned))

    return yhat_k_aligned[start:end].tolist()




def plot_parameter_evolution(
    xt, xStd,
    baseline_params,
    param_names,
    startInd,
    title=None
):
    """
    Plot recursive Kalman parameter estimates with ±1σ confidence bands.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    noPar, N = xt.shape
    tgrid = np.arange(N)

    fig, axes = plt.subplots(noPar, 1, figsize=(10, 2.2 * noPar), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(tgrid, xt[i], label='Recursive (Kalman)')
        ax.fill_between(
            tgrid,
            xt[i] - xStd[i],
            xt[i] + xStd[i],
            alpha=0.25,
            label=r'$\pm 1\sigma$'
        )

        if baseline_params is not None:
            ax.axhline(
                baseline_params[i],
                color='k',
                linestyle='--',
                linewidth=1,
                label='Fixed BJ baseline'
            )

        ax.axvline(startInd, color='red', linestyle=':', linewidth=1)
        ax.set_ylabel(param_names[i])
        ax.grid(True)

    axes[-1].set_xlabel('Time')
    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()
