import numpy as np
from scipy import signal
from tsa_lth.modelling import polydiv


def solutionA(payload):
    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])

    # --- fixed BJ model ---
    B = np.array([-1.2241, 1.0474])
    F = np.array([1.0, -0.9589])

    C = np.zeros(24)
    C[0]  = 1.0
    C[1]  = 0.3946
    C[2]  = 0.1956
    C[12] = 0.0514
    C[13] = 0.0610
    C[23] = 0.2649

    D = np.zeros(27)
    D[0]  = 1.0
    D[1]  = -0.7937
    D[24] = -0.3655
    D[25] = 0.0596
    D[26] = 0.0990

    y = data[:, 1]
    x = data[:, 2]
    test_idx = np.arange(start_idx, end_idx)

    # --- input ARMA model for x: A(z)x = C(z)e ---
    Ax = np.zeros(25)
    Ax[0]  = 1.0
    Ax[1]  = -1.5098
    Ax[2]  = 0.5622
    Ax[23] = -0.1898
    Ax[24] = 0.1389

    Cx = np.zeros(25)
    Cx[0]  = 1.0
    Cx[22] = 0.0772

    # k-step predicted input series (same length as x)
    Fx_k, Gx_k = polydiv(Cx, Ax, k)
    xhatk = signal.lfilter(Gx_k, Cx, x)

    # --- BJ k-step predictor ---
    A_eq = np.convolve(F, D)
    B_eq = np.convolve(D, B)
    C_eq = np.convolve(F, C)

    F_k, G_k = polydiv(C_eq, A_eq, k)
    Fhat, Ghat = polydiv(np.convolve(F_k, B_eq), C_eq, k)

    yhat = (
        signal.lfilter(Fhat, [1], xhatk) +
        signal.lfilter(Ghat, C_eq, x) +
        signal.lfilter(G_k, C_eq, y)
    )

    return yhat[test_idx].tolist()


def solutionB(payload):
    import numpy as np
    from scipy import signal
    from tsa_lth.modelling import polydiv

    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])

    # ================= Fixed BJ model (Part B) =================
    # Input 1
    B1 = np.array([-1.2789, -0.5303, -0.1148, 0.1033])
    F1 = np.array([1.0])

    # Input 2
    B2 = np.array([3.0642])
    F2 = np.array([1.0])

    # Noise model (max lag 169 -> length 170)
    C = np.zeros(170)
    C[0]  = 1.0
    C[1]  = 0.4230
    C[2]  = 0.1869
    C[23] = 0.1065

    D = np.zeros(170)
    D[0]   = 1.0
    D[1]   = -0.7819
    D[24]  = -0.1590
    D[72]  = -0.0310
    D[168] = -0.4266
    D[169] = 0.4003
    # ===========================================================

    y  = data[:, 1]
    x1 = data[:, 2]
    x2 = data[:, 3]
    test_idx = np.arange(start_idx, end_idx)

    # Predict input 1
    nabla = np.array([1.0, -1.0])
    C1 = np.array([1.0, -1.5755, 0.3283, 0.3343])
    A1 = np.zeros(25)
    A1[0]  = 1.0
    A1[1]  = -2.0167
    A1[2]  = 1.1923
    A1[3]  = -0.0915
    A1[13] = 0.0378
    A1[24] = 0.0166
    A1 = np.convolve(nabla, A1)
    Fx1, Gx1 = polydiv(C1, A1, k)
    xhatk1 = signal.lfilter(Gx1, C1, x1)

    # Predict input 2
    A2 = np.zeros(25)
    A2[0]  = 1.0
    A2[1]  = -1.8676
    A2[2]  = 1.4929
    A2[3]  = -0.5771
    A2[9]  = -0.0356
    A2[24] = -0.0126
    C2 = np.array([1.0, -0.9854, 0.5295])
    Fx2, Gx2 = polydiv(C2, A2, k)
    xhatk2 = signal.lfilter(Gx2, C2, x2)

    # --- Equivalent polynomials ---
    A_eq  = np.convolve(D, np.convolve(F1, F2))
    B1_eq = np.convolve(D, np.convolve(B1, F2))
    B2_eq = np.convolve(D, np.convolve(B2, F1))
    C_eq  = np.convolve(C, np.convolve(F1, F2))

    # --- k-step predictor ---
    Fk, Gk   = polydiv(C_eq, A_eq, k)
    Fh1, Gh1 = polydiv(np.convolve(Fk, B1_eq), C_eq, k)
    Fh2, Gh2 = polydiv(np.convolve(Fk, B2_eq), C_eq, k)

    yhat = (
        signal.lfilter(Fh1, [1], xhatk1) +
        signal.lfilter(Gh1, C_eq, x1) +
        signal.lfilter(Fh2, [1], xhatk2) +
        signal.lfilter(Gh2, C_eq, x2) +
        signal.lfilter(Gk,  C_eq, y)
    )

    return yhat[test_idx].tolist()


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




def precompute_input_preds(x1, x2, k):
    # --- x1 ---
    nabla = np.array([1.0, -1.0])
    C1 = np.array([1.0, -1.5755, 0.3283, 0.3343])
    A1 = np.zeros(25)
    A1[0]=1.0; A1[1]=-2.0167; A1[2]=1.1923; A1[3]=-0.0915; A1[13]=0.0378; A1[24]=0.0166
    A1 = np.convolve(nabla, A1)
    Fx1, Gx1 = polydiv(C1, A1, k)
    xhatk1 = signal.lfilter(Gx1, C1, x1)  # ≈ x1[t+k|t] stored at t

    # --- x2 ---
    A2 = np.zeros(25)
    A2[0]=1.0; A2[1]=-1.8676; A2[2]=1.4929; A2[3]=-0.5771; A2[9]=-0.0356; A2[24]=-0.0126
    C2 = np.array([1.0, -0.9854, 0.5295])
    Fx2, Gx2 = polydiv(C2, A2, k)
    xhatk2 = signal.lfilter(Gx2, C2, x2)

    return xhatk1, xhatk2




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

    # Predictions on inputs
    xhatk1, xhatk2 = precompute_input_preds(x1, x2, k)


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

        # ===== k-step prediction (prof-style) =====
        # store predicted future y; but y[t] is known now
        y_pred = {t: y[t]}   # IMPORTANT: observed y[t], not yhat_1

        Rx_k = Rx_t1.copy()
        for k0 in range(1, k + 1):

            Ck_vals = []
            for var, lag in desc:
                idx = t + k0 - lag

                if var == "y":
                    if idx <= t:
                        Ck_vals.append(-y[idx])
                    else:
                        Ck_vals.append(-y_pred[idx])

                elif var == "x1":
                    if idx <= t:
                        Ck_vals.append(x1[idx])
                    else:
                        Ck_vals.append(xhatk1[idx-k])

                elif var == "x2":
                    if idx <= t:
                        Ck_vals.append(x2[idx])
                    else:
                        Ck_vals.append(xhatk2[idx-k])

                elif var == "e":
                    if idx <= t:
                        Ck_vals.append(h_et[idx])
                    else:
                        Ck_vals.append(0.0)     # future noise = 0

            Ck = np.array(Ck_vals)[None, :]
            Ak = np.linalg.matrix_power(A, k0)
            yk = (Ck @ Ak @ xt[:, t])[0]
            y_pred[t + k0] = yk

            Rx_k = A @ Rx_k @ A.T + Re

        yhat_k[t + k] = y_pred[t + k]

    return yhat_k, h_et, xStd



def solutionC(payload):

    # Extract Data
    data = np.asarray(payload["data"])
    k = int(payload["k_steps"])
    start_idx = int(payload["start_idx"]) - 1
    end_idx = int(payload["end_idx"])
    test_idx = np.arange(start_idx, end_idx)

    y = data[:, 1]
    x1 = data[:, 2]
    x2 = data[:, 3]

    # Frozen
    desc = [
        ("y", 1),
        ("y", 24),
        ("y", 168),
        ("x1", 0),
        ("x1", 1),
        ("x2", 0),
        ("e", 1),
        ("e", 2),
    ]

    KB1 = np.array([-1.27889938, -0.5303092])
    KB2 = np.array([3.06419525])
    KC = np.array([1, 0.42302563, 0.18689207])
    KA = np.array([1, -0.78187552] + [0]*22 + [-0.15897257] + [0]*47 + [0] + [0]*95 + [-0.42658113])

    # Kalman
    N = len(y)
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

    return yhat_k[test_idx]
