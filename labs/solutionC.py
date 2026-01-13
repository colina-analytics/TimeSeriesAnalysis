import numpy as np
from scipy import signal
from tsa_lth.modelling import polydiv


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
        ("y", 72),
        ("y", 168),
        ("y", 169),
        ("x1", 0),
        ("x1", 1),
        ("x1", 2),
        ("x1", 3),
        ("x2", 0),
        ("e", 1),
        ("e", 2),
        ("e", 23),
    ]

    KB1 = np.array([-1.27889938, -0.5303092 , -0.11478266,  0.10328068])
    KB2 = np.array([3.06419525])
    KC = np.array([1, 0.42302563, 0.18689207] + [0]*20 + [0.10646353])
    KA = np.array([1, -0.78187552] + [0]*22 + [-0.15897257] + [0]*47 + [-0.03095736] + [0]*95 + [-0.42658113, 0.40030175])

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
