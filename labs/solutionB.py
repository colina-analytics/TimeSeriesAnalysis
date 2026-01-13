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
