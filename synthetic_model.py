import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

# ============================================================
# MODELO DE SIMULAÇÃO SINTÉTICA
# ============================================================

def pulse_signal(t, pulse):
    return pulse["amp"] * (((t % pulse["period"]) / pulse["period"]) < pulse["duty"])

def simulate_model(params, pulse, init, fixed, seed=2026):
    rng = np.random.default_rng(seed)
    t_end = 5000
    dt = 1

    t_eval = np.arange(0, t_end + dt, dt)
    H, Df, Temp, x = [np.zeros_like(t_eval, dtype=float) for _ in range(4)]

    H[0], Df[0], Temp[0], x[0] = init["H0"], init["Df0"], init["T0"], init["x0"]

    for i in range(len(t_eval) - 1):
        P = pulse_signal(t_eval[i], pulse)
        noise = rng.normal(0, params["sigma_t"], 4)

        dH = (
            params["k_load"] * P * (1 - H[i] / fixed["H_max"])
            - params["k_rel"] * H[i]
            + noise[0]
        )

        dDf = (
            params["k_def"] * P * (H[i] / fixed["H_max"])
            - params["k_r"] * Df[i]
            + noise[1]
        )

        dTemp = pulse["amp"] + noise[2]
        dX = noise[-1]

        H[i + 1] = np.clip(H[i] + dH, 0, fixed["H_max"])
        Df[i + 1] = np.clip(Df[i] + dDf, 0, fixed["Df_max"])

    return dict(t=t_eval, H=H, Df=Df, TempC=Temp, x=x)

def export_results(data):
    pd.DataFrame(data).to_csv("synthetic_data.csv")
    print("Resultados foram exportados para synthetic_data.csv")