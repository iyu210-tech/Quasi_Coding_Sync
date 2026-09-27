"Slow independent checks (not used by production solvers). Characteristic matrices multipled, transmission is 1/W22. Report bunchmarks must agree at two precision levels"

from __future__ import annotations

import numpy as np
from fibonacci_qc import config as cfg

def high_precision_transport(
    widths_nm,
    potentials_ev,
    energy_ev,
    *,
    dps=100,
    v_lead_ev=0
):
    import mpmath as mp
    with mp.workdps(int(dps)):
        # Preserve actual binary input floats
        h = mp.mpf(float(cfg.HBAR2_OVER_2M_EV_NM2)); e = mp.mpf(float(energy_ev)); lead = mp.mpf(float(v_lead_ev))
        if e <= lead: raise ValueError("propagating exterior required")
        k = mp.sqrt((e - lead) / h); m = mp.eye(2)
        if len(widths_nm) != len(potentials_ev): raise ValueError("array length mismatch")
        for width,pot in zip(widths_nm,potentials_ev):
            d,v = mp.mpf(float(width)), mp.mpf(float(pot))
            if d < 0: raise ValueError("negative width")
            q = mp.sqrt((e - v) / h); z = q * d
            sinc = d if q == 0 else mp.sin(z) / q
            a = mp.matrix([[mp.cos(z), mp.j * sinc], [mp.j * q * mp.sin(z), mp.cos(z)]])
            m = a * m
        b = mp.matrix([[1, 1], [k, -k]]); w = b ** -1 * m * b
        amplitude = 1 / w[1, 1]
        log_t = -2 * mp.log(abs(w[1, 1])); t = mp.exp(log_t)
        s = mp.matrix([[-w[1, 0] / w[1, 1], amplitude], [amplitude, w[0, 1] / w[1, 1]]])
        return {
            "T": float(t),
            "log_T": float(log_t),
            "log_T_decimal": mp.nstr(log_t, int(dps)-10),
            "S": np.array([[complex(s[i, j]) for j in range(2)] for i in range(2)]),
            "status": "underflow" if float(t) == 0 else "ok", "dps": int(dps)
        }


def checked_high_precision(
    widths,
    pots,
    energy,
    *,
    dps_low=80,
    dps_high=120
):
    low = high_precision_transport(widths, pots, energy, dps=dps_low)
    high = high_precision_transport(widths, pots, energy, dps=dps_high)
    if not np.isclose(low["log_T"], high["log_T"], atol = 1e-10, rtol = 1e-13):
        raise ArithmeticError("independent reference has not converged with precision")
    if not np.allclose(low["S"], high["S"], atol=1e-10, rtol=1e-10):
        raise ArithmeticError("independent scattering matrix has not converged with precision")
    return high