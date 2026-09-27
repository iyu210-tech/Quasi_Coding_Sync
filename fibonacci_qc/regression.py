"Frozen reference cases for checks"

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.parity_analysis import parity_contrast, position_parity, position_valid_mask, numerical_cell_mask
from fibonacci_qc.raw_data import run_metadata
from fibonacci_qc.scattering import clean_transmission_grid

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
REGRESSION_NAME = "regression_n04_substitution"

def regression_case_params() -> Dict[str, object]:
    "Locked small-case paramaters for regression tests"
    return {"generation": 4, "mode": cfg.MODE_SUBSTITUTION, "energies": np.linspace(0.5, 2.5, 8)}

def compute_regression_reference() -> Dict[str, np.ndarray]:
    "Compute the frozen reference arrays for small regression case"
    params = regression_case_params()
    geom = build_geometry(params["generation"]);
    sweep = fast_sweep(params["mode"],params["energies"],geometry=geom)
    contrast = parity_contrast(sweep["T"],geom.site_parity,sweep["valid"],energies=params["energies"],cell_valid=numerical_cell_mask(sweep["status"]))
    return {
        "generation": np.array(params["generation"]),
        "energies": params["energies"],
        "T_clean": sweep["T_clean"],
        "log_T_clean": sweep["log_T_clean"],
        "clean_status": sweep["clean_status"],
        "T": sweep["T"],
        "log_T": sweep["log_T"],
        "status": sweep["status"],
        "valid": sweep["valid"],
        "D": contrast.D
    }

def fixture_paths() -> Tuple[Path, Path]:
    return FIXTURE_DIR / f"{REGRESSION_NAME}.npz", FIXTURE_DIR / f"{REGRESSION_NAME}.meta.json"

def _require_physical(name,values):
    a = np.asarray(values, dtype = float)
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1 + cfg.TOLERANCES.t_bounds_atol):
        raise AssertionError(f"{name} must contain finite physical probabilities")

def write_regression_fixture(*, confirm=False) -> Path:
    "Write / refresh frozen regression fixture"
    if not confirm:
        raise RuntimeError("fixture regeneration is not a test; pass confirm=True only for deliberate independent regeneration")
    from fibonacci_qc.independent_reference import checked_high_precision
    params=regression_case_params(); e = params["energies"]
    # Reconstruct the tiny problem independently of geometry/defect/production helpers.
    letters = ["A","B","A"]; widths = {"A": cfg.D_A_NM, "B": cfg.D_B_NM}; pots = {"A": cfg.V_A_EV, "B": cfg.V_B_EV}
    clean = []; logclean = []; tc = np.empty((3,len(e))); lc = np.empty_like(tc)
    for j, energy in enumerate(e):
        r = checked_high_precision([widths[x] for x in letters], [pots[x] for x in letters], energy)
        clean.append(r["T"]); logclean.append(r["log_T"])
        for p in range(3):
            changed = letters.copy(); changed[p] = "B" if changed[p] == "A" else "A"
            r = checked_high_precision([widths[x] for x in changed], [pots[x] for x in changed], energy)
            tc[p, j], lc[p, j] = r["T"], r["log_T"]
    arrays = dict(generation=np.array(4),energies=e,T_clean=np.asarray(clean),log_T_clean=np.asarray(logclean),
        clean_status = np.zeros(len(e), dtype = np.uint8), T = tc, log_T = lc, status = np.zeros(tc.shape, dtype = np.uint8),
        valid = np.ones(3, dtype=bool), D = (tc[0] + tc[2]) / 2 - tc[1])
    for k in ("T", "T_clean"): _require_physical(k, arrays[k])
    npz, meta = fixture_paths(); npz.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(npz, **arrays)
    meta.write_text(json.dumps(dict(
        name=REGRESSION_NAME,
        method="independent mpmath characteristic matrices",
        precision_checks=[80,120],
        parameters=params|{"energies": e.tolist()},
        hbar2_over_2m_ev_nm2=cfg.HBAR2_OVER_2M_EV_NM2,
        note="Created independently"),indent = 2),encoding = "utf-8")
    return npz


def load_regression_fixture() -> Dict[str, np.ndarray]:
    "Load committed regression fixture"
    npz_path, _ = fixture_paths()
    if not npz_path.exists():
        raise FileNotFoundError(f"missing committed fixture {npz_path}.")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def assert_matches_regression_fixture(*, atol: float = 1e-12, rtol: float = 1e-10) -> None:
    "Recompute regression case and raise unless matches committed"
    fixture = load_regression_fixture()
    current = compute_regression_reference()
    for item in (fixture, current):
        for key in ("T","T_clean"): _require_physical(key, item[key])
        for key in ("D","log_T", "log_T_clean"):
            if not np.all(np.isfinite(item[key])): raise AssertionError(f"nonfinite regression {key}")
    for key in ("generation", "energies", "valid", "status", "clean_status"):
        if not np.array_equal(fixture[key], current[key]): raise AssertionError(f"regression mismatch: {key}")
    for key in ("T", "T_clean", "log_T", "log_T_clean", "D"):
        abs_tol = 1e-8 if key.startswith("log") else atol
        if not np.allclose(fixture[key], current[key], atol=abs_tol, rtol=rtol, equal_nan=False):
            raise AssertionError(f"regression mismatch: {key}")