"Frozen reference cases for checks"

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.parity_analysis import parity_contrast, position_parity, position_valid_mask
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
    generation = int(params["generation"])
    mode = str(params["mode"])
    energies = np.asarray(params["energies"], dtype=np.float64)
    geom = build_geometry(generation)
    sweep = fast_sweep(mode, energies, geometry=geom)
    t_clean, log_clean, _ = clean_transmission_grid(generation, energies, geometry=geom)
    parity = position_parity(geom, mode)
    valid = position_valid_mask(geom, mode, sweep_valid=sweep["valid"])
    contrast = parity_contrast(sweep["T"], parity, valid, energies=energies)
    return {
        "generation": np.array(generation, dtype=np.int64),
        "energies": energies,
        "T_clean": t_clean,
        "log_T_clean": log_clean,
        "T": sweep["T"],
        "valid": sweep["valid"],
        "D": contrast.D,
    }

def fixture_paths() -> Tuple[Path, Path]:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURE_DIR / f"{REGRESSION_NAME}.npz", FIXTURE_DIR / f"{REGRESSION_NAME}.meta.json"

def write_regression_fixture() -> Path:
    "Write / refresh frozen regression fixture"
    arrays = compute_regression_reference()
    npz_path, meta_path = fixture_paths()
    np.savez_compressed(npz_path, **arrays)
    meta = {
        "name": REGRESSION_NAME,
        "params": {
            "generation": int(arrays["generation"]),
            "mode": cfg.MODE_SUBSTITUTION,
            "n_energy": int(arrays["energies"].size),
        },
        "metadata": run_metadata(
            generations=[int(arrays["generation"])],
            energies=arrays["energies"],
            extra={"fixture": REGRESSION_NAME},
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return npz_path

def load_regression_fixture() -> Dict[str, np.ndarray]:
    "Load frozen regression fixture create if missing"
    npz_path, _ = fixture_paths()
    if not npz_path.exists():
        write_regression_fixture()
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}

def assert_matches_regression_fixture(
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> Dict[str, object]:
    "Recompute regression case and compare to frozen fixture"
    fixture = load_regression_fixture()
    current = compute_regression_reference()
    report = {"ok": True, "fields": {}}
    for key in ("T_clean", "T", "D"):
        a = np.asarray(fixture[key], dtype=np.float64)
        b = np.asarray(current[key], dtype=np.float64)
        match = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        report["fields"][key] = {
            "match": bool(match),
            "max_abs": float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else float("inf"),
        }
        if not match:
            report["ok"] = False
    return report