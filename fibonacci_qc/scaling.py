"""
How the parity transmission difference changes with size of system.
Fits log |D_n(E)| = log C - alpha_eff(E) log N_n at each energy where N_n is the fibonacci length F_n
The slope alpha is flat if its magnitude is less than 0.05, growing with size if alpha < -0.05 and shriking with size if alpha > 0.05.
Only counted as stable if passes LOO and RMSE checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import build_geometry, fibonacci_number
from fibonacci_qc.parity_analysis import parity_contrast, position_parity,position_valid_mask, numerical_cell_mask

DIRECTION_FLAT = "flat"
DIRECTION_INCREASING = "increasing_contrast"
DIRECTION_DECREASING = "decreasing_candidate"

DEFAULT_PAIRED_CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("sub-width", cfg.MODE_SUBSTITUTION, cfg.MODE_WIDTH_ONLY),
    ("sub-pot", cfg.MODE_SUBSTITUTION, cfg.MODE_POTENTIAL_ONLY),
    ("pot-width", cfg.MODE_POTENTIAL_ONLY, cfg.MODE_WIDTH_ONLY),
)

@dataclass(frozen=True)
class ScalingFitPoint:
    "Fit result at one energy"
    energy: float
    alpha: float
    intercept: float
    rmse: float
    n_usable: int
    support: bool
    direction: str
    stable: bool
    loo_alpha: Tuple[float, ...]
    loo_left_out: Tuple[int, ...]

@dataclass(frozen=True)
class ModeScalingResult:
    "Scaling curves for a defect mode and a generation window"
    mode: str
    window: Tuple[int, int]
    generations: Tuple[int, ...]
    sizes: Tuple[int, ...]
    energies: np.ndarray
    alpha: np.ndarray
    intercept: np.ndarray
    rmse: np.ndarray
    n_usable: np.ndarray
    support: np.ndarray
    direction: np.ndarray
    stable: np.ndarray
    loo_alpha: np.ndarray
    points: Tuple[ScalingFitPoint, ...]

def classify_direction(
    alpha: float,
    *,
    threshold: float = cfg.ALPHA_FLAT_THRESHOLD,
) -> str:
    "Map alpha_eff to a locked direction label."
    if not np.isfinite(alpha):
        return "undefined"
    if abs(alpha) <= threshold:
        return DIRECTION_FLAT
    if alpha < -threshold:
        return DIRECTION_INCREASING
    return DIRECTION_DECREASING

def generations_in_window(window: Tuple[int, int]) -> Tuple[int, ...]:
    "Inclusive generation list for locked fit window"
    lo, hi = window
    if hi < lo:
        raise ValueError(f"invalid generation window {window}")
    return tuple(range(lo, hi + 1))

def _usable_mask(
    abs_d: np.ndarray,
    *,
    d_floor: float = cfg.SCALING_D_FLOOR,
) -> np.ndarray:
    return np.isfinite(abs_d) & (abs_d > d_floor)

def fit_loglog_scaling(
    sizes: np.ndarray,
    abs_d: np.ndarray,
    *,
    min_usable: int = cfg.SCALING_MIN_USABLE_GENERATIONS,
    d_floor: float = cfg.SCALING_D_FLOOR,
) -> Tuple[float, float, float, int, bool]:
    "OLS fit of log|D| = intercept - alpha * log N"
    sizes = np.asarray(sizes, dtype=np.float64)
    abs_d = np.asarray(abs_d, dtype=np.float64)
    if sizes.shape != abs_d.shape:
        raise ValueError("sizes and abs_d need same shape")
    if sizes.ndim != 1:
        raise ValueError("sizes and abs_d must be 1D")
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("sizes must be finite & positive strictly")
    usable = _usable_mask(abs_d, d_floor=d_floor)
    n_usable = int(np.count_nonzero(usable))
    if n_usable < min_usable:
        return float("nan"), float("nan"), float("nan"), n_usable, False
    x = np.log(sizes[usable])
    if np.unique(x).size < 2:
        return float("nan"), float("nan"), float("nan"), n_usable, False
    y = np.log(abs_d[usable])
    if np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan"), n_usable, False
    # y=a+bx with b = -alpha, a = log C
    b, a = np.polyfit(x, y, 1)
    alpha = float(-b)
    intercept = float(a)
    resid = y - (a + b * x)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return alpha, intercept, rmse, n_usable, True

def leave_one_out_alphas(
    sizes: np.ndarray,
    abs_d: np.ndarray,
    generations: Sequence[int],
    *,
    min_usable: int = cfg.SCALING_MIN_USABLE_GENERATIONS,
    d_floor: float = cfg.SCALING_D_FLOOR,
) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    "Refit after dropping each generation"
    sizes = np.asarray(sizes, dtype=np.float64)
    abs_d = np.asarray(abs_d, dtype=np.float64)
    gens = tuple(int(g) for g in generations)
    alphas: List[float] = []
    left_out: List[int] = []
    for i, gen in enumerate(gens):
        mask = np.ones(sizes.size, dtype=bool)
        mask[i] = False
        alpha, _, _, _, support = fit_loglog_scaling(
            sizes[mask],
            abs_d[mask],
            min_usable=min_usable,
            d_floor=d_floor,
        )
        alphas.append(alpha if support else float("nan"))
        left_out.append(gen)
    return tuple(alphas), tuple(left_out)

def is_stable_fit(
    alpha: float,
    rmse: float,
    loo_alphas: Sequence[float],
    direction: str,
    *,
    loo_tol: float = cfg.SCALING_LOO_ALPHA_TOL,
    rmse_max: float = cfg.SCALING_RMSE_MAX,
) -> bool:
    "Locked stability. RMSE bound plus LOO alpha/direction agreement"
    if not np.isfinite(alpha) or not np.isfinite(rmse):
        return False
    if rmse > rmse_max:
        return False
    if direction == "undefined":
        return False
    finite_loo = [a for a in loo_alphas if np.isfinite(a)]
    if not finite_loo:
        return False
    if any(abs(a - alpha) > loo_tol for a in finite_loo):
        return False
    if any(classify_direction(a) != direction for a in finite_loo):
        return False
    return True

def fit_scaling_curve(
    sizes: np.ndarray,
    d_by_generation: np.ndarray,
    energies: np.ndarray,
    generations: Sequence[int],
    *,
    mode: str,
    window: Tuple[int, int],
) -> ModeScalingResult:
    "Fit alpha_eff(E) across a generation window for one mode. d_by_generation has shape (n_generations, n_energy)"
    sizes = np.asarray(sizes, dtype=np.float64)
    d_mat = np.asarray(d_by_generation, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    gens = tuple(int(g) for g in generations)
    if d_mat.shape != (len(gens), energies.size):
        raise ValueError(f"d_by_generation shape {d_mat.shape} does not match ({len(gens)}, {energies.size})"
        )
    if sizes.shape != (len(gens),):
        raise ValueError("sizes must align with generations")
    n_e = energies.size
    n_g = len(gens)
    alpha = np.full(n_e, np.nan, dtype=np.float64)
    intercept = np.full(n_e, np.nan, dtype=np.float64)
    rmse = np.full(n_e, np.nan, dtype=np.float64)
    n_usable = np.zeros(n_e, dtype=np.int64)
    support = np.zeros(n_e, dtype=bool)
    direction = np.full(n_e, "undefined", dtype=object)
    stable = np.zeros(n_e, dtype=bool)
    loo_alpha = np.full((n_e, n_g), np.nan, dtype=np.float64)
    points: List[ScalingFitPoint] = []
    abs_d = np.abs(d_mat)
    for e_idx in range(n_e):
        a, c, r, n_use, ok = fit_loglog_scaling(sizes, abs_d[:, e_idx])
        loo, left = leave_one_out_alphas(sizes, abs_d[:, e_idx], gens)
        direc = classify_direction(a) if ok else "undefined"
        stab = is_stable_fit(a, r, loo, direc) if ok else False

        alpha[e_idx] = a
        intercept[e_idx] = c
        rmse[e_idx] = r
        n_usable[e_idx] = n_use
        support[e_idx] = ok
        direction[e_idx] = direc
        stable[e_idx] = stab
        loo_alpha[e_idx, :] = loo
        points.append(
            ScalingFitPoint(
                energy=float(energies[e_idx]),
                alpha=a,
                intercept=c,
                rmse=r,
                n_usable=n_use,
                support=ok,
                direction=direc,
                stable=stab,
                loo_alpha=loo,
                loo_left_out=left,
            )
        )
    return ModeScalingResult(
        mode=mode,
        window=window,
        generations=gens,
        sizes=tuple(int(s) for s in sizes),
        energies=energies,
        alpha=alpha,
        intercept=intercept,
        rmse=rmse,
        n_usable=n_usable,
        support=support,
        direction=direction,
        stable=stable,
        loo_alpha=loo_alpha,
        points=tuple(points),
    )

def collect_parity_curves(
    mode: str,
    generations: Sequence[int],
    energies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    "Return (sizes, d_by_generation, energies) for a mode"
    energies = np.asarray(energies, dtype=np.float64)
    gens = tuple(int(g) for g in generations)
    sizes = np.array([fibonacci_number(g) for g in gens], dtype=np.float64)
    d_rows = []
    for gen in gens:
        geom = build_geometry(gen)
        sweep = fast_sweep(mode, energies, geometry=geom)
        parity = position_parity(geom, mode)
        valid = position_valid_mask(geom, mode, sweep_valid=np.asarray(sweep["valid"], dtype=bool))
        cell = numerical_cell_mask(sweep["status"] if "status" in sweep else None)
        d_rows.append(parity_contrast(sweep["T"], parity, valid, energies=energies, cell_valid=cell).D)
    return sizes, np.vstack(d_rows), energies

def collect_parity_curves_from_bundles(
    mode: str,
    bundles_by_generation: Mapping[int, Mapping[str, object]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...]]:
    "Build scaling inputs. Each bundle provide energies and either D or mode transmission + geometry from generation key."
    gens = tuple(sorted(bundles_by_generation))
    if not gens:
        raise ValueError("no generations provided")

    first = bundles_by_generation[gens[0]]
    energies = np.asarray(first["energies"], dtype=np.float64)
    sizes = np.array([fibonacci_number(g) for g in gens], dtype=np.float64)
    rows = []
    for gen in gens:
        bundle = bundles_by_generation[gen]
        e = np.asarray(bundle["energies"], dtype=np.float64)
        if e.shape != energies.shape or not np.allclose(e, energies):
            raise ValueError(f"energy grid mismatch at generation {gen}")
        if "D" in bundle:
            rows.append(np.asarray(bundle["D"], dtype=np.float64))
            continue
        modes = bundle["modes"]
        sweep = modes[mode]
        geom = build_geometry(gen)
        parity = position_parity(geom, mode)
        valid = position_valid_mask(geom, mode, sweep_valid=np.asarray(sweep["valid"], dtype=bool))
        rows.append(parity_contrast(sweep["T"], parity, valid, energies=energies).D)
    return sizes, np.vstack(rows), energies, gens

def common_support_mask(
    *supports: np.ndarray,
) -> np.ndarray:
    "Intersection of boolean support masks"
    if not supports:
        raise ValueError("at least one support mask is required")
    out = np.asarray(supports[0], dtype=bool).copy()
    for mask in supports[1:]:
        out &= np.asarray(mask, dtype=bool)
    return out

def paired_alpha_contrast(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    "delta(E) = alpha_a(E) - alpha_b(E) on common support"
    support = common_support_mask(support_a, support_b)
    delta = np.full_like(alpha_a, np.nan, dtype=np.float64)
    both = support & np.isfinite(alpha_a) & np.isfinite(alpha_b)
    delta[both] = np.asarray(alpha_a, dtype=np.float64)[both] - np.asarray(alpha_b, dtype=np.float64)[both]
    return delta, support

def analyze_mode_scaling(
    mode: str,
    *,
    window: Tuple[int, int],
    energies: Optional[np.ndarray] = None,
    bundles_by_generation: Optional[Mapping[int, Mapping[str, object]]] = None,
) -> ModeScalingResult:
    "Fit scaling for one mode over a generation window"
    gens = generations_in_window(window)
    if bundles_by_generation is not None:
        missing = [g for g in gens if g not in bundles_by_generation]
        if missing:
            raise KeyError(f"missing bundles for generations {missing}")
        subset = {g: bundles_by_generation[g] for g in gens}
        sizes, d_mat, energies_arr, _ = collect_parity_curves_from_bundles(mode, subset)
    else:
        e = (
            np.asarray(energies, dtype=np.float64)
            if energies is not None
            else cfg.energy_grid()
        )
        sizes, d_mat, energies_arr = collect_parity_curves(mode, gens, e)
    return fit_scaling_curve(sizes, d_mat, energies_arr, gens, mode=mode, window=window)

def analyze_scaling(
    *,
    modes: Optional[Sequence[str]] = None,
    windows: Optional[Sequence[Tuple[int, int]]] = None,
    energies: Optional[np.ndarray] = None,
    bundles_by_generation: Optional[Mapping[int, Mapping[str, object]]] = None,
    paired_contrasts: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> Dict[str, object]:
    "Full scaling analysis over modes & locked generation windows"
    modes = list(modes or cfg.DEFECT_MODES)
    windows = list(windows or cfg.FIT_GENERATION_WINDOWS)
    pairs = list(paired_contrasts or DEFAULT_PAIRED_CONTRASTS)

    by_window: Dict[str, object] = {}
    for window in windows:
        key = f"{window[0]}-{window[1]}"
        mode_fits: Dict[str, ModeScalingResult] = {}
        for mode in modes:
            mode_fits[mode] = analyze_mode_scaling(
                mode,
                window=window,
                energies=energies,
                bundles_by_generation=bundles_by_generation,
            )

        #Common support
        supports = [mode_fits[m].support for m in modes if m in mode_fits]
        common = common_support_mask(*supports) if supports else np.array([], dtype=bool)

        pair_results: Dict[str, object] = {}
        for name, mode_a, mode_b in pairs:
            if mode_a not in mode_fits or mode_b not in mode_fits:
                continue
            delta, support = paired_alpha_contrast(mode_fits[mode_a].alpha, mode_fits[mode_b].alpha, mode_fits[mode_a].support, mode_fits[mode_b].support)
            pair_results[name] = {
                "mode_a": mode_a,
                "mode_b": mode_b,
                "delta_alpha": delta,
                "support": support,
            }

        energies_out = next(iter(mode_fits.values())).energies if mode_fits else (
            np.asarray(energies) if energies is not None else cfg.energy_grid()
        )
        by_window[key] = {
            "window": window,
            "modes": mode_fits,
            "common_support": common,
            "paired": pair_results,
            "energies": energies_out,
        }
    return {
        "windows": by_window,
        "modes": modes,
        "fit_windows": windows,
        "paired_contrasts": pairs,
    }


def scaling_summary(
    analysis: Dict[str, object],
) -> List[Dict[str, object]]:
    "Flatten per-mode support fractions and median |alpha| on support"
    rows: List[Dict[str, object]] = []
    windows: Dict[str, object] = analysis["windows"]
    for key, payload in windows.items():
        mode_fits: Dict[str, ModeScalingResult] = payload["modes"]
        for mode, fit in mode_fits.items():
            supp = fit.support
            alphas = fit.alpha[supp]
            rows.append(
                {
                    "window": key,
                    "mode": mode,
                    "n_support": int(np.count_nonzero(supp)),
                    "support_fraction": float(np.mean(supp)) if supp.size else 0.0,
                    "n_stable": int(np.count_nonzero(fit.stable)),
                    "median_abs_alpha": float(np.nanmedian(np.abs(alphas)))
                    if alphas.size
                    else float("nan"),
                    "fraction_flat": float(
                        np.mean(fit.direction[supp] == DIRECTION_FLAT)
                    )
                    if np.any(supp)
                    else float("nan"),
                }
            )
    return rows