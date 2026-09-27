"Compute D_n(E)=<T>_even p - <T>_odd p"

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Sequence, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry

ArrayLike = Union[np.ndarray, Sequence[float]]

if hasattr(np, "trapezoid"):
    _TRAPEZOID = np.trapezoid
else:
    _TRAPEZOID = np.trapz

SITE_CONDITIONING_FAMILIES: Tuple[str, ...] = ("site_letter", "site_3", "site_5")
BOND_CONDITIONING_FAMILIES: Tuple[str, ...] = ("bond_pair", "bond_4", "bond_6")

@dataclass(frozen=True)
class ParityContrastResult:
    "Contrast curve D(E)"
    D: np.ndarray
    energies: np.ndarray
    n_even: int
    n_odd: int
    parity: np.ndarray
    valid: np.ndarray
    n_even_by_energy: np.ndarray  # int64, shape (n_energy,)
    n_odd_by_energy: np.ndarray
    coverage: float
    coverage_by_energy: np.ndarray

@dataclass(frozen=True)
class IntegratedContrast:
    "Integrated parity contrast metrics over an energy window"
    signed: float
    abs:float
    rms:float
    e_min:float
    e_max:float
    e_min_support: float
    e_max_support: float
    n_energy: int
    coverage: float

@dataclass(frozen=True)
class ResponseMagnitude:
    "Mean abs / RMS of ∆T over energy window"
    mean_abs: float
    rms: float
    e_min: float
    e_max: float
    n_cells: int
    n_intended: int = 0
    coverage: float = float("nan")

@dataclass(frozen=True)
class ParityCheckResult:
    "Outcome of an origin shift / consistency check"
    name:str
    max_abs_error: float
    expected_factor: float
    passed:bool
    detail: str

def conditioning_families(mode:str) -> Tuple[str, ...]:
    "Motif families for conditional parity contrasts"
    if mode == cfg.MODE_PHASON:
        return BOND_CONDITIONING_FAMILIES
    if mode in cfg.ONE_SITE_MODES:
        return SITE_CONDITIONING_FAMILIES
    raise ValueError(f"Unknown defect mode {mode!r}")

def position_parity(geometry:FibonacciGeometry, mode:str) -> np.ndarray:
    "Return full uncompact parity array for mode aligned to geom site / bond axis"
    if mode == cfg.MODE_PHASON:
        return np.asarray(geometry.bond_parity)
    if mode in cfg.ONE_SITE_MODES:
        return np.asarray(geometry.site_parity)
    raise ValueError(f"Unknown defect mode {mode!r}")

def position_valid_mask(
    geometry: FibonacciGeometry, mode: str, *, sweep_valid: Optional[np.ndarray]=None,
) -> np.ndarray:
    "Mask of positions that may enter parity avgs"
    if mode == cfg.MODE_PHASON:
        valid = np.asarray(geometry.phason_valid, dtype=bool)
    elif mode in cfg.ONE_SITE_MODES:
        valid = np.ones(geometry.n_sites, dtype=bool)
    else:
        raise ValueError(f"Unknown defect mode {mode!r}")

    if sweep_valid is not None:
        sweep_valid = np.asarray(sweep_valid, dtype=bool)
        if sweep_valid.shape != valid.shape:
            raise ValueError(f"Shape mismatch: sweep_valid {sweep_valid.shape} does not match {valid.shape}")
        valid = valid & sweep_valid
    return valid

def numerical_cell_mask(status: np.ndarray) -> np.ndarray:
    from fibonacci_qc.scattering import status_is_ok
    return status_is_ok(status)

def combine_analysis_mask(
    geometric_valid: np.ndarray,
    cell_valid: Optional[np.ndarray],
    *,
    n_energy: int,
) -> np.ndarray:
    "Broadcast geometric (n_pos,) validity onto cells & returns bool array w/ shape (n_pos, n_energy)"
    geo = np.asarray(geometric_valid, dtype=bool)
    if geo.ndim != 1:
        raise ValueError("geometric_valid must be 1D (n_pos,)")
    out = np.broadcast_to(geo[:, np.newaxis], (geo.size, n_energy)).copy()
    if cell_valid is None:
        return out
    cell = np.asarray(cell_valid, dtype=bool)
    if cell.shape != out.shape:
        raise ValueError(f"cell_valid shape {cell.shape} does not match {(geo.size, n_energy)}")
    return out & cell

def parity_group_masks(
    parity:np.ndarray, valid:np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    "Even/odd masks on full position axis; clean transmission position independent (D_clean=0). Decrease in parity contrast interpreted w/ generation, conditioning, or something else"
    parity = np.asarray(parity, dtype=np.int8)
    valid = np.asarray(valid, dtype=bool)
    if parity.shape != valid.shape:
        raise ValueError("parity and valid must have the same shape")
    even = valid & (parity == 0)
    odd = valid & (parity == 1)
    return even, odd

def _masked_mean_along_positions(
    transmission: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    "mean over axis 0 for each energy, ignoring NaNs outside mask"
    t=np.asarray(transmission, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if t.ndim != 2:
        raise ValueError(f"transmission must be 2D")
    if mask.ndim == 1:
        if mask.shape[0] != t.shape[0]:
            raise ValueError("mask and transmission position size differ")
        use = mask[:, np.newaxis] & np.isfinite(t)
    elif mask.shape == t.shape:
        use = mask & np.isfinite(t)
    else:
        raise ValueError("mask must be (n_pos,) or (n_pos, n_energy)")
    counts = np.count_nonzero(use, axis=0).astype(np.float64)
    summed = np.where(use, t, 0.0).sum(axis=0)
    mean = np.full(t.shape[1], np.nan, dtype=np.float64)
    good = counts > 0
    mean[good] = summed[good] / counts[good]
    return mean, counts

def parity_contrast(
    transmission: np.ndarray,
    parity: np.ndarray,
    valid:np.ndarray,
    *,
    energies=None,
    cell_valid=None,
) -> ParityContrastResult:
    "Compute D(E)=<T>_even - <T>_odd on the position axis that is frozen."
    t = np.asarray(transmission, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError("transmission has to have the shape (n_positions, n_energy)")

    geo_valid = np.asarray(valid, dtype=bool)
    analysis_mask = combine_analysis_mask(geo_valid, cell_valid, n_energy = t.shape[1])
    parity_arr = np.asarray(parity, dtype=np.int8)
    even_geo, odd_geo = parity_group_masks(parity_arr, geo_valid)
    n_even = int(np.count_nonzero(even_geo))
    n_odd = int(np.count_nonzero(odd_geo))
    even_cell = analysis_mask & (parity_arr[:, np.newaxis]==0)
    odd_cell = analysis_mask & (parity_arr[:, np.newaxis] == 1)
    if n_even == 0 or n_odd == 0:
        d = np.full(t.shape[1], np.nan, dtype=np.float64)
        n_even_e = np.zeros(t.shape[1], dtype=np.int64)
        n_odd_e = np.zeros(t.shape[1], dtype=np.int64)
    else:
        mean_even, n_even_e = _masked_mean_along_positions(t, even_cell)
        mean_odd, n_odd_e = _masked_mean_along_positions(t, odd_cell)
        d = np.where((n_even_e > 0) & (n_odd_e > 0), mean_even - mean_odd, np.nan)

    e_out = (
        np.asarray(energies, dtype=np.float64)
        if energies is not None
        else np.arange(t.shape[1], dtype=np.float64)
    )
    if e_out.shape[0] != t.shape[1]:
        raise ValueError("energy length has to match transmission energy axis")

    n_geo = int(np.count_nonzero(geo_valid)) * t.shape[1]
    finite_used = analysis_mask & np.isfinite(t)
    n_used = int(np.count_nonzero(finite_used))
    n_geo_positions = int(np.count_nonzero(geo_valid))
    coverage_by_energy = (np.count_nonzero(finite_used, axis=0) / n_geo_positions
                          if n_geo_positions else np.full(t.shape[1], np.nan))
    coverage = float(n_used / n_geo) if n_geo > 0 else float("nan")

    return ParityContrastResult(
        D=d,
        energies=e_out,
        n_even=n_even,
        n_odd=n_odd,
        parity=parity_arr,
        valid=geo_valid,
        n_even_by_energy=n_even_e,
        n_odd_by_energy=n_odd_e,
        coverage=coverage,
        coverage_by_energy=coverage_by_energy,
    )

def integrate_contrast(
    contrast: np.ndarray,
    energies: np.ndarray,
    e_min: float,
    e_max: float,
) -> IntegratedContrast:
    "Integrate D(E) over an energy window [e_min, e_max]"
    d = np.asarray(contrast, dtype=np.float64)
    e = np.asarray(energies, dtype=np.float64)
    if e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
        raise ValueError("integration needs at least two strictly increasing finite energies")
    if d.shape != e.shape:
        raise ValueError("contrast and energies must have same shape")
    if e_max < e_min:
        raise ValueError("e_max must be >= e_min")
    def _interp_at(x: float):
        if x < e[0] or x > e[-1]:
            return None
        return float(np.interp(x, e, d))
    sel = (e >= e_min) & (e <= e_max)
    e_list = list(e[sel])
    d_list = list(d[sel])
    d_lo = _interp_at(float(e_min))
    if d_lo is not None and np.isfinite(d_lo):
        if not e_list or e_list[0] > float(e_min) + 1e-15:
            e_list.insert(0, float(e_min))
            d_list.insert(0, d_lo)
    d_hi = _interp_at(float(e_max))
    if d_hi is not None and np.isfinite(d_hi):
        if not e_list or e_list[-1] < float(e_max) - 1e-15:
            e_list.append(float(e_max))
            d_list.append(d_hi)

    n_sel = len(e_list)
    if n_sel < 2:
        return IntegratedContrast(
            signed=float("nan"),
            abs=float("nan"),
            rms=float("nan"),
            e_min=float(e_min),
            e_max=float(e_max),
            e_min_support=float("nan"),
            e_max_support=float("nan"),
            n_energy=n_sel,
            coverage=0.0
        )
    e_win = np.asarray(e_list, dtype=np.float64)
    d_win = np.asarray(d_list, dtype=np.float64)
    e_min_s = float(e_win[0])
    e_max_s = float(e_win[-1])
    width = e_max_s - e_min_s
    req = float(e_max - e_min)
    coverage = float(width / req) if req > 0 else float("nan")
    signed = float(_TRAPEZOID(d_win, e_win))
    abs_int = float(_TRAPEZOID(np.abs(d_win), e_win))
    rms = float(np.sqrt(_TRAPEZOID(d_win ** 2, e_win) / width)) if width > 0 else 0.0
    return IntegratedContrast(
        signed=signed,
        abs=abs_int,
        rms=rms,
        e_min=float(e_min),
        e_max=float(e_max),
        e_min_support=e_min_s,
        e_max_support=e_max_s,
        n_energy=n_sel,
        coverage=coverage,
    )

def integrate_contrast_windows(
    contrast: np.ndarray,
    energies: np.ndarray,
    windows: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Dict[str, IntegratedContrast]:
    "Signed, absolute, and RMS integrals over all locked energy windows."
    windows = windows or cfg.ENERGY_WINDOWS
    return {
        name: integrate_contrast(contrast, energies, lo, hi)
        for name, (lo, hi) in windows.items()
    }

def summarize_delta_t(delta_t, energies, valid, *, cell_valid=None, e_min, e_max):
    """Unweighted finite-grid cell mean; certified zeros remain in denominator."""
    dt, e = np.asarray(delta_t, dtype=float), np.asarray(energies, dtype=float)
    if dt.ndim != 2 or e.shape != (dt.shape[1],):
        raise ValueError("delta_t and energies axes do not match")
    mask = combine_analysis_mask(np.asarray(valid, dtype=bool), cell_valid, n_energy=e.size)
    sel = (e >= e_min) & (e <= e_max)
    intended = int(np.count_nonzero(valid) * np.count_nonzero(sel))
    use = mask[:, sel] & np.isfinite(dt[:, sel])
    x = dt[:, sel][use]
    n = x.size
    return ResponseMagnitude(mean_abs=float(np.mean(np.abs(x))) if n else float("nan"),
        rms=float(np.sqrt(np.mean(x*x))) if n else float("nan"),
        e_min=float(e_min), e_max=float(e_max), n_cells=n, n_intended=intended,
        coverage=float(n/intended) if intended else float("nan"))

def summarize_delta_t_windows(
    delta_t: np.ndarray,
    energies: np.ndarray,
    valid: np.ndarray,
    *,
    cell_valid: Optional[np.ndarray] = None,
    windows: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Dict[str, ResponseMagnitude]:
    windows = windows or cfg.ENERGY_WINDOWS
    return {name: summarize_delta_t(delta_t, energies, valid, cell_valid=cell_valid, e_min=lo, e_max=hi) for name, (lo,hi) in windows.items()}

def conditional_parity_contrasts(
    transmission: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    *,
    families: Optional[Sequence[str]] = None,
    sweep_valid: Optional[np.ndarray] = None,
    energies: Optional[np.ndarray] = None,
    cell_valid: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, ParityContrastResult]]:
    "Parity contrasts on motif / stratum labels"
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
    families = list(families or conditioning_families(mode))

    out: Dict[str, Dict[str, ParityContrastResult]] = {}
    for family in families:
        labels = geometry.motif_labels(family)
        if labels.shape[0] != parity.shape[0]:
            raise ValueError(f"motif family {family!r} length {labels.shape[0]} does not match position axis {parity.shape[0]} for mode {mode!r}")
        strata: Dict[str, ParityContrastResult] = {}
        for value in sorted(set(labels.tolist())):
            stratum_valid = valid & (labels==value)
            strata[str(value)] = parity_contrast(
                transmission,
                parity,
                stratum_valid,
                energies=energies,
                cell_valid=cell_valid,
            )
        out[family] = strata
    return out

def origin_shift_check(
    transmission: np.ndarray,
    parity: np.ndarray,
    valid: np.ndarray,
    *,
    energies: Optional[np.ndarray] = None,
    cell_valid: Optional[np.ndarray] = None,
    atol: float = cfg.TOLERANCES.transmission_atol,
) -> ParityCheckResult:
    "Verify shifting origin by one site flips D(E)'s sign"
    base = parity_contrast(transmission, parity, valid, energies=energies, cell_valid=cell_valid)
    shifted = parity_contrast(
        transmission,
        (np.asarray(parity) + 1) % 2,
        valid,
        energies=energies,
        cell_valid=cell_valid,
    )
    err = np.nanmax(np.abs(shifted.D + base.D))
    passed = bool(np.isfinite(err) and err <= atol)
    return ParityCheckResult(
        name="origin_shift",
        max_abs_error=float(err) if np.isfinite(err) else float("inf"),
        expected_factor=-1.0,
        passed=passed,
        detail="one-site origin shift should reverse even/odd groups",
    )

def reversal_check(
    transmission: np.ndarray,
    parity: np.ndarray,
    valid: np.ndarray,
    *,
    energies: Optional[np.ndarray] = None,
    cell_valid: Optional[np.ndarray] = None,
    atol: float = cfg.TOLERANCES.solver_crosscheck_atol,
) -> ParityCheckResult:
    "Mirror position axis, compare D to original"
    t = np.asarray(transmission, dtype=np.float64)
    n_pos = t.shape[0]
    mirrored = t[::-1, :]
    base = parity_contrast(transmission, parity, valid, energies=energies, cell_valid=cell_valid)
    rev = parity_contrast(mirrored, parity, valid[::-1], energies=energies, cell_valid=None if cell_valid is None else np.asarray(cell_valid)[::-1, :])

    expected_factor = 1.0 if (n_pos % 2 == 1) else -1.0
    err = np.nanmax(np.abs(rev.D - expected_factor * base.D))
    passed = bool(np.isfinite(err) and err <= atol)
    return ParityCheckResult(
        name="reversal",
        max_abs_error=float(err) if np.isfinite(err) else float("inf"),
        expected_factor=expected_factor,
        passed=passed,
        detail=("mirrored axis - odd preserves, even flips sign"),
    )

def parity_checks(
    transmission: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    *,
    sweep_valid: Optional[np.ndarray] = None,
    energies: Optional[np.ndarray] = None,
    cell_valid: Optional[np.ndarray] = None,
) -> Dict[str, ParityCheckResult]:
    "Origin shift & reversal checks"
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
    return {
        "origin_shift": origin_shift_check(transmission, parity, valid, energies=energies, cell_valid=cell_valid),
        "reversal": reversal_check(transmission, parity, valid, energies=energies, cell_valid=cell_valid),
    }

def analyze_mode_parity(
    transmission: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    *,
    energies: ArrayLike,
    sweep_valid: Optional[np.ndarray] = None,
    families: Optional[Sequence[str]] = None,
    cell_valid: Optional[np.ndarray] = None,
    min_coverage: Optional[float] = None,
    delta_t: Optional[np.ndarray] = None,
    response_cell_valid: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    "Parity analysis for one defect mode in one generation"
    energies_arr = np.asarray(energies, dtype=np.float64)
    t = np.asarray(transmission, dtype=np.float64)
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
    unconditional = parity_contrast(t, parity, valid, energies=energies_arr, cell_valid=cell_valid)
    if min_coverage is not None and (not np.isfinite(unconditional.coverage) or unconditional.coverage < min_coverage):
        raise RuntimeError(f"parity coverage {unconditional.coverage} below min_coverage={min_coverage} for mode={mode!r} generation={geometry.generation}")
    conditional = conditional_parity_contrasts(
        t,
        geometry,
        mode,
        families=families,
        sweep_valid=sweep_valid,
        energies=energies_arr,
        cell_valid=cell_valid,
    )
    checks = parity_checks(t, geometry, mode, sweep_valid=sweep_valid, energies=energies_arr, cell_valid=cell_valid)

    response = (summarize_delta_t_windows(delta_t, energies_arr, valid,
                cell_valid=response_cell_valid if response_cell_valid is not None else cell_valid)
                if delta_t is not None else {})
    numerical_failure = bool(np.any(unconditional.coverage_by_energy < cfg.MIN_NUMERICAL_COVERAGE)
                             or not np.isfinite(unconditional.coverage))
    integrated_unconditional = integrate_contrast_windows(unconditional.D, energies_arr)
    integrated_conditional: Dict[str, Dict[str, Dict[str, IntegratedContrast]]] = {}
    for family, strata in conditional.items():
        integrated_conditional[family] = {
            label: integrate_contrast_windows(result.D, energies_arr)
            for label, result in strata.items()
        }

    return {
        "mode": mode,
        "generation": geometry.generation,
        "energies": energies_arr,
        "parity": parity,
        "valid": valid,
        "cell_valid": cell_valid,
        "coverage": unconditional.coverage,
        "numerical_failure": numerical_failure,
        "response": response,
        "unconditional": unconditional,
        "conditional": conditional,
        "integrated": {"unconditional": integrated_unconditional, "conditional": integrated_conditional},
        "checks": checks,
    }
def analyze_generation_parity(raw_bundle, geometry=None, *, modes=None):
    from fibonacci_qc.scattering import trusted_transmission_mask, encode_status
    generation = int(raw_bundle["generation"])
    geom = build_geometry(generation) if geometry is None else geometry
    if geom.generation != generation:
        raise ValueError("generation does not match geometry")
    e = np.asarray(raw_bundle["energies"], dtype=float)
    clean = np.asarray(raw_bundle["T_clean"], dtype=float)
    clean_ok = trusted_transmission_mask(raw_bundle["clean_status"], clean, raw_bundle["log_T_clean"])
    results = {}
    for mode in (list(raw_bundle["modes"]) if modes is None else modes):
        sweep = raw_bundle["modes"][mode]
        t = np.asarray(sweep["T"], dtype=float)
        cell = trusted_transmission_mask(sweep["status"], t, sweep["log_T"])
        joint = cell & clean_ok[np.newaxis, :]
        # Recompute from authoritative arrays; never trust a stale saved delta.
        delta = np.where(joint, t - clean[np.newaxis, :], np.nan)
        result = analyze_mode_parity(t, geom, mode, energies=e,
            sweep_valid=np.asarray(sweep["valid"], dtype=bool), cell_valid=cell,
            delta_t=delta, response_cell_valid=joint)
        geo = position_valid_mask(geom, mode, sweep_valid=sweep["valid"])
        code = encode_status(sweep["status"])
        result["n_certified_underflow"] = int(np.count_nonzero((code == 2) & geo[:, None]))
        result["n_invalid"] = int(np.count_nonzero(~cell & geo[:, None]))
        result["numerical_failure"] |= bool(np.any(~joint[geo]))
        results[mode] = result
    return {"generation": generation, "energies": e, "geometry_metadata": geom.to_metadata(),
            "modes": results, "numerical_failure": any(v["numerical_failure"] for v in results.values())}

def integrated_summary(
    analysis: Dict[str, object],
    *,
    window: str = "full",
) -> Dict[str, Dict[str, float]]:
    "Extract signed, absolute, and RMS integrals for each mode in one generation"
    summary: Dict[str, Dict[str, float]] = {}
    modes: Dict[str, object] = analysis["modes"]
    for mode, payload in modes.items():
        integrated = payload["integrated"]
        mode_summary: Dict[str, float] = {
            "signed": integrated["unconditional"][window].signed,
            "abs": integrated["unconditional"][window].abs,
            "rms": integrated["unconditional"][window].rms,
        }
        summary[mode] = mode_summary
    return summary