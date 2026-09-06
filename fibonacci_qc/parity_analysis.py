"Compute D_n(E)=<T>_even p - <T>_odd p"

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Sequence, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry

ArrayLike = Union[np.ndarray, Sequence[float]]

_TRAPEZOID = getattr(np, "trapezoid", np.trapz)

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

@dataclass(frozen=True)
class IntegratedContrast:
    "Integrated parity contrast metrics over an energy window"
    signed: float
    abs:float
    rms:float
    e_min:float
    e_max:float
    n_energy: int

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
    "Return uncompact parity array for mode"
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
        return np.asarray(geometry.phason_valid, dtype=bool)
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

def parity_group_masks(
    parity:np.ndarray, valid:np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    "Even/odd masks on full position axis"
    parity = np.asarray(parity, dtype=np.int8)
    valid = np.asarray(valid, dtype=bool)
    if parity.shape != valid.shape:
        raise ValueError("parity and valid must have the same shape")
    even = valid & (parity == 0)
    odd = valid & (parity == 1)
    return even, odd

def _masked_mean_along_positions(
    transmission: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    "mean over axis 0 for each energy, ignoring NaNs outside mask"
    t=np.asarray(transmission, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if t.ndim != 2:
        raise ValueError(f"transmission must be 2D")
    if t.shape[0] != mask.shape[0]:
        raise ValueError(f"transmission and mask must have the same number of positions (axis 0)")

    selected = np.where(mask[:, np.newaxis], t, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(selected, axis=0)

def parity_contrast(
    transmission: np.ndarray,
    parity: np.ndarray,
    valid:np.ndarray,
    *,
    energies: Optional[np.ndarray]=None,
) -> ParityContrastResult:
    "Compute D(E)=<T>_even - <T>_odd on the position axis that is frozen."
    t = np.asarray(transmission, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError("transmission has to have the shape (n_positions, n_energy)")

    even, odd = parity_group_masks(parity, valid)
    n_even = int(np.count_nonzero(even))
    n_odd = int(np.count_nonzero(odd))
    if n_even == 0 or n_odd == 0:
        d = np.full(t.shape[1], np.nan, dtype=np.float64)
    else:
        d = _masked_mean_along_positions(t, even) - _masked_mean_along_positions(t, odd)

    e_out = (
        np.asarray(energies, dtype=np.float64)
        if energies is not None
        else np.arange(t.shape[1], dtype=np.float64)
    )
    if e_out.shape[0] != t.shape[1]:
        raise ValueError("energy length has to match transmission energy axis")

    return ParityContrastResult(
        D=d,
        energies=e_out,
        n_even=n_even,
        n_odd=n_odd,
        parity=np.asarray(parity, dtype=np.int8),
        valid=np.asarray(valid, dtype=bool),
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
    if d.shape != e.shape:
        raise ValueError("contrast and energies must have same shape")
    sel = (e >= e_min) & (e <= e_max)
    n_sel = int(np.count_nonzero(sel))
    if n_sel < 2:
        return IntegratedContrast(
            signed=float("nan"),
            abs=float("nan"),
            rms=float("nan"),
            e_min=float(e_min),
            e_max=float(e_max),
            n_energy=n_sel,
        )
    e_win = e[sel]
    d_win = d[sel]
    width = float(e_win[-1] - e_win[0])
    signed = float(_TRAPEZOID(d_win, e_win))
    abs_int = float(_TRAPEZOID(np.abs(d_win), e_win))
    rms = float(np.sqrt(_TRAPEZOID(d_win ** 2, e_win) / width)) if width > 0 else 0.0
    return IntegratedContrast(
        signed=signed,
        abs=abs_int,
        rms=rms,
        e_min=float(e_min),
        e_max=float(e_max),
        n_energy=n_sel,
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

def conditional_parity_contrasts(
    transmission: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    *,
    families: Optional[Sequence[str]] = None,
    sweep_valid: Optional[np.ndarray] = None,
    energies: Optional[np.ndarray] = None,
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
            )
        out[family] = strata
    return out

def origin_shift_check(
    transmission: np.ndarray,
    parity: np.ndarray,
    valid: np.ndarray,
    *,
    energies: Optional[np.ndarray] = None,
    atol: float = cfg.TOLERANCES.transmission_atol,
) -> ParityCheckResult:
    "Verify shifting origin by one site flips D(E)'s sign"
    base = parity_contrast(transmission, parity, valid, energies=energies)
    shifted = parity_contrast(
        transmission,
        (np.asarray(parity) + 1) % 2,
        valid,
        energies=energies,
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
    atol: float = cfg.TOLERANCES.solver_crosscheck_atol,
) -> ParityCheckResult:
    "Mirror position axis, compare D to original"
    t = np.asarray(transmission, dtype=np.float64)
    n_pos = t.shape[0]
    mirrored = t[::-1, :]
    base = parity_contrast(transmission, parity, valid, energies=energies)
    rev = parity_contrast(mirrored, parity, valid[::-1], energies=energies)

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
) -> Dict[str, ParityCheckResult]:
    "Origin shift & reversal checks"
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
    return {
        "origin_shift": origin_shift_check(transmission, parity, valid, energies=energies),
        "reversal": reversal_check(transmission, parity, valid, energies=energies),
    }

def analyze_mode_parity(
    transmission: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    *,
    energies: ArrayLike,
    sweep_valid: Optional[np.ndarray] = None,
    families: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    "Parity analysis for one defect mode in one generation"
    energies_arr = np.asarray(energies, dtype=np.float64)
    t = np.asarray(transmission, dtype=np.float64)
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
    unconditional = parity_contrast(t, parity, valid, energies=energies_arr)
    conditional = conditional_parity_contrasts(
        t,
        geometry,
        mode,
        families=families,
        sweep_valid=sweep_valid,
        energies=energies_arr,
    )
    checks = parity_checks(t, geometry, mode, sweep_valid=sweep_valid, energies=energies_arr)

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
        "unconditional": unconditional,
        "conditional": conditional,
        "integrated": {"unconditional": integrated_unconditional, "conditional": integrated_conditional},
        "checks": checks,
    }

def analyze_generation_parity(
    raw_bundle: Dict[str, object],
    geometry: Optional[FibonacciGeometry] = None,
    *,
    modes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    "Parity analysis for all defect modes in one generation, from raw bundle"
    generation = int(raw_bundle["generation"])
    geom = geometry if geometry is not None else build_geometry(generation)
    if geom.generation != generation:
        raise ValueError(f"geometry.generation={geom.generation} does not match bundle={generation}")
    energies = np.asarray(raw_bundle["energies"], dtype=np.float64)
    modes_dict: Dict[str, Dict[str, np.ndarray]] = raw_bundle["modes"]
    modes = list(modes or modes_dict.keys())
    mode_results: Dict[str, object] = {}
    for mode in modes:
        sweep = modes_dict[mode]
        t = np.asarray(sweep["T"], dtype=np.float64)
        sweep_valid = np.asarray(sweep["valid"], dtype=bool)
        mode_results[mode] = analyze_mode_parity(t, geom, mode, energies=energies, sweep_valid=sweep_valid,)

    return {
        "generation": generation,
        "energies": energies,
        "geometry_metadata": geom.to_metadata(),
        "modes": mode_results,
    }

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