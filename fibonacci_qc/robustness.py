"Recompute key results with a finer energy grid and make sure they still match (parity difference, summary numbers, transmission peaks and how they line up)"

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry
from fibonacci_qc.parity_analysis import (integrate_contrast_windows, parity_contrast, position_parity, position_valid_mask,)
from fibonacci_qc.resonance import (detect_peaks, match_peaks_order_preserving, track_resonances,)
from fibonacci_qc.scattering import clean_transmission_grid

# Finer grid used
ROBUSTNESS_N_ENERGY = 2 * cfg.N_ENERGY

@dataclass(frozen=True)
class ContrastRobustness:
    "Baseline vs new parity-contrast comparison"
    max_abs_D: float
    rmse_D: float
    integrated_baseline: Dict[str, Dict[str, float]]
    integrated_fine: Dict[str, Dict[str, float]]
    integrated_abs_diff: Dict[str, Dict[str, float]]

@dataclass(frozen=True)
class ResonanceRobustness:
    "Baseline vs new resonance comparison for a position"
    position: int
    n_peaks_baseline: int
    n_peaks_fine: int
    n_matched: int
    mean_abs_peak_shift: float
    max_abs_peak_shift: float
    mean_abs_delta_E_baseline: float
    mean_abs_delta_E_fine: float
    delta_E_summary_diff: float

def finer_energy_grid(
    e_min: float = cfg.E_MIN_EV,
    e_max: float = cfg.E_MAX_EV,
    n_points: int = ROBUSTNESS_N_ENERGY,
) -> np.ndarray:
    "Inclusive denser linear energy grid"
    return cfg.energy_grid(e_min, e_max, n_points)

def interpolate_to_grid(
    values: np.ndarray,
    from_energies: np.ndarray,
    to_energies: np.ndarray,
) -> np.ndarray:
    "Linear interpolation of 1D curve onto to_energies"
    return np.interp(np.asarray(to_energies, dtype=np.float64), np.asarray(from_energies, dtype=np.float64), np.asarray(values, dtype=np.float64),)

def compare_parity_contrast(
    d_baseline: np.ndarray,
    energies_baseline: np.ndarray,
    d_fine: np.ndarray,
    energies_fine: np.ndarray,
) -> ContrastRobustness:
    "Compare D(E) and integrated measures"
    d_fine_on_base = interpolate_to_grid(d_fine, energies_fine, energies_baseline)
    diff = d_fine_on_base - np.asarray(d_baseline, dtype=np.float64)
    finite = np.isfinite(diff)
    if not np.any(finite):
        max_abs = float("nan")
        rmse = float("nan")
    else:
        max_abs = float(np.nanmax(np.abs(diff)))
        rmse = float(np.sqrt(np.nanmean(diff[finite] ** 2)))
    base_int = integrate_contrast_windows(d_baseline, energies_baseline)
    fine_int = integrate_contrast_windows(d_fine, energies_fine)
    integrated_baseline = {
        name: {"signed": m.signed, "abs": m.abs, "rms": m.rms}
        for name, m in base_int.items()
    }
    integrated_fine = {
        name: {"signed": m.signed, "abs": m.abs, "rms": m.rms}
        for name, m in fine_int.items()
    }
    integrated_abs_diff = {
        name: {
            key: abs(integrated_fine[name][key] - integrated_baseline[name][key])
            for key in ("signed", "abs", "rms")
        }
        for name in integrated_baseline
    }
    return ContrastRobustness(
        max_abs_D=max_abs,
        rmse_D=rmse,
        integrated_baseline=integrated_baseline,
        integrated_fine=integrated_fine,
        integrated_abs_diff=integrated_abs_diff,
    )

def compare_resonance_spectra(
    spectrum_baseline: np.ndarray,
    energies_baseline: np.ndarray,
    spectrum_fine: np.ndarray,
    energies_fine: np.ndarray,
    *,
    t_clean_baseline: np.ndarray,
    t_clean_fine: np.ndarray,
    position: int,
    mode: str = "robustness",
    parity: int = 0,
) -> ResonanceRobustness:
    "Compare peak locations and matched ΔE"
    peaks_base = detect_peaks(spectrum_baseline, energies_baseline)
    peaks_fine = detect_peaks(spectrum_fine, energies_fine)
    peak_match = match_peaks_order_preserving(peaks_base, peaks_fine)
    if peak_match.matched:
        shifts = [abs(m.delta_E) for m in peak_match.matched]
        mean_shift = float(np.mean(shifts))
        max_shift = float(np.max(shifts))
    else:
        mean_shift = float("nan")
        max_shift = float("nan")
    track_base = track_resonances(
        t_clean_baseline,
        spectrum_baseline,
        energies_baseline,
        mode=mode,
        position=position,
        parity=parity,
    )
    track_fine = track_resonances(
        t_clean_fine,
        spectrum_fine,
        energies_fine,
        mode=mode,
        position=position,
        parity=parity,
    )
    base_abs = track_base.mean_abs_delta_E
    fine_abs = track_fine.mean_abs_delta_E
    if np.isfinite(base_abs) and np.isfinite(fine_abs):
        summary_diff = abs(fine_abs - base_abs)
    else:
        summary_diff = float("nan")
    return ResonanceRobustness(
        position=position,
        n_peaks_baseline=len(peaks_base),
        n_peaks_fine=len(peaks_fine),
        n_matched=peak_match.n_matched,
        mean_abs_peak_shift=mean_shift,
        max_abs_peak_shift=max_shift,
        mean_abs_delta_E_baseline=base_abs,
        mean_abs_delta_E_fine=fine_abs,
        delta_E_summary_diff=summary_diff,
    )

def robustness_check_mode(
    geometry: FibonacciGeometry,
    mode: str,
    *,
    energies_baseline: Optional[np.ndarray] = None,
    energies_fine: Optional[np.ndarray] = None,
    positions: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    "Full check for a defect mode on one generation."
    e_base = (
        np.asarray(energies_baseline, dtype=np.float64)
        if energies_baseline is not None
        else cfg.energy_grid()
    )
    e_fine = (
        np.asarray(energies_fine, dtype=np.float64)
        if energies_fine is not None
        else finer_energy_grid()
    )
    sweep_base = fast_sweep(mode, e_base, geometry=geometry)
    sweep_fine = fast_sweep(mode, e_fine, geometry=geometry)
    t_clean_base, _, _ = clean_transmission_grid(
        geometry.generation, e_base, geometry=geometry
    )
    t_clean_fine, _, _ = clean_transmission_grid(
        geometry.generation, e_fine, geometry=geometry
    )
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(
        geometry, mode, sweep_valid=np.asarray(sweep_base["valid"], dtype=bool)
    )
    d_base = parity_contrast(
        sweep_base["T"], parity, valid, energies=e_base
    ).D
    d_fine = parity_contrast(
        sweep_fine["T"],
        parity,
        position_valid_mask(
            geometry, mode, sweep_valid=np.asarray(sweep_fine["valid"], dtype=bool)
        ),
        energies=e_fine,
    ).D
    contrast = compare_parity_contrast(d_base, e_base, d_fine, e_fine)
    n_pos = sweep_base["T"].shape[0]
    if positions is None:
        # Check a few valid positions
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            pos_list: Tuple[int, ...] = ()
        else:
            mid = valid_idx[valid_idx.size // 2]
            pos_list = tuple(
                sorted({int(valid_idx[0]), int(mid), int(valid_idx[-1])})
            )
    else:
        pos_list = tuple(int(p) for p in positions)
    resonance_rows = []
    for p in pos_list:
        if not valid[p]:
            continue
        resonance_rows.append(
            compare_resonance_spectra(
                sweep_base["T"][p],
                e_base,
                sweep_fine["T"][p],
                e_fine,
                t_clean_baseline=t_clean_base,
                t_clean_fine=t_clean_fine,
                position=p,
                mode=mode,
                parity=int(parity[p]),
            )
        )
    return {
        "mode": mode,
        "generation": geometry.generation,
        "n_energy_baseline": int(e_base.size),
        "n_energy_fine": int(e_fine.size),
        "contrast": contrast,
        "resonance": resonance_rows,
        "D_baseline": d_base,
        "D_fine": d_fine,
        "energies_baseline": e_base,
        "energies_fine": e_fine,
    }

def analyze_generation_robustness(
    generation: int,
    *,
    geometry: Optional[FibonacciGeometry] = None,
    modes: Optional[Sequence[str]] = None,
    energies_baseline: Optional[np.ndarray] = None,
    energies_fine: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    "Robustness checks for the defect modes."
    geom = geometry if geometry is not None else build_geometry(generation)
    modes = list(modes or cfg.DEFECT_MODES)
    mode_results = {
        mode: robustness_check_mode(
            geom,
            mode,
            energies_baseline=energies_baseline,
            energies_fine=energies_fine,
        )
        for mode in modes
    }
    return {"generation": geom.generation, "modes": mode_results, "geometry_metadata": geom.to_metadata(),}