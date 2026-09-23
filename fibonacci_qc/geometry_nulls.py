"Shuffle parity labels within geometry groups to test whether diffference between parities is larger than chance"

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry
from fibonacci_qc.parity_analysis import integrate_contrast, parity_contrast, position_parity, position_valid_mask, numerical_cell_mask

@dataclass(frozen=True)
class GlobalParityStats:
    "Scalars for parity contrast curve"
    S_max: float
    S_int: float

@dataclass(frozen=True)
class PermutationNullResult:
    "Observed statistics, null replicates, centered values, p values"
    stratum_family: str
    mode: str
    generation: int
    n_replicates: int
    observed: GlobalParityStats
    null_S_max: np.ndarray
    null_S_int: np.ndarray
    centered_S_max: float
    centered_S_int: float
    p_S_max: float
    p_S_int: float
    D_observed: np.ndarray
    energies: np.ndarray
    seed: int

def applicable_strata(mode: str) -> Tuple[str, ...]:
    "Return geometry null strata applicable to a mode"
    if mode == cfg.MODE_PHASON:
        return tuple(s for s in cfg.GEOMETRY_STRATA if s.startswith("bond_"))
    if mode in cfg.ONE_SITE_MODES:
        return tuple(s for s in cfg.GEOMETRY_STRATA if s.startswith("site_"))
    raise ValueError(f"unknown defect mode {mode!r}")

def permute_parity_within_strata(
    parity: np.ndarray,
    stratum_labels: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    "Shuffle parity inside each stratum, preserving parity counts."
    parity = np.asarray(parity, dtype=np.int8)
    labels = np.asarray(stratum_labels)
    valid = np.asarray(valid, dtype=bool)
    if parity.shape != labels.shape or parity.shape != valid.shape:
        raise ValueError("parity, stratum_labels, and valid must share one shape")
    out = parity.copy()
    for value in np.unique(labels):
        mask = valid & (labels == value)
        idx = np.flatnonzero(mask)
        if idx.size <= 1:
            continue
        shuffled = out[idx].copy()
        rng.shuffle(shuffled)
        # Preserve counts after shuffling
        out[idx] = shuffled
        n_even_before = int(np.count_nonzero(parity[idx] == 0))
        n_odd_before = int(np.count_nonzero(parity[idx] == 1))
        n_even_after = int(np.count_nonzero(out[idx] == 0))
        n_odd_after = int(np.count_nonzero(out[idx] == 1))
        if n_even_before != n_even_after or n_odd_before != n_odd_after:
            raise RuntimeError("stratum permutation failed to preserve even/odd counts")
    return out

def global_parity_stats(
    contrast: np.ndarray,
    energies: np.ndarray,
    *,
    e_min: float = cfg.E_MIN_EV,
    e_max: float = cfg.E_MAX_EV,
) -> GlobalParityStats:
    "Compute S_max and S_int for a D(E) curve."
    d = np.asarray(contrast, dtype=np.float64)
    e = np.asarray(energies, dtype=np.float64)
    finite = np.isfinite(d)
    if not np.any(finite):
        return GlobalParityStats(S_max=float("nan"), S_int=float("nan"))
    s_max = float(np.nanmax(np.abs(d)))
    integrated = integrate_contrast(d, e, e_min, e_max)
    return GlobalParityStats(S_max=s_max, S_int=integrated.abs)

def _two_sided_pvalue(observed: float, null_samples: np.ndarray) -> float:
    "Monte-Carlo p value for two-sided test of observed vs null samples."
    if not np.isfinite(observed):
        return float("nan")
    samples = np.asarray(null_samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan")
    # Compare absolute deviations
    null_mean = float(np.mean(samples))
    obs_dev = abs(observed - null_mean)
    null_dev = np.abs(samples - null_mean)
    return float((1 + np.count_nonzero(null_dev >= obs_dev - 1e-15)) / (samples.size + 1))

def stratified_permutation_null(
    transmission: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    stratum_family: str,
    *,
    energies: Optional[np.ndarray] = None,
    sweep_valid: Optional[np.ndarray] = None,
    n_replicates: int = cfg.PERMUTATION_REPLICATES,
    seed: int = 0,
) -> PermutationNullResult:
    "Run stratified parity permutation null for a stratum family."
    if stratum_family not in applicable_strata(mode):
        raise ValueError(f"stratum family {stratum_family!r} is not applicable to mode {mode!r}; use one of {applicable_strata(mode)}")
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")
    t = np.asarray(transmission, dtype=np.float64)
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
    labels = geometry.stratum_labels(stratum_family)
    if labels.shape[0] != parity.shape[0]:
        raise ValueError(f"stratum {stratum_family!r} length {labels.shape[0]} does not match parity axis {parity.shape[0]}")
    e_out = (
        np.asarray(energies, dtype=np.float64)
        if energies is not None
        else np.arange(t.shape[1], dtype=np.float64)
    )
    observed_contrast = parity_contrast(t, parity, valid, energies=e_out)
    observed = global_parity_stats(observed_contrast.D, e_out)
    rng = np.random.default_rng(seed)
    null_s_max = np.empty(n_replicates, dtype=np.float64)
    null_s_int = np.empty(n_replicates, dtype=np.float64)
    for b in range(n_replicates):
        # One fixed labeling for energy axis
        perm_parity = permute_parity_within_strata(parity, labels, valid, rng)
        d_b = parity_contrast(t, perm_parity, valid, energies=e_out).D
        stats_b = global_parity_stats(d_b, e_out)
        null_s_max[b] = stats_b.S_max
        null_s_int[b] = stats_b.S_int
    centered_max = float(observed.S_max - np.nanmean(null_s_max))
    centered_int = float(observed.S_int - np.nanmean(null_s_int))
    return PermutationNullResult(
        stratum_family=stratum_family,
        mode=mode,
        generation=geometry.generation,
        n_replicates=n_replicates,
        observed=observed,
        null_S_max=null_s_max,
        null_S_int=null_s_int,
        centered_S_max=centered_max,
        centered_S_int=centered_int,
        p_S_max=_two_sided_pvalue(observed.S_max, null_s_max),
        p_S_int=_two_sided_pvalue(observed.S_int, null_s_int),
        D_observed=observed_contrast.D,
        energies=e_out,
        seed=seed,
    )

def run_geometry_nulls(
    transmission: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    *,
    energies: Optional[np.ndarray] = None,
    sweep_valid: Optional[np.ndarray] = None,
    strata: Optional[Sequence[str]] = None,
    n_replicates: int = cfg.PERMUTATION_REPLICATES,
    seed: int = 0,
) -> Dict[str, PermutationNullResult]:
    "Run permutation nulls for every applicable stratum."
    families = list(strata or applicable_strata(mode))
    results: Dict[str, PermutationNullResult] = {}
    for offset, family in enumerate(families):
        results[family] = stratified_permutation_null(
            transmission,
            geometry,
            mode,
            family,
            energies=energies,
            sweep_valid=sweep_valid,
            n_replicates=n_replicates,
            seed=seed + 1009 * offset,
        )
    return results

def analyze_generation_nulls(
    raw_bundle: Dict[str, object],
    geometry: Optional[FibonacciGeometry] = None,
    *,
    modes: Optional[Sequence[str]] = None,
    strata_per_mode: Optional[Mapping[str, Sequence[str]]] = None,
    n_replicates: int = cfg.PERMUTATION_REPLICATES,
    seed: int = 0,
) -> Dict[str, object]:
    "Geometry-null analysis for all modes in raw generation bundle."
    generation = int(raw_bundle["generation"])
    geom = geometry if geometry is not None else build_geometry(generation)
    energies = np.asarray(raw_bundle["energies"], dtype=np.float64)
    modes_dict: Dict[str, Dict[str, np.ndarray]] = raw_bundle["modes"]
    modes = list(modes or modes_dict.keys())
    mode_results: Dict[str, object] = {}
    for m_idx, mode in enumerate(modes):
        sweep = modes_dict[mode]
        strata = None
        if strata_per_mode and mode in strata_per_mode:
            strata = strata_per_mode[mode]
        mode_results[mode] = run_geometry_nulls(
            sweep["T"],
            geom,
            mode,
            energies=energies,
            sweep_valid=np.asarray(sweep["valid"], dtype=bool),
            strata=strata,
            n_replicates=n_replicates,
            seed=seed + 10007 * m_idx,
        )
    return {"generation": generation, "energies": energies, "geometry_metadata": geom.to_metadata(), "n_replicates": n_replicates, "seed": seed, "modes": mode_results,}

def null_summary_table(
    analysis: Dict[str, object],
) -> List[Dict[str, object]]:
    "Flatten null results into one row per (mode, stratum)."
    rows: List[Dict[str, object]] = []
    modes: Dict[str, Dict[str, PermutationNullResult]] = analysis["modes"]
    for mode, strata in modes.items():
        for family, result in strata.items():
            rows.append(
                {
                    "mode": mode,
                    "stratum_family": family,
                    "S_max": result.observed.S_max,
                    "S_int": result.observed.S_int,
                    "centered_S_max": result.centered_S_max,
                    "centered_S_int": result.centered_S_int,
                    "p_S_max": result.p_S_max,
                    "p_S_int": result.p_S_int,
                    "n_replicates": result.n_replicates,
                }
            )
    return rows