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

def _masked_mean_along_position(
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
) -> ParityContrastResult: #unfinished