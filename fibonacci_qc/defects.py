"""
Four defect types
1. Substitution (A becomes B or vice versa) at site p (potential and width both)
2. Phason (AB becomes BA or vice versa) at bond with left site p (potential and width both)
3. Potential Only (V_A becomes V_B or vice versa) at site p (width fixed)
4. Width Only (D_A becomes D_B or vice versa) at site p (potential fixed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry

@dataclass(frozen=True)
class DefectedStack:
    "Physical stack after one local defect. letters variable track A/B sequence after substitution or phason."

    mode: str
    position: int
    letters: np.ndarray
    widths_nm: np.ndarray
    potentials_ev: np.ndarray
    valid: bool
    reason: str = ""

    @property
    def n_sites(self) -> int:
        return int(self.letters.size)

def _swap_ab_letter(letter:str) -> str:
    if letter == "A":
        return "B"
    elif letter == "B":
        return "A"
    else:
        raise ValueError(f"Invalid letter {letter}, expected A or B.")

def _other_potential(potential_ev: float) -> float:
    "Swap potentials"
    if abs(potential_ev - cfg.V_A_EV) <= abs(potential_ev - cfg.V_B_EV):
        return cfg.V_B_EV
    return cfg.V_A_EV

def _other_width(width_nm: float) -> float:
    "Swap widths"
    if abs(width_nm - cfg.D_A_NM) <= abs(width_nm - cfg.D_B_NM):
        return cfg.D_B_NM
    return cfg.D_A_NM

def validate_defect_position(mode: str, position: int, geom: FibonacciGeometry) -> Tuple[bool, str]:
    "Returns (valid, reason) for defect position in geometry"
    if mode not in cfg.DEFECT_MODES:
        return False, f"Unknown defect mode {mode!r}"

    if mode in cfg.ONE_SITE_MODES:
        if position < 0 or position >= geom.n_sites:
            return False, f"Position {position} out of bounds for n_sites={geom.n_sites}"
        return True, ""

    #Phason p left of the bond (p, p+1)
    if position < 0 or position >= geom.n_bonds:
        return False, (f"Position {position} out of bounds for n_bonds={geom.n_bonds}")
    if not bool(geom.phason_valid[position]):
        pair = str(geom.bond_pairs[position])
        return False, (f"Phason defect not allowed at bond {position} with pair {pair}")
    
    return True, ""

def apply_substitution(geometry: FibonacciGeometry, p: int) -> DefectedStack:
    "Substitution at site p"
    ok, reason = validate_defect_position(cfg.MODE_SUBSTITUTION, p, geometry)
    if not ok:
        return DefectedStack(
            mode = cfg.MODE_SUBSTITUTION,
            position = p,
            letters=geometry.letters.copy(),
            widths_nm=geometry.widths_nm.copy(),
            potentials_ev=geometry.potentials_ev.copy(),
            valid=False,
            reason=reason,
        )

    letters = geometry.letters.copy()
    widths = geometry.widths_nm.copy()
    pots = geometry.potentials_ev.copy()

    new_letter = _swap_ab_letter(str(letters[p]))
    letters[p] = new_letter
    widths[p] = cfg.LAYER_WIDTH_NM[new_letter]
    pots[p] = cfg.LAYER_POTENTIAL_EV[new_letter]

    return DefectedStack(
        mode = cfg.MODE_SUBSTITUTION,
        position = p,
        letters=letters,
        widths_nm=widths,
        potentials_ev=pots,
        valid=True,
    )

def apply_phason(geometry: FibonacciGeometry, p: int) -> DefectedStack:
    "Swap layers at bond (p, p+1). Phason defect."
    ok, reason = validate_defect_position(cfg.MODE_PHASON, p, geometry)
    if not ok:
        return DefectedStack(
            mode = cfg.MODE_PHASON,
            position = p,
            letters=geometry.letters.copy(),
            widths_nm=geometry.widths_nm.copy(),
            potentials_ev=geometry.potentials_ev.copy(),
            valid=False,
            reason=reason,
        )
    letters = geometry.letters.copy()
    widths = geometry.widths_nm.copy()
    pots = geometry.potentials_ev.copy()

    letters[p], letters[p+1] = letters[p+1], letters[p]
    widths[p], widths[p+1] = widths[p+1], widths[p]
    pots[p], pots[p+1] = pots[p+1], pots[p]

    return DefectedStack(
        mode = cfg.MODE_PHASON,
        position = p,
        letters=letters,
        widths_nm=widths,
        potentials_ev=pots,
        valid=True,
    )

def apply_potential_only(geometry: FibonacciGeometry, p: int) -> DefectedStack:
    "Potential Only applied"
    ok, reason = validate_defect_position(cfg.MODE_POTENTIAL_ONLY, p, geometry)
    if not ok:
        return DefectedStack(
            mode = cfg.MODE_POTENTIAL_ONLY,
            position = p,
            letters=geometry.letters.copy(),
            widths_nm=geometry.widths_nm.copy(),
            potentials_ev=geometry.potentials_ev.copy(),
            valid=False,
            reason=reason,
        )

    letters = geometry.letters.copy()
    widths = geometry.widths_nm.copy()
    pots = geometry.potentials_ev.copy()
    pots[p] = _other_potential(pots[p])

    return DefectedStack(
        mode = cfg.MODE_POTENTIAL_ONLY,
        position = p,
        letters=letters,
        widths_nm=widths,
        potentials_ev=pots,
        valid=True,
    )

def apply_width_only(geometry: FibonacciGeometry, p: int) -> DefectedStack:
    "Width Only applied"
    ok, reason = validate_defect_position(cfg.MODE_WIDTH_ONLY, p, geometry)
    if not ok:
        return DefectedStack(
            mode = cfg.MODE_WIDTH_ONLY,
            position = p,
            letters=geometry.letters.copy(),
            widths_nm=geometry.widths_nm.copy(),
            potentials_ev=geometry.potentials_ev.copy(),
            valid=False,
            reason=reason,
        )

    letters = geometry.letters.copy()
    widths = geometry.widths_nm.copy()
    pots = geometry.potentials_ev.copy()
    widths[p] = _other_width(widths[p])

    return DefectedStack(
        mode = cfg.MODE_WIDTH_ONLY,
        position = p,
        letters=letters,
        widths_nm=widths,
        potentials_ev=pots,
        valid=True,
    )

_APPLIERS = {cfg.MODE_SUBSTITUTION: apply_substitution,
             cfg.MODE_PHASON: apply_phason,
             cfg.MODE_POTENTIAL_ONLY: apply_potential_only,
             cfg.MODE_WIDTH_ONLY: apply_width_only}

def apply_defect(
    mode: str,
    geometry: FibonacciGeometry,
    p: int,
) -> DefectedStack:
    "Apply defect of given mode at position p in geometry"
    if mode not in _APPLIERS:
        raise ValueError(f"Unknown defect mode {mode!r}")
    return _APPLIERS[mode](geometry, p)

def allowed_positions(
    mode: str,
    geometry: FibonacciGeometry,
    *,
    only_valid: bool = True,
) -> np.ndarray:
    "Return allowed indices p for a given mode. For one site modes this is 0 to N-1, for phason it is 0 to N-2"
    if mode in cfg.ONE_SITE_MODES:
        return np.arange(geometry.n_sites, dtype = np.int64)

    if mode == cfg.MODE_PHASON:
        indices = np.arange(geometry.n_bonds, dtype = np.int64)
        if only_valid:
            return indices[geometry.phason_valid]
        return indices
    raise ValueError(f"unknown defect mode {mode!r}")
