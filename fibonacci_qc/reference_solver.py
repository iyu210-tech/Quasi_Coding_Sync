"""After each local defect the stack is recomposed. Solver for defected transmission

Defite T_m(p,E) to be transmission under defect mode m and position p.
∆T_m(p,E) = T_m(p,E) - T_clean(E), the change relative to the clean stack
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.defects import (
    DefectedStack,
    allowed_positions,
    apply_defect,
    validate_defect_position,
)
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry
from fibonacci_qc.scattering import TransmissionResult, clean_transmission, stack_transmission

PathLike = Union[str, Path]

@dataclass(frozen=True)
class DefectTransmissionResult:
    "T_m(p,E)"

    mode: str
    position: int
    energy_ev: float
    T: float
    log_T: float
    status:str
    T_clean: float
    delta_T: float
    valid: bool
    reason: str = ""

    @property
    def is_usable(self) -> bool:
        return self.valid and np.isfinite(self.T) and np.isfinite(self.T_clean)
    
def reference_transmission(
    mode: str,
    position: int,
    energy_ev: float,
    *,
    geometry: Optional[FibonacciGeometry] = None,
    generation: Optional[int] = None,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> Tuple[TransmissionResult, DefectedStack]:
    "Recompute the stack after using defect. Return transmission result & defect stack. Invalid return T = nan"
    geom = _resolve_geometry(geometry,generation)
    defected = apply_defect(mode, geom, position)
    if not defected.valid:
        return (
            TransmissionResult(
                T=float("nan"),
                log_T = float("nan"),
                status = "invalid",
                S21=None,
            ),
            defected
        )
    result = stack_transmission(
        defected.widths_nm,
        defected.potentials_ev,
        energy_ev,
        v_lead_ev,
    )
    return result, defected

def delta_t(
    mode: str,
    position: int,
    energy_ev: float,
    *,
    geometry: Optional[FibonacciGeometry] = None,
    generation: Optional[int] = None,
    t_clean: Optional[float] = None,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> DefectTransmissionResult:
    "Compute ∆T_m(p,E)"
    geom = _resolve_geometry(geometry, generation)
    if t_clean is None:
        clean = clean_transmission(
            geom.generation,
            energy_ev,
            geometry = geom,
            v_lead_ev=v_lead_ev,
        )
        t_clean_val = clean.T
    else:
        t_clean_val = float(t_clean)
    t_result, defected = reference_transmission(mode, position, energy_ev, geometry=geom, v_lead_ev=v_lead_ev)
    if not defected.valid:
        return DefectTransmissionResult(
            mode=mode,
            position=position,
            energy_ev=float(energy_ev),
            T=float("nan"),
            log_T=float("nan"),
            status = "invalid",
            T_clean = t_clean_val,
            delta_T=float("nan"),
            valid=False,
            reason=defected.reason,
        )
    return DefectTransmissionResult(
        mode=mode,
        position=position,
        energy_ev=float(energy_ev),
        T=t_result.T,
        log_T=t_result.log_T,
        status = t_result.status,
        T_clean = t_clean_val,
        delta_T=t_result.T-t_clean_val,
        valid=True,
    )

def reference_sweep(
    mode: str,
    energies_ev: np.ndarray,
    *,
    geometry: Optional[FibonacciGeometry] = None,
    generation: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
    only_valid: bool = True,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> Dict[str, np.ndarray]:
    "Positional x sweep w/ full stack recomputation, returns the arrays. Default layout is pos axis, invalid stay as valid=False"
    geom = _resolve_geometry(geometry, generation)
    energies = np.asarray(energies_ev, dtype=np.float64)
    if positions is None:
        pos = allowed_positions(mode, geom, only_valid=only_valid)
    else:
        pos = np.asarray(list(positions), dtype=np.int64)

    n_pos = pos.size
    n_e = energies.size

    t_clean = np.empty(n_e, dtype=np.float64)
    for e_idx, energy in enumerate(energies.flat):
        t_clean.flat[e_idx] = clean_transmission(geom.generation, float(energy), geometry = geom, v_lead_ev=v_lead_ev).T
    t_arr = np.full((n_pos, n_e), np.nan,dtype=np.float64)
    log_arr = np.full((n_pos, n_e), np.nan, dtype=np.float64)
    delta_arr = np.full((n_pos, n_e), np.nan, dtype=np.float64)
    status_arr = np.empty((n_pos, n_e), dtype=object)
    valid_arr = np.zeros(n_pos, dtype=bool)

    for p_idx, p in enumerate(pos):
        ok, reason = validate_defect_position(mode, int(p), geom)
        valid_arr[p_idx] = ok
        if not ok:
            status_arr[p_idx, :] = "invalid"
            continue
        for e_idx, energy in enumerate(energies.flat):
            result = delta_t(mode, int(p), float(energy), geometry=geom, t_clean=float(t_clean.flat[e_idx]), v_lead_ev=v_lead_ev)
            t_arr[p_idx, e_idx] = result.T
            log_arr[p_idx, e_idx] = result.log_T
            delta_arr[p_idx, e_idx] = result.delta_T
            status_arr[p_idx, e_idx] = result.status

    return {
        "mode": np.array(mode),
        "generation": np.array(geom.generation, dtype=np.int64),
        "positions": pos,
        "energies": energies,
        "T": t_arr,
        "log_T": log_arr,
        "delta_T": delta_arr,
        "T_clean": t_clean,
        "valid": valid_arr,
        "status": status_arr,
    }


def save_delta_t(
    sweep: Dict[str, np.ndarray],
    path: PathLike,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    "Save reference sweep"
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_RAW / out
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {key: value for key, value in sweep.items()}
    payload["config"] = np.array(cfg.CONFIG.as_dict(), dtype=object)
    if metadata:
        payload["metadata"] = np.array(metadata, dtype=object)

    np.savez_compressed(out, **payload)
    return out

def load_delta_t(path: PathLike) -> Dict[str, np.ndarray]:
    "Load saved sweep"
    out = Path(path)
    if not out.is_absolute() and not out.exists():
        candidate = cfg.OUTPUT_RAW / out
        if candidate.exists():
            out = candidate
    with np.load(out, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}

def _resolve_geometry(
    geometry: Optional[FibonacciGeometry],
    generation: Optional[int],
) -> FibonacciGeometry:
    if geometry is not None:
        if generation is not None and geometry.generation != generation:
            raise ValueError(f"geometry.generation={geometry.generation} does not match generation = {generation}")
        return geometry
    if generation is None:
        raise ValueError("either geometry or generation is needed")
    return build_geometry(generation)