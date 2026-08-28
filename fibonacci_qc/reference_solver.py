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
    t_result, defected = reference_transmission(
        mode, position, energy_ev, geometry=geom, v_lead_ev=v_lead_ev,
    )
    if not defected.valid:
        return DefectTransmissionResult(
            #add inputs and finish def delta_t
        )

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