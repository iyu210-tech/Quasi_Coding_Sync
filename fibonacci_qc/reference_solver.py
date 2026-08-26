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
