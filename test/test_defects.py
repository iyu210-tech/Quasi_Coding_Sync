"Test for defect transformations"

from __future__ import annotations

import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.defects import (
    allowed_positions,
    apply_defect,
    apply_phason,
    apply_potential_only,
    apply_substitution,
    apply_width_only,
    validate_defect_position,
)
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.reference_solver import (
    delta_t,
    load_delta_t,
    reference_sweep,
    reference_transmission,
    save_delta_t,
)
from fibonacci_qc.scattering import clean_transmission, stack_transmission

def test_substitution_swaps_letter_width_and_potential():
    geom = build_geometry(5) #ABAAB
    defected = apply_substitution(geom, 1)
    assert defected.valid
    assert defected.letters[1] == "A"
    assert defected.widths_nm[1] == cfg.D_A_NM
    assert defected.potentials_ev[1] == cfg.V_A_EV
    #unchanged keep original values.
    assert defected.letters[0] == "A"
    assert defected.widths_nm[0] == cfg.D_A_NM

#continue