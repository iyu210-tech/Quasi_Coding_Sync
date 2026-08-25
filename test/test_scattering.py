from __future__ import annotations

import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.scattering import (
    clean_transmission,
    clean_transmission_grid,
    interface_s_matrix,
    layer_s_matrix,
    propagation_s_matrix,
    q_number,
    redheffer_star,
    rectangular_barrier_transmission,
    safe_log_t,
    slab_s_matrix,
    stack_s_matrix,
    stack_transmission,
    transfer_matrix_stack,
    transmission_from_s,
    transmission_from_transfer,
    vacuum_passthrough_s_matrix,
)

def assert_in_unit_interval(t: float, *, label: str = "T") -> None:
    tol = cfg.TOLERANCES.t_bounds_atol
    assert np.isfinite(t), f"{label} is not finite"
    assert -tol <= t <= 1.0 + tol, f"{label}={t} outside [0, 1]"

def test_q_number_real_and_imaginary_branches():
    q_prop = q_number(1.0, 0.0).item()
    assert q_prop.real > 0.0
    assert abs(q_prop.imag) < 1e-15

    q_evan = q_number(0.5, 1.0).item()
    assert abs(q_evan.real) < 1e-15
    assert q_evan.imag > 0.0

    q_band = q_number(0.01, 0.01).item()
    assert abs(q_band) < 1e-6