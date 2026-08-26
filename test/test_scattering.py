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

def test_vacuum_passthrough_has_unit_transmission():
    s_vac = vacuum_passthrough_s_matrix()
    result = transmission_from_s(s_vac)
    assert result.T == pytest.approx(1.0, abs=1e-12)
    assert result.status =="ok"

def test_uniform_zero_potential_layers_have_unit_transmission():
    energies = np.linspace(0.2, 9.0, 25)
    widths = np.array([1.0,2.0,1.5,3.0])
    zeros = np.zeros_like(widths)
    for energy in energies: 
        result = stack_transmission(widths, zeros, float(energy))
        assert result.T == pytest.approx(1.0, abs=1e-10)
        assert result.status == "ok"

def test_single_barrier_matches_analytical_formula():
    barrier_height = 2.0
    width = 1.5
    energies = np.linspace(0.2,8.0,40)
    for e in energies:
        energy = float(e)
        numerical = stack_transmission(
            np.array([width]),
            np.array([barrier_height]),
            energy,
        ).T
        analytical = rectangular_barrier_transmission(energy, barrier_height, width)
        assert numerical == pytest.approx(analytical, rel=1e-9, abs=1e-10)
        assert_in_unit_interval(numerical)

def test_redheffer_matches_direct_two_layer_stack():
    geom = bulid_geometry(4) #ABA
    energy = 1.7
    s_direct = stack_s_matrix(geom.widths, geom.potentials, energy)

    q0 = q_number(energy, cfg.V_LEAD_EV).item()
    q_layers = q_number(energy, geom.potentials)
    s_manual = interface_s_matrix(q0, q_layers[0])
    s_manual = redheffer_star(s_manual, propagation_s_matrix(q_layers[0],geom.widths_nm[0]))
    s_manual = redheffer_star(s_manual, interface_s_matrix(q_layers[0], q_layers[1]))
    s_manual = redheffer_star(s_manual, propagation_s_matrix(q_layers[1], geom.widths_nm[1]))
    s_manual = redheffer_star(s_manual, interface_s_matrix(q_layers[1], q_layers[2]))
    s