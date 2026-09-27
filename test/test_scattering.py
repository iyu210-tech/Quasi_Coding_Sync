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
    exterior_unitarity_residuals,
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
    geom = build_geometry(4) #ABA
    energy = 1.7
    s_direct = stack_s_matrix(geom.widths_nm, geom.potentials_ev, energy)

    q0 = q_number(energy, cfg.V_LEAD_EV).item()
    q_layers = q_number(energy, geom.potentials_ev)
    s_manual = interface_s_matrix(q0, q_layers[0])
    s_manual = redheffer_star(s_manual, propagation_s_matrix(q_layers[0],geom.widths_nm[0]))
    s_manual = redheffer_star(s_manual, interface_s_matrix(q_layers[0], q_layers[1]))
    s_manual = redheffer_star(s_manual, propagation_s_matrix(q_layers[1], geom.widths_nm[1]))
    s_manual = redheffer_star(s_manual, interface_s_matrix(q_layers[1], q_layers[2]))
    s_manual = redheffer_star(s_manual, propagation_s_matrix(q_layers[2], geom.widths_nm[2]))
    s_manual = redheffer_star(s_manual, interface_s_matrix(q_layers[2], q0))

    t_direct = transmission_from_s(s_direct).T
    t_manual = transmission_from_s(s_manual).T
    assert t_direct == pytest.approx(t_manual, rel=1e-12, abs=1e-12)

def test_transfer_matrix_agrees_with_s_matrix():
    geom = build_geometry(6)
    energies = np.linspace(5.1,9.5,20)
    for e in energies:
        s_mat = stack_s_matrix(geom.widths_nm, geom.potentials_ev, float(e))
        m_mat = transfer_matrix_stack(geom.widths_nm, geom.potentials_ev, float(e))
        q0 = q_number(float(e), cfg.V_LEAD_EV).item()
        t_s = transmission_from_s(s_mat).T
        t_m = transmission_from_transfer(m_mat, q0)
        assert t_s == pytest.approx(t_m, rel=1e-10, abs=1e-10)

def test_small_fibonacci_words():
    cases = {
        2: "A", 
        3: "AB", 
        4: "ABA",
        5: "ABAAB",
    }
    energies = np.linspace(0.2,9.0, 15)
    for n, word in cases.items():
        geom = build_geometry(n)
        assert geom.word == word
        for e in energies:
            result = clean_transmission(n, float(e), geometry=geom)
            assert_in_unit_interval(result.T, label=f"T(n={n}, E={e})")
            assert result.status in ("ok","underflow")

def test_clean_transmission_grid_shape_and_bounds():
    geom = build_geometry(5)
    energies = np.linspace(0.2, 5.0, 50)
    t_arr, log_arr, status = clean_transmission_grid(5, energies, geometry=geom)
    assert t_arr.shape == energies.shape
    assert log_arr.shape == energies.shape
    assert status.shape == energies.shape
    for t_val in t_arr:
        assert_in_unit_interval(float(t_val))

def test_slab_at_band_edge_is_stable():
    energy = cfg.V_A_EV
    result = stack_transmission(
        np.array([cfg.D_A_NM]),
        np.array([cfg.V_A_EV]),
        energy,
    )
    assert np.isfinite(result.T)
    assert_in_unit_interval(result.T)

def test_safe_log_t_handles_underflow_and_invalid():
    log,status=safe_log_t(0.0)
    assert np.isnan(log) and status=="invalid"
    log,status=safe_log_t(-0.1)
    assert np.isnan(log) and status=="invalid"
    log,status=safe_log_t(np.exp(-720.0))
    assert status=="ok" and log==pytest.approx(-720.0,abs=1e-9)
    log,status=safe_log_t(.5)
    assert status=="ok" and log==pytest.approx(np.log(.5))

def test_transmission_bounded_on_default_energy_grid():
    geom = build_geometry(7)
    energies = np.linspace(cfg.E_MIN_EV, cfg.E_MAX_EV, 100)
    for e in energies:
        result = clean_transmission(7, float(e), geometry=geom)
        assert_in_unit_interval(result.T)

def test_slab_s_matrix_single_layer():
    energy = 3.0
    s_mat = slab_s_matrix(energy, cfg.V_B_EV, cfg.D_B_NM)
    result = transmission_from_s(s_mat)
    expected = stack_transmission(np.array([cfg.D_B_NM]), np.array([cfg.V_B_EV]), energy)
    assert result.T == pytest.approx(expected.T, rel=1e-12)

def test_exterior_stack_unitarity_and_reciprocity():
    geom = build_geometry(5)
    for energy in (0.3, 1.7, 8.0):
        s=stack_s_matrix(geom.widths_nm, geom.potentials_ev, float(energy))
        q0 = q_number(float(energy), cfg.V_LEAD_EV).item()
        res = exterior_unitarity_residuals(s, q0)
        assert res["R_plus_T_left"] == pytest.approx(1.0, abs=1e-10)
        assert res["R_plus_T_right"] == pytest.approx(1.0, abs=1e-10)
        assert res["reciprocity"] == pytest.approx(0.0, abs=1e-10)
        assert res["unitarity_frob"] == pytest.approx(0.0, abs=1e-10)

def test_barrier_log_transmission_matches_analytical():
    energy, v0, d = 1.2, 2.0, 1.5
    numerical = stack_transmission(np.array([d]), np.array([v0]), energy)
    analytical = rectangular_barrier_transmission(energy, v0, d)
    assert numerical.T == pytest.approx(analytical, rel=1e-9, abs=1e-12)
    assert numerical.log_T == pytest.approx(np.log(analytical), rel=1e-9, abs= 1e-12)