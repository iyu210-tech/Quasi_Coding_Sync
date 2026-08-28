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

def test_phason_swaps_left_and_right_of_bond_p():
    geom = build_geometry(5)
    defected = apply_phason(geom,0)
    assert defected.valid
    assert defected.position == 0
    np.testing.assert_array_equal(defected.letters[:2], ["B", "A"])
    assert defected.widths_nm[0] == cfg.D_B_NM
    assert defected.widths_nm[1] == cfg.D_A_NM

    invalid = apply_phason(geom,2)
    assert not invalid.valid
    np.testing.assert_array_equal(invalid.letters, geom.letters)

def test_potential_only_holds_width_fixed():
    geom = build_geometry(5)
    p=0
    defected = apply_potential_only(geom,p)
    assert defected.valid
    assert defected.letters[p] == "A"
    assert defected.widths_nm[p] == cfg.D_A_NM
    assert defected.potentials_ev[p] == cfg.V_B_EV

def test_width_only_holds_potential_fixed():
    geom = build_geometry(5)
    p=1
    defected = apply_width_only(geom,p)
    assert defected.valid
    assert defected.letters[p] == "B"
    assert defected.widths_nm[p] == cfg.D_A_NM
    assert defected.potentials_ev[p] == cfg.V_B_EV

def test_apply_defect_dispatches_all_modes():
    geom = build_geometry(4)
    for mode in cfg.DEFECT_MODES:
        p = 0 if mode != cfg.MODE_PHASON else 0
        result = apply_defect(mode, geom, p)
        assert result.mode == mode
        assert result.valid #ABA begin w/ AB, so phason at 0 valid

def test_phason_position_is_left_site_and_mask_uncompacted():
    geom = build_geometry(5)
    ok, _ = validate_defect_position(cfg.MODE_PHASON, 2, geom)
    assert not ok
    assert geom.bond_index[2]==2
    assert geom.phason_valid[2] is np.False_ or geom.phason_valid[2] == False

    valid_only = allowed_positions(cfg.MODE_PHASON, geom, only_valid=True)
    all_bonds = allowed_positions(cfg.MODE_PHASON, geom, only_valid=False)
    np.testing.assert_array_equal(all_bonds, [0,1,2,3])
    np.testing.assert_array_equal(valid_only, [0,1,3])
    assert 2 not in valid_only

def test_reference_solver_matches_manual_full_recompute():
    geom = build_geometry(5)
    energy = 1.5
    mode = cfg.MODE_SUBSTITUTION
    p=2
    defected = apply_substitution(geom,p)
    manual = stack_transmission(defected.widths_nm, defected.potentials_ev, energy)
    result, stack = reference_transmission(mode, p, energy, geometry = geom)
    assert stack.valid
    assert result.T == pytest.approx(manual.T, rel = 1e-12, abs = 1e-14)

def test_delta_t_equals_tm_minus_tclean():
    geom = build_geometry(4)
    energy = 2.3
    clean = clean_transmission(4,energy, geometry = geom)

    for mode in cfg.ONE_SITE_MODES:
        for p in range(geom.n_sites):
            out = delta_t(mode, p, energy, geometry=geom, t_clean=clean.T)
            assert out.valid
            assert out.delta_T == pytest.approx(out.T - clean.T, abs = 1e-14)
            assert out.T_clean == pytest.approx(clean.T)
    for p in allowed_positions(cfg.MODE_PHASON, geom):
        out = delta_t(cfg.MODE_PHASON, int(p), energy, geometry = geom, t_clean=clean.T)
        assert out.valid
        assert out.delta_T == pytest.approx(out.T - clean.T, abs = 1e-14)

def test_invalid_phason_delta_t_is_nan():
    geom = build_geometry(5)
    out = delta_t(cfg.MODE_PHASON, 2, 1.0, geometry=geom)
    assert not out.valid
    assert np.isnan(out.T)
    assert np.isnan(out.delta_T)

def test_reference_sweep_shapes_and_save_load(tmp_path):
    geom = build_geometry(4)
    energies = np.linspace(0.5,3.0, 5)
    sweep = reference_sweep(cfg.MODE_SUBSTITUTION, energies, geometry=geom)
    n_pos = geom.n_sites
    assert sweep["T"].shape == (n_pos, 5)
    assert sweep["delta_T"].shape == (n_pos, 5)
    assert sweep["T_clean"].shape == (5,)
    assert sweep["valid"].shape == (n_pos,)
    assert np.all(sweep["valid"])

    #one consistency check for ∆T
    p_idx = 1
    e_idx = 2
    assert sweep["delta_T"][p_idx, e_idx] == pytest.approx(sweep["T"][p_idx, e_idx]-sweep["T_clean"][e_idx])

    path = save_delta_t(sweep, tmp_path / "delta_sub_n4.npz", metadata={"note": "defect-test"},)
    assert path.exists()
    loaded = load_delta_t(path)
    np.testing.assert_allclose(loaded["delta_T"], sweep["delta_T"])
    np.testing.assert_array_equal(loaded["positions"], sweep["positions"])

def test_transmission_stays_in_unit_interval_for_small_defects():
    geom = build_geometry(5)
    energies = np.linspace(0.2,8.0,12)
    for mode in cfg.DEFECT_MODES:
        positions = allowed_positions(mode, geom, only_valid=True)
        for p in positions:
            for e in energies:
                result, defected = reference_transmission(mode, int(p), float(e), geometry=geom)
                assert defected.valid
                assert np.isfinite(result.T)
                assert -cfg.TOLERANCES.t_bounds_atol <=result.T <= 1.0 + cfg.TOLERANCES.t_bounds_atol

def test_substitution_equals_combined_letter_remap():
    "Sub at p must match remapping that site to other letter."
    geom = build_geometry(6)
    p=3
    defected = apply_substitution(geom,p)
    assert defected.letters[p] == ("B" if geom.letters[p] == "A" else "A")
    assert defected.widths_nm[p] == cfg.LAYER_WIDTH_NM[str(defected.letters[p])]
    assert defected.potentials_ev[p] == cfg.LAYER_POTENTIAL_EV[str(defected.letters[p])]