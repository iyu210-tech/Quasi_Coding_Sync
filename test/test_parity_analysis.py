"Test parity analysis functions"

from __future__ import annotations

import json
import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.analysis_io import (
    produce_parity_analysis,
    save_parity_analysis,
)
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.parity_analysis import (
    analyze_generation_parity,
    analyze_mode_parity,
    conditional_parity_contrasts,
    integrate_contrast,
    integrate_contrast_windows,
    origin_shift_check,
    parity_checks,
    parity_contrast,
    parity_group_masks,
    position_parity,
    position_valid_mask,
    reversal_check,
)
from fibonacci_qc.raw_data import save_generation_raw, sweep_generation

def _synthetic_transmission(n_pos: int, n_energy: int) -> np.ndarray:
    p = np.arange(n_pos, dtype=np.float64)[:, np.newaxis]
    e = np.arange(n_energy, dtype=np.float64)[np.newaxis, :]
    return 0.1 * p + 0.01 * e

def test_parity_contrast_manual_even_minus_odd():
    t = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]])
    parity = np.array([0, 1, 0, 1], dtype=np.int8)
    valid = np.ones(4, dtype=bool)
    result = parity_contrast(t, parity, valid)
    np.testing.assert_allclose(result.D, [-(3.0 - 1.0), -(30.0 - 10.0)])
    assert result.n_even == 2
    assert result.n_odd == 2

def test_phason_uses_uncompacted_bond_axis():
    geom = build_geometry(5)
    n_bonds = geom.n_bonds
    n_energy = 6
    t = _synthetic_transmission(n_bonds, n_energy)
    parity = position_parity(geom, cfg.MODE_PHASON)
    valid = position_valid_mask(geom, cfg.MODE_PHASON)
    assert parity.shape == (n_bonds,)
    assert valid.shape == (n_bonds,)
    assert valid[2] is np.False_ or valid[2] == False
    even, odd = parity_group_masks(parity, valid)
    assert even[2] == False
    assert odd[2] == False
    assert np.count_nonzero(even) + np.count_nonzero(odd) == np.count_nonzero(valid)

    result = parity_contrast(t, parity, valid)
    assert result.D.shape == (n_energy,)
    assert result.n_even + result.n_odd == int(np.count_nonzero(geom.phason_valid))

def test_conditional_site_letter_strata():
    geom = build_geometry(6)
    n_energy = 4
    t = _synthetic_transmission(geom.n_sites, n_energy)
    parity = position_parity(geom, cfg.MODE_SUBSTITUTION)
    valid = position_valid_mask(geom, cfg.MODE_SUBSTITUTION)
    conditional = conditional_parity_contrasts(t, geom, cfg.MODE_SUBSTITUTION, families=["site_letter"])
    labels = geom.site_labels("site_letter")
    for letter in ("A", "B"):
        mask = valid & (labels == letter)
        even, odd = parity_group_masks(parity, mask)
        expected = parity_contrast(t, parity, mask)
        np.testing.assert_allclose(conditional["site_letter"][letter].D, expected.D,)
        assert conditional["site_letter"][letter].n_even == int(np.count_nonzero(even))
        assert conditional["site_letter"][letter].n_odd == int(np.count_nonzero(odd))

def test_origin_shift_flips_sign():
    geom = build_geometry(7)
    t = _synthetic_transmission(geom.n_sites, 8)
    parity = position_parity(geom, cfg.MODE_POTENTIAL_ONLY)
    valid = position_valid_mask(geom, cfg.MODE_POTENTIAL_ONLY)
    check = origin_shift_check(t, parity, valid)
    assert check.passed
    assert check.expected_factor == -1.0

def test_reversal_check_even_and_odd_lengths():
  t_even = np.array([[1.0], [2.0], [3.0], [4.0]])
  parity_even = np.array([0, 1, 0, 1], dtype=np.int8)
  check_even = reversal_check(t_even, parity_even, np.ones(4, dtype=bool))
  assert check_even.expected_factor == -1.0
  assert check_even.passed
  t_odd = np.array([[1.0], [2.0], [3.0]])
  parity_odd = np.array([0, 1, 0], dtype=np.int8)
  check_odd = reversal_check(t_odd, parity_odd, np.ones(3, dtype=bool))
  assert check_odd.expected_factor == 1.0
  assert check_odd.passed

def test_integrate_contrast_windows():
    energies = np.linspace(0.0, 1.0, 101)
    contrast = np.ones_like(energies)
    full = integrate_contrast(contrast, energies, 0.0, 1.0)
    assert full.signed == pytest.approx(1.0)
    assert full.abs == pytest.approx(1.0)
    assert full.rms == pytest.approx(1.0)

    windows = integrate_contrast_windows(contrast, energies, {"unit": (0.0, 1.0)})
    assert windows["unit"].signed == pytest.approx(1.0)

def test_analyze_mode_parity_from_fast_sweep():
    geom = build_geometry(5)
    energies = np.linspace(0.5, 3.0, 7)
    sweep = fast_sweep(cfg.MODE_SUBSTITUTION, energies, geometry=geom)
    analysis = analyze_mode_parity(
        sweep["T"],
        geom,
        cfg.MODE_SUBSTITUTION,
        energies=energies,
        sweep_valid=sweep["valid"],
    )
    assert analysis["unconditional"].D.shape == energies.shape
    assert analysis["checks"]["origin_shift"].passed
    assert "site_letter" in analysis["conditional"]
    assert "full" in analysis["integrated"]["unconditional"]

def test_analyze_generation_and_save_roundtrip(tmp_path):
    geom = build_geometry(4)
    energies = np.linspace(0.4, 2.5, 5)
    bundle = sweep_generation(4, energies, geometry=geom, modes=[cfg.MODE_WIDTH_ONLY, cfg.MODE_PHASON],)
    analysis = analyze_generation_parity(bundle, geom)
    out = save_parity_analysis(analysis, tmp_path / "n04" / "parity.npz")
    assert out.exists()
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["generation"] == 4
    assert cfg.MODE_WIDTH_ONLY in meta["integrated"]
    assert meta["checks"][cfg.MODE_PHASON]["origin_shift"]["passed"]

def test_parity_checks_on_real_sweep():
    geom = build_geometry(6)
    energies = np.linspace(0.6, 4.0, 9)
    for mode in cfg.DEFECT_MODES:
        sweep = fast_sweep(mode, energies, geometry=geom)
        checks = parity_checks(sweep["T"], geom, mode, sweep_valid=sweep["valid"],energies=energies,)
        assert checks["origin_shift"].passed
        assert checks["reversal"].passed

def test_produce_parity_analysis_from_saved_raw(tmp_path):
    geom = build_geometry(4)
    energies = np.linspace(0.5, 2.0, 4)
    bundle = sweep_generation(4, energies, geometry=geom, modes=[cfg.MODE_SUBSTITUTION],)
    raw_dir = tmp_path / "mini"
    save_generation_raw(bundle, raw_dir / "n04" / "substitution.npz")
    manifest = produce_parity_analysis(raw_dir, generations=[4], output_dir=tmp_path / "parity_out",)
    assert manifest["files"]
    assert (tmp_path / "parity_out" / "manifest.json").exists()

def test_parity_contrast_respects_cell_valid():
    t = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    parity = np.array([0, 1, 0, 1], dtype=np.int8)
    valid = np.ones(4, dtype=bool)
    # Drop an even cell at energy 0 so even mean uses only position 2
    cell = np.ones((4, 2), dtype=bool)
    cell[0, 0] = False
    result = parity_contrast(t, parity, valid, cell_valid=cell)
    assert result.n_even_by_energy[0] == 1
    assert result.n_even_by_energy[1] == 2
    assert result.D[0] == pytest.approx(1.0 - 0.0)
    assert result.coverage < 1.0