"Test geometry nulls and numerical robustness"

from __future__ import annotations

import json
import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.geometry_nulls import (
    applicable_strata,
    global_parity_stats,
    permute_parity_within_strata,
    run_geometry_nulls,
    stratified_permutation_null,
)
from fibonacci_qc.nulls_io import (
    produce_nulls_analysis,
    produce_robustness_analysis,
    save_nulls_analysis,
    save_robustness_analysis,
)
from fibonacci_qc.parity_analysis import position_parity, position_valid_mask
from fibonacci_qc.raw_data import save_generation_raw, sweep_generation
from fibonacci_qc.robustness import (
    compare_parity_contrast,
    finer_energy_grid,
    interpolate_to_grid,
    robustness_check_mode,
)

def test_applicable_strata_split_by_mode():
    site = applicable_strata(cfg.MODE_SUBSTITUTION)
    bond = applicable_strata(cfg.MODE_PHASON)
    assert all(s.startswith("site_") for s in site)
    assert all(s.startswith("bond_") for s in bond)
    assert "site_letter" in site
    assert "bond_pair" in bond
    assert "bond_6" not in bond

def test_permute_preserves_even_odd_counts_within_strata():
    geom = build_geometry(7)
    parity = position_parity(geom, cfg.MODE_SUBSTITUTION)
    valid = position_valid_mask(geom, cfg.MODE_SUBSTITUTION)
    labels = geom.stratum_labels("site_letter")
    rng = np.random.default_rng(42)
    perm = permute_parity_within_strata(parity, labels, valid, rng)

    for letter in np.unique(labels):
        mask = valid & (labels == letter)
        assert np.count_nonzero(parity[mask] == 0) == np.count_nonzero(perm[mask] == 0)
        assert np.count_nonzero(parity[mask] == 1) == np.count_nonzero(perm[mask] == 1)
    # Invalid positions
    assert perm.shape == parity.shape

def test_permute_phason_ignores_invalid_bonds():
    geom = build_geometry(5)
    parity = position_parity(geom, cfg.MODE_PHASON)
    valid = position_valid_mask(geom, cfg.MODE_PHASON)
    labels = geom.stratum_labels("bond_pair")
    rng = np.random.default_rng(7)
    perm = permute_parity_within_strata(parity, labels, valid, rng)
    # AA bond at index 2 stays out of pool
    assert valid[2] is np.False_ or valid[2] == False
    assert perm[2] == parity[2]

def test_global_parity_stats_on_known_curve():
    energies = np.linspace(0.0, 1.0, 11)
    contrast = np.full_like(energies, 2.0)
    stats = global_parity_stats(contrast, energies, e_min=0.0, e_max=1.0)
    assert stats.S_max == pytest.approx(2.0)
    assert stats.S_int == pytest.approx(2.0)

def test_global_parity_stats_common_energy_mask():
    energies = np.array([0.0, 0.5, 1.0, 2.0])
    contrast = np.array([10.0,1.0,1.0,10.0]) # peaks outside [0.5, 1.0]
    stats = global_parity_stats(contrast, energies, e_min=0.5, e_max=1.0)
    assert stats.S_max == pytest.approx(1.0)

def test_stratified_null_uses_fixed_labeling_and_returns_centered_stats():
    geom = build_geometry(5)
    energies = np.linspace(0.5, 3.0, 10)
    sweep = fast_sweep(cfg.MODE_SUBSTITUTION, energies, geometry=geom)
    result = stratified_permutation_null(
        sweep["T"],
        geom,
        cfg.MODE_SUBSTITUTION,
        "site_letter",
        energies=energies,
        sweep_valid=sweep["valid"],
        n_replicates=24,
        seed=1,
    )
    assert result.n_replicates == 24
    assert result.null_S_max.shape == (24,)
    assert result.null_S_int.shape == (24,)
    assert result.centered_S_max == pytest.approx(result.observed.S_max - float(np.mean(result.null_S_max)))
    assert result.centered_S_int == pytest.approx(result.observed.S_int - float(np.mean(result.null_S_int)))
    assert 0.0 < result.p_S_max <= 1.0
    assert 0.0 < result.p_S_int <= 1.0
    assert result.D_observed.shape == energies.shape

def test_run_geometry_nulls_all_applicable_strata():
    geom = build_geometry(5)
    energies = np.linspace(0.4, 2.0, 8)
    sweep = fast_sweep(cfg.MODE_POTENTIAL_ONLY, energies, geometry=geom)
    results = run_geometry_nulls(
        sweep["T"],
        geom,
        cfg.MODE_POTENTIAL_ONLY,
        energies=energies,
        sweep_valid=sweep["valid"],
        n_replicates=8,
        seed=0,
    )
    assert set(results) == set(applicable_strata(cfg.MODE_POTENTIAL_ONLY))

def test_finer_grid_and_interpolate():
    fine = finer_energy_grid(0.0, 1.0, n_points=21)
    assert fine.size == 21
    base = np.linspace(0.0, 1.0, 11)
    values = base ** 2
    interp = interpolate_to_grid(values, base, fine)
    assert interp.shape == fine.shape
    np.testing.assert_allclose(interp[::2], values, atol=1e-12)

def test_compare_parity_contrast_identical_grids():
    energies = np.linspace(0.1, 2.0, 20)
    d = np.sin(3 * energies)
    cmp = compare_parity_contrast(d, energies, d, energies)
    assert cmp.max_abs_D == pytest.approx(0.0, abs=1e-12)
    assert cmp.rmse_D == pytest.approx(0.0, abs=1e-12)
    assert cmp.integrated_abs_diff["full"]["abs"] == pytest.approx(0.0, abs=1e-10)

def test_robustness_check_mode_small_grids():
    geom = build_geometry(4)
    e_base = np.linspace(0.5, 2.5, 16)
    e_fine = np.linspace(0.5, 2.5, 31)
    result = robustness_check_mode(
        geom,
        cfg.MODE_WIDTH_ONLY,
        energies_baseline=e_base,
        energies_fine=e_fine,
        positions=[0, 1],
    )
    assert result["n_energy_baseline"] == 16
    assert result["n_energy_fine"] == 31
    assert np.isfinite(result["contrast"].rmse_D)
    assert len(result["resonance"]) >= 1

def test_produce_nulls_and_robustness_io(tmp_path):
    geom = build_geometry(4)
    energies = np.linspace(0.5, 2.0, 8)
    modes = [cfg.MODE_SUBSTITUTION, cfg.MODE_PHASON]
    bundle = sweep_generation(4, energies, geometry=geom, modes=modes)
    raw_dir = tmp_path / "raw"
    for mode in modes:
        mode_bundle = {
            "generation": 4,
            "energies": bundle["energies"],
            "T_clean": bundle["T_clean"],
            "log_T_clean": bundle["log_T_clean"],
            "clean_status": bundle["clean_status"],
            "modes": {mode: bundle["modes"][mode]},
            "geometry_metadata": bundle["geometry_metadata"],
        }
        save_generation_raw(mode_bundle, raw_dir / "n04" / f"{mode}.npz")
    manifest = produce_nulls_analysis(
        raw_dir,
        generations=[4],
        modes=modes,
        n_replicates=6,
        seed=3,
        output_dir=tmp_path / "nulls",
    )
    assert manifest["files"]
    assert (tmp_path / "nulls" / "n04" / "nulls.npz").exists()
    meta = json.loads((tmp_path / "nulls" / "n04" / "nulls.meta.json").read_text())
    assert meta["summary"]

    from fibonacci_qc.geometry_nulls import analyze_generation_nulls

    analysis = analyze_generation_nulls(bundle, geom, modes=modes, n_replicates=4, seed=0)
    save_nulls_analysis(analysis, tmp_path / "direct_nulls.npz")

    rob_manifest = produce_robustness_analysis(
        generations=[4],
        modes=[cfg.MODE_SUBSTITUTION],
        energies_baseline=np.linspace(0.5, 2.0, 10),
        energies_fine=np.linspace(0.5, 2.0, 19),
        output_dir=tmp_path / "robustness",
    )
    assert rob_manifest["files"]
    assert (tmp_path / "robustness" / "n04" / "robustness.npz").exists()