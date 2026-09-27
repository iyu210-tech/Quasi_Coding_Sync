"Test for finite-size scaling and comparisons of paired exponents"

from __future__ import annotations

import json
import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import fibonacci_number
from fibonacci_qc.raw_data import save_generation_raw, sweep_generation
from fibonacci_qc.scaling import (
    DIRECTION_DECREASING,
    DIRECTION_FLAT,
    DIRECTION_INCREASING,
    analyze_scaling,
    classify_direction,
    collect_parity_curves_from_bundles,
    common_support_mask,
    fit_loglog_scaling,
    fit_scaling_curve,
    generations_in_window,
    assess_fit_stability,
    paired_alpha_contrast,
)
from fibonacci_qc.scaling_io import produce_scaling_analysis, save_scaling_analysis

def test_classify_direction_thresholds():
    assert classify_direction(0.0) == DIRECTION_FLAT
    assert classify_direction(0.05) == DIRECTION_FLAT
    assert classify_direction(-0.05) == DIRECTION_FLAT
    assert classify_direction(-0.06) == DIRECTION_INCREASING
    assert classify_direction(0.06) == DIRECTION_DECREASING

def test_generations_in_locked_windows():
    assert generations_in_window((10, 15)) == tuple(range(10, 16))
    assert cfg.FIT_GENERATION_WINDOWS == ((10, 15), (10, 17), (10, 20))

def test_fit_loglog_recovers_known_alpha():
    # |D| = C * N^{-alpha} with alpha = 0.4, C = 2
    sizes = np.array([8.0, 13.0, 21.0, 34.0, 55.0])
    alpha_true = 0.4
    c_true = 2.0
    abs_d = c_true * sizes ** (-alpha_true)
    alpha, intercept, rmse, n_usable, support = fit_loglog_scaling(sizes, abs_d)
    assert support
    assert n_usable == 5
    assert alpha == pytest.approx(alpha_true, rel=1e-10)
    assert intercept == pytest.approx(np.log(c_true), rel=1e-10)
    assert rmse == pytest.approx(0.0, abs=1e-12)

def test_fit_increasing_contrast_negative_alpha():
    sizes = np.array([5.0, 8.0, 13.0, 21.0])
    abs_d = 0.1 * sizes ** (0.3)  # alpha_eff = -0.3
    alpha, _, _, _, support = fit_loglog_scaling(sizes, abs_d)
    assert support
    assert alpha == pytest.approx(-0.3, rel=1e-10)
    assert classify_direction(alpha) == DIRECTION_INCREASING

def test_stability_requires_loo_and_rmse():
    stable,indeterminate=assess_fit_stability(.2,.01,(.19,.21,.20),(True,True,True),DIRECTION_DECREASING)
    assert stable and not indeterminate
    stable,indeterminate=assess_fit_stability(.2,.01,(.19,.4,.2),(True,True,True),DIRECTION_DECREASING)
    assert not stable and not indeterminate
    stable,indeterminate=assess_fit_stability(.2,cfg.SCALING_RMSE_MAX+.1,(.2,.2),(True,True),DIRECTION_DECREASING)
    assert not stable and not indeterminate
    stable,indeterminate=assess_fit_stability(.2,.01,(float("nan"),.2),(True,True),DIRECTION_DECREASING)
    assert not stable and indeterminate
    stable,indeterminate=assess_fit_stability(.2,.01,(),(),DIRECTION_DECREASING)
    assert not stable and indeterminate

def test_fit_scaling_curve_shapes_and_loo():
    gens = (4, 5, 6, 7)
    sizes = np.array([fibonacci_number(g) for g in gens], dtype=np.float64)
    energies = np.linspace(0.5, 2.0, 6)
    # Power law at every energy w/ mild E-dependent alpha
    alphas_true = 0.2 + 0.01 * np.arange(energies.size)
    # sizes**(-a) for each energy column
    d_mat = np.empty((len(gens), energies.size), dtype=np.float64)
    for e_idx, a in enumerate(alphas_true):
        d_mat[:, e_idx] = 2.0 * sizes ** (-a)
    result = fit_scaling_curve(sizes, d_mat, energies, gens, mode=cfg.MODE_SUBSTITUTION, window=(4, 7))
    assert result.alpha.shape == energies.shape
    assert result.loo_alpha.shape == (energies.size, len(gens))
    assert np.all(result.support)
    assert np.all(result.stable)
    np.testing.assert_allclose(result.alpha, alphas_true, rtol=1e-8)
    assert result.direction[0] == DIRECTION_DECREASING

def test_common_support_and_paired_contrast():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    sa = np.array([True, True, False, True])
    sb = np.array([True, False, True, True])
    common = common_support_mask(sa, sb)
    np.testing.assert_array_equal(common, [True, False, False, True])
    pair = paired_alpha_contrast(a, b, sa, sb)
    delta, support = pair["delta_alpha"], pair["support"]
    np.testing.assert_array_equal(support, common)
    assert delta[0] == pytest.approx(0.5)
    assert np.isnan(delta[1])
    assert delta[3] == pytest.approx(0.5)

def test_collect_from_bundles_and_analyze_scaling():
    energies = np.linspace(0.5, 1.5, 5)
    gens = (5, 6, 7, 8)
    sizes = np.array([fibonacci_number(g) for g in gens], dtype=np.float64)
    # Synthetic |D| with alpha = 0.25
    d = 3.0 * sizes[:, None] ** (-0.25) * np.ones((len(gens), energies.size))
    bundles = {
        g: {"energies": energies, "D_by_mode": {cfg.MODE_SUBSTITUTION: d[i]}}
        for i, g in enumerate(gens)
    }
    sizes_out, d_out, e_out, gens_out = collect_parity_curves_from_bundles(
        cfg.MODE_SUBSTITUTION, bundles
    )
    np.testing.assert_array_equal(gens_out, gens)
    np.testing.assert_allclose(sizes_out, sizes)
    np.testing.assert_allclose(d_out, d)

    # Synthetic D(E) curves for 2 modes (known alpha = 0.25 and 0.10)
    bundles_sub = {g: {"energies": energies, "D_by_mode": {cfg.MODE_SUBSTITUTION: d[i]}} for i, g in enumerate(gens)}
    bundles_width = {
        g: {"energies": energies, "D_by_mode": {cfg.MODE_WIDTH_ONLY: (1.0 * sizes[i] ** (-0.10)) * np.ones(energies.size)}}
        for i, g in enumerate(gens)
    }

    # Fit modes separately, compare alpha curves
    from fibonacci_qc.scaling import analyze_mode_scaling

    fit_sub = analyze_mode_scaling(cfg.MODE_SUBSTITUTION, window=(5, 8), bundles_by_generation=bundles_sub)
    fit_width = analyze_mode_scaling(cfg.MODE_WIDTH_ONLY, window=(5, 8), bundles_by_generation=bundles_width)
    assert np.all(fit_sub.support)
    np.testing.assert_allclose(fit_sub.alpha, 0.25, rtol=1e-8)
    np.testing.assert_allclose(fit_width.alpha, 0.10, rtol=1e-8)
    pair = paired_alpha_contrast(fit_sub.alpha, fit_width.alpha, fit_sub.support, fit_width.support)
    delta, support = pair["delta_alpha"], pair["support"]
    assert np.all(support)
    np.testing.assert_allclose(delta, 0.15, rtol=1e-8)

def test_produce_scaling_from_raw(tmp_path):
    energies = np.linspace(0.5, 2.0, 6)
    gens = [4, 5, 6, 7]
    modes = [cfg.MODE_SUBSTITUTION, cfg.MODE_WIDTH_ONLY]
    raw_dir = tmp_path / "raw"
    for gen in gens:
        geom_bundle = sweep_generation(gen, energies, modes=modes)
        for mode in modes:
            mode_bundle = {
                "generation": gen,
                "energies": geom_bundle["energies"],
                "T_clean": geom_bundle["T_clean"],
                "log_T_clean": geom_bundle["log_T_clean"],
                "clean_status": geom_bundle["clean_status"],
                "modes": {mode: geom_bundle["modes"][mode]},
                "geometry_metadata": geom_bundle["geometry_metadata"],
            }
            save_generation_raw(mode_bundle, raw_dir / f"n{gen:02d}" / f"{mode}.npz")

    manifest = produce_scaling_analysis(
        raw_dir,
        generations=gens,
        modes=modes,
        windows=[(4, 7)],
        energies=energies,
        output_dir=tmp_path / "scaling",
    )
    assert manifest["files"]
    assert (tmp_path / "scaling" / "scaling.npz").exists()
    assert (tmp_path / "scaling" / "manifest.json").exists()
    meta = json.loads((tmp_path / "scaling" / "scaling.meta.json").read_text())
    assert meta["summary"]
    assert "4-7" in meta["windows"]
    assert "sub-width" in meta["windows"]["4-7"]["paired"]
    assert any(row["window"] == "4-7" for row in manifest["summary"])

    # Direct save of in-memory analysis.
    analysis = analyze_scaling(
        modes=modes,
        windows=[(4, 7)],
        energies=energies,
        bundles_by_generation={
            gen: {
                "energies": energies,
                "modes": sweep_generation(gen, energies, modes=modes)["modes"],
            }
            for gen in gens
        },
    )
    save_scaling_analysis(analysis, tmp_path / "direct_scaling.npz")

def test_config_scaling_constants_locked():
    assert cfg.SCALING_MIN_USABLE_GENERATIONS == 3
    assert cfg.SCALING_LOO_ALPHA_TOL == pytest.approx(0.05)
    assert cfg.SCALING_RMSE_MAX == pytest.approx(0.5)
    assert cfg.ALPHA_FLAT_THRESHOLD == pytest.approx(0.05)
    assert "scaling_rmse_max" in cfg.CONFIG.as_dict()