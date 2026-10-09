"Tests for the newest round of fixes"

from __future__ import annotations

from dataclasses import replace
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.fast_solver import _resolve_geometry, fast_sweep
from fibonacci_qc.scattering import clean_transmission, stack_transmission
from fibonacci_qc.parity_analysis import parity_contrast
from fibonacci_qc.geometry_nulls import (
    geometry_expected_contrast,
    stratified_permutation_null,
    permute_parity_within_strata,
    analyze_generation_nulls,
)
from fibonacci_qc.nulls_io import save_nulls_analysis
from fibonacci_qc.resonance import Peak, chain_peaks_across_generations, DEFAULT_MAX_MATCH_DELTA_E_EV
from fibonacci_qc.cross_generation import compare_clean_anchor_grids
from fibonacci_qc.raw_data import produce_raw_dataset
from fibonacci_qc.resonance_io import produce_resonance_analysis
import fibonacci_qc.resolution as resolution

def test_geometry_dispatch_all_supported_forms():
    geom = build_geometry(4)
    assert _resolve_geometry(geom, None) is geom
    assert _resolve_geometry(geom, 4) is geom
    assert _resolve_geometry(None, 4).word == "ABA"
    with pytest.raises(ValueError):
        _resolve_geometry(geom, 5)
    with pytest.raises(ValueError):
        _resolve_geometry(None, None)

def test_clean_transmission_does_not_discard_supplied_geometry():
    geom = build_geometry(4)
    changed = replace(geom, widths_nm=np.array([1.1, 2.2, 1.3]))
    direct = stack_transmission(changed.widths_nm, changed.potentials_ev, 8.)
    actual = clean_transmission(4, 8., geometry=changed)
    assert actual.log_T == pytest.approx(direct.log_T, abs=1e-12)
    with pytest.raises(ValueError):
        clean_transmission(5, 8., geometry=changed)

def test_geometry_expectation_matches_exhaustive_stratified_relabelings():
    g = build_geometry(5)
    t = np.array([[.15, .22], [.3, .4], [.55, .52], [.7, .65], [.9, .82]])
    labels = g.letters
    valid = np.ones(5, dtype=bool)
    pieces = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        count = np.count_nonzero(g.site_parity[idx] == 0)
        pieces.append((idx, list(itertools.combinations(idx, count))))
    curves = []
    for assignments in itertools.product(*(v[1] for v in pieces)):
        parity = np.ones(5, dtype=np.int8)
        for evens in assignments:
            parity[list(evens)] = 0
        curves.append(parity_contrast(t, parity, valid).D)
    actual = geometry_expected_contrast(t, g.site_parity, labels, valid)
    np.testing.assert_allclose(actual, np.mean(curves, axis=0), atol=1e-15)

def test_geometry_center_is_invariant_to_within_stratum_shuffling():
    g = build_geometry(10)
    t = np.random.default_rng(1).random((g.n_sites, 3))
    valid = np.ones(g.n_sites, dtype=bool)
    labels = g.stratum_labels("site_3")
    reference = geometry_expected_contrast(t, g.site_parity, labels, valid)
    rng = np.random.default_rng(2)
    for _ in range(5):
        parity = permute_parity_within_strata(g.site_parity, labels, valid, rng)
        np.testing.assert_allclose(geometry_expected_contrast(t, parity, labels, valid), reference, atol=1e-15)

def test_residual_diagnostic_detects_offsetting_parity_contrast():
    g = build_geometry(10)
    valid = np.ones(g.n_sites, dtype=bool)
    levels = np.where(g.letters == "A", .2, .8)
    raw_difference = levels[g.site_parity == 0].mean()-levels[g.site_parity == 1].mean()
    levels = levels-raw_difference*(g.site_parity == 0)
    t = np.repeat(levels[:, None], 5, axis=1)
    r = stratified_permutation_null(t, g, "substitution", "site_letter", energies=np.linspace(.1, 10., 5), n_replicates=499, seed=7)
    assert np.max(np.abs(r.D_observed)) < 1e-14
    assert np.max(np.abs(r.D_residual)) > .01
    assert r.p_S_max < .05 and r.p_S_int < .05
    np.testing.assert_allclose(r.D_observed-r.D_geometry_expected, r.D_residual)
    assert r.residual_abs_S_max == r.observed.S_max

def test_geometry_only_difference_has_zero_residual():
    g = build_geometry(10)
    t = np.repeat(np.where(g.letters == "A", .2, .8)[:, None], 3, axis=1)
    r = stratified_permutation_null(t, g, "substitution", "site_letter", energies=np.array([.1, 5., 10.]), n_replicates=19)
    assert r.raw_observed.S_max > .01
    assert r.residual_abs_S_max < 1e-14
    assert r.p_S_max == 1.0

def test_incomplete_data_does_not_silently_change_null_denominators():
    g = build_geometry(5)
    t = np.full((5, 3), .5)
    t[0, 1] = np.nan
    r = stratified_permutation_null(t, g, "substitution", "site_letter", energies=np.array([.1, 5., 10.]), n_replicates=9)
    assert not r.inference_valid
    assert np.isnan(r.p_S_max) and np.isnan(r.p_S_int)
    assert np.isnan(r.D_geometry_expected[1])

def test_no_permutation_freedom_is_unidentified_not_evidence_of_no_effect():
    g = build_geometry(4)
    r = stratified_permutation_null(np.full((3, 3), .5), g, "substitution", "site_3", energies=np.array([.1, 5., 10.]), n_replicates=9)
    assert r.n_free_strata == 0
    assert not r.inference_valid and np.isnan(r.p_S_max)

def test_invalid_phason_bonds_are_not_missing_observations():
    g = build_geometry(5)
    t = np.full((g.n_bonds, 3), .5)
    t[~g.phason_valid] = np.nan
    r = stratified_permutation_null(t, g, "phason", "bond_pair", energies=np.array([.1, 5., 10.]), n_replicates=9)
    assert r.inference_valid and np.all(np.isfinite(r.D_residual))

def test_residual_curve_and_statistic_definition_survive_serialization(tmp_path):
    g = build_geometry(5)
    e = np.array([.1, 5., 10.])
    s = fast_sweep("substitution", e, geometry=g)
    a = analyze_generation_nulls({"generation": 5, "energies": e, "modes": {"substitution": s}}, g, n_replicates=9)
    path = save_nulls_analysis(a, tmp_path/"nulls.npz")
    with np.load(path, allow_pickle=False) as saved:
        k = "substitution__site_letter__"
        np.testing.assert_allclose(saved[k+"D_observed"]-saved[k+"D_geometry_expected"], saved[k+"D_residual"])
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["modes"]["substitution"]["site_letter"]["statistic_revision"] == "geometry-centered-curve-v2"

def peak(index, energy):
    return Peak(index, energy, .9, .5)

def test_cross_generation_keeps_births_and_terminated_tracks():
    tracks = chain_peaks_across_generations([
        [peak(1, 1.), peak(2, 3.)],
        [peak(10, 1.05), peak(20, 4.)],
        [peak(100, 1.1), peak(200, 4.05)],
    ], [4, 5, 6])
    assert len(tracks) == 3
    assert [len(t["path"]) for t in tracks] == [3, 1, 2]
    assert tracks[1]["end_reason"] == "unmatched"
    assert tracks[2]["birth_generation"] == 5

def test_cross_generation_does_not_bridge_an_unobserved_generation():
    tracks = chain_peaks_across_generations([[peak(1, 1.)], [peak(2, 1.01)]], [4, 6])
    assert len(tracks) == 2 and all(len(t["path"]) == 1 for t in tracks)
    assert tracks[0]["end_reason"] == "generation_gap"

def test_peak_birth_after_an_empty_generation_is_retained():
    tracks = chain_peaks_across_generations([[], [peak(1, 1.)]], [4, 5])
    assert len(tracks) == 1 and tracks[0]["birth_generation"] == 5

def test_cross_generation_rejects_duplicate_generations():
    with pytest.raises(ValueError, match="unique"):
        chain_peaks_across_generations([[], []], [4, 4])

def row(index, energy):
    return dict(index=index, energy=energy, height=.9, prominence=.5)

def test_cross_generation_numeric_gate_maps_changed_grid_indices():
    base = {4: [row(10, 1.)], 5: [row(20, 1.1)]}
    fine = {4: [row(21, 1.0001)], 5: [row(39, 1.1001)]}
    r = compare_clean_anchor_grids([4, 5], base, fine)
    assert r["gate"] == "pass" and r["n_links"] == 1

def test_cross_generation_link_change_fails_even_when_local_peaks_are_close():
    gate = DEFAULT_MAX_MATCH_DELTA_E_EV
    base = {4: [row(10, 1.)], 5: [row(20, 1.+gate-.0004)]}
    fine = {4: [row(21, .9996)], 5: [row(39, 1.+gate)]}
    assert compare_clean_anchor_grids([4, 5], base, fine)["gate"] == "fail"

def test_cross_generation_shift_change_has_its_own_tolerance():
    base = {4: [row(10, 1.)], 5: [row(20, 1.1)]}
    fine = {4: [row(21, .9992)], 5: [row(39, 1.1008)]}
    assert compare_clean_anchor_grids([4, 5], base, fine)["gate"] == "fail"

def test_cross_generation_numerical_failure_is_unresolved():
    assert compare_clean_anchor_grids([4, 5], {}, {}, complete=False)["gate"] == "unresolved"

def test_cross_generation_products_include_actual_shift_assignments(tmp_path):
    raw = tmp_path/"raw"
    produce_raw_dataset(generations=[4, 5], energies_ev=np.linspace(.1, 10., 101), modes=["substitution"], output_dir=raw, validate=False)
    output = tmp_path/"resonance"
    result = produce_resonance_analysis(raw, generations=[4, 5], modes=["substitution"], output_dir=output)
    meta = json.loads(Path(result["cross_generation"]["manifest_path"]).read_text())
    assert meta["n_links"] > 0 and meta["n_assignments"] > 0
    tracks = json.loads((output/"cross_generation_tracks.json").read_text())
    lookup = {(p["generation"], p["peak_index"]): t["track_id"] for t in tracks for p in t["path"]}
    for line in (output/"cross_generation_assignments.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r["clean_track_id"] == lookup[(r["generation"], r["clean_index"])]
        assert r["delta_E"] == pytest.approx(r["energy_defect"]-r["energy_clean"])
    assert not meta["numerically_accepted"]

def fake_evidence(gate):
    return dict(primary_gate=gate, generation_gates={"10": gate}, cross_generation_gate={"gate": "not_applicable"}, manifest_path="synthetic-test-only")

def test_resolution_pilot_requires_consecutive_passes_and_recommends_tested_baseline(tmp_path, monkeypatch):
    gates = iter(["fail", "pass", "pass"])
    def simulate(**kw):
        np.testing.assert_array_equal(kw["energies_fine"][::2], kw["energies_baseline"])
        return fake_evidence(next(gates))
    monkeypatch.setattr(resolution, "produce_robustness_analysis", simulate)
    r = resolution.run_resolution_pilot(start_points=5, max_rounds=3, max_points=33, output_dir=tmp_path/"pilot")
    assert r["recommended_n_energy"] == 17
    assert r["status"] == "converged_on_requested_checks"
    assert r["scientific_claim"] is False
    assert cfg.N_ENERGY == 2000

def test_resolution_pilot_budget_never_becomes_a_false_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(resolution, "produce_robustness_analysis", lambda **kw: fake_evidence("fail"))
    r = resolution.run_resolution_pilot(start_points=5, max_rounds=2, max_points=17, output_dir=tmp_path/"pilot")
    assert r["status"] == "unresolved" and r["recommended_n_energy"] is None

def test_resolution_pilot_does_not_overwrite_existing_evidence(tmp_path):
    target = tmp_path/"pilot"
    target.mkdir()
    with pytest.raises(FileExistsError):
        resolution.run_resolution_pilot(output_dir=target)

def test_resolution_pilot_records_errors_without_recommending_a_grid(tmp_path, monkeypatch):
    def broken(**kw):
        raise ArithmeticError("synthetic test")
    monkeypatch.setattr(resolution, "produce_robustness_analysis", broken)
    with pytest.raises(ArithmeticError):
        resolution.run_resolution_pilot(output_dir=tmp_path/"pilot")
    saved = json.loads((tmp_path/"pilot"/"resolution_manifest.json").read_text())
    assert saved["status"] == "error" and saved["recommended_n_energy"] is None

def test_real_small_pilot_reports_unresolved_without_relaxing_gates(tmp_path):
    before = dict(cfg.GATE_TOLERANCES)
    r = resolution.run_resolution_pilot(generations=[4], start_points=12, max_rounds=1,
                                       max_points=23, required_passes=1, output_dir=tmp_path/"pilot")
    assert r["status"] == "unresolved" and r["recommended_n_energy"] is None
    assert dict(cfg.GATE_TOLERANCES) == before
    evidence = Path(r["rounds"][0]["evidence_manifest"])
    assert evidence.exists()

def _launcher_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1]/"tools"/"run_from_pilot.py"
    spec = importlib.util.spec_from_file_location("run_from_pilot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _passing_pilot():
    from fibonacci_qc.raw_data import config_sha256
    return dict(status="converged_on_requested_checks", recommended_n_energy=3999,
                generations=[10, 20], modes=list(cfg.DEFECT_MODES),
                config_sha256=config_sha256(), gate_tolerances=dict(cfg.GATE_TOLERANCES),
                analysis_revisions={"geometry_null": "geometry-centered-curve-v2",
                                    "cross_generation_resonance": "clean-anchor-continuation-v1"},
                required_consecutive_passes=2,
                rounds=[dict(primary_gate="pass", n_energy_baseline=2000),
                        dict(primary_gate="pass", n_energy_baseline=3999)])

def test_production_launcher_preserves_production_profile_and_selected_grid(tmp_path, monkeypatch):
    import fibonacci_qc.pipeline as pipeline
    monkeypatch.setattr(pipeline, "run_pipeline", lambda **kw: kw)
    path = tmp_path/"pilot.json"
    path.write_text(json.dumps(_passing_pilot()))
    assert _launcher_module().launch_from_pilot(path) == dict(profile="production", n_energy=3999)

def test_production_launcher_rejects_a_generation10_only_pilot(tmp_path):
    report = _passing_pilot()
    report["generations"] = [10]
    path = tmp_path/"pilot.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="both generations"):
        _launcher_module().launch_from_pilot(path)

def test_production_launcher_rejects_relaxed_tolerances(tmp_path):
    report = _passing_pilot()
    report["gate_tolerances"]["peak_energy_ev"] *= 100
    path = tmp_path/"pilot.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="tolerances"):
        _launcher_module().launch_from_pilot(path)