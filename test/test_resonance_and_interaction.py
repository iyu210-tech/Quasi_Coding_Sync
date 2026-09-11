"Tests for non-additivity and resonance tracking."

from __future__ import annotations

import json
import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.interaction import (
    interaction_from_mode_arrays,
    interaction_from_sweeps,
    interaction_remainder,
    interaction_remainder_log,
    interaction_summary,
)
from fibonacci_qc.raw_data import save_generation_raw, sweep_generation
from fibonacci_qc.resonance import (
    Peak,
    analyze_generation_resonances,
    analyze_mode_resonances,
    detect_peaks,
    match_peaks_order_preserving,
    track_resonances,
)
from fibonacci_qc.resonance_io import (produce_resonance_analysis, save_interaction_remainder, save_resonance_analysis,)
from fibonacci_qc.scattering import clean_transmission_grid

def test_interaction_remainder_formula():
    sub = np.array([[1.0, 2.0], [3.0, 4.0]])
    pot = np.array([[0.2, 0.3], [0.4, 0.5]])
    width = np.array([[0.1, 0.1], [0.2, 0.2]])
    expected = sub - pot - width
    np.testing.assert_allclose(interaction_remainder(sub, pot, width), expected)

def test_interaction_remainder_log_formula():
    sub = np.array([[-1.0, -2.0]])
    pot = np.array([[-0.2, -0.3]])
    width = np.array([[-0.1, -0.1]])
    np.testing.assert_allclose(interaction_remainder_log(sub, pot, width), sub - pot - width,
    )

def test_interaction_from_mode_arrays_shapes():
    n_pos, n_e = 4, 5
    energies = np.linspace(0.5, 2.5, n_e)
    t_clean = np.full(n_e, 0.2)
    log_clean = np.log(t_clean)
    rng = np.random.default_rng(0)
    t_sub = rng.random((n_pos, n_e))
    t_pot = rng.random((n_pos, n_e))
    t_width = rng.random((n_pos, n_e))
    rem = interaction_from_mode_arrays(
        t_sub=t_sub,
        t_pot=t_pot,
        t_width=t_width,
        log_t_sub=np.log(np.clip(t_sub, 1e-300, None)),
        log_t_pot=np.log(np.clip(t_pot, 1e-300, None)),
        log_t_width=np.log(np.clip(t_width, 1e-300, None)),
        t_clean=t_clean,
        log_t_clean=log_clean,
        energies=energies,
    )
    assert rem.I.shape == (n_pos, n_e)
    assert rem.I_log.shape == (n_pos, n_e)
    np.testing.assert_allclose(rem.I, (t_sub - t_clean) - (t_pot - t_clean) - (t_width - t_clean),)
    summary = interaction_summary(rem)
    assert "full" in summary
    assert np.isfinite(summary["full"]["rms_I"])

def test_interaction_from_real_sweeps():
    geom = build_geometry(5)
    energies = np.linspace(0.5, 3.0, 8)
    modes = {}
    for mode in (cfg.MODE_SUBSTITUTION, cfg.MODE_POTENTIAL_ONLY, cfg.MODE_WIDTH_ONLY,):
        modes[mode] = fast_sweep(mode, energies, geometry=geom)
    t_clean, log_clean, _ = clean_transmission_grid(geom.generation, energies, geometry=geom)
    rem = interaction_from_sweeps(modes, t_clean=t_clean, log_t_clean=log_clean, energies=energies)
    assert rem.I.shape == (geom.n_sites, energies.size)
    assert rem.valid.shape == (geom.n_sites,)
    assert np.all(rem.valid)

def test_detect_peaks_finds_local_maxima():
    energies = np.linspace(0.0, 10.0, 101)
    values = (np.exp(-((energies - 2.0) ** 2) / 0.1) + 0.7 * np.exp(-((energies - 6.0) ** 2) / 0.15) + 0.05 * np.sin(20 * energies))
    peaks = detect_peaks(values, energies, min_prominence=0.01)
    assert len(peaks) >= 2
    peak_energies = [p.energy for p in peaks]
    assert any(abs(e - 2.0) < 0.2 for e in peak_energies)
    assert any(abs(e - 6.0) < 0.2 for e in peak_energies)

def test_match_maximizes_count_then_minimizes_displacement():
    clean = [
        Peak(0, 1.0, 1.0, 0.1),
        Peak(1, 3.0, 1.0, 0.1),
        Peak(2, 5.0, 1.0, 0.1),
    ]
    defect = [
        Peak(0, 1.1, 1.0, 0.1),
        Peak(1, 4.9, 1.0, 0.1),
    ]
    result = match_peaks_order_preserving(clean, defect)
    assert result.n_matched == 2
    deltas = sorted(abs(m.delta_E) for m in result.matched)
    assert deltas[0] == pytest.approx(0.1)
    assert deltas[1] == pytest.approx(0.1)
    assert len(result.unmatched_clean) == 1
    assert result.unmatched_clean[0].energy == pytest.approx(3.0)

def test_match_prefers_more_matches_over_smaller_displacement():
    # Matching all three costs more total |ΔE| than matching only the closest pair, mainly want match count.
    clean = [
        Peak(0, 0.0, 1.0, 0.1),
        Peak(1, 2.0, 1.0, 0.1),
        Peak(2, 4.0, 1.0, 0.1),
    ]
    defect = [
        Peak(0, 0.5, 1.0, 0.1),
        Peak(1, 2.5, 1.0, 0.1),
        Peak(2, 4.5, 1.0, 0.1),
    ]
    result = match_peaks_order_preserving(clean, defect)
    assert result.n_matched == 3
    assert result.total_displacement == pytest.approx(1.5)

def test_match_is_order_preserving():
    clean = [Peak(i, float(i), 1.0, 0.1) for i in range(4)]
    defect = [Peak(i, float(i) + 0.2, 1.0, 0.1) for i in range(4)]
    result = match_peaks_order_preserving(clean, defect)
    clean_idx = [m.clean_index for m in result.matched]
    defect_idx = [m.defect_index for m in result.matched]
    assert clean_idx == sorted(clean_idx)
    assert defect_idx == sorted(defect_idx)

def test_track_resonances_reports_delta_e():
    energies = np.linspace(0.0, 5.0, 51)
    clean = np.exp(-((energies - 2.0) ** 2) / 0.05)
    defect = np.exp(-((energies - 2.3) ** 2) / 0.05)
    track = track_resonances(clean, defect, energies, mode="substitution", position=0, parity=0)
    assert track.valid
    assert track.match.n_matched >= 1
    assert track.mean_delta_E == pytest.approx(0.3, abs=0.15)

def test_analyze_mode_resonances_includes_parity_summary():
    geom = build_geometry(5)
    energies = np.linspace(0.4, 4.0, 40)
    sweep = fast_sweep(cfg.MODE_SUBSTITUTION, energies, geometry=geom)
    t_clean, _, _ = clean_transmission_grid(geom.generation, energies, geometry=geom)
    result = analyze_mode_resonances(sweep["T"], t_clean, energies, geom, cfg.MODE_SUBSTITUTION, sweep_valid=sweep["valid"],)
    assert len(result["tracks"]) == geom.n_sites
    assert "even" in result["summary"]["by_parity"]
    assert "odd" in result["summary"]["by_parity"]
    assert len(result["summary"]["by_position"]) == geom.n_sites

def test_phason_resonance_uses_uncompacted_axis():
    geom = build_geometry(5)
    energies = np.linspace(0.5, 3.0, 20)
    sweep = fast_sweep(cfg.MODE_PHASON, energies, geometry=geom)
    t_clean, _, _ = clean_transmission_grid(geom.generation, energies, geometry=geom)
    result = analyze_mode_resonances(sweep["T"], t_clean, energies, geom, cfg.MODE_PHASON, sweep_valid=sweep["valid"],)
    assert len(result["tracks"]) == geom.n_bonds
    invalid = [t for t in result["tracks"] if not t.valid]
    assert len(invalid) == int(np.count_nonzero(~geom.phason_valid))
    assert result["tracks"][2].valid is False

def test_save_and_produce_section6(tmp_path):
    geom = build_geometry(4)
    energies = np.linspace(0.5, 2.5, 12)
    modes = [
        cfg.MODE_SUBSTITUTION,
        cfg.MODE_POTENTIAL_ONLY,
        cfg.MODE_WIDTH_ONLY,
        cfg.MODE_PHASON,
    ]
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
    rem = interaction_from_sweeps(bundle["modes"], t_clean=bundle["T_clean"], log_t_clean=bundle["log_T_clean"], energies=energies,)
    save_interaction_remainder(rem, tmp_path / "interaction.npz", generation=4)
    resonance = analyze_generation_resonances(bundle, geom)
    save_resonance_analysis(resonance, tmp_path / "resonance.npz")
    meta = json.loads((tmp_path / "resonance.meta.json").read_text(encoding="utf-8"))
    assert cfg.MODE_SUBSTITUTION in meta["modes"]
    manifest = produce_resonance_analysis(
        raw_dir,
        generations=[4],
        output_dir=tmp_path / "section6_out",
    )
    assert manifest["files"]
    assert (tmp_path / "section6_out" / "manifest.json").exists()
    assert (tmp_path / "section6_out" / "n04" / "interaction.npz").exists()
    assert (tmp_path / "section6_out" / "n04" / "resonance.npz").exists()