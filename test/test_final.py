"Tests plotting, drivers, pipeline, frozen regression"

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.drivers import (
    main_analyze,
    main_figures,
    main_raw,
    run_figures_driver,
    run_raw_driver,
)
from fibonacci_qc.pipeline import (
    create_freeze_tag,
    run_pipeline,
    verify_reproducibility,
    write_freeze_snapshot,
)
from fibonacci_qc.plotting import (
    plot_clean_transmission,
    plot_parity_curve,
    plot_transmission_heatmap,
    produce_figures,
    save_figure,
)
from fibonacci_qc.regression import (
    assert_matches_regression_fixture,
    compute_regression_reference,
)

@pytest.fixture(scope="module")
def tiny_raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw_study")
    from fibonacci_qc.raw_data import produce_raw_dataset
    produce_raw_dataset(
        generations=[4, 5],
        energies_ev=np.linspace(0.5, 2.0, 6),
        output_dir=root,
        modes=[cfg.MODE_SUBSTITUTION, cfg.MODE_WIDTH_ONLY],
        validate=False,
    )
    return root

def test_plot_helpers_write_png(tmp_path):
    energies = np.linspace(0.5, 2.0, 20)
    t = np.exp(-energies)
    fig = plot_clean_transmission(energies, t)
    out = save_figure(fig, tmp_path / "clean.png")
    assert out.exists()
    assert out.stat().st_size > 0
    positions = np.arange(4)
    heat = np.clip(np.outer(0.2 + 0.1 * positions, t), 1e-6, 1.0)
    fig2 = plot_transmission_heatmap(energies, positions, heat)
    out2 = save_figure(fig2, tmp_path / "heat.png")
    assert out2.exists()
    fig3 = plot_parity_curve(energies, np.sin(energies))
    out3 = save_figure(fig3, tmp_path / "parity.png")
    assert out3.exists()

def test_produce_figures_from_raw_bundle(tiny_raw, tmp_path):
    from fibonacci_qc.raw_data import load_generation_raw
    raw = load_generation_raw(tiny_raw / "n04" / "substitution.npz")
    modes = {}
    for path in (tiny_raw / "n04").glob("*.npz"):
        loaded = load_generation_raw(path)
        modes.update(loaded["modes"])
    bundle = {
        "generation": 4,
        "energies": raw["energies"],
        "T_clean": raw["T_clean"],
        "modes": modes,
    }
    written = produce_figures(
        raw_bundle=bundle,
        output_dir=tmp_path / "figs",
        mode=cfg.MODE_SUBSTITUTION,
    )
    assert "clean" in written
    assert "heatmap" in written
    assert Path(written["clean"]).exists()

def test_run_raw_driver_cli(tmp_path, monkeypatch):
    manifest = run_raw_driver(
        generations=[4],
        modes=[cfg.MODE_SUBSTITUTION],
        n_energy=5,
        output_dir=str(tmp_path / "cli_raw"),
        validate=False,
    )
    assert manifest["files"]
    assert main_raw(
        [
            "--generations",
            "4",
            "--modes",
            "substitution",
            "--n-energy",
            "5",
            "--output-dir",
            str(tmp_path / "cli_raw2"),
            "--no-validate",
        ]
    ) == 0

def test_run_figures_driver(tiny_raw, tmp_path):
    result = run_figures_driver(
        raw_dir=str(tiny_raw),
        generation=4,
        mode=cfg.MODE_SUBSTITUTION,
        output_dir=str(tmp_path / "figs"),
        n_replicates=6,
    )
    assert result["files"]
    assert Path(result["manifest_path"]).exists()
    meta = json.loads(Path(result["manifest_path"]).read_text())
    assert "metadata" in meta
    assert "git" in meta["metadata"]
    assert "config" in meta["metadata"]

def test_main_figures_cli(tiny_raw, tmp_path):
    assert (
        main_figures(
            [
                "--raw-dir",
                str(tiny_raw),
                "--generation",
                "4",
                "--mode",
                "substitution",
                "--output-dir",
                str(tmp_path / "figs_cli"),
                "--n-replicates",
                "4",
            ]
        )
        == 0
    )

def test_regression_fixture_matches_committed_reference():
    assert_matches_regression_fixture()
    ref = compute_regression_reference()
    assert ref["T"].shape[0] == 3

def test_verify_reproducibility_small():
    report = verify_reproducibility(generations=[4], modes=[cfg.MODE_SUBSTITUTION], n_energy=5)
    assert report["matched"]
    assert report["n_finite"] > 0
    assert report["logs_matched"]
    assert report["valid_matched"]
    assert report["statuses_matched"]
    assert report["scientific_claim"] is False
    assert Path(report["report_path"]).exists()

def test_pipeline_and_freeze_snapshot(tmp_path, monkeypatch):
    # Pipeline outputs under output tree, use small parameters
    payload = run_pipeline(
        generations=[4, 5],
        modes=[cfg.MODE_SUBSTITUTION, cfg.MODE_WIDTH_ONLY],
        n_energy=5,
        n_replicates=4,
        raw_dir="pipeline_test",
        figure_dir="pipeline_test",
        clear_outputs=False,
        validate=False,
    )
    assert Path(payload["pipeline_manifest_path"]).exists()
    assert "metadata" in payload
    assert payload["metadata"]["git"]
    assert payload["metadata"]["config"]
    assert payload["scaling_supported"] is False
    assert payload["scientific_claim"] is False
    snap = write_freeze_snapshot("unit_test_freeze")
    assert snap.exists()
    data = json.loads(snap.read_text())
    assert data["label"] == "unit_test_freeze"
    assert "config" in data
    monkeypatch.setattr(
        "fibonacci_qc.pipeline._git_metadata",
        lambda: {"commit": "abc", "branch": "main", "dirty": ""},
    )
    freeze = create_freeze_tag("test-freeze-dry", dry_run=True)
    assert freeze["dry_run"] is True
    assert freeze["created"] is False
    assert Path(freeze["metadata_path"]).exists()
    assert "dependencies" in json.loads(Path(freeze["metadata_path"]).read_text())

def test_freeze_tag_rejects_dirty_tree(monkeypatch):
    monkeypatch.setattr(
        "fibonacci_qc.pipeline._git_metadata",
        lambda: {"commit": "abc", "branch": "main", "dirty": " M fibonacci_qc/pipeline.py"},
    )
    with pytest.raises(RuntimeError, match="dirty"):
        create_freeze_tag("test-freeze-dirty", dry_run=True)

def test_main_analyze_cli(tiny_raw):
    # Copy fixture data into output/raw
    from fibonacci_qc.drivers import run_analyze_driver

    dest = cfg.OUTPUT_RAW / "study_test"
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    import shutil

    shutil.copytree(tiny_raw, dest)
    result = run_analyze_driver(raw_dir="study_test", generations=[4, 5], modes=[cfg.MODE_SUBSTITUTION, cfg.MODE_WIDTH_ONLY], n_replicates=4)
    assert Path(result["analyze_manifest_path"]).exists()
    assert main_analyze(
        [
            "--raw-dir",
            "study_test",
            "--generations",
            "4,5",
            "--modes",
            "substitution,width_only",
            "--n-replicates",
            "4",
        ]
    ) == 0