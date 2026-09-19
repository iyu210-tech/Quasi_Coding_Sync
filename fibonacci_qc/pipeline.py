"Full pipeline"

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.drivers import run_analyze_driver, run_figures_driver, run_raw_driver
from fibonacci_qc.raw_data import _git_metadata, load_generation_raw, run_metadata

PathLike = Union[str, Path]

def clear_output_tree(*, keep_gitkeep: bool = True) -> None:
    "Remove contents of output for emptiness at start"
    cfg.ensure_output_dirs()
    for root in cfg.OUTPUT_DIRS:
        if not root.exists():
            continue
        for child in root.iterdir():
            if keep_gitkeep and child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

def run_pipeline(
    *,
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    n_energy: int = 12,
    n_replicates: int = 12,
    raw_dir: str = "pipeline",
    figure_dir: str = "pipeline",
    clear_outputs: bool = True,
    validate: bool = True,
) -> Dict[str, object]:
    "Rerun raw -> analyze -> figures from an empty output dir"
    if clear_outputs:
        clear_output_tree()
    gens = list(generations or [4, 5, 6, 7])
    mode_list = list(modes or [cfg.MODE_SUBSTITUTION, cfg.MODE_WIDTH_ONLY])
    raw_manifest = run_raw_driver(generations=gens, modes=mode_list, n_energy=n_energy, output_dir=raw_dir, validate=validate)
    analyze_manifest = run_analyze_driver(raw_dir=raw_dir, generations=gens, modes=mode_list, n_replicates=n_replicates)
    figures_manifest = run_figures_driver(raw_dir=raw_dir, generation=gens[0], mode=mode_list[0], output_dir=figure_dir, n_replicates=n_replicates)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generations": gens,
        "modes": mode_list,
        "n_energy": n_energy,
        "n_replicates": n_replicates,
        "metadata": run_metadata(generations=gens, energies=cfg.energy_grid(n_points=n_energy), extra={"driver": "pipeline"}),
        "raw": {"manifest_path": raw_manifest.get("manifest_path"), "files": raw_manifest.get("files")},
        "analyze": {"manifest_path": analyze_manifest.get("analyze_manifest_path")},
        "figures": figures_manifest,
    }
    out = cfg.OUTPUT_METADATA / "pipeline_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["pipeline_manifest_path"] = str(out)
    return payload

def verify_reproducibility(
    *,
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    n_energy: int = 8,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> Dict[str, object]:
    "Run raw generation twice, compare heatmaps"
    gens = list(generations or [4, 5])
    mode_list = list(modes or [cfg.MODE_SUBSTITUTION])
    dir_a = "repro_a"
    dir_b = "repro_b"

    # Clear only these folders if present.
    for name in (dir_a, dir_b):
        path = cfg.OUTPUT_RAW / name
        if path.exists():
            shutil.rmtree(path)

    run_raw_driver(generations=gens, modes=mode_list, n_energy=n_energy, output_dir=dir_a, validate=False)
    run_raw_driver(generations=gens, modes=mode_list, n_energy=n_energy, output_dir=dir_b, validate=False)
    mismatches = []
    for gen in gens:
        for mode in mode_list:
            path_a = cfg.OUTPUT_RAW / dir_a / f"n{gen:02d}" / f"{mode}.npz"
            path_b = cfg.OUTPUT_RAW / dir_b / f"n{gen:02d}" / f"{mode}.npz"
            a = load_generation_raw(path_a)
            b = load_generation_raw(path_b)
            ta = np.asarray(a["modes"][mode]["T"], dtype=np.float64)
            tb = np.asarray(b["modes"][mode]["T"], dtype=np.float64)
            if not np.allclose(ta, tb, atol=atol, rtol=rtol, equal_nan=True):
                mismatches.append(
                    {
                        "generation": gen,
                        "mode": mode,
                        "max_abs": float(np.nanmax(np.abs(ta - tb))),
                    }
                )
    report = {
        "matched": len(mismatches) == 0,
        "mismatches": mismatches,
        "metadata": run_metadata(
            generations=gens,
            energies=cfg.energy_grid(n_points=n_energy),
            extra={"driver": "reproducibility"},
        ),
    }
    out = cfg.OUTPUT_METADATA / "reproducibility.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(out)
    return report

def create_freeze_tag(
    tag: str,
    *,
    message: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    "Create an annotated Git tag for the research freeze, write a freeze metadata JSON under output/metadata/"
    git = _git_metadata()
    payload = {
        "tag": tag,
        "message": message or f"Research freeze {tag}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "config": cfg.CONFIG.as_dict(),
        "dry_run": dry_run,
    }
    out = cfg.OUTPUT_METADATA / f"freeze_{tag.replace('/', '_')}.json"
    cfg.ensure_output_dirs()
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["metadata_path"] = str(out)
    if dry_run:
        payload["created"] = False
        return payload
    try:
        subprocess.check_call(
            [
                "git",
                "tag",
                "-a",
                tag,
                "-m",
                payload["message"],
            ],
            cwd=cfg.PROJECT_ROOT,
        )
        payload["created"] = True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        payload["created"] = False
        payload["error"] = str(exc)
    return payload

def write_freeze_snapshot(
    label: str = "research_freeze",
) -> Path:
    "Write config & git freeze snapshot w/o creating Git tag"
    cfg.ensure_output_dirs()
    payload = {
        "label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "config": cfg.CONFIG.as_dict(),
    }
    out = cfg.OUTPUT_METADATA / f"{label}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out