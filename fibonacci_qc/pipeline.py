"Full pipeline"

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, Tuple
import numpy as np
import hashlib
import sys

from fibonacci_qc import config as cfg
from fibonacci_qc.drivers import run_analyze_driver, run_figures_driver, run_raw_driver
from fibonacci_qc.raw_data import _git_metadata, load_generation_raw, run_metadata, config_sha256

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
    generations=None,
    modes=None,
    n_energy=None,
    n_replicates=None,
    raw_dir=None,
    figure_dir=None,
    clear_outputs=False,
    profile="smoke",
    validate=True
):
    if profile not in {"smoke","production"}: raise ValueError("unknown profile")
    prod = profile == "production"
    gens = list((cfg.GENERATIONS if prod else [4,5,6,7]) if generations is None else generations)
    wanted = list((cfg.DEFECT_MODES if prod else [cfg.MODE_SUBSTITUTION,cfg.MODE_WIDTH_ONLY]) if modes is None else modes)
    ne = (cfg.N_ENERGY if prod else 12) if n_energy is None else int(n_energy)
    nr = (cfg.PERMUTATION_REPLICATES if prod else 12) if n_replicates is None else int(n_replicates)

    if prod and (set(gens) != set(cfg.GENERATIONS) or set(wanted)!=set(cfg.DEFECT_MODES) or ne<cfg.N_ENERGY or nr<cfg.PERMUTATION_REPLICATES):
        raise ValueError("production must cover generations 10-20, all four modes, >=2000 energies, and >=500 permutations")
    if prod and not validate:raise ValueError("production requires numerical reference cross-checks")
    if clear_outputs:raise ValueError("automatic deletion disabled; supply fresh run name")
    
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_name = raw_dir or f"{profile}_{stamp}"
    figure_name = figure_dir or raw_name
    raw = run_raw_driver(generations=gens,modes=wanted,n_energy=ne,output_dir=raw_name,validate=validate)
    analysis = run_analyze_driver(raw_dir=raw_name,generations=gens,modes=wanted,n_replicates=nr)
    figures = run_figures_driver(raw_dir=raw_name,generation=gens[0],mode=wanted[0],output_dir=figure_name,n_replicates=None)
    scaling_supported = any(int(r.get("n_support", 0)) > 0 for r in analysis["scaling"].get("summary",[]))
    gate = analysis["robustness"].get("primary_gate","unresolved")
    primary_ok = gate == "pass" and not analysis["parity"].get("numerical_failure",True)
    
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "scientific_claim": bool(prod and primary_ok),
        "numerically_accepted": bool(prod and primary_ok),
        "acceptance_meaning": "sampled numerical criteria passed",
        "primary_gate": gate,
        "scaling_supported": scaling_supported,
        "generations": gens,
        "modes": wanted,
        "n_energy":ne,
        "n_replicates":nr,
        "raw_dir":str(raw_name),
        "metadata":run_metadata(generations=gens, energies=cfg.energy_grid(n_points=ne), extra={"driver" : "pipeline", "profile": profile}),
        "raw":{k:raw.get(k) for k in ("manifest_path", "files", "peak_rss_bytes")},
        "analyze":{"manifest_path":analysis["analyze_manifest_path"]},"figures":figures}
    path = cfg.OUTPUT_METADATA/f"pipeline_{profile}_{stamp}.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2),encoding="utf-8"); payload["pipeline_manifest_path"]=str(path)
    if prod and not primary_ok:raise RuntimeError(f"production numerical acceptance blocked by {gate}; diagnostics saved to {path}")
    return payload

def _finite_match(left: np.ndarray, right: np.ndarray, *, atol: float, rtol: float):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return False, 0
    finite = np.isfinite(a)
    n_finite = int(np.count_nonzero(finite))
    if n_finite == 0:
        return False, 0
    return bool(np.allclose(a[finite], b[finite], atol=atol, rtol=rtol)), n_finite

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
    n_finite = 0
    logs_matched = True
    valid_matched = True
    statuses_matched = True
    for gen in gens:
        for mode in mode_list:
            path_a = cfg.OUTPUT_RAW / dir_a / f"n{gen:02d}" / f"{mode}.npz"
            path_b = cfg.OUTPUT_RAW / dir_b / f"n{gen:02d}" / f"{mode}.npz"
            a = load_generation_raw(path_a)
            b = load_generation_raw(path_b)
            sweep_a = a["modes"][mode]
            sweep_b = b["modes"][mode]
            ok_t, n_here = _finite_match(sweep_a["T"], sweep_b["T"], atol=atol, rtol=rtol)
            ok_log, _ = _finite_match(sweep_a["log_T"], sweep_b["log_T"], atol=atol, rtol=rtol)
            ok_clean, _ = _finite_match(a["T_clean"], b["T_clean"], atol=atol, rtol=rtol)
            ok_valid = np.array_equal(np.asarray(sweep_a["valid"]), np.asarray(sweep_b["valid"]))
            ok_status = np.array_equal(np.asarray(sweep_a["status"]), np.asarray(sweep_b["status"]))
            n_finite += n_here
            logs_matched = logs_matched and ok_log
            valid_matched = valid_matched and ok_valid
            statuses_matched = statuses_matched and ok_status
            if not (ok_t and ok_log and ok_clean and ok_valid and ok_status):
                mismatches.append({"generation": gen, "mode": mode, "n_finite": n_here})
    report = {
        "matched": len(mismatches) == 0 and n_finite > 0 and logs_matched and valid_matched and statuses_matched,
        "mismatches": mismatches,
        "n_finite": n_finite,
        "logs_matched": logs_matched,
        "valid_matched": valid_matched,
        "statuses_matched": statuses_matched,
        "scientific_claim": False,
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

def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "missing"

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

def _data_identity() -> Dict[str, str]:
    identities = {}
    if cfg.OUTPUT_RAW.exists():
        for manifest in sorted(cfg.OUTPUT_RAW.glob("*/manifest.json")):
            identities[str(manifest.relative_to(cfg.PROJECT_ROOT))] = _file_sha256(manifest)
    return identities

def create_freeze_tag(
    tag: str,
    *,
    message: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    "Create an annotated Git tag for the research freeze, write a freeze metadata JSON under output/metadata/"
    git = _git_metadata()
    dirty = str(git.get("dirty", "")).strip()
    if dirty:
        raise RuntimeError(f"refusing freeze tag on a dirty working tree:\n{dirty}")
    if git.get("commit") in (None, "", "unknown"):
        raise RuntimeError("refusing freeze tag without a resolved commit")
    created = False
    error = None
    tag_message = message or f"Research freeze {tag}"
    if not dry_run:
        try:
            subprocess.check_call(["git", "tag", "-a", tag, "-m", tag_message], cwd=cfg.PROJECT_ROOT)
            created = True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            created = False
            error = str(exc)
    payload = {
        "tag": tag,
        "message": tag_message,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created": created,
        "dry_run": dry_run,
        "error": error,
        "git": git,
        "config": cfg.CONFIG.as_dict(),
        "config_sha256": config_sha256(),
        "python": sys.version,
        "dependencies": {
            "numpy": _package_version("numpy"),
            "matplotlib": _package_version("matplotlib"),
        },
        "data_identity": _data_identity(),
    }
    out = cfg.OUTPUT_METADATA / f"freeze_{tag.replace('/', '_')}.json"
    cfg.ensure_output_dirs()
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["metadata_path"] = str(out)
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