"Producing raw data for the later heatmaps. Create and save various T_m(n,p,E) with clean transmission, validity, + metadata for generations 10 to 20"

from __future__ import annotations

import json 
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import StackCache, fast_sweep, validate_against_reference
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry, iter_study_geometries
from fibonacci_qc.scattering import clean_transmission_grid

PathLike = Union[str, Path]

def _git_metadata() -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        root = cfg.PROJECT_ROOT
        meta["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd = root, text=True
        ).strip()
        meta["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd = root, text=True
        ).strip()
        meta["dirty"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        meta["commit"] = "unknown"
        meta["branch"] = "unknown"
        meta["dirty"] = "unknown"
    return meta

def run_metadata(
    *,
    generations:Sequence[int],
    energies:np.ndarray,
    extra:Optional[Dict[str,object]] = None,
) -> Dict[str, object]:
    "Assembe metadata w/ outputs"
    payload: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generations": list(generations),
        "n_energy": int(energies.size),
        "energy_min_ev": float(energies[0]) if energies.size else None,
        "energy_max_ev": float(energies[-1]) if energies.size else None,
        "defect_modes": list(cfg.DEFECT_MODES),
        "config": cfg.CONFIG.as_dict(),
        "git": _git_metadata(),
    }
    if extra:
        payload.update(extra)
    return payload

def sweep_generation(
        generation: int,
        energies_ev: np.ndarray,
        *,
        modes: Optional[Sequence[str]] = None,
        geometry: Optional[FibonacciGeometry] = None,
        v_lead_ev: float = cfg.V_LEAD_EV,
) -> Dict[str, object]:
    "Sweep all positions and energies for a generation"
    geom = geometry if geometry is not None else build_geometry(generation)
    if geom.generation != generation:
        raise ValueError(f"geometry.generation={geom.generation} does not match generation={generation}")
    energies = np.asarray(energies_ev, dtype=np.float64)
    modes = list(modes or cfg.DEFECT_MODES)

    t_clean, log_clean, clean_status = clean_transmission_grid(generation, energies, geometry = geom, v_lead_ev=v_lead_ev)
    mode_data: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in modes:
        mode_data[mode] = fast_sweep(mode, energies, geometry = geom, v_lead_ev=v_lead_ev)
    return {
        "generation": generation,
        "energies": energies,
        "T_clean": t_clean,
        "log_T_clean": log_clean,
        "clean_status": clean_status,
        "modes": mode_data,
        "geometry_metadata": geom.to_metadata(),
    }

def save_generation_raw(
    bundle: Dict[str, object],
    path: PathLike,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    "Save a generations heatmaps and JSON metadata"
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.isabsolute():
        out = cfg.OUTPUT_RAW / out
    out.parent.mkdir(parents=True, exist_ok = True)

    generation = int(bundle["generation"])
    energies = np.asarray(bundle["energies"], dtype=np.float64)
    np_payload: Dict[str, object] = {
        "generation": np.array(generation, dtype=np.int64),
        "energies": energies,
        "T_clean": np.asarray(bundle["T_clean"], dtype=np.float64),
        "log_T_clean": np.asarray(bundle["log_T_clean"], dtype=np.float64),
        "clean_status": np.asarray(bundle["clean_status"], dtype=object),
    }
    modes: Dict[str, Dict[str, np.ndarray]] = bundle["modes"]
    for mode, sweep in modes.items():
        prefix = f"{mode}__"
        for key, value in sweep.items():
            if key in {"mode", "generation"}:
                continue
            np_payload[prefix + key] = value

    np.savez_compressed(out, **np_payload)

    meta_path = out.with_suffix(".meta.json")
    meta = {
        "npz": str(out.name),
        "generation": generation,
        "geometry_metadata": bundle.get("geometry_metadata"),
        "run_metadata": metadata or {},
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding = "utf-8")
    return out

def load_generation_raw(path: PathLike) -> Dict[str, object]:
    "Load a saved generation bundle"
    out = Path(path)
    if not out.is_absolute() and not out.exists():
        candidate = cfg.OUTPUT_RAW / out
        if candidate.exists():
            out = candidate

    with np.load(out, allow_pickle=True) as data:
        result: Dict[str, object] = {
            "generation": int(data["generation"]),
            "energies": data["energies"],
            "T_clean": data["T_clean"],
            "log_T_clean": data["log_T_clean"],
            "clean_status": data["clean_status"],
            "modes": {},
        }
        modes: Dict[str, Dict[str, np.ndarray]] = {}
        for key in data.files:
            if "__" not in key:
                continue
            mode, field = key.split("__", 1)
            modes.setdefault(mode, {})[field] = data[key]
        result["modes"] = modes
        return result

def produce_raw_dataset(
    *,
    generations:Optional[Iterable[int]] = None,
    energies_ev: Optional[np.ndarray] = None,
    output_dir: PathLike = "study",
    modes: Optional[Sequence[str]] = None,
    validate: bool = True,
    validation_energies: Optional[Sequence[float]] = None,
    v_lead_ev: float=cfg.V_LEAD_EV,
) -> Dict[str, object]:
    "generate heatmaps for all generations and defect modes"
    cfg.ensure_output_dirs()
    gens = list(generations or cfg.GENERATIONS)
    energies = (
        np.asarray(energies_ev, dtype = np.float64)
        if energies_ev is not None else cfg.energy_grid()
    )
    modes = list(modes or cfg.DEFECT_MODES)

    root = Path(output_dir)
    if not root.is_absolute():
        root = cfg.OUTPUT_RAW / root
    root.mkdir(parents=True, exist_ok=True)

    t0=time.perf_counter()
    manifest: Dict[str, object] = {
        "output_dir": str(root),
        "files": [],
        "validation": [],
    }
    if validate: #check smallest generation
        check_geom = build_geometry(min(gens))
        check_energies = list(
            validation_energies or [float(energies[0]), float(energies[len(energies) // 2]), float(energies[-1])]
        )
        mismatches = validate_against_reference(
            check_geom,
            check_energies, modes = modes,
            v_lead_ev=v_lead_ev,
        )
        manifest["validation"].append({"generation": check_geom.generation, "mismatches": mismatches})
        if mismatches:
            raise RuntimeError(f"validation failed with {len(mismatches)} mismatches")
    for generation in gens:
        bundle = sweep_generation(
            generation,
            energies,
            modes=modes,
            v_lead_ev=v_lead_ev,
        )
        gen_dir = root / f"n{generation:02d}"
        for mode in modes:
            mode_bundle = {
                "generation": generation,
                "energies": bundle["energies"],
                "T_clean": bundle["T_clean"],
                "log_T_clean": bundle["log_T_clean"],
                "clean_status": bundle["clean_status"],
                "modes": {mode: bundle["modes"][mode]},
                "geometry_metadata": bundle["geometry_metadata"],
            }
            out = save_generation_raw(
                mode_bundle,
                gen_dir / f"{mode}.npz",
                metadata = run_metadata(
                    generations = [generation], energies = energies, extra={"mode":mode, "solver": "fast_prefix_suffix"},
                ),
            )
            manifest["files"].append(str(out.relative_to(root)))
    manifest["run_metadata"] = run_metadata(
        generations = gens,
        energies = energies,
        extra = {"modes": modes, "elapsed_s": time.perf_counter()-t0, "solver": "fast_prefix_suffix",},
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding = "utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest