"Producing raw data for the later heatmaps. Create and save various T_m(n,p,E) with clean transmission, validity, + metadata for generations 10 to 20"

from __future__ import annotations

import json 
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import hashlib
import resource
import shutil
import sys

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import StackCache, fast_sweep, validate_against_reference
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry, iter_study_geometries
from fibonacci_qc.scattering import clean_transmission_grid

PathLike = Union[str, Path]

def config_sha256() -> str:
    payload = cfg.CONFIG.as_dict()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _config_hash_of(bundle: Dict[str, object]) -> str:
    run = bundle.get("run_metadata") or {}
    digest = run.get("config_sha256")
    if not digest:
        raise ValueError("run_metadata.config_sha256 is missing")
    return str(digest)

def select_payload_modes(bundle: Dict[str, object], modes: Optional[Sequence[str]]) -> Dict[str, Dict[str, np.ndarray]]:
    available: Dict[str, Dict[str, np.ndarray]] = dict(bundle["modes"])
    unknown = [name for name in available if name not in cfg.DEFECT_MODES]
    if unknown:
        raise ValueError(f"payload mode keys {unknown} are outside {list(cfg.DEFECT_MODES)}")
    wanted = list(modes) if modes is not None else list(available)
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(f"requested modes {missing} are not in the payload; found {list(available)}")
    return {name: available[name] for name in wanted}

def assert_mode_axes(sweep: Dict[str, np.ndarray], energies: np.ndarray, *, where: str) -> None:
    transmission = np.asarray(sweep["T"])
    positions = np.asarray(sweep["positions"])
    valid = np.asarray(sweep["valid"])
    if transmission.ndim != 2 or transmission.shape[1] != int(energies.size):
        raise ValueError(f"transmission shape {transmission.shape} does not match energies at {where}")
    if positions.shape != (transmission.shape[0],) or valid.shape != (transmission.shape[0],):
        raise ValueError(f"positions/valid shape does not match transmission rows at {where}")
    if not np.array_equal(positions, np.arange(transmission.shape[0])):
        raise ValueError(f"positions are not 0..N-1 at {where}")
    
def assert_same_run(anchor: Dict[str, object], candidate: Dict[str, object], *, where: str, same_generation: bool) -> None:
    anchor_e = np.asarray(anchor["energies"], dtype=np.float64)
    candidate_e = np.asarray(candidate["energies"], dtype=np.float64)
    if not np.array_equal(anchor_e, candidate_e):
        raise ValueError(f"energy grid mismatch at {where}")
    if _config_hash_of(anchor) != _config_hash_of(candidate):
        raise ValueError(f"config hash mismatch at {where}")
    for mode, sweep in dict(candidate["modes"]).items():
        assert_mode_axes(sweep, candidate_e, where=f"{where} mode {mode}")
        if mode in dict(anchor["modes"]):
            other = dict(anchor["modes"])[mode]
            if not np.array_equal(np.asarray(sweep["positions"]), np.asarray(other["positions"])):
                raise ValueError(f"position mismatch for {mode} at {where}")
    if same_generation:
        if int(anchor["generation"]) != int(candidate["generation"]):
            raise ValueError(f"generation id mismatch at {where}")
        for key in ("T_clean", "log_T_clean"):
            if not np.array_equal(np.asarray(anchor[key]), np.asarray(candidate[key]), equal_nan=True):
                raise ValueError(f"{key} mismatch at {where}")
        if (anchor.get("geometry_metadata") or {}) != (candidate.get("geometry_metadata") or {}):
            raise ValueError(f"geometry metadata mismatch at {where}")
        
def validate_loaded_bundle(bundle: Dict[str, object], *, generation: int, modes: Optional[Sequence[str]]) -> Dict[str, object]:
    if int(bundle["generation"]) != int(generation):
        raise ValueError(f"payload generation {bundle['generation']} does not match directory n{generation:02d}")
    energies = np.asarray(bundle["energies"], dtype=np.float64)
    if np.asarray(bundle["T_clean"]).shape != energies.shape or np.asarray(bundle["log_T_clean"]).shape != energies.shape:
        raise ValueError("clean curves do not match the energy grid")
    selected = select_payload_modes(bundle, modes)
    for mode, sweep in selected.items():
        assert_mode_axes(sweep, energies, where=f"n{generation:02d}/{mode}")
    checked = dict(bundle)
    checked["modes"] = selected
    return checked

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
        "config_sha256": config_sha256(),
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

    t_clean, log_clean, clean_status = None
    mode_data: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in modes:
        sweep = fast_sweep(mode, energies, geometry=geom, v_lead_ev=v_lead_ev)
        mode_data[mode] = sweep
        if t_clean is None:
            t_clean = sweep["T_clean"]
            log_clean = sweep["log_T_clean"]
            clean_status = np.array(["ok" if np.isfinite(t) else "invalid" for t in t_clean], dtype=object)
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
    if not out.is_absolute():
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
        meta_path = out.with_suffix(".meta.json")
        run_metadata = {}
        geometry_metadata = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            run_metadata = meta.get("run_metadata") or {}
            geometry_metadata = meta.get("geometry_metadata")
        modes: Dict[str, Dict[str, np.ndarray]] = {}
        for key in data.files:
            if "__" not in key:
                continue
            mode, field = key.split("__", 1)
            modes.setdefault(mode, {})[field] = data[key]
        result["modes"] = modes
        result["run_metadata"] = run_metadata
        result["geometry_metadata"] = geometry_metadata
        return result

def assert_same_energy_grid(a: np.ndarray, b: np.ndarray, *, where: str) -> None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=0.0):
        raise ValueError(f"energy grid mismatch at {where}")

def load_generation_bundle(
    gen_dir: PathLike,
    *,
    modes: Optional[Sequence[str]] = None,
    require_modes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    "Loads one generation & validates generation, energies, T_clean across files."
    gen_dir = Path(gen_dir)
    wanted = list(require_modes or modes or cfg.DEFECT_MODES)
    mode_data: Dict[str, Dict[str, np.ndarray]] = {}
    energies = None
    t_clean = None
    log_clean = None
    generation = None
    meta_run = None

    for path in sorted(gen_dir.glob("*.npz")):
        raw = load_generation_raw(path)
        gen = int(raw["generation"])
        if generation is None:
            generation = gen
            energies = np.asarray(raw["energies"], dtype=np.float64)
            t_clean = np.asarray(raw["T_clean"], dtype=np.float64)
            log_clean = np.asarray(raw["log_T_clean"], dtype=np.float64)
            meta_run = raw.get("run_metadata")
        else:
            if gen != generation:
                raise ValueError(f"generation id mismatch in {path}")
            assert_same_energy_grid(energies, raw["energies"], where=str(path))
            if not np.allclose(t_clean, raw["T_clean"]):
                raise ValueError(f"T_clean mismatch in {path}")

        for mode_name, sweep in raw["modes"].items():
            if modes is not None and mode_name not in modes:
                continue
            if mode_name in mode_data:
                raise ValueError(f"duplicate mode {mode_name!r} while loading {gen_dir}")
            mode_data[mode_name] = sweep

    missing = [m for m in wanted if m not in mode_data]
    if missing:
        raise FileNotFoundError(f"missing modes {missing} under {gen_dir}")

    return {
        "generation": generation,
        "energies": energies,
        "T_clean": t_clean,
        "log_T_clean": log_clean,
        "modes": mode_data,
        "run_metadata": meta_run,
    }

def validate_bundle_config(bundle: Dict[str, object]) -> None:
    "Refuse analysis if the saved model disagrees w/ current"
    stored = (bundle.get("run_metadata") or {}).get("config")
    if not stored:
        raise ValueError("raw bundle missing run_metadata.config; refuse to analyze")
    current = cfg.CONFIG.as_dict()
    for key in ("d_a_nm", "d_b_nm", "v_a_ev", "v_b_ev", "v_lead_ev", "hbar2_over_2m_ev_nm2", "energy_windows", "scaling_d_floor", "scaling_min_usable_generations", "scaling_loo_alpha_tol", "scaling_rmse_max"):
        if stored.get(key) != current.get(key):
            raise ValueError(f"config drift at {key!r}: stored={stored.get(key)!r} current={current.get(key)!r}")

def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss) * 1024

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
    peak_rss = _rss_bytes()
    for generation in gens:
        geom = build_geometry(generation)
        clean_reference = None
        log_reference = None
        clean_status = None
        gen_dir = root / f"n{generation:02d}"
        for mode in modes:
            spill = gen_dir / "_spill" / mode
            sweep = fast_sweep(
                mode,
                energies,
                geometry=geom,
                v_lead_ev=v_lead_ev,
                energy_chunk=64,
                spill_dir=spill,
            )
            if clean_reference is None:
                clean_reference = np.array(sweep["T_clean"], copy=True)
                log_reference = np.array(sweep["log_T_clean"], copy=True)
                clean_status = np.array(["ok" if np.isfinite(value) else "invalid" for value in clean_reference], dtype="<U16")
            elif not np.array_equal(clean_reference, np.asarray(sweep["T_clean"])):
                raise RuntimeError(f"clean curve changed between modes at generation {generation}")
            mode_bundle = {
                "generation": generation,
                "energies": energies,
                "T_clean": clean_reference,
                "log_T_clean": log_reference,
                "clean_status": clean_status,
                "modes": {mode: sweep},
                "geometry_metadata": geom.to_metadata(),
            }
            out = save_generation_raw(
                mode_bundle,
                gen_dir / f"{mode}.npz",
                metadata=run_metadata(generations=[generation], energies=energies, extra={"mode": mode, "solver": "fast_prefix_suffix"}),
            )
            manifest["files"].append(str(out.relative_to(root)))
            peak_rss = max(peak_rss, _rss_bytes())
            del sweep, mode_bundle
            shutil.rmtree(spill, ignore_errors=True)
    manifest["peak_rss_bytes"] = peak_rss
    manifest["run_metadata"] = run_metadata(generations = gens, energies = energies, extra = {"modes": modes, "elapsed_s": time.perf_counter()-t0, "solver": "fast_prefix_suffix",})
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding = "utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest