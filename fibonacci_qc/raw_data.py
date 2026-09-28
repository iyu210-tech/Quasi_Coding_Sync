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
import uuid

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import StackCache, fast_sweep, validate_against_reference
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry, iter_study_geometries
from fibonacci_qc.scattering import encode_status, SOLVER_REVISION, trusted_transmission_mask, trusted_log_mask

SCHEMA_VERSION = 2
MODEL_KEYS = ("length_unit", 
            "energy_unit", 
            "hbar2_over_2m_ev_nm2", 
            "v_lead_ev",
            "d_a_nm", 
            "d_b_nm", 
            "v_a_ev",
            "v_b_ev",
            "word_w1",
            "word_w2",
            "substitution",
            "layer_width_nm",
            "layer_potential_ev",
            "defect_modes",
            "site_motif_radius",
            "bond_motif_radius",
            "lead_left_token",
            "lead_right_token"
            )

def _model_hash(snapshot):
    data = {key: snapshot[key] for key in MODEL_KEYS}
    data["solver_revision"] = SOLVER_REVISION
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def config_sha256():
    "Physical model/geometry/solver identity"
    return _model_hash(cfg.CONFIG.as_dict())

def _config_hash_of(bundle):
    digest = (bundle.get("run_metadata") or {}).get("config_sha256")
    if not digest:
        raise ValueError("run_metadata.config_sha256 is missing")
    return str(digest)

def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()

def _git_metadata():
    try:
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=cfg.PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return dict(commit=git("rev-parse","HEAD"), branch=git("rev-parse","--abbrev-ref","HEAD"), dirty=git("status","--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return dict(commit="unknown", branch="unknown", dirty="unknown")

def run_metadata(*, generations, energies, extra=None):
    e = np.asarray(energies, dtype=np.float64)
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(), generations=list(generations),
                n_energy=int(e.size), energy_min_ev=float(e[0]) if e.size else None,
                energy_max_ev=float(e[-1]) if e.size else None,
                energy_sha256=hashlib.sha256(e.tobytes()).hexdigest(),
                defect_modes=list(cfg.DEFECT_MODES), config=cfg.CONFIG.as_dict(), git=_git_metadata(),
                config_sha256=config_sha256(), schema_version=SCHEMA_VERSION,
                solver_revision=SOLVER_REVISION)
    if extra: data.update(extra)
    return data

def select_payload_modes(bundle, modes):
    available = dict(bundle["modes"])
    wanted = list(available) if modes is None else list(modes)
    if not wanted or any(m not in cfg.DEFECT_MODES for m in available):
        raise ValueError("empty or unknown modes in raw bundle")
    missing = [m for m in wanted if m not in available]
    if missing: raise FileNotFoundError(f"missing requested modes: {missing}")
    return {m: available[m] for m in wanted}

def _check_energy(e):
    e = np.asarray(e, dtype=float)
    if e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0):
        raise ValueError("raw energy coordinates must be finite, strictly increasing, and length >= 2")
    return e

def assert_same_energy_grid(a, b, *, where):
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise ValueError(f"energy grid mismatch at {where}")

def assert_mode_axes(sweep, energies, *, where):
    t = np.asarray(sweep["T"]); p = np.asarray(sweep["positions"]); valid = np.asarray(sweep["valid"])
    if t.ndim != 2 or t.shape[1] != len(energies) or p.shape != (t.shape[0],) or valid.shape != p.shape:
        raise ValueError(f"mode axes disagree at {where}")
    if not np.array_equal(p, np.arange(p.size)):
        raise ValueError(f"positions must be the full, uncompacted axis at {where}")
    for key in ("log_T", "status"):
        if key not in sweep or np.asarray(sweep[key]).shape != t.shape:
            raise ValueError(f"{key} missing or wrong shape at {where}")
    if "delta_T" in sweep and np.asarray(sweep["delta_T"]).shape != t.shape:
        raise ValueError(f"delta_T shape disagrees at {where}")
    if "energies" in sweep: assert_same_energy_grid(sweep["energies"], energies, where=where)
    code = encode_status(sweep["status"]); log = np.asarray(sweep["log_T"])
    certified = (code == 0) | (code == 2)
    good = trusted_transmission_mask(code, t, log) & trusted_log_mask(code, log)
    if np.any(certified & ~good):
        raise ValueError(f"certified status contradicts T/log_T at {where}")

def validate_bundle_config(bundle):
    meta = bundle.get("run_metadata") or {}
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("solver_revision") != SOLVER_REVISION:
        raise ValueError("raw bundle is not from the repaired solver/schema; regenerate")
    stored = meta.get("config")
    if not stored or _model_hash(stored) != _config_hash_of(bundle) or _config_hash_of(bundle) != config_sha256():
        raise ValueError("raw physical-model/config identity does not match current solver")

def validate_loaded_bundle(bundle, *, generation, modes=None):
    if int(bundle["generation"]) != int(generation): raise ValueError("generation id mismatch")
    e = _check_energy(bundle["energies"]); geom = build_geometry(int(generation))
    for key in ("T_clean", "log_T_clean", "clean_status"):
        if key not in bundle or np.asarray(bundle[key]).shape != e.shape:
            raise ValueError(f"{key} does not match energy axis")
    clean_code = encode_status(bundle["clean_status"])
    clean_good = trusted_transmission_mask(clean_code, bundle["T_clean"], bundle["log_T_clean"]) & trusted_log_mask(clean_code, bundle["log_T_clean"])
    if np.any(((clean_code == 0)|(clean_code == 2)) & ~clean_good):
        raise ValueError("clean status contradicts clean numerical values")
    if bundle.get("geometry_metadata") != geom.to_metadata():
        raise ValueError("geometry metadata does not match generation")
    selected = select_payload_modes(bundle, modes)
    for mode, sweep in selected.items():
        assert_mode_axes(sweep, e, where=f"n{generation}/{mode}")
        n = geom.n_bonds if mode == cfg.MODE_PHASON else geom.n_sites
        if np.asarray(sweep["T"]).shape[0] != n: raise ValueError("geometry and position count disagree")
        expected = geom.phason_valid if mode == cfg.MODE_PHASON else np.ones(n, dtype=bool)
        if not np.array_equal(np.asarray(sweep["valid"],dtype=bool), expected):
            raise ValueError("saved geometric eligibility differs from model")
    validate_bundle_config(bundle)
    return {**bundle, "modes": selected, "clean_status": clean_code}

def assert_same_run(anchor, candidate, *, where, same_generation):
    assert_same_energy_grid(anchor["energies"], candidate["energies"], where=where)
    if _config_hash_of(anchor) != _config_hash_of(candidate): raise ValueError(f"config hash mismatch at {where}")
    ar, cr = (anchor.get("run_metadata") or {}).get("run_id"), (candidate.get("run_metadata") or {}).get("run_id")
    if ar != cr: raise ValueError(f"run id mismatch at {where}")
    for mode, sweep in candidate["modes"].items(): assert_mode_axes(sweep, candidate["energies"], where=where)
    if not same_generation: return
    if int(anchor["generation"]) != int(candidate["generation"]): raise ValueError(f"generation mismatch at {where}")
    for key in ("T_clean","log_T_clean"):
        if not np.array_equal(anchor[key], candidate[key], equal_nan=True): raise ValueError(f"{key} mismatch at {where}")
    if not np.array_equal(encode_status(anchor["clean_status"]), encode_status(candidate["clean_status"])):
        raise ValueError(f"clean status mismatch at {where}")
    if anchor["geometry_metadata"] != candidate["geometry_metadata"]: raise ValueError(f"geometry mismatch at {where}")
    for mode in set(anchor["modes"]) & set(candidate["modes"]):
        if not np.array_equal(anchor["modes"][mode]["positions"], candidate["modes"][mode]["positions"]):
            raise ValueError(f"position mismatch at {where}")

def sweep_generation(generation, energies_ev, *, modes=None, geometry=None, v_lead_ev=cfg.V_LEAD_EV):
    if v_lead_ev != cfg.V_LEAD_EV: raise ValueError("saved study uses the configured exterior potential")
    geom = build_geometry(generation) if geometry is None else geometry
    if geom.generation != generation: raise ValueError("generation/geometry mismatch")
    e = _check_energy(energies_ev); wanted = list(cfg.DEFECT_MODES if modes is None else modes)
    if not wanted: raise ValueError("no modes requested")
    sweeps = {m: fast_sweep(m,e,geometry=geom,v_lead_ev=v_lead_ev) for m in wanted}
    first = next(iter(sweeps.values()))
    return dict(generation=generation, energies=e, T_clean=first["T_clean"],
                log_T_clean=first["log_T_clean"], clean_status=first["clean_status"], modes=sweeps,
                geometry_metadata=geom.to_metadata(), run_metadata=run_metadata(generations=[generation],energies=e))

def save_generation_raw(bundle, path, *, metadata=None):
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute(): out = cfg.OUTPUT_RAW/out
    if out.exists(): raise FileExistsError(f"refusing to replace frozen raw data: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(bundle.get("run_metadata") or run_metadata(generations=[int(bundle["generation"])],energies=bundle["energies"]))
    if metadata: meta.update(metadata)
    checked = validate_loaded_bundle({**bundle,"run_metadata":meta},generation=int(bundle["generation"]))
    payload = {k: np.asarray(checked[k]) for k in ("generation","energies","T_clean","log_T_clean")}
    payload["clean_status"] = encode_status(checked["clean_status"])
    for mode, sweep in checked["modes"].items():
        for key, value in sweep.items():
            if key in ("mode","generation"): continue
            payload[f"{mode}__{key}"] = encode_status(value) if key in ("status","clean_status") else value
    np.savez_compressed(out, **payload)
    sidecar = dict(npz=out.name, generation=int(bundle["generation"]), geometry_metadata=checked["geometry_metadata"],
                   run_metadata=meta, data_sha256=_file_hash(out))
    out.with_suffix(".meta.json").write_text(json.dumps(sidecar,indent=2),encoding="utf-8")
    return out

def load_generation_raw(path):
    out = Path(path)
    if not out.is_absolute() and not out.exists(): out = cfg.OUTPUT_RAW/out
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("data_sha256") and meta["data_sha256"] != _file_hash(out):
        raise ValueError(f"raw data checksum mismatch: {out}")
    with np.load(out,allow_pickle=True) as data:
        result = {key: data[key] for key in ("energies","T_clean","log_T_clean","clean_status")}
        result["generation"] = int(data["generation"]); mode_data = {}
        for key in data.files:
            if "__" in key:
                mode, field = key.split("__",1)
                mode_data.setdefault(mode,{})[field] = encode_status(data[key]) if field in ("status","clean_status") else data[key]
        result.update(modes=mode_data, clean_status=encode_status(result["clean_status"]),
                      geometry_metadata=meta.get("geometry_metadata"), run_metadata=meta.get("run_metadata") or {})
        return result

def load_generation_bundle(gen_dir, *, modes=None, require_modes=None):
    folder = Path(gen_dir); anchor = None; mode_data = {}
    paths = sorted(folder.glob("*.npz"))
    if not paths: raise FileNotFoundError(f"no raw files under {folder}")
    for path in paths:
        raw = load_generation_raw(path)
        generation = int(raw["generation"])
        if folder.name.startswith("n") and folder.name[1:].isdigit() and int(folder.name[1:]) != generation:
            raise ValueError("generation directory and file disagree")
        raw = validate_loaded_bundle(raw,generation=generation)
        if anchor is None: anchor = raw
        else: assert_same_run(anchor,raw,where=str(path),same_generation=True)
        for mode, sweep in raw["modes"].items():
            if mode in mode_data: raise ValueError(f"duplicate mode {mode!r}")
            mode_data[mode] = sweep
    wanted = require_modes if require_modes is not None else modes
    result = {**anchor,"modes":mode_data}
    if wanted is not None: select_payload_modes(result,wanted)
    return validate_loaded_bundle(result,generation=int(anchor["generation"]),modes=modes)

def _rss_bytes():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss)*1024

def produce_raw_dataset(*, generations=None, energies_ev=None, output_dir="study", modes=None,
                        validate=True, validation_energies=None, v_lead_ev=cfg.V_LEAD_EV):
    if v_lead_ev != cfg.V_LEAD_EV: raise ValueError("study raw files require configured exterior potential")
    gens = list(cfg.GENERATIONS if generations is None else generations)
    wanted = list(cfg.DEFECT_MODES if modes is None else modes)
    if not gens or not wanted or len(set(wanted)) != len(wanted): raise ValueError("nonempty unique modes/generations required")
    e = _check_energy(cfg.energy_grid() if energies_ev is None else energies_ev)
    root = Path(output_dir)
    if not root.is_absolute(): root = cfg.OUTPUT_RAW/root
    if root.exists() and any(root.iterdir()): raise FileExistsError(f"use a fresh raw run directory: {root}")
    root.mkdir(parents=True,exist_ok=True); run_id = uuid.uuid4().hex; start = time.perf_counter()
    meta = run_metadata(generations=gens,energies=e,extra={"run_id":run_id,"modes":wanted})
    manifest = dict(output_dir=str(root.resolve()),files=[],validation=[],run_metadata=meta)
    for gen in gens:
        geom = build_geometry(gen)
        if validate:
            check_e = sorted(set(validation_energies if validation_energies is not None else [e[0],e[len(e)//2],e[-1],5.0]))
            subset = {}
            for mode in wanted:
                pos = np.flatnonzero(geom.phason_valid) if mode == cfg.MODE_PHASON else np.arange(geom.n_sites)
                subset[mode] = sorted(set(int(pos[i]) for i in (0,len(pos)//2,len(pos)-1))) if pos.size else []
            errors = validate_against_reference(geom,check_e,modes=wanted,positions_per_mode=subset)
            manifest["validation"].append(dict(generation=gen,positions=subset,energies=check_e,mismatches=errors))
            if errors: raise RuntimeError(f"fast/reference cross-check failed: {errors[:3]}")
        for mode in wanted:
            spill = root/f"n{gen:02d}"/"_spill"/mode
            sweep = fast_sweep(mode,e,geometry=geom,energy_chunk=64,spill_dir=spill)
            bundle = dict(generation=gen,energies=e,T_clean=sweep["T_clean"],log_T_clean=sweep["log_T_clean"],
                          clean_status=sweep["clean_status"],modes={mode:sweep},geometry_metadata=geom.to_metadata(),run_metadata=meta)
            out = save_generation_raw(bundle,root/f"n{gen:02d}"/(mode+".npz"))
            manifest["files"].append(str(out.relative_to(root)))
            del sweep,bundle
            shutil.rmtree(spill)
    manifest.update(peak_rss_bytes=_rss_bytes(),elapsed_s=time.perf_counter()-start)
    path = root/"manifest.json";path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    manifest["manifest_path"] = str(path)
    return manifest