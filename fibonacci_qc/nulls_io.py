"Save and load geometry-null and robustness analysis results"

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import numpy as np
from dataclasses import asdict

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.geometry_nulls import (PermutationNullResult, analyze_generation_nulls, null_summary_table,)
from fibonacci_qc.raw_data import load_generation_raw, load_generation_bundle
from fibonacci_qc.robustness import (ContrastRobustness, ResonanceRobustness, analyze_generation_robustness,)

PathLike = Union[str, Path]

def _contrast_to_dict(contrast):
    return asdict(contrast)

def _resonance_to_dict(row):
    return asdict(row)

def save_nulls_analysis(
    analysis: Dict[str, object],
    path: PathLike,
    *,
    metadata: Optional[Dict[str, object]] = None,
    n_replicates: int = cfg.PERMUTATION_REPLICATES,
) -> Path:
    "Permutation-null results to .npz + JSON metadata."
    cfg.ensure_output_dirs()
    n_replicates = int(analysis["n_replicates"])
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_ANALYSIS / out
    out.parent.mkdir(parents=True, exist_ok=True)
    generation = int(analysis["generation"])
    energies = np.asarray(analysis["energies"], dtype=np.float64)
    np_payload: Dict[str, object] = {
        "generation": np.array(generation, dtype=np.int64),
        "energies": energies,
        "n_replicates": np.array(analysis["n_replicates"], dtype=np.int64),
        "seed": np.array(analysis["seed"], dtype=np.int64),
    }
    json_modes: Dict[str, object] = {}
    modes: Dict[str, Dict[str, PermutationNullResult]] = analysis["modes"]  # type: ignore[assignment]
    for mode, strata in modes.items():
        json_modes[mode] = {}
        for family, result in strata.items():
            prefix = f"{mode}__{family}__"
            np_payload[prefix + "null_S_max"] = result.null_S_max
            np_payload[prefix + "null_S_int"] = result.null_S_int
            np_payload[prefix + "D_observed"] = result.D_observed
            np_payload[prefix + "D_geometry_expected"] = result.D_geometry_expected
            np_payload[prefix + "D_residual"] = result.D_residual
            json_modes[mode][family] = {
                "statistic_revision": result.statistic_revision,
                "statistic_curve": "D_observed - D_geometry_expected",
                "inference_valid": result.inference_valid,
                "inference_reason": result.inference_reason,
                "raw_S_max": result.raw_observed.S_max,
                "raw_S_int": result.raw_observed.S_int,
                "residual_abs_S_max": result.residual_abs_S_max,
                "residual_abs_S_int": result.residual_abs_S_int,
                "S_max": result.observed.S_max,
                "S_int": result.observed.S_int,
                "centered_S_max": result.centered_S_max,
                "centered_S_int": result.centered_S_int,
                "p_S_max": result.p_S_max,
                "p_S_int": result.p_S_int,
                "n_replicates": result.n_replicates,
                "seed": result.seed,
            }
    np.savez_compressed(out, **np_payload)
    meta = {
        "npz": out.name,
        "generation": generation,
        "geometry_metadata": analysis.get("geometry_metadata"),
        "summary": null_summary_table(analysis),
        "modes": json_modes,
        "run_metadata": metadata or {},
        "primary_stratum_families": list(cfg.GEOMETRY_STRATA),
        "mc_p_resolution": float(1.0 / (n_replicates + 1)) if n_replicates >= 1 else float("nan"),
        "pvalue_note": ("Families not in primary_stratum_families are exploratory (mc_p_resolution is 1/(B+1)"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out

def load_nulls_analysis(path: PathLike) -> Dict[str, object]:
    "Load saved geometry null analysis bundle"
    out = Path(path)
    if not out.is_absolute() and not out.exists():
        candidate = cfg.OUTPUT_ANALYSIS / out
        if candidate.exists():
            out = candidate
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    with np.load(out, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    return {"arrays": arrays, "metadata": meta}

def save_robustness_analysis(
    analysis: Dict[str, object],
    path: PathLike,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    "Persist robustness comparisons to .npz + JSON metadata."
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_ANALYSIS / out
    out.parent.mkdir(parents=True, exist_ok=True)
    generation = int(analysis["generation"])
    np_payload: Dict[str, object] = {"generation": np.array(generation, dtype=np.int64),}
    json_modes: Dict[str, object] = {}
    modes: Dict[str, object] = analysis["modes"]
    for mode, payload in modes.items():
        prefix = f"{mode}__"
        np_payload[prefix + "D_baseline"] = payload["D_baseline"]
        np_payload[prefix + "D_fine"] = payload["D_fine"]
        np_payload[prefix + "energies_baseline"] = payload["energies_baseline"]
        np_payload[prefix + "energies_fine"] = payload["energies_fine"]
        contrast: ContrastRobustness = payload["contrast"]
        resonance_rows: Sequence[ResonanceRobustness] = payload["resonance"]
        json_modes[mode] = {
            "n_energy_baseline": payload["n_energy_baseline"],
            "n_energy_fine": payload["n_energy_fine"],
            "contrast": _contrast_to_dict(contrast),
            "resonance": [_resonance_to_dict(r) for r in resonance_rows],
            "contrast_gate": payload["contrast_gate"],
            "primary_gate": payload["primary_gate"],
            "numerical_gate": payload["numerical_gate"],
            "response_baseline": payload["response_baseline"],
            "response_fine": payload["response_fine"],
            "response_comparison": payload["response_comparison"],
            "all_valid_resonance_positions": payload["all_valid_resonance_positions"],
        }
    np.savez_compressed(out, **np_payload)
    meta = {
        "npz": out.name,
        "generation": generation,
        "geometry_metadata": analysis.get("geometry_metadata"),
        "modes": json_modes,
        "run_metadata": metadata or {},
        "primary_gate": analysis["primary_gate"],
        "clean_peaks_baseline": analysis["clean_peaks_baseline"],
        "clean_peaks_fine": analysis["clean_peaks_fine"],
        "clean_peaks_complete": analysis["clean_peaks_complete"],
        "interaction": analysis["interaction"],
        "interaction_gate": analysis["interaction_gate"],
        "gate_tolerances": analysis["gate_tolerances"],
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out

def load_robustness_analysis(path: PathLike) -> Dict[str, object]:
    "Load saved robustness analysis bundle"
    out = Path(path)
    if not out.is_absolute() and not out.exists():
        candidate = cfg.OUTPUT_ANALYSIS / out
        if candidate.exists():
            out = candidate
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    with np.load(out, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    return {"arrays": arrays, "metadata": meta}

def produce_nulls_analysis(
    raw_dir: PathLike,
    *,
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    n_replicates: int = cfg.PERMUTATION_REPLICATES,
    seed: int = 0,
    output_dir: PathLike = "nulls",
) -> Dict[str, object]:
    "Batch geometry-null analysis over the study directory."
    cfg.ensure_output_dirs()
    raw_root = Path(raw_dir)
    if not raw_root.is_absolute():
        raw_root = cfg.OUTPUT_RAW / raw_root
    out_root = Path(output_dir)
    if not out_root.is_absolute():
        out_root = cfg.OUTPUT_ANALYSIS / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if generations is None:
        generations = sorted(
            int(p.name[1:])
            for p in raw_root.glob("n*")
            if p.is_dir() and p.name[1:].isdigit()
        )
    manifest: Dict[str, object] = {
        "files": [],
        "generations": list(generations),
        "n_replicates": n_replicates,
        "seed": seed,
        "primary_stratum_families": list(cfg.GEOMETRY_STRATA),
        "mc_p_resolution": float(1.0 / (n_replicates + 1)) if n_replicates >= 1 else float("nan"),
        "pvalue_note": ("Families not in primary_stratum_families are exploratory (mc_p_resolution is 1/(B+1)"),
    }
    for generation in generations:
        gen_dir = raw_root / f"n{generation:02d}"
        loaded = load_generation_bundle(gen_dir, modes=modes)
        bundle = {
            "generation": int(loaded["generation"]),
            "energies": loaded["energies"],
            "T_clean": loaded["T_clean"],
            "modes": loaded["modes"],
            "geometry_metadata": loaded.get("geometry_metadata"),
            "run_metadata": loaded.get("run_metadata"),
        }
        geom = build_geometry(generation)
        analysis = analyze_generation_nulls(
            bundle,
            geom,
            modes=list(bundle["modes"]),
            n_replicates=n_replicates,
            seed=seed,
        )
        out = save_nulls_analysis(analysis, out_root / f"n{generation:02d}" / "nulls.npz", metadata={"source_raw_dir": str(raw_root)},)
        manifest["files"].append(str(out.relative_to(out_root)))
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest

def produce_robustness_analysis(
    *, 
    generations=None,
    modes=None,
    energies_baseline=None,
    energies_fine=None,
    output_dir="robustness",
    raw_dir=None
):
    from fibonacci_qc.robustness import worst_gate
    cfg.ensure_output_dirs()
    gens=list(cfg.GENERATIONS if generations is None else generations)
    if len(gens) != len(set(gens)) or not gens:
        raise ValueError("nonempty unique generations required")
    clean_base, clean_fine, clean_complete = {}, {}, True
    wanted=list(cfg.DEFECT_MODES if modes is None else modes)
    root=Path(output_dir)
    if not root.is_absolute():root=cfg.OUTPUT_ANALYSIS/root
    root.mkdir(parents=True,exist_ok=True)
    rawroot=Path(raw_dir) if raw_dir is not None else None
    if rawroot is not None and not rawroot.is_absolute():rawroot=cfg.OUTPUT_RAW/rawroot
    manifest={
        "files":[],
        "generations":gens,
        "modes":wanted,
        "generation_gates":{},
        "gate_tolerances":dict(cfg.GATE_TOLERANCES)
    }
    for gen in gens:
        bundle=load_generation_bundle(rawroot/f"n{gen:02d}",modes=wanted) if rawroot is not None else None
        analysis=analyze_generation_robustness(gen,modes=wanted,energies_baseline=energies_baseline, energies_fine=energies_fine,baseline_bundle=bundle)
        path=save_robustness_analysis(analysis,root/f"n{gen:02d}"/"robustness.npz", metadata={"source_raw_dir":str(rawroot) if rawroot is not None else None})
        manifest["files"].append(str(path.relative_to(root)))
        manifest["generation_gates"][str(gen)]=analysis["primary_gate"]
        clean_base[gen] = analysis["clean_peaks_baseline"]
        clean_fine[gen] = analysis["clean_peaks_fine"]
        clean_complete &= analysis["clean_peaks_complete"]

    from fibonacci_qc.cross_generation import compare_clean_anchor_grid

    cross = compare_clean
    manifest["primary_gate"]=worst_gate(manifest["generation_gates"].values())
    path=root/"manifest.json";path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    manifest["manifest_path"]=str(path)
    return manifest