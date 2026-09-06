"Save and load parity analysis products"

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union, Sequence
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.parity_analysis import(IntegratedContrast, ParityCheckResult, ParityContrastResult, analyze_generation_parity,)
from fibonacci_qc.raw_data import load_generation_raw

PathLike = Union[str, Path]

def _integrated_to_dict(metrics: Dict[str, IntegratedContrast]) -> Dict[str, object]:
    return {name: asdict(value) for name, value in metrics.items()}

def _check_to_dict(check: ParityCheckResult) -> Dict[str, object]:
    return asdict(check)

def _contrast_to_arrays(result: ParityContrastResult) -> Dict[str, np.ndarray]:
    return {
        "D": result.D,
        "n_even": np.array(result.n_even, dtype=np.int64),
        "n_odd": np.array(result.n_odd, dtype=np.int64),
    }

def save_parity_analysis(
    analysis: Dict[str, object],
    path: PathLike,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    "Persist a generation parity analysis to .npz and JSON metadata"
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_ANALYSIS / out
    out.parent.mkdir(parents=True, exist_ok=True)
    generation = int(analysis["generation"])
    energies = np.asarray(analysis["energies"], dtype=np.float64)
    np_payload: Dict[str, object] = {
        "generation": np.array(generation, dtype=np.int64),
        "energies": energies,
    }

    modes: Dict[str, object] = analysis["modes"]  # type: ignore[assignment]
    for mode, payload in modes.items():
        prefix = f"{mode}__"
        unconditional: ParityContrastResult = payload["unconditional"]  # type: ignore[index]
        for key, value in _contrast_to_arrays(unconditional).items():
            np_payload[prefix + "unconditional__" + key] = value
        conditional: Dict[str, Dict[str, ParityContrastResult]] = payload["conditional"]  # type: ignore[index]
        for family, strata in conditional.items():
            for label, result in strata.items():
                for key, value in _contrast_to_arrays(result).items():
                    field = f"{prefix}conditional__{family}__{label}__{key}"
                    np_payload[field] = value
        np_payload[prefix + "parity"] = np.asarray(payload["parity"])  # type: ignore[index]
        np_payload[prefix + "valid"] = np.asarray(payload["valid"])  # type: ignore[index]
    np.savez_compressed(out, **np_payload)
    meta = {
        "npz": out.name,
        "generation": generation,
        "geometry_metadata": analysis.get("geometry_metadata"),
        "integrated": {
            mode: {
                "unconditional": _integrated_to_dict(
                    payload["integrated"]["unconditional"]  # type: ignore[index]
                ),
                "conditional": {
                    family: {
                        label: _integrated_to_dict(metrics)
                        for label, metrics in strata.items()
                    }
                    for family, strata in payload["integrated"]["conditional"].items()  # type: ignore[index]
                },
            }
            for mode, payload in modes.items()
        },
        "checks": {
            mode: {
                name: _check_to_dict(check)
                for name, check in payload["checks"].items()  # type: ignore[index]
            }
            for mode, payload in modes.items()
        },
        "run_metadata": metadata or {},
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out

def load_parity_analysis(path: PathLike) -> Dict[str, object]:
    "Load saved parity analysis bundle"
    out = Path(path)
    if not out.is_absolute() and not out.exists():
        candidate = cfg.OUTPUT_ANALYSIS / out
        if candidate.exists():
            out = candidate
    meta_path = out.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(out, allow_pickle=True) as data:
        generation = int(data["generation"])
        energies = data["energies"]
        modes: Dict[str, object] = {}
        mode_names = sorted(
            {
                key.split("__", 1)[0]
                for key in data.files
                if "__" in key and not key.startswith("generation")
            }
        )
        for mode in mode_names:
            prefix = f"{mode}__"
            modes[mode] = {
                "generation": generation,
                "energies": energies,
                "parity": data[prefix + "parity"],
                "valid": data[prefix + "valid"],
                "unconditional_D": data[prefix + "unconditional__D"],
            }
        return {
            "generation": generation,
            "energies": energies,
            "modes": modes,
            "metadata": meta,
        }

def analyze_raw_file(
    raw_path: PathLike,
    *,
    modes: Optional[Sequence[str]] = None,
    save_path: Optional[PathLike] = None,
) -> Dict[str, object]:
    "Load raw generation file, run parity analysis, optional save"
    raw = load_generation_raw(raw_path)
    generation = int(raw["generation"])
    geom = build_geometry(generation)
    analysis = analyze_generation_parity(raw, geom, modes=modes)
    if save_path is not None:
        save_parity_analysis(analysis, save_path)
    return analysis

def produce_parity_analysis(
    raw_dir: PathLike,
    *,
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    output_dir: PathLike = "parity",
) -> Dict[str, object]:
    "Parity analysis for all generations under raw study directory"
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
    manifest: Dict[str, object] = {"files": [], "generations": list(generations)}
    for generation in generations:
        gen_dir = raw_root / f"n{generation:02d}"
        raw_files = sorted(gen_dir.glob("*.npz"))
        if not raw_files:
            continue
        mode_data: Dict[str, Dict[str, np.ndarray]] = {}
        energies = None
        for raw_file in raw_files:
            mode_name = raw_file.stem
            if modes is not None and mode_name not in modes:
                continue
            raw = load_generation_raw(raw_file)
            if energies is None:
                energies = raw["energies"]
            mode_data[mode_name] = raw["modes"][mode_name]  # type: ignore[index]
        if not mode_data:
            continue
        bundle = {
            "generation": generation,
            "energies": energies,
            "modes": mode_data,
        }
        geom = build_geometry(generation)
        analysis = analyze_generation_parity(bundle, geom)
        out = save_parity_analysis(
            analysis,
            out_root / f"n{generation:02d}" / "parity.npz",
            metadata={"source_raw_dir": str(raw_root)},
        )
        manifest["files"].append(str(out.relative_to(out_root)))
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest