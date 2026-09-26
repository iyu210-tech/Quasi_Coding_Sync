"Save and load scaling analysis products"

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.raw_data import load_generation_raw
from fibonacci_qc.scaling import ModeScalingResult, analyze_scaling, scaling_summary

PathLike = Union[str, Path]

def _mode_fit_to_meta(fit: ModeScalingResult) -> Dict[str, object]:
    return {
        "mode": fit.mode,
        "window": list(fit.window),
        "generations": list(fit.generations),
        "sizes": list(fit.sizes),
        "n_support": int(np.count_nonzero(fit.support)),
        "n_stable": int(np.count_nonzero(fit.stable)),
        "support_fraction": float(np.mean(fit.support)) if fit.support.size else 0.0,
    }

def save_scaling_analysis(
    analysis: Dict[str, object],
    path: PathLike,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    "Persist scaling fits to .npz + JSON metadata"
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_ANALYSIS / out
    out.parent.mkdir(parents=True, exist_ok=True)

    np_payload: Dict[str, object] = {}
    json_windows: Dict[str, object] = {}
    windows: Dict[str, object] = analysis["windows"]

    if "energies" in analysis:
        np_payload["energies"] = np.asarray(analysis["energies"], dtype=np.float64)
    if "global_support" in analysis:
        np_payload["global_support"] = np.asarray(analysis["global_support"], dtype=bool)
    for key, payload in windows.items():
        energies = np.asarray(payload["energies"], dtype=np.float64)
        np_payload[f"{key}__energies"] = energies
        np_payload[f"{key}__common_support"] = np.asarray(payload["common_support"], dtype=bool)
        mode_meta: Dict[str, object] = {}
        mode_fits: Dict[str, ModeScalingResult] = payload["modes"]
        for mode, fit in mode_fits.items():
            prefix = f"{key}__{mode}__"
            np_payload[prefix + "alpha"] = fit.alpha
            np_payload[prefix + "intercept"] = fit.intercept
            np_payload[prefix + "rmse"] = fit.rmse
            np_payload[prefix + "n_usable"] = fit.n_usable
            np_payload[prefix + "support"] = fit.support
            np_payload[prefix + "stable"] = fit.stable
            np_payload[prefix + "direction"] = np.asarray(fit.direction, dtype=object)
            np_payload[prefix + "loo_alpha"] = fit.loo_alpha
            np_payload[prefix + "sizes"] = np.asarray(fit.sizes, dtype=np.int64)
            np_payload[prefix + "generations"] = np.asarray(fit.generations, dtype=np.int64)
            if hasattr(fit, "generation_usable"):
                np_payload[prefix + "generation_usable"] = fit.generation_usable
            mode_meta[mode] = _mode_fit_to_meta(fit)
        paired_meta: Dict[str, object] = {}
        paired: Dict[str, object] = payload["paired"]
        for name, pair in paired.items():
            prefix = f"{key}__pair__{name}__"
            np_payload[prefix + "delta_alpha"] = pair["delta_alpha"]
            np_payload[prefix + "support"] = pair["support"]
            if "support_stable" in pair:
                np_payload[prefix + "support_stable"] = pair["support_stable"]
            if "delta_alpha_stable" in pair:
                np_payload[prefix + "delta_alpha_stable"] = pair["delta_alpha_stable"]
            if "support_global" in pair:
                np_payload[prefix + "support_global"] = pair["support_global"]
            if "delta_alpha_global" in pair:
                np_payload[prefix + "delta_alpha_global"] = pair["delta_alpha_global"]
            paired_meta[name] = {
                "mode_a": pair["mode_a"],
                "mode_b": pair["mode_b"],
            }
        json_windows[key] = {
            "window": list(payload["window"]),
            "modes": mode_meta,
            "paired": paired_meta,
        }
    np.savez_compressed(out, **np_payload)
    meta = {
        "npz": out.name,
        "modes": list(analysis.get("modes", [])),
        "fit_windows": [list(w) for w in analysis.get("fit_windows", [])],  # type: ignore[arg-type]
        "summary": scaling_summary(analysis),
        "windows": json_windows,
        "run_metadata": metadata or {},
        "has_global_support": "global_support" in analysis,
        "n_global_support": (int(np.count_nonzero(analysis["global_support"])) if "global_support" in analysis else None),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out

def load_scaling_analysis(path: PathLike) -> Dict[str, object]:
    "Load saved scaling analysis bundle"
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

def _load_raw_bundles(
    raw_dir: Path,
    generations: Sequence[int],
    modes: Sequence[str],
) -> Dict[int, Dict[str, object]]:
    from fibonacci_qc.raw_data import assert_same_run, load_generation_bundle
    bundles: Dict[int, Dict[str, object]] = {}
    anchor = None
    for generation in generations:
        bundle = load_generation_bundle(raw_dir / f"n{generation:02d}", modes=modes)
        if anchor is None:
            anchor = bundle
        else:
            assert_same_run(anchor, bundle, where=f"generation {generation}", same_generation=False)
        bundles[generation] = bundle
    return bundles

def produce_scaling_analysis(
    raw_dir: Optional[PathLike] = None,
    *,
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    windows: Optional[Sequence[Tuple[int, int]]] = None,
    energies: Optional[np.ndarray] = None,
    output_dir: PathLike = "scaling",
) -> Dict[str, object]:
    "Run scaling analysis, and build parity curves from raw heatmaps if raw_dir is given, else recomputed with fast solver"
    cfg.ensure_output_dirs()
    modes = list(modes or cfg.DEFECT_MODES)
    windows = list(windows or cfg.FIT_GENERATION_WINDOWS)
    if generations is None:
        gens = sorted({g for lo, hi in windows for g in range(lo, hi + 1)})
    else:
        gens = list(generations)
    bundles = None
    if raw_dir is not None:
        raw_root = Path(raw_dir)
        if not raw_root.is_absolute():
            raw_root = cfg.OUTPUT_RAW / raw_root
        bundles = _load_raw_bundles(raw_root, gens, modes)
        if not bundles:
            raise FileNotFoundError(f"no usable raw bundles under {raw_root}")
    analysis = analyze_scaling(modes=modes, windows=windows, energies=energies, bundles_by_generation=bundles)
    out_root = Path(output_dir)
    if not out_root.is_absolute():
        out_root = cfg.OUTPUT_ANALYSIS / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    out = save_scaling_analysis(
        analysis,
        out_root / "scaling.npz",
        metadata={
            "source_raw_dir": str(raw_dir) if raw_dir is not None else None,
            "generations": gens,
        },
    )
    manifest = {
        "files": [str(out.relative_to(out_root))],
        "modes": modes,
        "windows": [list(w) for w in windows],
        "summary": scaling_summary(analysis),
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["analysis"] = analysis
    return manifest