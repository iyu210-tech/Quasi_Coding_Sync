"Save and load interaction and resonance products."

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.interaction import (InteractionRemainder, interaction_from_sweeps, interaction_summary,)
from fibonacci_qc.raw_data import load_generation_raw
from fibonacci_qc.resonance import (Peak, PeakMatch, ResonanceTrack, analyze_generation_resonances,)

PathLike = Union[str, Path]

def _peak_to_dict(peak: Peak) -> Dict[str, object]:
    return {
        "index": peak.index,
        "energy": peak.energy,
        "height": peak.height,
        "prominence": peak.prominence,
    }

def _match_to_dict(match: PeakMatch) -> Dict[str, object]:
    return {
        "clean_index": match.clean_index,
        "defect_index": match.defect_index,
        "energy_clean": match.energy_clean,
        "energy_defect": match.energy_defect,
        "height_clean": match.height_clean,
        "height_defect": match.height_defect,
        "delta_E": match.delta_E,
    }

def _track_to_dict(track: ResonanceTrack) -> Dict[str, object]:
    return {
        "mode": track.mode,
        "position": track.position,
        "parity": track.parity,
        "valid": track.valid,
        "clean_peaks": [_peak_to_dict(p) for p in track.clean_peaks],
        "defect_peaks": [_peak_to_dict(p) for p in track.defect_peaks],
        "matched": [_match_to_dict(m) for m in track.match.matched],
        "unmatched_clean": [_peak_to_dict(p) for p in track.match.unmatched_clean],
        "unmatched_defect": [_peak_to_dict(p) for p in track.match.unmatched_defect],
        "n_matched": track.match.n_matched,
        "total_displacement": track.match.total_displacement,
        "mean_delta_E": track.mean_delta_E,
        "mean_abs_delta_E": track.mean_abs_delta_E,
        "rms_delta_E": track.rms_delta_E,
    }

def save_interaction_remainder(
    remainder: InteractionRemainder,
    path: PathLike,
    *,
    generation: Optional[int] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    "Persist I_p(E) arrays to .npz + JSON metadata."
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_ANALYSIS / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        I=remainder.I,
        I_log=remainder.I_log,
        energies=remainder.energies,
        positions=remainder.positions,
        valid=remainder.valid,
        delta_T_sub=remainder.delta_T_sub,
        delta_T_pot=remainder.delta_T_pot,
        delta_T_width=remainder.delta_T_width,
        delta_log_T_sub=remainder.delta_log_T_sub,
        delta_log_T_pot=remainder.delta_log_T_pot,
        delta_log_T_width=remainder.delta_log_T_width,
    )
    meta = {"npz": out.name, "generation": generation, "summary": interaction_summary(remainder), "run_metadata": metadata or {},}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out

def load_interaction_remainder(path: PathLike) -> Dict[str, object]:
    "Load saved interaction remainder."
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

def save_resonance_analysis(
    analysis: Dict[str, object],
    path: PathLike,
    *,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    "Persist resonance tracks & summaries for one generation."
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_ANALYSIS / out
    out.parent.mkdir(parents=True, exist_ok=True)
    generation = int(analysis["generation"])
    energies = np.asarray(analysis["energies"], dtype=np.float64)
    np_payload: Dict[str, object] = {"generation": np.array(generation, dtype=np.int64), "energies": energies,}
    json_modes: Dict[str, object] = {}
    modes: Dict[str, object] = analysis["modes"]  # type: ignore[assignment]
    for mode, payload in modes.items():
        tracks: List[ResonanceTrack] = payload["tracks"]  # type: ignore[index]
        prefix = f"{mode}__"
        n_pos = len(tracks)
        n_matched = np.array([t.match.n_matched for t in tracks], dtype=np.int64)
        mean_abs = np.array([t.mean_abs_delta_E for t in tracks], dtype=np.float64)
        mean_d = np.array([t.mean_delta_E for t in tracks], dtype=np.float64)
        rms = np.array([t.rms_delta_E for t in tracks], dtype=np.float64)
        parity = np.array([t.parity for t in tracks], dtype=np.int8)
        valid = np.array([t.valid for t in tracks], dtype=bool)
        positions = np.array([t.position for t in tracks], dtype=np.int64)
        np_payload[prefix + "n_matched"] = n_matched
        np_payload[prefix + "mean_delta_E"] = mean_d
        np_payload[prefix + "mean_abs_delta_E"] = mean_abs
        np_payload[prefix + "rms_delta_E"] = rms
        np_payload[prefix + "parity"] = parity
        np_payload[prefix + "valid"] = valid
        np_payload[prefix + "positions"] = positions
        np_payload[prefix + "n_positions"] = np.array(n_pos, dtype=np.int64)
        json_modes[mode] = {"summary": payload["summary"]}

    np.savez_compressed(out, **np_payload)
    meta = {
        "npz": out.name,
        "generation": generation,
        "geometry_metadata": analysis.get("geometry_metadata"),
        "modes": json_modes,
        "run_metadata": metadata or {},
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out

def load_resonance_analysis(path: PathLike) -> Dict[str, object]:
    "Load saved resonance analysis bundle."
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

def analyze_raw_interaction_and_resonance(
    raw_path: PathLike,
    *,
    modes: Optional[Sequence[str]] = None,
    save_dir: Optional[PathLike] = None,
) -> Dict[str, object]:
    "Run interaction + resonance analysis on one generation's raw files."
    raw = load_generation_raw(raw_path)
    generation = int(raw["generation"])
    geom = build_geometry(generation)

    # Interaction needs the three one-site modes
    modes_dict: Dict[str, Dict[str, np.ndarray]] = dict(raw["modes"])
    parent = Path(raw_path).parent
    for mode in (cfg.MODE_SUBSTITUTION, cfg.MODE_POTENTIAL_ONLY, cfg.MODE_WIDTH_ONLY,):
        if mode in modes_dict:
            continue
        sibling = parent / f"{mode}.npz"
        if sibling.exists():
            sibling_raw = load_generation_raw(sibling)
            modes_dict[mode] = sibling_raw["modes"][mode]
    result: Dict[str, object] = {"generation": generation, "interaction": None, "resonance": None,}
    have_interaction = all(
        m in modes_dict
        for m in (cfg.MODE_SUBSTITUTION, cfg.MODE_POTENTIAL_ONLY, cfg.MODE_WIDTH_ONLY,)
    )
    if have_interaction:
        remainder = interaction_from_sweeps(
            modes_dict,
            t_clean=raw["T_clean"],
            log_t_clean=raw["log_T_clean"],
            energies=raw["energies"],
        )
        result["interaction"] = remainder
        if save_dir is not None:
            save_interaction_remainder(
                remainder,
                Path(save_dir) / f"n{generation:02d}" / "interaction.npz",
                generation=generation,
            )
    resonance = analyze_generation_resonances(
        {
            "generation": generation,
            "energies": raw["energies"],
            "T_clean": raw["T_clean"],
            "modes": modes_dict if modes is None else {
                m: modes_dict[m] for m in modes if m in modes_dict
            },
        },
        geom,
        modes=modes,
    )
    result["resonance"]=resonance
    if save_dir is not None:
        save_resonance_analysis(resonance, Path(save_dir) / f"n{generation:02d}" / "resonance.npz",)
    return result

def produce_resonance_analysis(
    raw_dir: PathLike,
    *,
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    output_dir: PathLike = "resonance",
) -> Dict[str, object]:
    "Batch interaction + resonance analysis over raw study directory."
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

        seed = gen_dir / f"{cfg.MODE_SUBSTITUTION}.npz"
        if not seed.exists():
            seed = raw_files[0]
        result = analyze_raw_interaction_and_resonance(seed, modes=modes, save_dir=out_root,)
        manifest["files"].append(
            {
                "generation": generation,
                "interaction": result["interaction"] is not None,
                "resonance": True,
            }
        )

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest