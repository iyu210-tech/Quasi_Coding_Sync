"""
Three command line drivers:
fibonacci-qc-raw generates raw heatmaps
fibonacci-qc-analyze analyzes data
fibonacci-qc-figures produces figures from analysis products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.analysis_io import produce_parity_analysis
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.interaction import interaction_from_sweeps
from fibonacci_qc.nulls_io import produce_nulls_analysis
from fibonacci_qc.parity_analysis import analyze_mode_parity
from fibonacci_qc.plotting import produce_figures
from fibonacci_qc.raw_data import load_generation_raw, produce_raw_dataset, run_metadata
from fibonacci_qc.resonance import analyze_generation_resonances
from fibonacci_qc.resonance_io import produce_resonance_analysis
from fibonacci_qc.scaling_io import produce_scaling_analysis

def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]

def _parse_str_list(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]

def run_raw_driver(
    *,
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    n_energy: Optional[int] = None,
    output_dir: str = "study",
    validate: bool = True,
) -> dict:
    "Generate raw heatmaps, write run metadata"
    energies = cfg.energy_grid(n_points=n_energy) if n_energy is not None else cfg.energy_grid()
    return produce_raw_dataset(
        generations=generations,
        energies_ev=energies,
        output_dir=output_dir,
        modes=modes,
        validate=validate,
    )

def _windows_for_generations(generations: Sequence[int]) -> list:
    "Use locked fit windows when available, otherwise the span of generations"
    gens = sorted(int(g) for g in generations)
    gen_set = set(gens)
    usable = [
        (lo, hi)
        for lo, hi in cfg.FIT_GENERATION_WINDOWS
        if all(g in gen_set for g in range(lo, hi + 1))
    ]
    if usable:
        return usable
    return [(gens[0], gens[-1])]

def run_analyze_driver(
    *,
    raw_dir: str = "study",
    generations: Optional[Sequence[int]] = None,
    modes: Optional[Sequence[str]] = None,
    n_replicates: int = cfg.PERMUTATION_REPLICATES,
) -> dict:
    "Run parity, resonance, nulls, and scaling analysis on frozen raw data"
    gens = list(generations) if generations is not None else None
    windows = _windows_for_generations(gens) if gens else None
    from fibonacci_qc.nulls_io import produce_nulls_analysis, produce_robustness_analysis
    results = {
        "parity": produce_parity_analysis(
            raw_dir,
            generations=gens,
            modes=modes,
            output_dir="parity",
        ),
        "resonance": produce_resonance_analysis(
            raw_dir,
            generations=gens,
            modes=modes,
            output_dir="resonance",
        ),
        "nulls": produce_nulls_analysis(
            raw_dir,
            generations=gens,
            modes=modes,
            n_replicates=n_replicates,
            output_dir="nulls",
        ),
        "scaling": produce_scaling_analysis(
            raw_dir,
            generations=gens,
            modes=modes,
            windows=windows,
            output_dir="scaling",
        ),
        "robustness": produce_robustness_analysis(
            generations =gens,
            modes=modes,
            output_dir="robustness"
        )
    }
    summary_path = cfg.OUTPUT_ANALYSIS / "analyze_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_dir": raw_dir,
        "generations": list(generations) if generations else None,
        "modes": list(modes) if modes else list(cfg.DEFECT_MODES),
        "metadata": run_metadata(generations=list(generations or cfg.GENERATIONS), energies=cfg.energy_grid(), extra={"driver": "analyze"}),
        "products": {
            name: {"manifest_path": value.get("manifest_path"), "files": value.get("files")}
            for name, value in results.items()
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    results["analyze_manifest_path"] = str(summary_path)
    return results

def run_figures_driver(
    *,
    raw_dir: str = "study",
    generation: Optional[int] = None,
    mode: str = cfg.MODE_SUBSTITUTION,
    output_dir: str = "study",
    n_replicates: int = 20,
) -> dict:
    "Build standard figure set for a generation/mode"
    raw_root = Path(raw_dir)
    if not raw_root.is_absolute():
        raw_root = cfg.OUTPUT_RAW / raw_root

    if generation is None:
        gen_dirs = sorted(
            p for p in raw_root.glob("n*") if p.is_dir() and p.name[1:].isdigit()
        )
        if not gen_dirs:
            raise FileNotFoundError(f"no generation folders under {raw_root}")
        generation = int(gen_dirs[0].name[1:])

    mode_path = raw_root / f"n{generation:02d}" / f"{mode}.npz"
    if not mode_path.exists():
        # Use mode file to rebuild bundle
        candidates = sorted((raw_root / f"n{generation:02d}").glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(f"no raw files for generation {generation}")
        mode_path = candidates[0]

    # Load all modes for this generation
    gen_dir = raw_root / f"n{generation:02d}"
    modes_data = {}
    energies = None
    t_clean = None
    log_clean = None
    for path in sorted(gen_dir.glob("*.npz")):
        raw = load_generation_raw(path)
        if energies is None:
            energies = raw["energies"]
            t_clean = raw["T_clean"]
            log_clean = raw["log_T_clean"]
        modes_data.update(raw["modes"])
    bundle = {
        "generation": generation,
        "energies": energies,
        "T_clean": t_clean,
        "log_T_clean": log_clean,
        "modes": modes_data,
    }
    geom = build_geometry(generation)
    parity = None
    if mode in modes_data:
        parity = analyze_mode_parity(modes_data[mode]["T"], geom, mode, energies=energies, sweep_valid=modes_data[mode]["valid"])
    interaction = None
    needed = cfg.MODE_SUBSTITUTION, cfg.MODE_POTENTIAL_ONLY, cfg.MODE_WIDTH_ONLY
    if all(m in modes_data for m in needed):
        interaction = interaction_from_sweeps(modes_data, t_clean=t_clean, log_t_clean=log_clean, energies=energies)
    resonance = None
    if modes_data:
        resonance_full = analyze_generation_resonances(bundle, geom, modes=[mode] if mode in modes_data else None)
        if mode in resonance_full["modes"]:
            resonance = resonance_full["modes"][mode]
    nulls = None
    if mode in modes_data:
        from fibonacci_qc.geometry_nulls import run_geometry_nulls
        nulls = run_geometry_nulls(modes_data[mode]["T"], geom, mode, energies=energies, sweep_valid=modes_data[mode]["valid"], n_replicates=n_replicates, seed=0)

    scaling = None
    # Load saved scaling product if present
    scaling_path = cfg.OUTPUT_ANALYSIS / "scaling" / "scaling.npz"
    if scaling_path.exists():
        from fibonacci_qc.scaling_io import load_scaling_analysis

        loaded = load_scaling_analysis(scaling_path)
        # Make a scaling dict if metadata has windows (for plotting)
        meta = loaded.get("metadata", {})
        arrays = loaded.get("arrays", {})
        windows = {}
        for key in meta.get("windows", {}):
            energies_key = f"{key}__energies"
            if energies_key not in arrays:
                continue
            mode_alphas = {}
            for mode_name in meta["windows"][key].get("modes", {}):
                alpha_key = f"{key}__{mode_name}__alpha"
                support_key = f"{key}__{mode_name}__support"
                if alpha_key in arrays:
                    class _Fit:
                        pass
                    fit = _Fit()
                    fit.alpha = arrays[alpha_key]
                    fit.support = arrays.get(support_key)
                    mode_alphas[mode_name] = fit
            windows[key] = {
                "energies": arrays[energies_key],
                "modes": mode_alphas,
                "common_support": arrays.get(f"{key}__common_support"),
            }
        if windows:
            scaling = {"windows": windows}
    written = produce_figures(
        raw_bundle=bundle,
        parity=parity,
        interaction=interaction,
        resonance=resonance,
        nulls=nulls,
        scaling=scaling,
        output_dir=output_dir,
        mode=mode if mode in modes_data else next(iter(modes_data)),
    )
    meta_path = Path(output_dir)
    if not meta_path.is_absolute():
        meta_path = cfg.OUTPUT_FIGURES / meta_path
    meta_path.mkdir(parents=True, exist_ok=True)
    figure_meta = {
        "generation": generation,
        "mode": mode,
        "files": written,
        "metadata": run_metadata(generations=[generation], energies=np.asarray(energies), extra={"driver": "figures"}),
    }
    meta_file = meta_path / "figures_manifest.json"
    meta_file.write_text(json.dumps(figure_meta, indent=2, default=str), encoding="utf-8")
    return {"files": written, "manifest_path": str(meta_file)}

def _build_raw_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate raw Fibonacci defect heatmaps.")
    p.add_argument("--generations", type=_parse_int_list, default=None)
    p.add_argument("--modes", type=_parse_str_list, default=None)
    p.add_argument("--n-energy", type=int, default=None)
    p.add_argument("--output-dir", default="study")
    p.add_argument("--no-validate", action="store_true")
    return p

def _build_analyze_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyse frozen raw Fibonacci data.")
    p.add_argument("--raw-dir", default="study")
    p.add_argument("--generations", type=_parse_int_list, default=None)
    p.add_argument("--modes", type=_parse_str_list, default=None)
    p.add_argument("--n-replicates", type=int, default=cfg.PERMUTATION_REPLICATES)
    return p

def _build_figures_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Produce Fibonacci research figures.")
    p.add_argument("--raw-dir", default="study")
    p.add_argument("--generation", type=int, default=None)
    p.add_argument("--mode", default=cfg.MODE_SUBSTITUTION)
    p.add_argument("--output-dir", default="study")
    p.add_argument("--n-replicates", type=int, default=20)
    return p

def main_raw(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_raw_parser().parse_args(argv)
    manifest = run_raw_driver(
        generations=args.generations,
        modes=args.modes,
        n_energy=args.n_energy,
        output_dir=args.output_dir,
        validate=not args.no_validate,
    )
    print(json.dumps({"manifest_path": manifest.get("manifest_path"), "files": manifest.get("files")}, indent=2))
    return 0

def main_analyze(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_analyze_parser().parse_args(argv)
    result = run_analyze_driver(
        raw_dir=args.raw_dir,
        generations=args.generations,
        modes=args.modes,
        n_replicates=args.n_replicates,
    )
    print(json.dumps({"analyze_manifest_path": result.get("analyze_manifest_path")}, indent=2))
    return 0

def main_figures(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_figures_parser().parse_args(argv)
    result = run_figures_driver(
        raw_dir=args.raw_dir,
        generation=args.generation,
        mode=args.mode,
        output_dir=args.output_dir,
        n_replicates=args.n_replicates,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main_raw())