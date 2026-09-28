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

def _windows_for_generations(generations):
    gens=sorted(set(int(g) for g in generations))
    if not gens:raise ValueError("no generations")
    use=[(a,b) for a,b in cfg.FIT_GENERATION_WINDOWS if set(range(a,b+1)).issubset(gens)]
    if use:return use
    spans=[];start=prev=gens[0]
    for g in gens[1:]:
        if g!=prev+1:spans.append((start,prev));start=g
        prev=g
    spans.append((start,prev))
    return spans

def _study_energies(raw_dir: str, generations: Optional[Sequence[int]]) -> np.ndarray:
    from fibonacci_qc.raw_data import load_generation_bundle

    raw_root = Path(raw_dir)
    if not raw_root.is_absolute():
        raw_root = cfg.OUTPUT_RAW / raw_root
    if generations:
        gen_dir = raw_root / f"n{int(list(generations)[0]):02d}"
    else:
        gen_dirs = sorted(p for p in raw_root.glob("n*") if p.is_dir() and p.name[1:].isdigit())
        if not gen_dirs:
            raise FileNotFoundError(f"no generation folders under {raw_root}")
        gen_dir = gen_dirs[0]
    bundle = load_generation_bundle(gen_dir)
    return np.asarray(bundle["energies"], dtype=np.float64)

def run_analyze_driver(
    *,raw_dir="study",
    generations=None,
    modes=None,
    n_replicates=cfg.PERMUTATION_REPLICATES):
    from fibonacci_qc.raw_data import load_generation_bundle,config_sha256,_file_hash
    from fibonacci_qc.nulls_io import produce_robustness_analysis
    import hashlib
    raw_root=_resolve_raw(raw_dir)
    gens=(sorted(int(p.name[1:]) for p in raw_root.glob("n*") if p.is_dir() and p.name[1:].isdigit())
          if generations is None else list(generations))
    if not gens:raise FileNotFoundError("no requested raw generations")
    first=load_generation_bundle(raw_root/f"n{gens[0]:02d}",modes=modes)
    wanted=list(first["modes"]) if modes is None else list(modes)
    energies=np.asarray(first["energies"],dtype=float)
    root=cfg.OUTPUT_ANALYSIS/raw_root.name
    root.mkdir(parents=True,exist_ok=True)
    files=sorted(raw_root.glob("n*/*.npz"))
    identity=hashlib.sha256("".join(str(p.relative_to(raw_root))+_file_hash(p) for p in files).encode()).hexdigest()
    source={"source_raw_dir":str(raw_root),"source_data_sha256":identity, "source_run_id":first["run_metadata"].get("run_id"),"config_sha256":config_sha256(), "analysis_config":cfg.CONFIG.as_dict(),"generations":gens,"modes":wanted, "n_replicates":int(n_replicates)}
    source["analysis_revisions"] = {
        "geometry_null": "geometry-centered-curve-v2",
        "cross_generation_resonance": "clean-anchor-continuation-v1",
    }
    guard=root/"source.json"
    if guard.exists() and json.loads(guard.read_text())!=source:
        raise ValueError(f"analysis destination belongs to different data/settings: {root}; use a new run name")
    guard.write_text(json.dumps(source,indent=2),encoding="utf-8")
    results={}
    results["parity"]=produce_parity_analysis(str(raw_root),generations=gens,modes=wanted,output_dir=root/"parity")
    results["resonance"]=produce_resonance_analysis(str(raw_root),generations=gens,modes=wanted,output_dir=root/"resonance")
    results["nulls"]=produce_nulls_analysis(str(raw_root),generations=gens,modes=wanted,n_replicates=n_replicates,output_dir=root/"nulls")
    # Partition non-contiguous sample lists
    windows=_windows_for_generations(gens)
    results["scaling"]=produce_scaling_analysis(str(raw_root),generations=gens,modes=wanted,windows=windows,output_dir=root/"scaling")
    results["robustness"]=produce_robustness_analysis(generations=gens,modes=wanted,
        energies_baseline=energies,output_dir=root/"robustness",raw_dir=str(raw_root))
    cross_path = root/"resonance"/"cross_generation.meta.json"
    cross_meta = json.loads(cross_path.read_text())
    cross_meta["numerical_gate"] = results["robustness"]["cross_generation_gate"]["gate"]
    cross_meta["numerically_accepted"] = bool(
        results["robustness"]["primary_gate"] == "pass"
        and not results["parity"]["numerical_failure"]
        and cross_meta["numerical_gate"] == "pass"
    )
    cross_path.write_text(json.dumps(cross_meta, indent=2), encoding="utf-8")
    for sidecar in root.glob("**/*.meta.json"):
        data=json.loads(sidecar.read_text())
        data.setdefault("run_metadata",{}).update(source)
        sidecar.write_text(json.dumps(data,indent=2),encoding="utf-8")
    summary=root/"analyze_manifest.json"
    payload={
        "raw_dir":str(raw_root),
        "generations":gens,"modes":wanted,
        "primary_gate":results["robustness"]["primary_gate"],
        "numerical_failure":results["parity"]["numerical_failure"],
        "metadata":run_metadata(generations=gens,energies=energies,extra={"driver":"analyze",**source}),
        "products":{key:{k:value.get(k) for k in ("manifest_path","files","primary_gate","numerical_failure")} for key,value in results.items()}}
    summary.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    results["analyze_manifest_path"]=str(summary)
    return results

def _resolve_raw(raw_dir: str) -> Path:
    raw_root = Path(raw_dir)
    if not raw_root.is_absolute():
        raw_root = cfg.OUTPUT_RAW / raw_root
    return raw_root.resolve()

def _require_same_raw(meta: dict, raw_root: Path) -> None:
    run = meta.get("run_metadata") or {}
    recorded = run.get("source_raw_dir", meta.get("source_raw_dir"))
    if not recorded:
        raise ValueError("saved product has no source_raw_dir")
    recorded_path = Path(str(recorded))
    if not recorded_path.is_absolute():
        recorded_path = cfg.OUTPUT_RAW / recorded_path
    if run.get("source_data_sha256"):
        from fibonacci_qc.raw_data import _file_hash
        import hashlib
        identity=hashlib.sha256("".join(str(p.relative_to(raw_root))+_file_hash(p) for p in sorted(raw_root.glob("n*/*.npz"))).encode()).hexdigest()
        if identity!=run["source_data_sha256"]:raise ValueError("raw data changed since saved analysis")
    if recorded_path.resolve() != raw_root:
        raise ValueError(f"saved source {recorded_path} does not match selected raw run {raw_root}")

class _Observed:
    def __init__(self, s_max: float) -> None:
        self.S_max = s_max

class _NullPlot:
    def __init__(self, samples: np.ndarray, s_max: float) -> None:
        self.null_S_max = samples
        self.observed = _Observed(s_max)

def _nulls_for_plots(loaded: dict, mode: str, n_replicates: Optional[int]) -> dict:
    arrays = loaded["arrays"]
    meta = loaded["metadata"]
    families = (meta.get("modes") or {}).get(mode) or {}
    plotted = {}
    for family, row in families.items():
        samples_key = f"{mode}__{family}__null_S_max"
        if samples_key not in arrays:
            continue
        saved = int(row["n_replicates"])
        if n_replicates is not None and int(n_replicates) != saved:
            raise ValueError(f"requested n_replicates={n_replicates} but {mode}/{family} was saved with {saved}")
        plotted[family] = _NullPlot(arrays[samples_key], float(row["S_max"]))
    return plotted

def _scaling_for_plots(loaded: dict) -> dict:
    meta = loaded["metadata"]
    arrays = loaded["arrays"]
    windows = {}
    for key, window_meta in (meta.get("windows") or {}).items():
        energies_key = f"{key}__energies"
        if energies_key not in arrays:
            continue
        mode_fits = {}
        for mode_name in window_meta.get("modes", {}):
            alpha_key = f"{key}__{mode_name}__alpha"
            if alpha_key not in arrays:
                continue
            fit = type("Fit", (), {})()
            fit.alpha = arrays[alpha_key]
            fit.support = arrays.get(f"{key}__{mode_name}__support")
            fit.stable = arrays.get(f"{key}__{mode_name}__stable")
            mode_fits[mode_name] = fit
        paired = {}
        for name in window_meta.get("paired", {}):
            delta_key = f"{key}__pair__{name}__delta_alpha"
            if delta_key in arrays:
                paired[name] = {
                    "delta_alpha": arrays[delta_key],
                    "support": arrays.get(f"{key}__pair__{name}__support"),
                }
        windows[key] = {
            "energies": arrays[energies_key],
            "modes": mode_fits,
            "common_support": arrays.get(f"{key}__common_support"),
            "paired": paired,
        }
    return {"windows": windows}

def run_figures_driver(*, raw_dir: str = "study", generation: Optional[int] = None, mode: str = cfg.MODE_SUBSTITUTION, output_dir: str = "study", n_replicates: Optional[int] = None) -> dict:
    from fibonacci_qc.analysis_io import load_parity_analysis
    from fibonacci_qc.nulls_io import load_nulls_analysis
    from fibonacci_qc.raw_data import load_generation_bundle
    from fibonacci_qc.resonance_io import load_interaction_remainder, load_resonance_analysis
    from fibonacci_qc.scaling_io import load_scaling_analysis

    raw_root = _resolve_raw(raw_dir)
    if generation is None:
        gen_dirs = sorted(p for p in raw_root.glob("n*") if p.is_dir() and p.name[1:].isdigit())
        if not gen_dirs:
            raise FileNotFoundError(f"no generation folders under {raw_root}")
        generation = int(gen_dirs[0].name[1:])
    bundle = load_generation_bundle(raw_root / f"n{generation:02d}")
    energies = np.asarray(bundle["energies"], dtype=np.float64)

    def _saved(kind: str, filename: str) -> Optional[dict]:
        path = cfg.OUTPUT_ANALYSIS / raw_root.name/kind / f"n{generation:02d}" / filename
        return path

    parity = resonance = interaction = nulls = scaling = None
    parity_path = cfg.OUTPUT_ANALYSIS / raw_root.name / "parity" / f"n{generation:02d}" / "parity.npz"
    if parity_path.exists():
        parity = load_parity_analysis(parity_path)
        _require_same_raw(parity["metadata"], raw_root)
    resonance_path = cfg.OUTPUT_ANALYSIS / raw_root.name / "resonance" / f"n{generation:02d}" / "resonance.npz"
    if resonance_path.exists():
        resonance = load_resonance_analysis(resonance_path)
        _require_same_raw(resonance["metadata"], raw_root)
    interaction_path = cfg.OUTPUT_ANALYSIS / raw_root.name / "resonance" / f"n{generation:02d}" / "interaction.npz"
    if interaction_path.exists():
        interaction = load_interaction_remainder(interaction_path)
        _require_same_raw(interaction["metadata"], raw_root)
    null_path = cfg.OUTPUT_ANALYSIS/ raw_root.name / "nulls" / f"n{generation:02d}" / "nulls.npz"
    if null_path.exists():
        loaded_nulls = load_nulls_analysis(null_path)
        _require_same_raw(loaded_nulls["metadata"], raw_root)
        nulls = _nulls_for_plots(loaded_nulls, mode if mode in bundle["modes"] else next(iter(bundle["modes"])), n_replicates)
    scaling_path = cfg.OUTPUT_ANALYSIS / raw_root.name / "scaling" / "scaling.npz"
    if scaling_path.exists():
        loaded_scaling = load_scaling_analysis(scaling_path)
        _require_same_raw(loaded_scaling["metadata"], raw_root)
        saved_gens = list((loaded_scaling["metadata"].get("run_metadata") or {}).get("generations") or [])
        if generation not in saved_gens and saved_gens:
            raise ValueError(f"scaling product generations {saved_gens} do not include generation {generation}")
        scaling = _scaling_for_plots(loaded_scaling)

    written = produce_figures(
        raw_bundle=bundle,
        parity=parity,
        interaction=interaction,
        resonance=resonance,
        nulls=nulls,
        scaling=scaling,
        output_dir=output_dir,
        mode=mode if mode in bundle["modes"] else next(iter(bundle["modes"])),
        run_id=raw_root.name,
        generation=generation,
    )
    meta_path = Path(output_dir)
    if not meta_path.is_absolute():
        meta_path = cfg.OUTPUT_FIGURES / meta_path
    meta_path.mkdir(parents=True, exist_ok=True)
    figure_meta = {
        "generation": generation,
        "mode": mode,
        "raw_dir": str(raw_root),
        "files": written,
        "metadata": run_metadata(generations=[generation], energies=energies, extra={"driver": "figures", "raw_dir": str(raw_root)}),
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
    p.add_argument("--n-replicates", type=int, default=None)
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