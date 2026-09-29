"Uniform refinement pilot"

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fibonacci_qc import config as cfg
from fibonacci_qc.nulls_io import produce_robustness_analysis
from fibonacci_qc.raw_data import config_sha256
from fibonacci_qc.robustness import nested_fine_grid

def run_resolution_pilot(
    *,
    generations=(10,),
    modes=None,
    start_points=2000,
    max_rounds=4,
    max_points=31985,
    required_passes=2,
    output_dir,
):
    "Compare N -> 2N-1 on common uniform grid for the generations"
    gens = sorted(int(g) for g in generations)
    wanted = list(cfg.DEFECT_MODES if modes is None else modes)
    if not gens or len(set(gens)) != len(gens) or min(gens) < 2:
        raise ValueError("nonempty unique generations >= 2 required")
    if not wanted or len(set(wanted)) != len(wanted) or not set(wanted) <= set(cfg.DEFECT_MODES):
        raise ValueError("nonempty unique configured defect modes required")
    if any(isinstance(v, bool) or int(v) != v for v in (start_points, max_rounds, max_points, required_passes)):
        raise ValueError("grid and iteration limits must be integers")
    if start_points < 3 or max_rounds < 1 or max_points < 2 * start_points - 1:
        raise ValueError("limits must allow at least one refinement of >= 3 baseline points")
    if required_passes < 1 or required_passes > max_rounds:
        raise ValueError("required_passes must be between 1 and max_rounds")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)  # Do not replace prev. pilot's evidence.
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256(), 
        "generations": gens, 
        "modes": wanted,
        "gate_tolerances": dict(cfg.GATE_TOLERANCES),
        "analysis_revisions": {
            "geometry_null": "geometry-centered-curve-v2",
            "cross_generation_resonance": "clean-anchor-continuation-v1"
        },
        "required_consecutive_passes": int(required_passes),
        "max_rounds": int(max_rounds), 
        "max_points": int(max_points),
        "status": "running", 
        "recommended_n_energy": None,
        "scientific_claim": False,
        "scope": "requested generations/modes only",
        "rounds": [],
    }
    path = root/"resolution_manifest.json"

    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    ne = int(start_points)
    streak = 0
    try:
        for iteration in range(1, int(max_rounds) + 1):
            if 2*ne-1 > max_points:
                break
            eb = cfg.energy_grid(n_points=ne)
            ef = nested_fine_grid(eb)
            report["current_comparison"] = {"baseline": ne, "fine": int(ef.size)}
            save()
            evidence = produce_robustness_analysis(generations=gens, modes=wanted, energies_baseline=eb, energies_fine=ef, output_dir=root/f"round_{iteration:02d}")
            gate = evidence["primary_gate"]
            streak = streak+1 if gate == "pass" else 0
            report["rounds"].append({
                "round": iteration, 
                "n_energy_baseline": ne, 
                "n_energy_fine": int(ef.size),
                "primary_gate": gate, 
                "consecutive_passes": streak,
                "generation_gates": evidence["generation_gates"],
                "cross_generation_gate": evidence["cross_generation_gate"],
                "evidence_manifest": evidence["manifest_path"],
            })
            if streak >= required_passes:
                report["status"] = "converged_on_requested_checks"
                report["recommended_n_energy"] = ne
                break
            ne = 2 * ne - 1
            save()
        if report["status"] == "running":
            report["status"] = "unresolved"
            report["reason"] = "refinement budget exhausted w/out required consecutive passes"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    report.pop("current_comparison", None)
    save()
    return {**report, "manifest_path": str(path)}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", default="10")
    parser.add_argument("--modes", default=",".join(cfg.DEFECT_MODES))
    parser.add_argument("--start-points", type=int, default=2000)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=31985)
    parser.add_argument("--required-passes", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run_resolution_pilot(
        generations=[int(g) for g in args.generations.split(",")],
        modes=args.modes.split(","), 
        start_points=args.start_points,
        max_rounds=args.max_rounds,
        max_points=args.max_points,
        required_passes=args.required_passes,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["recommended_n_energy"] is not None else 2

if __name__ == "__main__":
    raise SystemExit(main())