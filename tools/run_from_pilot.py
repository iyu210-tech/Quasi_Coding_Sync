"Launch workflow from checked pilot"

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fibonacci_qc import config as cfg
from fibonacci_qc.raw_data import config_sha256
from fibonacci_qc.geometry_nulls import NULL_STATISTIC_REVISION
from fibonacci_qc.cross_generation import CROSS_GENERATION_REVISION

def launch_from_pilot(manifest_path):
    report = json.loads(Path(manifest_path).read_text())
    n = report.get("recommended_n_energy")
    if report.get("status") != "converged_on_requested_checks" or type(n) is not int or n < cfg.N_ENERGY:
        raise ValueError("pilot has no passing candidate grid for production")
    if not {cfg.N_GENERATION_MIN, cfg.N_GENERATION_MAX} <= set(report.get("generations", [])):
        raise ValueError("first run one pilot covering both generations 10 and 20 at a common grid")
    if set(report.get("modes", [])) != set(cfg.DEFECT_MODES):
        raise ValueError("pilot must include all four defect modes")
    if report.get("config_sha256") != config_sha256():
        raise ValueError("pilot physical model/solver differs from the current project")
    if report.get("gate_tolerances") != dict(cfg.GATE_TOLERANCES):
        raise ValueError("pilot tolerances differ from current acceptance criteria")
    revisions = {"geometry_null": NULL_STATISTIC_REVISION,
                 "cross_generation_resonance": CROSS_GENERATION_REVISION
                }
    if report.get("analysis_revisions") != revisions:
        raise ValueError("pilot uses a different analysis revision")
    rounds = report.get("rounds", [])
    needed = report.get("required_consecutive_passes", 0)
    if type(needed) is not int or needed < 2 or len(rounds) < needed:
        raise ValueError("production launcher requires at least two consecutive passing comparisons")
    if any(r.get("primary_gate") != "pass" for r in rounds[-needed:]):
        raise ValueError("pilot does not end with the required consecutive passes")
    if rounds[-1].get("n_energy_baseline") != n:
        raise ValueError("recommended grid is not the final tested baseline")
    from fibonacci_qc.pipeline import run_pipeline
    return run_pipeline(profile="production", n_energy=n)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    args = parser.parse_args()
    result = launch_from_pilot(args.manifest)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()