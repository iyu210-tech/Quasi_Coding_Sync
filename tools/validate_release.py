"Reproduce independent benchmarks w/o rewriting fixtures / raw data. Run from project root: python tools/validate_release.py --output report.json"

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import importlib.metadata
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.scattering import clean_transmission,stack_s_matrix
from fibonacci_qc.fast_solver import StackCache,validate_against_reference
from fibonacci_qc.independent_reference import checked_high_precision
from fibonacci_qc.regression import assert_matches_regression_fixture

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="validation_report.json"); args = parser.parse_args()
    report = {
        "python": sys.version,
        "dependencies": {p:importlib.metadata.version(p) for p in ("numpy", "matplotlib", "pytest", "mpmath")}, "benchmarks": []}
    cases = [(3, 5.0), (5, 0.2), (10, 0.5), (10, 5.0), (15, 0.5), (15, 8.0), (20, 0.5), (20, 8.0)]
    for n, e in cases:
        g = build_geometry(n)
        expected = checked_high_precision(g.widths_nm, g.potentials_ev, e)
        reference = clean_transmission(n, e, geometry=g)
        cached = StackCache.build(g, e).clean_transmission()
        errors = {name: abs(value.log_T-expected["log_T"]) for name, value in [("reference", reference), ("cached", cached)]}
        if not (reference.is_valid and cached.is_valid): raise AssertionError(f"invalid solver result n={n} E={e}")
        if any(x > 1e-7 + 1e-12 * abs(expected["log_T"]) for x in errors.values()): raise AssertionError(f"log benchmark failure: {errors}")
        dense = stack_s_matrix(g.widths_nm, g.potentials_ev, e)
        matrix_error = float(np.max(np.abs(dense-expected["S"])))
        if matrix_error > 1e-7: raise AssertionError(f"full-S benchmark failure: {matrix_error}")
        row = dict(
            generation=n,
            energy_ev=e,
            expected_T=expected["T"],
            expected_log_T=expected["log_T"],
            reference_T=reference.T,
            reference_log_T=reference.log_T,
            cached_T=cached.T,
            cached_log_T=cached.log_T,
            reference_status=reference.status,
            cached_status=cached.status,
            log_absolute_errors=errors,
            full_S_absolute_error=matrix_error,
            independent_precisions=[80,120]
        )
        report["benchmarks"].append(row); print(json.dumps(row), flush=True)
    g = build_geometry(10)
    errors = validate_against_reference(g, [0.2, 0.5, 4.9999999999, 5.0, 5.0000000001, 7.0, 9.0])
    if errors: raise AssertionError(errors[:5])
    report["all_position_four_mode_crosscheck"] = {"generation": 10, "energies": [0.2, 0.5, 4.9999999999, 5.0, 5.0000000001, 7.0, 9.0], "mismatches": 0}
    assert_matches_regression_fixture(); report["independent_fixture_matches"] = True
    report["scope"] = "Selected energies and small all-mode tests only"
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VALIDATION COMPLETE", args.output)

if __name__ == "__main__": main()