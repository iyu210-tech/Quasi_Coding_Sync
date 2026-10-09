"Read-only endpoint audit plus optional, time-bounded extension of existing pilots."
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def assess(baseline, fine, tolerance, complete):
    if not complete or not all(finite(x) for x in (baseline, fine, tolerance)) or tolerance < 0:
        return None, "unresolved"
    difference = abs(fine - baseline)
    return difference, "pass" if difference <= tolerance else "fail"

def stable_pair(previous, current):
    return bool(previous and current and previous["status"] == current["status"] == "pass"
                and previous["round"] + 1 == current["round"]
                and previous["n_fine"] == current["n_baseline"]
                and (str(current.get("family", "")).endswith(":pointwise") or math.isclose(previous["fine"], current["baseline"], rel_tol=1e-10, abs_tol=1e-10)))

def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (v if not isinstance(v, float) or math.isfinite(v) else "") for k, v in row.items()})

def audit(inputs, output):
    from fibonacci_qc import config as cfg
    from fibonacci_qc.raw_data import config_sha256
    from fibonacci_qc.geometry_nulls import NULL_STATISTIC_REVISION, applicable_strata
    from fibonacci_qc.cross_generation import CROSS_GENERATION_REVISION
    revisions = {
        "geometry_null": NULL_STATISTIC_REVISION,
        "cross_generation_resonance": CROSS_GENERATION_REVISION
    }
    all_rows, resonance, strict, sources = [], [], [], []
    rounds_by_gen, seen_generations = {}, set()
    for source in inputs:
        source = Path(source).expanduser().resolve()
        manifest = source / "resolution_manifest.json" if source.is_dir() else source
        report = load(manifest)

        if report.get("config_sha256") != config_sha256():
            raise ValueError(f"Different physical model/solver: {manifest}")
        if report.get("gate_tolerances") != dict(cfg.GATE_TOLERANCES):
            raise ValueError(f"Different tolerances: {manifest}")
        if report.get("analysis_revisions") != revisions:
            raise ValueError(f"Different analysis revision: {manifest}")

        gens = report["generations"]
        if len(gens) != len(set(gens)) or seen_generations.intersection(gens):
            raise ValueError("Supply only one pilot per generation; do not select favorable rounds across pilots")
        seen_generations.update(gens)

        sources.append({
            "file": str(manifest),
            "status": report.get("status"),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        })
        rounds = sorted(report.get("rounds", []), key=lambda x: x["round"])
        if len({r["round"] for r in rounds}) != len(rounds):
            raise ValueError("Duplicate round numbers")

        for gen in gens:
            rounds_by_gen[gen] = [r["round"] for r in rounds]

        for round_info in rounds:
            number = round_info["round"]
            for gen in gens:
                path = manifest.parent / f"round_{number:02d}" / f"n{gen:02d}" / "robustness.meta.json"
                data = load(path)

                if data["generation"] != gen or data.get("gate_tolerances") != dict(cfg.GATE_TOLERANCES):
                    raise ValueError(f"Inconsistent generation/tolerances: {path}")
                sources.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

                modes = data["modes"]
                if set(modes) != set(report["modes"]):
                    raise ValueError(f"Mode coverage mismatch: {path}")

                common = dict(generation=gen, round=number, n_baseline=round_info["n_energy_baseline"],
                              n_fine=round_info["n_energy_fine"], source=str(path))
                strict.append({
                    **common,
                    "original_primary_gate": data.get("primary_gate", "missing"),
                    "original_interaction_gate": data.get("interaction_gate", "missing"),
                    "cross_generation_gate": round_info.get("cross_generation_gate", {}).get("gate", "missing")})
                tol = cfg.GATE_TOLERANCES

                def add(mode, family, window, metric, b, f, limit, complete, units):
                    difference, status = assess(b, f, limit, complete)
                    all_rows.append({
                        **common,
                        "mode": mode,
                        "family": family,
                        "window": window,
                        "metric": metric,
                        "units": units,
                        "baseline": b,
                        "fine": f,
                        "abs_difference": difference,
                        "tolerance": limit,
                        "status": status
                    })

                def contrast(mode, family, comparison, valid):
                    complete = valid and comparison.get("numerical_complete") is True

                    for window in cfg.ENERGY_WINDOWS:
                        coverage = comparison.get("window_coverage", {}).get(window)
                        ok = complete and finite(coverage) and coverage >= 1 - 1e-12
                        b = comparison.get("integrated_baseline", {}).get(window, {})
                        f = comparison.get("integrated_fine", {}).get(window, {})

                        for metric in ("signed", "abs", "rms"):
                            add(mode, family, window, metric, b.get(metric), f.get(metric),
                                tol["parity_rms" if metric == "rms" else "parity_integral_ev"], ok,
                                "dimensionless" if metric == "rms" else "eV")

                    for metric in ("max_abs_new_nodes", "rmse_new_nodes"):
                        add(mode, family + ":pointwise", "all", metric, 0.0, comparison.get(metric),
                            tol["parity_pointwise"], complete, "dimensionless")

                for mode, item in modes.items():
                    if (item.get("n_energy_baseline"), item.get("n_energy_fine")) != (common["n_baseline"], common["n_fine"]):
                        raise ValueError(f"Energy counts disagree: {path}/{mode}")

                    valid = item.get("numerical_gate") == "pass"
                    contrast(mode, "parity", item.get("contrast", {}), valid)

                    for family in applicable_strata(mode):
                        comparison = item.get("residual_parity_checks", {}).get(family, {}).get("comparison", {})
                        contrast(mode, "residual:" + family, comparison, valid)

                    for window in cfg.ENERGY_WINDOWS:
                        b = item.get("response_baseline", {}).get(window, {})
                        f = item.get("response_fine", {}).get(window, {})
                        coverage = (b.get("coverage"), f.get("coverage"))
                        ok = valid and all(finite(x) and x >= cfg.MIN_NUMERICAL_COVERAGE for x in coverage)

                        for metric in ("mean_abs", "rms"):
                            add(mode, "response", window, metric, b.get(metric), f.get(metric),
                                tol["response"], ok, "dimensionless")

                    for row in item.get("resonance", []):
                        resonance.append({**common, "mode": mode, **row})

                interaction = data.get("interaction", {})
                valid = all(m.get("numerical_gate") == "pass" for m in modes.values())

                if set(cfg.ONE_SITE_MODES).issubset(modes):
                    for window in cfg.ENERGY_WINDOWS:
                        b = interaction.get("baseline", {}).get(window, {})
                        f = interaction.get("fine", {}).get(window, {})

                        for metric in ("mean_abs_I", "rms_I", "mean_abs_I_log", "rms_I_log"):
                            is_log = metric.endswith("_log")
                            a, z = b.get(metric), f.get(metric)

                            limit = (tol["interaction_log_atol"] + tol["interaction_log_rtol"] * max(abs(a), abs(z))
                                     if is_log and finite(a) and finite(z)
                                     else (None if is_log else tol["interaction"]))

                            key = "coverage_I_log" if is_log else "coverage_I"
                            coverage = (b.get(key), f.get(key))
                            ok = valid and all(finite(x) and x >= cfg.MIN_NUMERICAL_COVERAGE for x in coverage)

                            add("one_site_combination", "interaction_log" if is_log else "interaction_linear",
                                window, metric, a, z, limit, ok, "dimensionless")

    grouped = defaultdict(list)
    for row in all_rows:
        key = tuple(row[k] for k in ("generation", "mode", "family", "window", "metric"))
        grouped[key].append(row)

    summary = []
    for key, rows in sorted(grouped.items()):
        sequence = rounds_by_gen[key[0]]
        lookup = {r["round"]: r for r in rows}
        current = lookup.get(sequence[-1])
        previous = lookup.get(sequence[-2]) if len(sequence) >= 2 else None
        if current is None:
            continue

        stable = stable_pair(previous, current)
        summary.append({k: current[k] for k in ("generation", "mode", "family", "window", "metric", "units")}
                       | {
                        "finest_sampled_value": current["fine"],
                        "n_energy": current["n_fine"],
                        "previous_change": previous["abs_difference"] if previous else None,
                        "latest_change": current["abs_difference"],
                        "latest_tolerance": current["tolerance"],
                        "previous_status": previous["status"] if previous else "missing",
                        "latest_status": current["status"],
                        "stable_on_last_two_sampled_comparisons": stable,
                        "stability_note": ("two adjacent sampled comparisons passed" if stable else
                            "fewer than two completed comparisons" if previous is None else
                            "at least one comparison failed or was unresolved"
                            if previous["status"] != "pass" or current["status"] != "pass" else
                            "comparison sequence or repeated intermediate value does not match")
                        })

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    write_csv(output / "endpoint_summary.csv", summary)
    write_csv(output / "all_comparisons.csv", all_rows)
    write_csv(output / "resonance_checks.csv", resonance)
    write_csv(output / "original_gates.csv", strict)

    conclusion = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "generations_with_completed_comparisons": [g for g, rs in rounds_by_gen.items() if rs],
        "full_original_study_accepted": False,
        "permutation_pvalue_convergence": "not_established_by_this_pilot",
        "meaning": "Per-endpoint agreement on the last two sampled comparisons only",
        "original_acceptance_rules_modified": False,
        "n_endpoints_stable_on_last_two_comparisons": sum(r["stable_on_last_two_sampled_comparisons"] for r in summary),
        "n_endpoints_listed": len(summary),
    }
    (output / "audit.json").write_text(json.dumps(conclusion, indent=2, allow_nan=False), encoding="utf-8")

    print(f"AUDIT WRITTEN: {output}")
    print(f"Endpoint checks stable on the last two comparisons: "
          f"{conclusion['n_endpoints_stable_on_last_two_comparisons']}/{len(summary)}")
    print("Original full-study acceptance has NOT been granted. No simulation was rerun by the audit.")
    return summary


def extend(existing, output, first, last, hours):
    "Run unchanged, two-comparison pilots sequentially"
    if not (11 <= first <= last <= 20) or not finite(hours) or hours <= 0:
        raise ValueError("Require 11 <= first <= last <= 20 and a positive time budget")

    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)

    audit([existing], root / "initial_audit")

    deadline = time.monotonic() + hours * 3600
    pilots = [str(Path(existing).expanduser().resolve())]
    jobs = []
    journal = root / "jobs.json"

    for gen in range(first, last + 1):
        if time.monotonic() >= deadline:
            break

        job = root / f"n{gen:02d}"
        command = [
            sys.executable,
            "-m",
            "fibonacci_qc.resolution",
            "--generations",
            str(gen),
            "--start-points",
            "7997",
            "--max-rounds",
            "2",
            "--max-points",
            "31985",
            "--required-passes",
            "2",
            "--output-dir",
            str(job),
        ]

        started = time.monotonic()
        record = {"generation": gen, "status": "running", "command": command}
        jobs.append(record)
        journal.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

        print(f"Starting generation {gen}; remaining budget "
              f"{(deadline-time.monotonic())/3600:.2f} h", flush=True)

        with (root / f"n{gen:02d}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)

            try:
                code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
                record["returncode"] = code
                record["status"] = "finished" if code in (0, 2) else "error"

            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                record["status"] = "time_budget_exhausted" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        record["elapsed_seconds"] = time.monotonic() - started
        if (job / "resolution_manifest.json").exists():
            pilots.append(str(job))
        journal.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        if record["status"] != "finished":
            print(f"Stopped: {record['status']}; inspect {root / f'n{gen:02d}.log'}", flush=True)
            break
        print(f"Generation {gen} finished", flush=True)
        audit(pilots, root / f"audit_through_n{gen:02d}")

    audit(pilots, root / "final_audit")
    print("Computation stopped. Use final_audit")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    a = sub.add_parser("audit")
    a.add_argument("pilots", nargs="+")
    a.add_argument("--output", required=True)

    e = sub.add_parser("extend")
    e.add_argument("--existing-pilot", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--first", type=int, default=11)
    e.add_argument("--last", type=int, default=15)
    e.add_argument("--hours", type=float, default=24.0)

    args = parser.parse_args()
    if args.action == "audit":
        audit(args.pilots, args.output)
    else:
        extend(args.existing_pilot, args.output, args.first, args.last, args.hours)

if __name__ == "__main__":
    main()