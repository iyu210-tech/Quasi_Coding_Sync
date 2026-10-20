#!/usr/bin/env python3
"Local, read-only figures from the Fibonacci projects saved results."

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
import numpy as np

MODES = ("substitution", "phason", "potential_only", "width_only")
LABELS = {"substitution": "Substitution", "phason": "Phason", "potential_only": "Potential-only", "width_only": "Width-only"}
REVISION = "geometry-centered-curve-v2"
WINDOWS = {"full": (0.1, 10.0), "low": (0.1, 1.0), "mid": (1.0, 5.0), "high": (5.0, 10.0)}
TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
VERSION = "qc-local-figures-1.0"

def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def write_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in json_safe(row).items()})

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def array_digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()

def family_default(mode: str) -> str:
    return "bond_4" if mode == "phason" else "site_5"

def integrate_window(energy: np.ndarray, curve: np.ndarray, lo: float, hi: float) -> dict:
    "Match the saved analysis: interpolate D at boundaries, then trapezoid D, |D|, D^2."
    e, d = np.asarray(energy), np.asarray(curve)
    if (e.ndim != 1 or e.size < 2 or d.shape != e.shape or
            not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0)):
        raise ValueError("Invalid integration axes.")
    if not e[0] <= lo < hi <= e[-1]:
        raise ValueError("Requested integration window is outside the saved energy span.")
    interior = (e > lo) & (e < hi)
    x = np.r_[lo, e[interior], hi]
    y = np.r_[np.interp(lo, e, d), d[interior], np.interp(hi, e, d)]
    if not np.all(np.isfinite(y)):
        return dict(signed=math.nan, absolute=math.nan, rms=math.nan, mean_signed=math.nan, mean_absolute=math.nan)
    signed, absolute = float(TRAP(y, x)), float(TRAP(np.abs(y), x))
    return dict(signed=signed, absolute=absolute,
                rms=float(np.sqrt(TRAP(y * y, x) / (hi - lo))),
                mean_signed=signed / (hi - lo), mean_absolute=absolute / (hi - lo))

@dataclass
class NullRecord:
    generation: int
    size: int
    mode: str
    family: str
    npz: Path
    meta: Path
    item: dict
    source: dict
    signature: str
    energy_hash: str
    model_hash: str
    repaired: bool

    def row(self) -> dict:
        return dict(
            generation=self.generation,
            n_energy=self.size,
            mode=self.mode,
            family=self.family,
            null_npz=str(self.npz),
            null_meta=str(self.meta),
            inference_valid=self.item.get("inference_valid"),
            seed=self.item.get("seed"),
            n_replicates=self.item.get("n_replicates"),
            energy_sha256=self.energy_hash,
            config_sha256=self.model_hash,
            raw_file_recorded=self.source.get("source_raw_file"),
            raw_sha256_recorded=self.source.get("source_data_sha256"),
            uses_corrected_raw=self.repaired)

def enclosing_model(meta_path: Path, stop: Path) -> str:
    "Read the run model identity without importing any solver code."
    for parent in meta_path.parents:
        p = parent / "run_manifest.json"
        if p.is_file():
            h = read_json(p).get("config_sha256")
            if h:
                return h
        if parent == stop:
            break
    return ""

def discover_records(roots: list[Path], issues: list[dict]) -> dict[tuple, NullRecord]:
    "Discover COMPLETE checkpoints. Copied numerical duplicates are collapsed."
    candidates: dict[tuple, list[NullRecord]] = {}
    seen: set[Path] = set()
    for root in roots:
        for directory, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in
                       {"raw",
                        "code_snapshot",
                        "__pycache__",
                        ".git",
                        "local_figures",
                        "fixtures",
                        "test",
                        "tests",
                        "node_modules"
                        }]
            if "COMPLETE.json" not in names or "nulls.npz" not in names:
                continue
            folder = Path(directory)
            npz = (folder / "nulls.npz").resolve()
            if npz in seen:
                continue
            seen.add(npz)
            try:
                meta_path = npz.with_suffix(".meta.json")
                meta = read_json(meta_path)
                checkpoint = read_json(folder / "COMPLETE.json")
                checks = checkpoint.get("rows", [])
                if not checks:
                    raise ValueError("Empty COMPLETE checkpoint.")
                digest, meta_digest = sha256(npz), sha256(meta_path)
                for c in checks:
                    if c.get("npz_sha256") != digest or c.get("meta_sha256") != meta_digest:
                        raise ValueError("COMPLETE checksum does not match saved result pair.")
                with np.load(npz, allow_pickle=False) as a:
                    e = a["energies"]
                    if e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
                        raise ValueError("Invalid saved null energy coordinates.")
                    for mode, families in meta["modes"].items():
                        if mode not in MODES:
                            raise ValueError("Unknown mechanism in null output.")
                        for family, item in families.items():
                            if item.get("statistic_revision") != REVISION:
                                raise ValueError("Not the corrected residual-statistic revision.")
                            prefix = f"{mode}__{family}__"
                            arrays = [a[prefix + k] for k in
                                      ("D_observed", "D_geometry_expected", "D_residual",
                                       "null_S_int", "null_S_max")]
                            d, mu, residual, ni, nm = arrays
                            if any(x.shape != e.shape for x in (d, mu, residual)):
                                raise ValueError("Null curve axes disagree.")
                            if not np.allclose(residual, d - mu, atol=1e-12, rtol=1e-12, equal_nan=True):
                                raise ValueError("Residual identity failed.")
                            B = int(item["n_replicates"])
                            if ni.shape != (B,) or nm.shape != (B,):
                                raise ValueError("Permutation replicate count mismatch.")
                            if item.get("inference_valid") is True:
                                if not all(np.all(np.isfinite(x)) for x in arrays):
                                    raise ValueError("Nonfinite input in an allegedly usable analysis.")
                                si = integrate_window(e, residual, .1, 10.)["absolute"]
                                sm = float(np.max(np.abs(residual)))
                                for k, val, null in [("S_int", si, ni), ("S_max", sm, nm)]:
                                    if not math.isclose(float(item[k]), val, rel_tol=1e-10, abs_tol=1e-12):
                                        raise ValueError(f"{k} disagrees with the saved curve.")
                                    pv = (1 + int(np.count_nonzero(null >= val - 1e-15))) / (B + 1)
                                    if not math.isclose(pv, float(item["p_" + k]), abs_tol=1e-12, rel_tol=0):
                                        raise ValueError("Saved p-value disagrees with shuffled samples.")
                            source = meta.get("run_metadata", {})
                            record = NullRecord(
                                int(meta["generation"]), int(e.size), mode, family, npz, meta_path,
                                item, source, array_digest(e, *arrays),
                                hashlib.sha256(e.tobytes()).hexdigest(),
                                source.get("config_sha256") or enclosing_model(meta_path, root),
                                bool(source.get("repair_revision") or source.get("fallback_revision")))
                            key = (record.generation, record.size, mode, family)
                            candidates.setdefault(key, []).append(record)
            except Exception as exc:
                issues.append(dict(file=str(npz), problem=str(exc)))
    chosen = {}
    for key, group in candidates.items():
        usable = [r for r in group if r.item.get("inference_valid") is True]
        pool = usable or group
        identities = {(r.signature, r.item.get("seed"), r.item.get("n_replicates"),
                       r.model_hash) for r in pool}
        if len(identities) != 1:
            issues.append(dict(file=" | ".join(str(r.npz) for r in pool), problem=f"Conflicting results for {key}; use --run to select intended runs."))
            continue
        chosen[key] = sorted(pool, key=lambda r: (not r.repaired, str(r.npz)))[0]
    return chosen

def null_arrays(record: NullRecord) -> tuple[np.ndarray, ...]:
    prefix = f"{record.mode}__{record.family}__"
    with np.load(record.npz, allow_pickle=False) as a:
        return (a["energies"],) + tuple(a[prefix + k] for k in
            ("D_observed", "D_geometry_expected", "D_residual", "null_S_int", "null_S_max"))

def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def save_plot(fig, out: Path, name: str, footer: str) -> None:
    fig.set_size_inches(11.5, 6.8)
    fig.subplots_adjust(left=.10, right=.97, bottom=.20, top=.86)
    fig.text(.10, .05, footer, fontsize=9, va="bottom")
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.svg")
    pyplot().close(fig)

def make_overview(records: dict, out: Path, size: int) -> None:
    "Recompute parity integrals from compact saved curves"
    rows = []
    for key, rec in sorted(records.items()):
        g, n, mode, family = key
        if family != family_default(mode):
            continue
        e, d, mu, residual, ni, nm = null_arrays(rec)
        for window, (lo, hi) in WINDOWS.items():
            vals = integrate_window(e, d, lo, hi)
            other = integrate_window(e, residual, lo, hi)
            row = dict(generation=g, n_energy=n, mode=mode, window=window,
                       window_width_ev=hi-lo, **vals,
                       residual_absolute=other["absolute"],
                       saved_input_usable=rec.item.get("inference_valid") is True,
                       original_combined_gate="not_reassessed",
                       source_null_npz=str(rec.npz), uses_corrected_raw=rec.repaired)
            row.update(p_int=rec.item.get("p_S_int"), p_max=rec.item.get("p_S_max"))
            rows.append(row)
    write_csv(out / "parity_all_saved_grids.csv", rows)
    lookup = {(r["generation"], r["n_energy"], r["mode"], r["window"]): r for r in rows}
    final = []
    for r in rows:
        if r["n_energy"] != size:
            continue
        row = dict(r)
        n_mid, n_low = (size+1)//2, ((size+1)//2+1)//2
        b = lookup.get((r["generation"], n_mid, r["mode"], r["window"]))
        a = lookup.get((r["generation"], n_low, r["mode"], r["window"]))
        recs = [records.get((r["generation"], n, r["mode"], family_default(r["mode"])))
                for n in (n_low, n_mid, size)]
        paired = False
        if all(recs):
            energies = [null_arrays(x)[0] for x in recs]
            paired = (all(x.item.get("inference_valid") is True for x in recs)
                      and len({x.model_hash for x in recs}) == 1 and bool(recs[0].model_hash)
                      and np.array_equal(energies[0], energies[1][::2])
                      and np.array_equal(energies[1], energies[2][::2]))
        row["exact_nested_three_grid_input_available"] = paired
        for metric in ("signed", "absolute", "rms"):
            d1 = abs(b[metric]-a[metric]) if a and b else math.nan
            d2 = abs(r[metric]-b[metric]) if b else math.nan
            row[f"{metric}_previous_change"] = d1
            row[f"{metric}_latest_change"] = d2
            row[f"{metric}_max_observed_change"] = max(d1, d2) if math.isfinite(d1+d2) else math.nan
            row[f"{metric}_two_changes_le_0p001"] = bool(paired and d1 <= .001 and d2 <= .001)
        row["interpretation"] = "Saved-curve diagnostic only; not a rigorous error bound or original gate pass."
        final.append(row)
    write_csv(out / "parity_finest_with_refinement.csv", final)
    p = pyplot()
    for metric, ylabel, name in [
        ("absolute", "Absolute parity integral (eV)", "overview_parity_absolute"),
        ("signed", "Signed parity integral (eV)", "overview_parity_signed"),
        ("rms", "Window-normalized parity RMS", "overview_parity_rms")]:
        fig, ax = p.subplots()
        available = sorted({r["generation"] for r in final})
        generations = list(range(available[0], available[-1]+1)) if available else []
        for mode in MODES:
            by_g = {r["generation"]: r for r in final if r["mode"] == mode and r["window"] == "full"}
            vals = [by_g[g][metric] if g in by_g and by_g[g]["saved_input_usable"] else np.nan
                    for g in generations]
            if not np.any(np.isfinite(vals)):
                continue
            ax.plot(generations, vals, marker="o", markersize=5, label=LABELS[mode])
        ax.set(title=f"Parity contrast across generations | {size:,} saved energies",
               xlabel="Fibonacci generation n", ylabel=ylabel)
        ax.set_xticks(generations)
        ax.grid(alpha=.2)
        if metric == "signed":
            ax.axhline(0, linewidth=.6, linestyle=":")
        ax.legend(frameon=False)
        save_plot(fig, out, name, "Window: 0.1 - 10 eV. Finest sampled values")

def make_null_detail(rec: NullRecord, out: Path) -> None:
    e, d, mu, residual, ni, nm = null_arrays(rec)
    prefix = f"n{rec.generation}_{rec.mode}_{rec.family}"
    np.savetxt(out / f"{prefix}_parity_curves.csv", np.column_stack([e,d,mu,residual]), delimiter=",", header="energy_ev,D_observed,D_geometry_expected,D_residual", comments="")
    np.savetxt(out / f"{prefix}_permutation_samples.csv",
               np.column_stack([np.arange(1, ni.size+1), ni, nm]),
               delimiter=",",
               header="replicate,S_int_null_ev,S_max_null",
               comments="",
               fmt=["%d","%.17g","%.17g"])
    write_json(out / f"{prefix}_permutation_metadata.json", {**rec.row(), "statistics": rec.item})
    p = pyplot()
    fig, ax = p.subplots()
    for y, label in [(d,"Observed parity contrast"), (mu,"Geometry expectation"), (residual,"Residual")]:
        ax.plot(e, y, linewidth=.7, label=label)
    ax.set(title=f"Generation {rec.generation}: {LABELS[rec.mode]} / {rec.family}", xlabel="Energy (eV)", ylabel="Transmission contrast")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save_plot(fig, out, f"{prefix}_parity_curves", "Saved sampled curves; narrow spectral structure and maxima may remain resolution-sensitive.")
    if rec.item.get("inference_valid") is not True:
        return
    for key, null, xlabel in [("S_int",ni,"Integrated absolute residual (eV)"), ("S_max",nm,"Maximum sampled absolute residual")]:
        fig, ax = p.subplots()
        ax.hist(null, bins=30, label=f"{len(null)} shuffled labelings")
        obs = float(rec.item[key])
        ax.axvline(obs, linewidth=1.8, linestyle="--", label=f"Observed = {obs:.6g}")
        low, high = min(obs, float(np.min(null))), max(obs, float(np.max(null)))
        margin = max((high-low)*.06, abs(obs)*.01, 1e-15)
        ax.set_xlim(low-margin, high+margin)
        ax.set(title=f"n={rec.generation}, {LABELS[rec.mode]}, {rec.family} | {key}",
               xlabel=xlabel, ylabel="Number of shuffled labelings")
        ax.legend(frameon=False)
        save_plot(fig, out, f"{prefix}_permutation_{key}",
                  f"Upper-tail value = {float(rec.item["p_"+key]):.6g}; B={len(null)}; seed={rec.item["seed"]}.\n Random-labeling benchmark")

class RowMember:
    "Sequential bounded-memory reader for one numeric C-order .npy inside an NPZ"
    def __init__(self, archive: zipfile.ZipFile, key: str):
        self.info = archive.getinfo(key + ".npy")
        self.stream = archive.open(self.info, "r")
        version = np.lib.format.read_magic(self.stream)
        if version == (1, 0):
            self.shape, fortran, self.dtype = np.lib.format.read_array_header_1_0(self.stream)
        elif version == (2, 0):
            self.shape, fortran, self.dtype = np.lib.format.read_array_header_2_0(self.stream)
        else:
            self.close()
            raise ValueError(f"{key}: unsupported NPY version {version}.")
        if self.dtype.hasobject or self.dtype.fields or self.dtype.kind not in "fiub":
            self.close()
            raise ValueError(f"{key}: unsafe or unsupported nonnumeric array.")
        if len(self.shape) != 2 or fortran:
            self.close()
            raise ValueError(f"{key}: requires the project\"s C-order 2D raw format; got {self.shape}, Fortran={fortran}.")
        expected = self.stream.tell() + math.prod(self.shape) * self.dtype.itemsize
        if expected != self.info.file_size:
            self.close()
            raise ValueError(f"{key}: header/data length mismatch.")
        self.position = 0

    def read_rows(self, n: int) -> np.ndarray:
        n = min(n, self.shape[0] - self.position)
        count = n * self.shape[1] * self.dtype.itemsize
        data = self.stream.read(count)
        if len(data) != count:
            raise EOFError("Truncated NPZ member.")
        self.position += n
        result = np.frombuffer(data, dtype=self.dtype).reshape(n, self.shape[1])
        if self.position == self.shape[0] and self.stream.read(1):
            raise ValueError("Unexpected bytes after array payload.")
        return result

    def close(self):
        self.stream.close()

def load_small(archive: zipfile.ZipFile, key: str) -> np.ndarray:
    info = archive.getinfo(key+".npy")
    if info.file_size > 64 * 1024**2:
        raise ValueError(f"{key} is unexpectedly large for a coordinate/reference vector.")
    with archive.open(info) as f:
        return np.lib.format.read_array(f, allow_pickle=False)

def map_recorded_path(path: str, project: Path) -> Path:
    original = Path(path).expanduser()
    parts = original.parts
    if "output" in parts:
        i = parts.index("output")
        candidate = project.joinpath(*parts[i:])
        if candidate.is_file():
            return candidate.resolve()
    if original.is_file():
        return original.resolve()
    raise FileNotFoundError(f"Cannot locate recorded raw file: {path}. Use --raw-file with its actual path.")

def select_raw(rec: NullRecord, project: Path, override: str | None) -> Path:
    recorded = rec.source.get("source_raw_file")
    if override:
        path = Path(override).expanduser().resolve()
    elif recorded:
        path = map_recorded_path(recorded, project)
    else:
        raise ValueError("No source_raw_file in this completed result. Supply --raw-file explicitly.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def trusted_cells(status, T, logT, bound: float) -> np.ndarray:
    status, T, logT = np.asarray(status), np.asarray(T), np.asarray(logT)
    if status.dtype.kind not in "iu":
        raise ValueError("Expected encoded integer status array")
    good = ((status == 0) | (status == 2)) & np.isfinite(T) & np.isfinite(logT)
    return good & (T >= 0) & (T <= 1+bound) & ((status != 2) | (T == 0)) & (logT <= math.log1p(bound))

def chosen_positions(text: str, valid: np.ndarray) -> list[int]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("No geometrically eligible position.")
    result = []
    for part in text.split(","):
        part = part.strip().lower()
        if part == "middle":
            p = int(indices[np.argmin(np.abs(indices - len(valid)//2))])
        elif part == "first":
            p = int(indices[0])
        elif part == "last":
            p = int(indices[-1])
        else:
            p = int(part)
        if not 0 <= p < len(valid) or not valid[p]:
            raise ValueError(f"Position {p} is outside the axis or is ineligible (e.g. an AA phason pair).")
        if p not in result:
            result.append(p)
    if not result or len(result) > 8:
        raise ValueError("Choose between 1 and 8 positions.")
    return result

def summarize_raw(path: Path, rec: NullRecord, out: Path, args) -> dict:
    "Scan only this modes T/log_T/status members. All quantities remain on the recorded grid"
    sidecar = path.with_suffix(".meta.json")
    meta = read_json(sidecar)
    expected = meta.get("data_sha256")
    if not expected:
        raise ValueError("Raw sidecar lacks data_sha256; refusing unverified selection.")
    linked = rec.source.get("source_data_sha256")
    if not linked or linked != expected:
        raise ValueError("Raw checksum identity differs from the file used for this saved null result.")
    print(f"Checking raw-file checksum: {path}", flush=True)
    started = time.monotonic()
    before = path.stat()
    if sha256(path) != expected:
        raise ValueError("Raw checksum failed. Do not use this file for figures.")
    rm = meta["run_metadata"]
    if int(meta["generation"]) != rec.generation:
        raise ValueError("Raw generation differs from selected result.")
    if rec.model_hash and rm.get("config_sha256") != rec.model_hash:
        raise ValueError("Raw physical model differs from selected result.")
    bound = float(rm["config"]["tolerances"]["t_bounds_atol"])
    mode = rec.mode
    prefix = f"{mode}__"
    with ExitStack() as stack:
        z = stack.enter_context(zipfile.ZipFile(path, "r"))
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate NPZ member names.")
        e = load_small(z, "energies")
        tc, lc, sc = (load_small(z, k) for k in ("T_clean","log_T_clean","clean_status"))
        positions, valid = (load_small(z, prefix+k) for k in ("positions","valid"))
        valid = np.asarray(valid, dtype=bool)
        if int(load_small(z, "generation")) != rec.generation:
            raise ValueError("Raw generation field mismatch.")
        if (e.ndim != 1 or e.size != rec.size or not np.all(np.isfinite(e)) or
                not np.all(np.diff(e) > 0) or any(x.shape != e.shape for x in (tc,lc,sc))):
            raise ValueError("Raw energy/reference vectors are inconsistent.")
        if hashlib.sha256(e.tobytes()).hexdigest() != rec.energy_hash:
            raise ValueError("Raw energy coordinates differ from the saved null analysis.")
        if positions.shape != valid.shape or not np.array_equal(positions, np.arange(len(valid))):
            raise ValueError("Raw positions must be the full, uncompacted original index axis.")
        psel = chosen_positions(args.positions, valid)
        clean_good = trusted_cells(sc, tc, lc, bound)
        plot_sel = np.flatnonzero((e >= args.energy_window[0]) & (e <= args.energy_window[1]))
        if len(plot_sel) < 2:
            raise ValueError("Fewer than two recorded energies in the requested plotting window.")
        ne, nr = len(e), len(valid)
        eranges = np.linspace(0, len(plot_sel), min(args.heatmap_cols, len(plot_sel))+1, dtype=int)
        rranges = np.linspace(0, nr, min(args.heatmap_rows,nr)+1, dtype=int)
        shape = (len(rranges)-1, len(eranges)-1)
        heat_t, heat_d = np.zeros(shape), np.zeros(shape)
        count_t, count_d = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
        selected = {}
        sum_even, sum_odd = np.zeros(ne), np.zeros(ne)
        n_even, n_odd = np.zeros(ne,dtype=np.int64), np.zeros(ne,dtype=np.int64)
        response_acc = {w: [0., 0., 0, int(valid.sum()*np.count_nonzero((e>=lo)&(e<=hi)))] for w,(lo,hi) in WINDOWS.items()}
        counts = dict(rejected_eligible_cells=0,
                      underflow_eligible_cells=0,
                      rejected_clean_cells=int(np.count_nonzero(~clean_good)),
                      eligible_cells=int(valid.sum()*ne)
                    )
        streams = []
        for key in ("T","log_T","status"):
            reader = RowMember(z,prefix+key)
            stack.callback(reader.close)
            streams.append(reader)
        if any(reader.shape != (nr,ne) for reader in streams):
            raise ValueError("Raw matrix axes disagree.")
        for start in range(0, nr, args.chunk_rows):
            t, logt, code = [reader.read_rows(args.chunk_rows) for reader in streams]
            stop = start+len(t)
            vg = valid[start:stop,None]
            cell = trusted_cells(code,t,logt,bound) & vg
            if np.any(vg & ((code==0)|(code==2)) & ~cell):
                raise ValueError("A certified status contradicts its T/log_T values.")
            counts["rejected_eligible_cells"] += int(np.count_nonzero(vg & ~cell))
            counts["underflow_eligible_cells"] += int(np.count_nonzero(cell & (code==2)))
            for p0 in psel:
                if start <= p0 < stop:
                    j=p0-start
                    selected[p0]=(t[j].copy(),logt[j].copy(),code[j].copy(),cell[j].copy())
            even = cell & ((positions[start:stop,None] % 2)==0)
            odd = cell & ((positions[start:stop,None] % 2)==1)
            sum_even += np.where(even,t,0).sum(axis=0)
            sum_odd += np.where(odd,t,0).sum(axis=0)
            n_even += even.sum(axis=0)
            n_odd += odd.sum(axis=0)
            good_delta = cell & clean_good[None,:]
            delta = t-tc[None,:]
            for w,(lo,hi) in WINDOWS.items():
                select=(e>=lo)&(e<=hi)
                values=delta[:,select][good_delta[:,select]]
                acc=response_acc[w]
                acc[0] += float(np.abs(values).sum())
                acc[1] += float(np.square(values).sum())
                acc[2] += len(values)
            t_window=t[:,plot_sel]
            d_window=np.abs(delta[:,plot_sel])
            ct=cell[:,plot_sel]
            cd=good_delta[:,plot_sel]
            sum_t=np.add.reduceat(np.where(ct,t_window,0),eranges[:-1],axis=1)
            sum_d=np.add.reduceat(np.where(cd,d_window,0),eranges[:-1],axis=1)
            cnt_t=np.add.reduceat(ct.astype(np.int64),eranges[:-1],axis=1)
            cnt_d=np.add.reduceat(cd.astype(np.int64),eranges[:-1],axis=1)
            row_bins=np.searchsorted(rranges,positions[start:stop],side="right")-1
            np.add.at(heat_t,row_bins,sum_t)
            np.add.at(heat_d,row_bins,sum_d)
            np.add.at(count_t,row_bins,cnt_t)
            np.add.at(count_d,row_bins,cnt_d)
            if start==0 or stop==nr or (start//args.chunk_rows)%50==0:
                print(f"  Read {stop:,}/{nr:,} positions",flush=True)
        mean_even=np.divide(sum_even,n_even,out=np.full(ne,np.nan),where=n_even>0)
        mean_odd=np.divide(sum_odd,n_odd,out=np.full(ne,np.nan),where=n_odd>0)
        d=mean_even-mean_odd
        observed_saved=null_arrays(rec)[1]
        if not np.allclose(d,observed_saved,atol=1e-12,rtol=1e-10,equal_nan=True):
            raise ValueError("Recomputed parity does not match the saved null result linked to this raw file.")
        heat_t=np.divide(heat_t,count_t,out=np.full(shape,np.nan),where=count_t>0)
        heat_d=np.divide(heat_d,count_d,out=np.full(shape,np.nan),where=count_d>0)
    if (path.stat().st_size, path.stat().st_mtime_ns) != (before.st_size,before.st_mtime_ns):
        raise ValueError("Raw file changed while being read.")
    stem=f"n{rec.generation}_{mode}_e{rec.size:05d}"
    columns=[e,tc,lc,sc.astype(float),clean_good.astype(float)]
    headers=["energy_ev","T_clean","ln_T_clean","clean_status","clean_usable"]
    for p0 in psel:
        t,logt,code,good=selected[p0]
        columns.extend([t,logt,code.astype(float),good.astype(float), np.where(good&clean_good,t-tc,np.nan)])
        headers.extend([f"T_p{p0}",f"ln_T_p{p0}",f"status_p{p0}",f"usable_p{p0}",f"delta_T_p{p0}"])
    np.savetxt(out/f"{stem}_selected_spectra.csv",np.column_stack(columns),delimiter=",", header=",".join(headers),comments="",fmt="%.17g")
    np.savetxt(out/f"{stem}_parity_from_raw.csv",np.column_stack([e,d,n_even,n_odd]),delimiter=",", header="energy_ev,D_raw,n_even_used,n_odd_used",comments="",fmt="%.17g")
    response=[]
    for w,acc in response_acc.items():
        total,squared,used,intended=acc
        response.append(dict(generation=rec.generation,
                            mode=mode,
                            n_energy=rec.size,
                            window=w,
                            mean_abs=total/used if used else math.nan,
                            rms=math.sqrt(squared/used) if used else math.nan,
                            used_cells=used,
                            intended_cells=intended,
                            coverage=used/intended if intended else None,
                            original_combined_gate="not_reassessed",
                            source_raw_file=str(path),
                            data_sha256=expected,
                            uses_corrected_raw=rec.repaired
                            )
                        )
    write_csv(out/f"{stem}_response_summaries.csv",response)
    ep=e[plot_sel]
    edges=np.r_[ep[0],(ep[eranges[1:-1]-1]+ep[eranges[1:-1]])/2,ep[-1]]
    yedges=rranges.astype(float)-.5
    np.savez_compressed(out/f"{stem}_heatmap_blocks.npz",
                        energy_edges_ev=edges,
                        position_edges=yedges,
                        mean_T=heat_t,
                        mean_abs_delta_T=heat_d,
                        usable_T_counts=count_t,
                        usable_delta_T_counts=count_d,
                        energy_bin_start_indices=plot_sel[eranges[:-1]],
                        energy_bin_stop_indices_exclusive=plot_sel[eranges[1:]-1]+1,
                        position_bin_edges=rranges
                        )
    provenance={**rec.row(),
                "raw_file_used":str(path),
                "raw_sha256_verified":expected,
                "raw_sidecar_sha256":sha256(sidecar),
                "positions":psel,**counts,
                "heatmap":"Mean of all usable sampled cells within each displayed block.",
                "rows_per_chunk":args.chunk_rows,
                "elapsed_seconds":time.monotonic()-started,
                "status_codes":{"0":"ok","1":"invalid","2":"underflow","3":"overflow","4":"non_unitary"},
                "source_run_metadata":rm,
                "parity_matches_saved_null_curve":True,
                "scope":"Read-only extraction and postprocessing"}
    write_json(out/f"{stem}_raw_provenance.json",provenance)
    make_raw_plots(e,tc,lc,clean_good,selected,plot_sel,edges,yedges,heat_t,heat_d,out,stem,rec)
    return provenance

def make_raw_plots(e,tc,lc,clean_good,selected,sel,edges,yedges,heat_t,heat_d,out,stem,rec):
    p=pyplot()
    for kind,ylabel in [("T","Transmission probability"),("log10_T","Retained log10 transmission"), ("delta_T","Defect minus clean transmission")]:
        fig,ax=p.subplots()
        if kind!="delta_T":
            clean = tc if kind=="T" else lc/np.log(10.)
            ax.plot(e[sel],np.where(clean_good,clean,np.nan)[sel],linewidth=.75,label="Clean")
        for p0,(t,logt,code,good) in selected.items():
            values=t if kind=="T" else logt/np.log(10.) if kind=="log10_T" else t-tc
            mask=good if kind!="delta_T" else good&clean_good
            ax.plot(e[sel],np.where(mask,values,np.nan)[sel],linewidth=.75,label=f"Defect p={p0}")
        ax.set(title=f"n={rec.generation}: {LABELS[rec.mode]} | {len(e):,} recorded energies",
               xlabel="Energy (eV)",ylabel=ylabel,xlim=(e[sel][0],e[sel][-1]))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        save_plot(fig,out,f"{stem}_{kind}", "p is the original zero-based layer index (left index of an eligible pair for phasons).")
    for name,values,label in [("T",heat_t,"Mean T within display block"), ("abs_delta_T",heat_d,"Mean |T_defect - T_clean| within display block")]:
        fig,ax=p.subplots()
        mesh=ax.pcolormesh(edges,yedges,np.ma.masked_invalid(values),shading="flat",rasterized=True)
        fig.colorbar(mesh,ax=ax,label=label,pad=.02)
        ax.set(title=f"n={rec.generation}: {LABELS[rec.mode]} | block-averaged heatmap",
               xlabel="Energy (eV)", ylabel="Original defect-position index p")
        save_plot(fig,out,f"{stem}_heatmap_{name}", "Display bins average ALL usable recorded cells.\n Ineligible positions and rejected cells are excluded. Exact counts are in heatmap_blocks.npz.")

def package_output(out: Path) -> Path:
    archive=out/"shareable_figures_and_extracts.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.iterdir()):
            if file.is_file() and file!=archive and not file.name.endswith(".tmp"):
                z.write(file,file.name)
    return archive

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project",default=".",help="Project root.")
    parser.add_argument("--run",action="append",help="Optional run folder to scan; repeat for several runs. Default: project/output.")
    parser.add_argument("--generation",type=int,default=15,help="Detailed spectra/heatmaps/permutation figures for this generation.")
    parser.add_argument("--mode",choices=MODES,default="substitution")
    parser.add_argument("--family",help="Default: site_5 for site defects, bond_4 for phasons.")
    parser.add_argument("--n-energy",type=int,default=31985)
    parser.add_argument("--positions",default="middle",help="Comma-separated indices or first,middle,last; at most 8.")
    parser.add_argument("--energy-window",nargs=2,type=float,default=[.1,10.],metavar=("MIN","MAX"),
                        help="Display interval only; summaries retain all four predefined windows.")
    parser.add_argument("--chunk-rows",type=int,default=8,help="Maximum raw rows held per chunk (default 8).")
    parser.add_argument("--heatmap-rows",type=int,default=240)
    parser.add_argument("--heatmap-cols",type=int,default=720)
    parser.add_argument("--overview-only",action="store_true",help="Use compact results only.")
    parser.add_argument("--raw-file",help="Explicit raw location if files moved. Checksum must still match selected result.")
    parser.add_argument("--out",help="New output directory")
    args=parser.parse_args()
    if not (1 <= args.chunk_rows <= 128 and 1 <= args.heatmap_rows <= 1000 and
            1 <= args.heatmap_cols <= 2000):
        parser.error("Chunk rows must be 1-128; heatmap rows 1-1000 and columns 1-2000.")
    if not .1 <= args.energy_window[0] < args.energy_window[1] <= 10.:
        parser.error("Choose .1 <= MIN < MAX <= 10 for --energy-window.")
    project=Path(args.project).expanduser().resolve()
    roots=[Path(x).expanduser().resolve() for x in args.run] if args.run else [project/"output"]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"No extracted run folder found: {root}. Use existing unzipped project folders.")
    out=Path(args.out).expanduser().resolve() if args.out else project/"output"/"local_figures"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    print("OUTPUT:",out,flush=True)
    manifest=dict(version=VERSION,arguments=vars(args),started=datetime.now().isoformat(),
                  status="running",raw_files_modified=False,simulations_run=False,
                  numpy_version=np.__version__,python=sys.version,issues=[])
    write_json(out/"figure_manifest.json",manifest)
    shutil.copy2(Path(__file__).resolve(),out/"plotting_script_snapshot.py")
    try:
        print("Reading compact completed results and verifying their hashes/statistics...",flush=True)
        records=discover_records(roots,manifest["issues"])
        write_csv(out/"source_results_index.csv",[r.row() for _,r in sorted(records.items())])
        write_csv(out/"selection_issues.csv",manifest["issues"])
        if not records:
            raise ValueError("No usable completed result checkpoints found; inspect selection_issues.csv.")
        hashes={r.model_hash for r in records.values() if r.model_hash}
        if len(hashes)>1:
            raise ValueError("Different model configurations were found; use --run to select compatible run folders.")
        print(f"Found {len(records)} distinct grouping/grid records; {len(manifest["issues"])} selection warnings.",flush=True)
        make_overview(records,out,args.n_energy)
        family=args.family or family_default(args.mode)
        rec=records.get((args.generation,args.n_energy,args.mode,family))
        if rec is None:
            raise ValueError(f"No unambiguous completed result for n{args.generation}/{args.mode}/{family}/{args.n_energy}. See source_results_index.csv and selection_issues.csv.")
        make_null_detail(rec,out)
        if not args.overview_only:
            raw=select_raw(rec,project,args.raw_file)
            manifest["raw_processing"]=summarize_raw(raw,rec,out,args)
        manifest["status"]="completed_with_selection_warnings" if manifest["issues"] else "completed"
        manifest["selected_detail"]=rec.row()
        (out/"READ_FIRST.txt").write_text(
            "These are local figures and compact extracts.\n"
            "PNG and SVG versions are provided for each figure. CSV/NPZ files contain plotted values.\n"
            "All input files were opened read-only. Raw source selection follows the saved null metadata.\n"
            "The raw file checksum is verified before large-array processing.\n"
            "Heatmaps are explicitly block-averaged displays.\n"
            "Certified underflow remains usable. Logarithmic plots use retained logarithms.\n"
            "Parity trend CSVs include actual adjacent-grid changes where compatible grids exist.\n"
            "The original combined gates are NOT rerun or changed. \n"
            "This version recomputes response summaries only for the selected raw mode/grid.\n",
            encoding="utf-8")
    except KeyboardInterrupt:
        manifest["status"]="interrupted"
        manifest["error"]="KeyboardInterrupt; original data unchanged."
    except Exception as exc:
        manifest["status"]="error"
        manifest["error"]=f"{type(exc).__name__}: {exc}"
        print("STOP:",manifest["error"],file=sys.stderr,flush=True)
    finally:
        write_json(out/"figure_manifest.json",manifest)
    archive=package_output(out)
    print("STATUS:",manifest["status"],flush=True)
    print("FIGURES / CSV / NPZ:",out,flush=True)
    print("SMALL UPLOAD PACKAGE:",archive,flush=True)
    return 0 if manifest["status"].startswith("completed") else 1

if __name__=="__main__":
    raise SystemExit(main())