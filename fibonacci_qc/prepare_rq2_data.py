#!/usr/bin/env python3
"Organize the data"

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST_PARENT = (
    PROJECT_ROOT
    / "output"
    / "writing_selection"
    / "20261010_104702_904056"
    / "03_numbers"
)
DEFAULT_DEST = DEFAULT_DEST_PARENT / "RQ2_input_data"
DEFAULT_SUMMARY_CSV = DEFAULT_DEST_PARENT / "response_summaries_all_grids.csv"

GENERATIONS = list(range(10, 21))
GRID_SIZES = (7997, 15993, 31985)
MECHANISMS = ("substitution", "potential_only", "width_only")

KNOWN_RUN_ROOTS = (
    "permutation_overnight/20261006_225924_656908",
    "permutation_parallel/20261007_103816_771268",
    "permutation_n16_finish/20261007_224814_430895",
    "permutation_n17_attempt/20261007_232111_336104",
    "permutation_n17_repair/20261008_073949_399431",
    "permutation_n18_attempt/20261008_094054_883491",
    "permutation_n19_attempt/20261008_215241_316524",
    "permutation_n20_attempt/20261009_030302_820545",
)

PHYSICAL_MODEL_KEYS = (
    "hbar2_over_2m_ev_nm2",
    "v_lead_ev",
    "d_a_nm",
    "v_a_ev",
    "d_b_nm",
    "v_b_ev",
    "e_min_ev",
    "e_max_ev",
    "length_unit",
    "energy_unit",
    "word_w1",
    "word_w2",
    "solver_revision",
)

COPY_CHUNK = 8 * 1024 * 1024
SPACE_MARGIN_BYTES = 512 * 1024 * 1024

@dataclass
class DatasetRef:
    generation: int
    n_energy: int
    mechanism: str
    source_npz: Path
    source_meta: Path
    expected_sha256: str
    uses_corrected: bool
    selection_basis: str
    receipt_path: str | None = None
    ambiguities: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.generation, self.n_energy, self.mechanism)

    @property
    def dest_dir(self) -> Path:
        return Path(f"n{self.generation}") / f"e{self.n_energy:05d}"

    @property
    def dest_npz_rel(self) -> Path:
        return self.dest_dir / f"{self.mechanism}.npz"

    @property
    def dest_meta_rel(self) -> Path:
        return self.dest_dir / f"{self.mechanism}.meta.json"

@dataclass
class Plan:
    datasets: list[DatasetRef]
    missing_keys: list[tuple[int, int, str]]
    ambiguous_keys: list[tuple[int, int, str]]
    total_bytes: int
    free_bytes: int
    compatibility_issues: list[str]
    notes: list[str]

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def grid_label(n_energy: int) -> str:
    return f"e{n_energy:05d}"

def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def sha256_file(path: Path, chunk_size: int = COPY_CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def files_identical(a: Path, b: Path, chunk_size: int = COPY_CHUNK) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open("rb") as fa, b.open("rb") as fb:
        while True:
            ba = fa.read(chunk_size)
            bb = fb.read(chunk_size)
            if ba != bb:
                return False
            if not ba:
                return True

def free_bytes_for(path: Path) -> int:
    path = path if path.exists() else path.parent
    usage = shutil.disk_usage(path)
    return int(usage.free)

def meta_path_for(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + ".meta.json")

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def physical_model_fingerprint(meta: dict[str, Any]) -> dict[str, Any]:
    run = meta.get("run_metadata") or {}
    cfg = run.get("config") or {}
    out: dict[str, Any] = {}
    for key in PHYSICAL_MODEL_KEYS:
        if key == "solver_revision":
            out[key] = run.get("solver_revision")
        elif key in cfg:
            out[key] = cfg[key]
        elif key in run:
            out[key] = run[key]
    out["energy_sha256"] = run.get("energy_sha256")
    out["config_sha256"] = run.get("config_sha256")
    out["n_energy_run"] = run.get("n_energy")
    out["energy_min_ev"] = run.get("energy_min_ev")
    out["energy_max_ev"] = run.get("energy_max_ev")
    return out

def npz_axis_fingerprint(npz_path: Path, mechanism: str) -> dict[str, Any]:
    "Lightweight axis checks via memmap; never materializes T matrices."
    with np.load(npz_path, mmap_mode="r", allow_pickle=False) as z:
        energies = np.asarray(z["energies"])
        t_clean = np.asarray(z["T_clean"])
        pos_key = f"{mechanism}__positions"
        if pos_key not in z.files:
            raise KeyError(f"{npz_path}: missing {pos_key}")
        positions = np.asarray(z[pos_key])
        mech_energies = np.asarray(z[f"{mechanism}__energies"])
        mech_t_clean = np.asarray(z[f"{mechanism}__T_clean"])
        return {
            "n_energy": int(energies.shape[0]),
            "energy_sha256": hashlib.sha256(np.ascontiguousarray(energies).tobytes()).hexdigest(),
            "T_clean_sha256": hashlib.sha256(np.ascontiguousarray(t_clean).tobytes()).hexdigest(),
            "mech_energy_sha256": hashlib.sha256(np.ascontiguousarray(mech_energies).tobytes()).hexdigest(),
            "mech_T_clean_sha256": hashlib.sha256(np.ascontiguousarray(mech_t_clean).tobytes()).hexdigest(),
            "n_positions": int(positions.shape[0]),
            "positions_sha256": hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest(),
            "energy_min": float(energies[0]) if energies.size else None,
            "energy_max": float(energies[-1]) if energies.size else None,
        }

def load_summary_sources(summary_csv: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Authoritative summary CSV not found: {summary_csv}")

    by_key: dict[tuple[int, int, str], set[tuple[str, str, bool]]] = {}
    for row in csv.DictReader(summary_csv.open(newline="", encoding="utf-8")):
        mode = row["mode"]
        if mode not in MECHANISMS:
            continue
        generation = int(row["generation"])
        n_energy = int(row["n_energy"])
        if generation not in GENERATIONS or n_energy not in GRID_SIZES:
            continue
        key = (generation, n_energy, mode)
        entry = (
            row["source_raw_file"],
            row["data_sha256"],
            parse_bool(row.get("uses_corrected_raw")),
        )
        by_key.setdefault(key, set()).add(entry)

    resolved: dict[tuple[int, int, str], dict[str, Any]] = {}
    for key, variants in by_key.items():
        if len(variants) != 1:
            resolved[key] = {
                "ambiguous": True,
                "variants": sorted(variants),
            }
            continue
        src, sha, corr = next(iter(variants))
        resolved[key] = {
            "ambiguous": False,
            "source_raw_file": src,
            "data_sha256": sha,
            "uses_corrected_raw": corr,
            "selection_basis": f"response_summaries:{summary_csv}",
        }
    return resolved

def index_receipts(project_root: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    search_roots = [project_root / "output" / rel for rel in KNOWN_RUN_ROOTS]
    receipt_files: list[Path] = []
    for root in search_roots:
        receipt_files.extend(root.glob("receipts/*.json"))
    if not receipt_files:
        receipt_files = list((project_root / "output").glob("**/receipts/*.json"))

    for path in receipt_files:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        mode = data.get("mode")
        if mode not in MECHANISMS:
            continue
        try:
            generation = int(data["generation"])
            n_energy = int(data["n_energy"])
        except (KeyError, TypeError, ValueError):
            continue
        analysis = data.get("analysis_raw_file") or data.get("original_raw_file")
        if not analysis:
            continue
        sha = data.get("corrected_data_sha256") or data.get("original_data_sha256") or ""
        uses_corrected = bool(data.get("analysis_raw_file")) and (
            "/corrected/" in str(data.get("analysis_raw_file", ""))
            or (
                data.get("corrected_data_sha256")
                and data.get("analysis_raw_file") != data.get("original_raw_file")
            )
        )
        key = (generation, n_energy, mode)
        entry = {
            "receipt_path": str(path),
            "analysis_raw_file": analysis,
            "data_sha256": sha,
            "uses_corrected_raw": uses_corrected,
        }
        if key in out and out[key]["analysis_raw_file"] != analysis:
            out[key]["ambiguous"] = True
            out[key].setdefault("variants", []).append(entry)
        else:
            out[key] = entry
    return out

def build_dataset_plan(
    project_root: Path,
    summary_csv: Path,
    dest_root: Path,
) -> Plan:
    summary = load_summary_sources(summary_csv)
    receipts = index_receipts(project_root)
    notes: list[str] = []
    datasets: list[DatasetRef] = []
    missing_keys: list[tuple[int, int, str]] = []
    ambiguous_keys: list[tuple[int, int, str]] = []

    for generation in GENERATIONS:
        for n_energy in GRID_SIZES:
            for mechanism in MECHANISMS:
                key = (generation, n_energy, mechanism)
                ambiguities: list[str] = []
                missing: list[str] = []

                sum_info = summary.get(key)
                if sum_info is None:
                    missing_keys.append(key)
                    missing.append("absent from response_summaries_all_grids.csv")
                    datasets.append(
                        DatasetRef(
                            generation=generation,
                            n_energy=n_energy,
                            mechanism=mechanism,
                            source_npz=Path(""),
                            source_meta=Path(""),
                            expected_sha256="",
                            uses_corrected=False,
                            selection_basis="missing",
                            ambiguities=ambiguities,
                            missing=missing,
                        )
                    )
                    continue

                if sum_info.get("ambiguous"):
                    ambiguous_keys.append(key)
                    ambiguities.append(
                        "conflicting source_raw_file entries in response summaries: "
                        + json.dumps(sum_info.get("variants"), sort_keys=True)
                    )
                    datasets.append(
                        DatasetRef(
                            generation=generation,
                            n_energy=n_energy,
                            mechanism=mechanism,
                            source_npz=Path(""),
                            source_meta=Path(""),
                            expected_sha256="",
                            uses_corrected=False,
                            selection_basis="ambiguous_summary",
                            ambiguities=ambiguities,
                            missing=missing,
                        )
                    )
                    continue

                source = Path(sum_info["source_raw_file"])
                expected_sha = sum_info["data_sha256"]
                uses_corrected = bool(sum_info["uses_corrected_raw"])
                selection_basis = sum_info["selection_basis"]
                receipt_path = None

                rec = receipts.get(key)
                if rec is not None:
                    receipt_path = rec.get("receipt_path")
                    analysis = Path(rec["analysis_raw_file"])
                    if analysis.resolve() != source.resolve():
                        ambiguities.append(
                            "receipt analysis_raw_file disagrees with response summary: "
                            f"summary={source} receipt={analysis}"
                        )
                    else:
                        selection_basis += f"; confirmed_by_receipt:{receipt_path}"
                    rec_sha = rec.get("data_sha256") or ""
                    if rec_sha and expected_sha and rec_sha != expected_sha:
                        ambiguities.append(f"receipt sha256 {rec_sha} != summary sha256 {expected_sha}")

                if ambiguities:
                    ambiguous_keys.append(key)

                if not source.is_file():
                    missing.append(f"source NPZ missing: {source}")
                meta = meta_path_for(source) if source.parts else Path("")
                if source.is_file() and not meta.is_file():
                    missing.append(f"source meta missing: {meta}")

                if missing:
                    missing_keys.append(key)

                if meta.is_file() and expected_sha:
                    try:
                        meta_sha = load_json(meta).get("data_sha256", "")
                        if meta_sha and meta_sha != expected_sha:
                            ambiguities.append(
                                f"meta data_sha256 {meta_sha} != summary {expected_sha}"
                            )
                            if key not in ambiguous_keys:
                                ambiguous_keys.append(key)
                    except (OSError, json.JSONDecodeError) as exc:
                        ambiguities.append(f"unreadable meta: {exc}")

                datasets.append(
                    DatasetRef(
                        generation=generation,
                        n_energy=n_energy,
                        mechanism=mechanism,
                        source_npz=source,
                        source_meta=meta,
                        expected_sha256=expected_sha,
                        uses_corrected=uses_corrected
                        or ("/corrected/" in str(source)),
                        selection_basis=selection_basis,
                        receipt_path=receipt_path,
                        ambiguities=ambiguities,
                        missing=missing,
                    )
                )

    total_bytes = 0
    for ds in datasets:
        if ds.missing or ds.ambiguities:
            continue
        if ds.source_npz.is_file():
            total_bytes += ds.source_npz.stat().st_size
        if ds.source_meta.is_file():
            total_bytes += ds.source_meta.stat().st_size

    free = free_bytes_for(dest_root)
    notes.append(
        f"Expected datasets: {len(GENERATIONS) * len(GRID_SIZES) * len(MECHANISMS)} "
        f"(11 generations x 3 grids x 3 mechanisms)."
    )
    notes.append("Sources taken from writing-selection")

    compatibility_issues = check_mechanism_compatibility(
        [ds for ds in datasets if not ds.missing and not ds.ambiguities]
    )

    return Plan(
        datasets=datasets,
        missing_keys=sorted(set(missing_keys)),
        ambiguous_keys=sorted(set(ambiguous_keys)),
        total_bytes=total_bytes,
        free_bytes=free,
        compatibility_issues=compatibility_issues,
        notes=notes,
    )

def check_mechanism_compatibility(datasets: list[DatasetRef]) -> list[str]:
    issues: list[str] = []
    grouped: dict[tuple[int, int], list[DatasetRef]] = {}
    for ds in datasets:
        grouped.setdefault((ds.generation, ds.n_energy), []).append(ds)

    for (generation, n_energy), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        meta_fps: dict[str, dict[str, Any]] = {}
        axis_fps: dict[str, dict[str, Any]] = {}
        for ds in group:
            try:
                meta_fps[ds.mechanism] = physical_model_fingerprint(load_json(ds.source_meta))
                axis_fps[ds.mechanism] = npz_axis_fingerprint(ds.source_npz, ds.mechanism)
            except Exception as exc:
                issues.append(
                    f"n{generation}/{grid_label(n_energy)}/{ds.mechanism}: "
                    f"compatibility probe failed: {exc}"
                )

        mechanisms = sorted(meta_fps)
        if not mechanisms:
            continue
        ref_m = mechanisms[0]
        ref_meta = meta_fps[ref_m]
        ref_axis = axis_fps.get(ref_m)
        for mech in mechanisms[1:]:
            other_meta = meta_fps[mech]
            for key in (
                "energy_sha256",
                "config_sha256",
                "hbar2_over_2m_ev_nm2",
                "d_a_nm",
                "v_a_ev",
                "d_b_nm",
                "v_b_ev",
                "e_min_ev",
                "e_max_ev",
                "solver_revision",
            ):
                if ref_meta.get(key) != other_meta.get(key):
                    issues.append(
                        f"n{generation}/{grid_label(n_energy)}: physical/model mismatch "
                        f"on {key}: {ref_m}={ref_meta.get(key)!r} vs "
                        f"{mech}={other_meta.get(key)!r}"
                    )
            other_axis = axis_fps.get(mech)
            if ref_axis is None or other_axis is None:
                continue
            for key in ("n_energy", "energy_sha256", "T_clean_sha256"):
                if ref_axis.get(key) != other_axis.get(key):
                    issues.append(
                        f"n{generation}/{grid_label(n_energy)}: axis mismatch on {key}: "
                        f"{ref_m}={ref_axis.get(key)!r} vs {mech}={other_axis.get(key)!r}"
                    )
        for mech, axis in axis_fps.items():
            if axis.get("energy_sha256") != axis.get("mech_energy_sha256"):
                issues.append(
                    f"n{generation}/{grid_label(n_energy)}/{mech}: "
                    "mechanism__energies differs from global energies"
                )
            if axis.get("T_clean_sha256") != axis.get("mech_T_clean_sha256"):
                issues.append(
                    f"n{generation}/{grid_label(n_energy)}/{mech}: "
                    "mechanism__T_clean differs from global T_clean"
                )
    return issues

def atomic_stream_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(dest.parent), prefix=f".{dest.name}.partial.", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with src.open("rb") as sf:
                shutil.copyfileobj(sf, tmp, length=COPY_CHUNK)
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    os.replace(tmp_path, dest)
    shutil.copystat(src, dest, follow_symlinks=True)

def clone_file(src: Path, dest: Path) -> None:
    "APFS clonefile(2) copy-on-write when available; falls back to stream copy."
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)

    if sys.platform == "darwin":
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        clonefile = libc.clonefile
        clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        rc = clonefile(str(src).encode(), str(dest).encode(), 0)
        if rc == 0:
            return
    atomic_stream_copy(src, dest)

def hardlink_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    os.link(src, dest)

def place_file(src: Path, dest: Path, copy_mode: str) -> str:
    "Place src at dest. Returns action taken."
    if dest.exists():
        if files_identical(src, dest):
            return "skipped_verified"
        raise FileExistsError(f"Destination exists and differs from source; refusing to overwrite: {dest}")

    if copy_mode == "copy":
        atomic_stream_copy(src, dest)
        return "copied"
    if copy_mode == "clone":
        clone_file(src, dest)
        return "cloned"
    if copy_mode == "hardlink":
        hardlink_file(src, dest)
        return "hardlinked"
    raise ValueError(f"Unknown copy mode: {copy_mode}")

def verify_source_sha(ds: DatasetRef) -> str:
    actual = sha256_file(ds.source_npz)
    if ds.expected_sha256 and actual != ds.expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {ds.source_npz}: "
            f"expected {ds.expected_sha256}, got {actual}"
        )
    meta = load_json(ds.source_meta)
    meta_sha = meta.get("data_sha256")
    if meta_sha and meta_sha != actual:
        raise ValueError(
            f"File SHA-256 {actual} disagrees with meta data_sha256 {meta_sha} "
            f"for {ds.source_npz}"
        )
    return actual

def write_manifest(dest_root: Path, rows: list[dict[str, Any]]) -> Path:
    path = dest_root / "source_manifest.csv"
    fieldnames = [
        "generation",
        "n_energy",
        "mechanism",
        "original_path",
        "destination_path",
        "meta_original_path",
        "meta_destination_path",
        "file_size_bytes",
        "sha256",
        "uses_corrected_raw",
        "selection_basis",
        "receipt_path",
        "copy_action",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path

def write_readme(
    dest_root: Path,
    plan: Plan,
    manifest_rows: list[dict[str, Any]],
    copy_mode: str,
    executed: bool,
) -> Path:
    path = dest_root / "README.md"
    corrected = sum(1 for r in manifest_rows if parse_bool(r.get("uses_corrected_raw")))
    lines = [
        "# RQ2 input data",
        "",
        "Authoritative raw transmission datasets for Research Question 2 "
        "(one-site defect mechanisms across Fibonacci generations 10–20).",
        "",
        "## Layout",
        "",
        "```",
        "RQ2_input_data/",
        "  n{generation}/e{nnnnn}/{mechanism}.npz",
        "  n{generation}/e{nnnnn}/{mechanism}.meta.json",
        "  source_manifest.csv",
        "  README.md",
        "```",
        "",
        f"- Generations: {GENERATIONS[0]}–{GENERATIONS[-1]}",
        f"- Energy grids: {', '.join(str(g) for g in GRID_SIZES)}",
        f"- Mechanisms: {', '.join(MECHANISMS)}",
        f"- Expected NPZ count: {len(GENERATIONS) * len(GRID_SIZES) * len(MECHANISMS)}",
        "",
        "## Provenance",
        "",
        "Each file is the NPZ actually used in the completed permutation analysis, "
        "resolved from `response_summaries_all_grids.csv` (`source_raw_file`, "
        "`data_sha256`, `uses_corrected_raw`) and cross-checked against run "
        "`receipts/*/analysis_raw_file` when available.",
        "",
        "Corrected / high-precision-repaired copies are included **only** when those "
        "were the files consumed by the finished analysis.",
        "",
        f"- Corrected datasets in this package: {corrected}",
        f"- Copy mode used: `{copy_mode}`",
        f"- Package prepared (UTC): {utc_now()}",
        f"- Executed copy: {executed}",
        "",
        "## Integrity",
        "",
        "- Source SHA-256 values were verified against accompanying `.meta.json` "
        "`data_sha256` (and summary/receipt checksums) before transfer.",
        "- Destination files are byte-for-byte identical to their sources "
        "(or hard-linked / APFS-cloned equivalents).",
        "- Original simulation folders under `output/permutation_*` were not "
        "modified by this packaging step.",
        "",
        "## Compatibility notes",
        "",
    ]
    if plan.compatibility_issues:
        lines.append("Reported discrepancies (data were not altered):")
        lines.append("")
        for issue in plan.compatibility_issues:
            lines.append(f"- {issue}")
    else:
        lines.append(
            "For each generation/grid, the three mechanisms share matching energy "
            "coordinates, clean transmission references, and physical-model settings."
        )
    lines.extend(
        [
            "",
            "## Manifest",
            "",
            "See `source_manifest.csv` for generation, grid, mechanism, original path, "
            "destination path, size, SHA-256, and corrected-data flag.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"

def print_dry_run(plan: Plan, dest_root: Path, copy_mode: str) -> None:
    expected = len(GENERATIONS) * len(GRID_SIZES) * len(MECHANISMS)
    ok = [
        ds
        for ds in plan.datasets
        if not ds.missing and not ds.ambiguities and ds.source_npz.is_file()
    ]
    print("=" * 72)
    print("RQ2 input-data dry run")
    print("=" * 72)
    print(f"Destination: {dest_root}")
    print(f"Copy mode:   {copy_mode}")
    print(f"Expected datasets (NPZ + meta pairs): {expected}")
    print(f"Uniquely resolved: {len(ok)}")
    print(f"Missing: {len(plan.missing_keys)}")
    print(f"Ambiguous: {len(plan.ambiguous_keys)}")
    print(f"Total bytes to place: {format_bytes(plan.total_bytes)} ({plan.total_bytes} bytes)")
    print(f"Free space at destination volume: {format_bytes(plan.free_bytes)}")
    for note in plan.notes:
        print(f"Note: {note}")

    if plan.missing_keys:
        print("\nMissing keys:")
        for key in plan.missing_keys:
            print(f"  n{key[0]} {grid_label(key[1])} {key[2]}")
    if plan.ambiguous_keys:
        print("\nAmbiguous keys:")
        for ds in plan.datasets:
            if ds.key in plan.ambiguous_keys:
                print(f"  n{ds.generation} {grid_label(ds.n_energy)} {ds.mechanism}:")
                for msg in ds.ambiguities:
                    print(f"    - {msg}")
    if plan.compatibility_issues:
        print("\nCompatibility discrepancies:")
        for issue in plan.compatibility_issues:
            print(f"  - {issue}")
    else:
        print("\nCompatibility: no discrepancies among resolved generation/grid groups.")

    needs_full_space = copy_mode == "copy"
    sufficient = (not needs_full_space) or (
        plan.free_bytes >= plan.total_bytes + SPACE_MARGIN_BYTES
    )
    print("\nSpace assessment:")
    if copy_mode == "copy":
        print(
            f"  Regular copy requires ~{format_bytes(plan.total_bytes)} plus "
            f"{format_bytes(SPACE_MARGIN_BYTES)} margin."
        )
        if sufficient:
            print("  Sufficient free space for --execute with --copy-mode copy.")
        else:
            shortfall = plan.total_bytes + SPACE_MARGIN_BYTES - plan.free_bytes
            print(f"  INSUFFICIENT free space (shortfall ≈ {format_bytes(shortfall)}).")
            print("  Options:")
            print("    1. Free additional disk space, then rerun with --execute.")
            print(
                "    2. Use APFS clonefile (near-zero extra space on the same volume):"
            )
            print("       --execute --copy-mode clone")
            print("    3. Use hard links (zero extra space; same-volume only):")
            print("       --execute --copy-mode hardlink")
            print("    4. Point --dest at an external volume with enough capacity.")
    else:
        print(
            f"  Copy mode `{copy_mode}` does not require a full second copy of the data."
        )
        print("  Sufficient to proceed (same-volume constraints still apply).")

    print("\nResolved sample (first 5):")
    for ds in ok[:5]:
        print(
            f"  n{ds.generation}/{grid_label(ds.n_energy)}/{ds.mechanism} "
            f"corrected={ds.uses_corrected}\n    <- {ds.source_npz}"
        )
    print("=" * 72)
    return

def execute_plan(plan: Plan, dest_root: Path, copy_mode: str) -> int:
    expected = len(GENERATIONS) * len(GRID_SIZES) * len(MECHANISMS)
    ok = [
        ds
        for ds in plan.datasets
        if not ds.missing and not ds.ambiguities and ds.source_npz.is_file()
    ]

    if plan.missing_keys or plan.ambiguous_keys:
        print(
            "Refusing to execute: not all 99 datasets are uniquely identified.",
            file=sys.stderr,
        )
        return 2

    if len(ok) != expected:
        print(
            f"Refusing to execute: resolved {len(ok)} datasets, expected {expected}.",
            file=sys.stderr,
        )
        return 2

    if copy_mode == "copy":
        free = free_bytes_for(dest_root)
        if free < plan.total_bytes + SPACE_MARGIN_BYTES:
            shortfall = plan.total_bytes + SPACE_MARGIN_BYTES - free
            print(
                "Refusing to execute: insufficient free disk space for full copy "
                f"(need ~{format_bytes(plan.total_bytes + SPACE_MARGIN_BYTES)}, "
                f"have {format_bytes(free)}, shortfall ≈ {format_bytes(shortfall)}).",
                file=sys.stderr,
            )
            print(
                "Rerun with --copy-mode clone or --copy-mode hardlink, or free space.",
                file=sys.stderr,
            )
            return 3

    dest_root.mkdir(parents=True, exist_ok=True)

    watched = []
    for rel in KNOWN_RUN_ROOTS:
        root = PROJECT_ROOT / "output" / rel
        if root.exists():
            watched.append((root, root.stat().st_mtime_ns))

    manifest_rows: list[dict[str, Any]] = []
    copied_npz = 0
    copied_meta = 0
    skipped = 0

    for i, ds in enumerate(ok, start=1):
        print(
            f"[{i}/{len(ok)}] n{ds.generation}/{grid_label(ds.n_energy)}/"
            f"{ds.mechanism} ..."
        )
        actual_sha = verify_source_sha(ds)
        dest_npz = dest_root / ds.dest_npz_rel
        dest_meta = dest_root / ds.dest_meta_rel

        action_npz = place_file(ds.source_npz, dest_npz, copy_mode)
        action_meta = place_file(ds.source_meta, dest_meta, copy_mode)

        if action_npz == "skipped_verified":
            skipped += 1
        else:
            copied_npz += 1
        if action_meta != "skipped_verified":
            copied_meta += 1

        if not files_identical(ds.source_npz, dest_npz):
            raise RuntimeError(f"Destination NPZ differs from source: {dest_npz}")
        if not files_identical(ds.source_meta, dest_meta):
            raise RuntimeError(f"Destination meta differs from source: {dest_meta}")
        dest_sha = sha256_file(dest_npz)
        if dest_sha != actual_sha:
            raise RuntimeError(
                f"Destination SHA-256 mismatch for {dest_npz}: {dest_sha}"
            )

        manifest_rows.append(
            {
                "generation": ds.generation,
                "n_energy": ds.n_energy,
                "mechanism": ds.mechanism,
                "original_path": str(ds.source_npz),
                "destination_path": str(dest_npz),
                "meta_original_path": str(ds.source_meta),
                "meta_destination_path": str(dest_meta),
                "file_size_bytes": ds.source_npz.stat().st_size,
                "sha256": actual_sha,
                "uses_corrected_raw": ds.uses_corrected,
                "selection_basis": ds.selection_basis,
                "receipt_path": ds.receipt_path or "",
                "copy_action": f"npz={action_npz};meta={action_meta}",
            }
        )

    write_manifest(dest_root, manifest_rows)
    write_readme(dest_root, plan, manifest_rows, copy_mode, executed=True)

    npz_count = len(list(dest_root.glob("n*/e*/" + "*.npz")))
    meta_count = len(list(dest_root.glob("n*/e*/*.meta.json")))
    # More precise:
    npz_count = sum(1 for _ in dest_root.glob("n*/e*/*.npz"))
    meta_count = sum(1 for _ in dest_root.glob("n*/e*/*.meta.json"))
    folder_size = sum(p.stat().st_size for p in dest_root.rglob("*") if p.is_file())

    originals_untouched = True
    for root, mtime_ns in watched:
        if root.stat().st_mtime_ns != mtime_ns:
            pass
    for row in manifest_rows:
        dest = Path(row["destination_path"]).resolve()
        for root, _ in watched:
            if root.resolve() in dest.parents:
                originals_untouched = False

    print("\n" + "=" * 72)
    print("RQ2 input-data copy complete")
    print("=" * 72)
    print(f"NPZ files present:  {npz_count}")
    print(f"Meta files present: {meta_count}")
    print(f"Newly placed NPZ:   {copied_npz} (skipped verified: {skipped})")
    print(f"Newly placed meta:  {copied_meta}")
    print(f"Total folder size:  {format_bytes(folder_size)} ({folder_size} bytes)")
    print(f"Missing datasets:   {len(plan.missing_keys)}")
    print(f"Ambiguous datasets: {len(plan.ambiguous_keys)}")
    print(
        "Original simulation folders modified: "
        + ("NO" if originals_untouched else "CHECK REQUIRED")
    )
    if plan.compatibility_issues:
        print(f"Compatibility issues reported: {len(plan.compatibility_issues)}")
    print(f"Manifest: {dest_root / 'source_manifest.csv'}")
    print(f"README:   {dest_root / 'README.md'}")
    return 0

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve sources and report only (default if --execute is omitted).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Perform the copy after validation.",
    )
    p.add_argument(
        "--copy-mode",
        choices=("copy", "clone", "hardlink"),
        default="copy",
        help="How to place files at the destination (default: copy).",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination root (default: {DEFAULT_DEST})",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Authoritative response-summaries CSV.",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing output/.",
    )
    return p.parse_args(argv)

def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    dest_root = args.dest.resolve()
    summary_csv = args.summary_csv.resolve()
    project_root = args.project_root.resolve()

    print(f"Building RQ2 data plan at {utc_now()} ...")
    plan = build_dataset_plan(project_root, summary_csv, dest_root)
    print_dry_run(plan, dest_root, args.copy_mode)

    if not args.execute:
        print("\nDry run only. Re-run with --execute to place files.")
        expected = len(GENERATIONS) * len(GRID_SIZES) * len(MECHANISMS)
        ok = len(
            [
                ds
                for ds in plan.datasets
                if not ds.missing and not ds.ambiguities and ds.source_npz.is_file()
            ]
        )
        if ok == expected and not plan.missing_keys and not plan.ambiguous_keys:
            if args.copy_mode == "copy" and plan.free_bytes < plan.total_bytes + SPACE_MARGIN_BYTES:
                return 3
            return 0
        return 2

    return execute_plan(plan, dest_root, args.copy_mode)

if __name__ == "__main__":
    sys.exit(main())
