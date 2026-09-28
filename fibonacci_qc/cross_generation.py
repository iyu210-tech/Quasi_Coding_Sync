"Clean anchor continuation and numerical grid consistency"

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from fibonacci_qc import config as cfg
from fibonacci_qc.resonance import(
    Peak,
    chain_peaks_across_generations,
    match_peaks_order_preserving,
    DEFAULT_MAX_MATCH_DELTA_E_EV,
)

CROSS_GENERATION_REVISION = "clean-anchor-continuation-v1"

def _peaks(rows):
    return [Peak(int(p["index"]), float(p["energy"]), float(p["height"]), float(p["prominence"])) for p in rows]

def _edges(tracks):
    return {((a["generation"], a["peak_index"]), (b["generation"], b["peak_index"])): b["energy"]-a["energy"] for track in tracks for a, b in zip(track["path"], track["path"][1:])}

def compare_clean_anchor_grid(generations, baseline, fine, *, complete=True):
    "Check edge identities and their energy shifts"
    gens = list(generations)
    if not complete:
        return {"gate": "unresolved", "reason": "untrusted clean spectrum"}
    if len(gens) < 2:
        return {"gate": "not_applicable", "reason": "only one generation"}
    b = [_peaks(baseline[g]) for g in gens]
    f = [_peaks(fine[g]) for g in gens]
    mapping = {}
    for gen, pb, pf in zip(gens, b, f):
        match = match_peaks_order_preserving(pb, pf, max_delta_e=cfg.GATE_TOLERANCES["peak_energy_ev"])
        if len(pb) != len(pf) or match.n_matched != len(pb):
            return {"gate": "fail", "reason": f"clean peaks changed under refinement at n={gen}"}
        mapping.update({(gen, m.clean_index): (gen, m.defect_index) for m in match.matched})
    eb = _edges(chain_peaks_across_generations(b, gens))
    ef = _edges(chain_peaks_across_generations(f, gens))
    expected = {(mapping[a], mapping[z]): shift for (a, z), shift in eb.items()}
    if set(expected) != set(ef):
        return {"gate": "fail", "reason": "cross-generation clean-anchor links changed"}
    if not expected:
        return {"gate": "not_applicable", "reason": "no adjacent-generation links", "n_links": 0}
    error = max(abs(ef[k]-v) for k, v in expected.items())
    return {"gate": "pass" if error <= cfg.GATE_TOLERANCES["peak_shift_ev"] else "fail",
            "reason": "same links; energy-shift tolerance applied",
            "n_links": len(expected), "max_shift_change_ev": float(error)}

def write_clean_anchor_tracks(resonance_root, generations, modes):
    "Persist clean tracks + streamed assignment table"
    root = Path(resonance_root)
    gens = sorted(int(g) for g in generations)
    if len(set(gens)) != len(gens) or not gens:
        raise ValueError("nonempty, unique generations required")
    wanted = list(modes)
    if not wanted:
        raise ValueError("at least one defect mode required")
    lists = []
    source = None
    numerical_failure = False
    for gen in gens:
        meta = json.loads((root/f"n{gen:02d}"/"resonance.meta.json").read_text())
        if int(meta["generation"]) != gen:
            raise ValueError("resonance generation mismatch")
        current_source = meta.get("run_metadata", {}).get("source_raw_dir")
        if source is None:
            source = current_source
        elif current_source != source:
            raise ValueError("cross-generation products come from different raw runs")
        rows = meta["modes"][wanted[0]]["clean_peaks"]
        for mode in wanted:
            if meta["modes"][mode]["clean_peaks"] != rows:
                raise ValueError("clean peak lists disagree across modes")
        numerical_failure |= bool(meta.get("numerical_failure", False))
        lists.append(_peaks(rows))
    tracks = chain_peaks_across_generations(lists, gens)
    lookup = {(p["generation"], p["peak_index"]): t["track_id"]
              for t in tracks for p in t["path"]}
    tracks_path = root/"cross_generation_tracks.json"
    tracks_path.write_text(json.dumps(tracks, indent=2), encoding="utf-8")
    assignment_path = root/"cross_generation_assignments.jsonl"
    count = 0
    with assignment_path.open("w", encoding="utf-8") as handle:
        for gen in gens:
            meta = json.loads((root/f"n{gen:02d}"/"resonance.meta.json").read_text())
            for mode in wanted:
                for track in meta["modes"][mode]["tracks"]:
                    if not track["valid"]:
                        continue
                    for pair in track["matched"]:
                        record = {
                            "generation": gen,
                            "mode": mode,
                            "position": int(track["position"]),
                            "parity": int(track["parity"]),
                            "clean_track_id": lookup[(gen, int(pair["clean_index"]))],
                            **pair
                        }
                        handle.write(json.dumps(record)+"\n")
                        count += 1
    report = {
        "analysis_revision": CROSS_GENERATION_REVISION,
        "generations": gens, "modes": wanted,
        "tracking_scope": "cross_generation_clean_anchors_for_within_generation_defect_shifts",
        "position_convention": "fixed left anchored site/bond index p",
        "identity_method": "adjacent-generation energy-ordered max-count/min-shift matching",
        "physical_eigenstate_identity_proven": False,
        "defected_peak_cross_generation_identity": False,
        "match_gate_max_delta_e_ev": DEFAULT_MAX_MATCH_DELTA_E_EV,
        "n_clean_tracks": len(tracks), 
        "n_links": len(_edges(tracks)), "n_assignments": count,
        "numerical_failure": numerical_failure, 
        "numerical_gate": "not_checked",
        "numerically_accepted": False,
        "run_metadata": {"source_raw_dir": source},
        "files": [tracks_path.name, assignment_path.name],
        "interpretation": "Births/terminations retained",
    }
    path = root/"cross_generation.meta.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(path),
        "files": report["files"],
        "n_links": report["n_links"]
    }