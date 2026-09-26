"Resonance and peak detection w/ order preserving matching. Maximize number of matched peaks then minimize total resonance displacement."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry
from fibonacci_qc.parity_analysis import position_parity, position_valid_mask

_BASELINE_DE_EV = (cfg.E_MAX_EV - cfg.E_MIN_EV) / (cfg.N_ENERGY - 1)
DEFAULT_MAX_MATCH_DELTA_E_EV = 50.0 * _BASELINE_DE_EV

DEFAULT_MIN_PROMINENCE = 1e-4
DEFAULT_MIN_DISTANCE_EV = _BASELINE_DE_EV

@dataclass(frozen=True)
class Peak:
    "One detected peak in energy grid"
    index: int
    energy: float
    height: float
    prominence: float

@dataclass(frozen=True)
class PeakMatch:
    "One order preserving matched pair"
    clean_index: int
    defect_index: int
    energy_clean: float
    energy_defect: float
    height_clean: float
    height_defect: float
    delta_E: float

@dataclass
class MatchResult:
    "Matched / unmatched pair"
    matched: List[PeakMatch] = field(default_factory=list)
    unmatched_clean: List[Peak] = field(default_factory=list)
    unmatched_defect: List[Peak] = field(default_factory=list)
    n_matched: int = 0
    total_displacement: float = 0.0
    def __post_init__(self) -> None:
        self.n_matched = len(self.matched)
        self.total_displacement = float(
            sum(abs(m.delta_E) for m in self.matched)
        )

@dataclass(frozen=True)
class ResonanceTrack:
    "Resonance tracking for one mode and position"
    mode: str
    position: int
    parity: int
    valid: bool
    clean_peaks: Tuple[Peak, ...]
    defect_peaks: Tuple[Peak, ...]
    match: MatchResult
    mean_delta_E: float
    mean_abs_delta_E: float
    rms_delta_E: float

def _refine_peak_energy(values: np.ndarray, energies: np.ndarray, index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(energies[index])
    y0, y1, y2 = float(values[index - 1]), float(values[index]), float(values[index + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-300 or not np.isfinite(denom):
        return float(energies[index])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    e0, e1, e2 = float(energies[index - 1]), float(energies[index]), float(energies[index + 1])
    if delta <= 0.0:
        return e1 + delta * (e1 - e0)
    return e1 + delta * (e2 - e1)


def detect_peaks(
    values: np.ndarray,
    energies: np.ndarray,
    *,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_distance_ev: int = DEFAULT_MIN_DISTANCE_EV,
) -> List[Peak]:
    "Detect local maxima with optional filters."
    y = np.asarray(values, dtype=np.float64)
    e = np.asarray(energies, dtype=np.float64)
    if y.shape != e.shape:
        raise ValueError("values and energies must have the same shape")
    if y.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if y.size < 3:
        return []
    candidates: List[Peak] = []
    for i in range(1, y.size - 1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i - 1]) and np.isfinite(y[i + 1])):
            continue
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            prominence = float(y[i] - max(y[i - 1], y[i + 1]))
            if prominence + 1e-15 < min_prominence:
                continue
            candidates.append(
                Peak(
                    index=i,
                    energy=_refine_peak_energy(y,e,i),
                    height=float(y[i]),
                    prominence=prominence,
                )
            )
    if min_distance_ev <= 0.0 or len(candidates) <= 1:
        return candidates
    #Keep highest within min_distance (in grid index units).
    ordered = sorted(candidates, key=lambda p: p.height, reverse=True)
    kept: List[Peak] = []
    taken = np.zeros(y.size, dtype=bool)
    for peak in ordered:
        if any(abs(peak.energy - k.energy) < min_distance_ev - 1e-15 for k in kept):
            continue
        kept.append(peak)
    kept.sort(key=lambda p: p.index)
    return kept

def match_peaks_order_preserving(
    clean_peaks: Sequence[Peak],
    defect_peaks: Sequence[Peak],
    *,
    max_delta_e: float = DEFAULT_MAX_MATCH_DELTA_E_EV,
) -> MatchResult:
    "Max matches min displacement among admissible pairs (|E_clean - E_defect| <= max_delta_e)"
    a = list(clean_peaks)
    b = list(defect_peaks)
    m, n = len(a), len(b)
    # best[i][j] = (n_matches, -total_cost) using a[:i], b[:j]
    best = [[(0, 0.0) for _ in range(n + 1)] for _ in range(m + 1)]
    choice = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    # choice: 0=skip a, 1=skip b, 2=match
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            candidates: List[Tuple[Tuple[int, float], int]] = []
            if i > 0:
                candidates.append((best[i - 1][j], 0))
            if j > 0:
                candidates.append((best[i][j - 1], 1))
            if i > 0 and j > 0:
                cost = abs(a[i - 1].energy - b[j - 1].energy)
                if cost <= max_delta_e + 1e-15:
                    prev_m, prev_c = best[i-1][j-1]
                    candidates.append((prev_m + 1, prev_c - cost),2)
            chosen_state, chosen_op = max(candidates, key=lambda item: item[0])
            best[i][j] = chosen_state
            choice[i][j] = chosen_op
    matched: List[PeakMatch] = []
    used_a = set()
    used_b = set()
    i, j = m, n
    while i > 0 or j > 0:
        op = choice[i][j]
        if op == 0:
            i -= 1
        elif op == 1:
            j -= 1
        else:
            ca = a[i - 1]
            db = b[j - 1]
            matched.append(
                PeakMatch(
                    clean_index=ca.index,
                    defect_index=db.index,
                    energy_clean=ca.energy,
                    energy_defect=db.energy,
                    height_clean=ca.height,
                    height_defect=db.height,
                    delta_E=db.energy - ca.energy,
                )
            )
            used_a.add(i - 1)
            used_b.add(j - 1)
            i -= 1
            j -= 1
    matched.reverse()
    unmatched_clean = [p for idx, p in enumerate(a) if idx not in used_a]
    unmatched_defect = [p for idx, p in enumerate(b) if idx not in used_b]
    return MatchResult(
        matched=matched,
        unmatched_clean=unmatched_clean,
        unmatched_defect=unmatched_defect,
    )

def chain_peaks_across_generations(
    peaks_by_generation: Sequence[Sequence[Peak]],
    generations: Sequence[int],
    *,
    max_delta_e: float = DEFAULT_MAX_MATCH_DELTA_E_EV,
) -> List[Dict[str, object]]:
    "Order preserving matches chaining peak lists over generations, return one record"
    if len(peaks_by_generation) != len(generations):
        raise ValueError("peaks_by_generation and generations length are mismatched")
    if not peaks_by_generation:
        return []
    #identites are each peak in first generation
    identities: List[Dict[str, object]] = [
        {
            "path": [
                {
                    "generation": int(generations[0]),
                    "peak_index": p.index,
                    "energy": p.energy,
                }
            ]
        }
        for p in peaks_by_generation[0]
    ]
    current = list(peaks_by_generation[0])
    for g_idx in range(1, len(peaks_by_generation)):
        nxt = list(peaks_by_generation[g_idx])
        match = match_peaks_order_preserving(current, nxt, max_delta_e=max_delta_e)
        forward: Dict[int, Peak] = {}
        # rebuild w/ energies
        used_prev = set()
        new_current: List[Peak] = []
        new_identities: List[Dict[str, object]] = []
        # Rematch
        prev_by_index = {p.index: i for i, p in enumerate(current)}
        next_by_index = {p.index: j for j, p in enumerate(nxt)}
        for m in match.matched:
            if m.clean_index not in prev_by_index or m.defect_index not in next_by_index:
                continue
            i = prev_by_index[m.clean_index]
            j = next_by_index[m.defect_index]
            if i in used_prev:
                continue
            used_prev.add(i)
            ident = identities[i]
            ident = {
                "path": list(ident["path"])
                + [
                    {
                        "generation": int(generations[g_idx]),
                        "peak_index": nxt[j].index,
                        "energy": nxt[j].energy,
                    }
                ]
            }
            new_identities.append(ident)
            new_current.append(nxt[j])
        identities = new_identities
        current = new_current
        if not current:
            break
    return identities

def _shift_stats(matches: Sequence[PeakMatch]) -> Tuple[float, float, float]:
    if not matches:
        return float("nan"), float("nan"), float("nan")
    deltas = np.array([m.delta_E for m in matches], dtype=np.float64)
    return (
        float(np.mean(deltas)),
        float(np.mean(np.abs(deltas))),
        float(np.sqrt(np.mean(deltas ** 2))),
    )

def track_resonances(
    clean_spectrum: np.ndarray,
    defect_spectrum: np.ndarray,
    energies: np.ndarray,
    *,
    mode: str,
    position: int,
    parity: int,
    valid: bool = True,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_distance_ev: int = DEFAULT_MIN_DISTANCE_EV,
    max_delta_e: float = DEFAULT_MAX_MATCH_DELTA_E_EV,
    clean_peaks: Optional[Sequence[Peak]] = None,
) -> ResonanceTrack:
    "Detect and match peaks"
    if not valid:
        empty = MatchResult()
        return ResonanceTrack(
            mode=mode,
            position=position,
            parity=parity,
            valid=False,
            clean_peaks=(),
            defect_peaks=(),
            match=empty,
            mean_delta_E=float("nan"),
            mean_abs_delta_E=float("nan"),
            rms_delta_E=float("nan"),
        )
    clean_peaks = detect_peaks(
        clean_spectrum,
        energies,
        min_prominence=min_prominence,
        min_distance_ev=min_distance_ev,
    )
    if clean_peaks is None:
        clean_peaks_list = detect_peaks(clean_spectrum, energies, min_prominence=min_prominence, min_distance_ev=min_distance_ev)
    else:
        clean_peaks_list = list(clean_peaks)
    defect_peaks = detect_peaks(
        defect_spectrum,
        energies,
        min_prominence=min_prominence,
        min_distance_ev=min_distance_ev,
    )
    match = match_peaks_order_preserving(clean_peaks_list, defect_peaks, max_delta_e=max_delta_e)
    mean_d, mean_abs, rms = _shift_stats(match.matched)
    return ResonanceTrack(
        mode=mode,
        position=position,
        parity=parity,
        valid=True,
        clean_peaks=tuple(clean_peaks_list),
        defect_peaks=tuple(defect_peaks),
        match=match,
        mean_delta_E=mean_d,
        mean_abs_delta_E=mean_abs,
        rms_delta_E=rms,
    )

def analyze_mode_resonances(
    transmission: np.ndarray,
    t_clean: np.ndarray,
    energies: np.ndarray,
    geometry: FibonacciGeometry,
    mode: str,
    *,
    sweep_valid: Optional[np.ndarray] = None,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_distance_ev: int = DEFAULT_MIN_DISTANCE_EV,
) -> Dict[str, object]:
    "Track resonance for every position of one mode"
    t = np.asarray(transmission, dtype=np.float64)
    t_clean = np.asarray(t_clean, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
    clean_peaks = detect_peaks(t_clean, energies, min_prominence=min_prominence, min_distance_ev=min_distance_ev)
    tracks: List[ResonanceTrack] = []
    for p in range(t.shape[0]):
        tracks.append(
            track_resonances(
                t_clean,
                t[p],
                energies,
                mode=mode,
                position=p,
                parity=int(parity[p]),
                valid=bool(valid[p]),
                min_prominence=min_prominence,
                min_distance_ev=min_distance_ev,
                clean_peaks=clean_peaks,
            )
        )
    return {
        "mode": mode,
        "generation": geometry.generation,
        "energies": energies,
        "clean_peaks": clean_peaks,
        "tracks": tracks,
        "summary": summarize_resonance_tracks(tracks),
    }

def summarize_resonance_tracks(
    tracks: Sequence[ResonanceTrack],
) -> Dict[str, object]:
    "Summary measures (by position and parity)"
    by_position: List[Dict[str, object]] = []
    for track in tracks:
        by_position.append(
            {
                "position": track.position,
                "parity": track.parity,
                "valid": track.valid,
                "n_clean_peaks": len(track.clean_peaks),
                "n_defect_peaks": len(track.defect_peaks),
                "n_matched": track.match.n_matched,
                "n_unmatched_clean": len(track.match.unmatched_clean),
                "n_unmatched_defect": len(track.match.unmatched_defect),
                "total_displacement": track.match.total_displacement,
                "mean_delta_E": track.mean_delta_E,
                "mean_abs_delta_E": track.mean_abs_delta_E,
                "rms_delta_E": track.rms_delta_E,
            }
        )
    by_parity: Dict[str, Dict[str, float]] = {}
    for parity_value in (0, 1):
        group = [t for t in tracks if t.valid and t.parity == parity_value]
        deltas: List[float] = []
        per_position_mean_abs: List[float]=[]
        n_matched = 0
        n_unmatched_clean = 0
        n_unmatched_defect = 0
        n_clean_peaks = 0
        n_defect_peaks = 0
        n_positions_with_match = 0
        for track in group:
            n_matched += track.match.n_matched
            n_unmatched_clean += len(track.match.unmatched_clean)
            n_unmatched_defect += len(track.match.unmatched_defect)
            n_clean_peaks += len(track.clean_peaks)
            n_defect_peaks += len(track.defect_peaks)
            deltas.extend(m.delta_E for m in track.match.matched)
            if track.match.n_matched > 0 and np.isfinite(track.mean_abs_delta_E):
                per_position_mean_abs.append(track.mean_abs_delta_E)
                n_positions_with_match += 1
        # avg over all matched pairs
        if deltas:
            arr = np.asarray(deltas, dtype=np.float64)
            pooled = {
                "mean_delta_E": float(np.mean(arr)),
                "mean_abs_delta_E": float(np.mean(np.abs(arr))),
                "rms_delta_E": float(np.sqrt(np.mean(arr ** 2))),
            }
        else:
            pooled = {
                "mean_delta_E": float("nan"),
                "mean_abs_delta_E": float("nan"),
                "rms_delta_E": float("nan"),
            }
        if per_position_mean_abs:
            equal_position_mean_abs_delta_E = float(np.mean(per_position_mean_abs))
        else:
            equal_position_mean_abs_delta_E = float("nan")
        stats = {
        "n_positions": float(len(group)),
        "n_matched": float(n_matched),
        "n_unmatched_clean": float(n_unmatched_clean),
        "n_unmatched_defect": float(n_unmatched_defect),
        "match_coverage_peaks": (
            float(n_matched / n_clean_peaks) if n_clean_peaks > 0 else float("nan")
        ),
        "match_coverage_positions": (
            float(n_positions_with_match / len(group)) if group else float("nan")
        ),
        "equal_position_mean_abs_delta_E": equal_position_mean_abs_delta_E,
        "estimand_pooled_matches": pooled,
        # keep prior keys as the pooled estimand for back-compat:
        **pooled,
    }
    by_parity["even" if parity_value == 0 else "odd"] = stats
    return {"by_position": by_position, "by_parity": by_parity}

def analyze_generation_resonances(
    raw_bundle: Dict[str, object],
    geometry: Optional[FibonacciGeometry] = None,
    *,
    modes: Optional[Sequence[str]] = None,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_distance_ev: int = DEFAULT_MIN_DISTANCE_EV,
) -> Dict[str, object]:
    "Resonance track for all modes"
    generation = int(raw_bundle["generation"])
    geom = geometry if geometry is not None else build_geometry(generation)
    if geom.generation != generation:
        raise ValueError(f"geometry.generation={geom.generation} does not match bundle={generation}")
    energies = np.asarray(raw_bundle["energies"], dtype=np.float64)
    t_clean = np.asarray(raw_bundle["T_clean"], dtype=np.float64)
    modes_dict: Dict[str, Dict[str, np.ndarray]] = raw_bundle["modes"]  # type: ignore[assignment]
    modes = list(modes or modes_dict.keys())
    mode_results: Dict[str, object] = {}
    for mode in modes:
        sweep = modes_dict[mode]
        mode_results[mode] = analyze_mode_resonances(
            sweep["T"],
            t_clean,
            energies,
            geom,
            mode,
            sweep_valid=np.asarray(sweep["valid"], dtype=bool),
            min_prominence=min_prominence,
            min_distance_ev=min_distance_ev,
        )
    return {
        "generation": generation,
        "energies": energies,
        "geometry_metadata": geom.to_metadata(),
        "modes": mode_results,
    }