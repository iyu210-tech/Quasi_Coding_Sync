"Resonance and peak detection w/ order preserving matching. Maximize number of matched peaks then minimize total resonance displacement."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry
from fibonacci_qc.parity_analysis import position_parity, position_valid_mask

DEFAULT_MIN_PROMINENCE = 0.0
DEFAULT_MIN_DISTANCE = 1

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

def detect_peaks(
    values: np.ndarray,
    energies: np.ndarray,
    *,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_distance: int = DEFAULT_MIN_DISTANCE,
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
                    energy=float(e[i]),
                    height=float(y[i]),
                    prominence=prominence,
                )
            )
    if min_distance <= 1 or len(candidates) <= 1:
        return candidates
    #Keep highest within min_distance (in grid index units).
    ordered = sorted(candidates, key=lambda p: p.height, reverse=True)
    kept: List[Peak] = []
    taken = np.zeros(y.size, dtype=bool)
    for peak in ordered:
        lo = max(0, peak.index - min_distance + 1)
        hi = min(y.size, peak.index + min_distance)
        if np.any(taken[lo:hi]):
            continue
        taken[peak.index] = True
        kept.append(peak)
    kept.sort(key=lambda p: p.index)
    return kept

def match_peaks_order_preserving(
    clean_peaks: Sequence[Peak],
    defect_peaks: Sequence[Peak],
) -> MatchResult:
    "Max matches min displacement"
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
                prev_m, prev_c = best[i - 1][j - 1]
                cost = abs(a[i - 1].energy - b[j - 1].energy)
                candidates.append(((prev_m + 1, prev_c - cost), 2))
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
    min_distance: int = DEFAULT_MIN_DISTANCE,
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
        min_distance=min_distance,
    )
    defect_peaks = detect_peaks(
        defect_spectrum,
        energies,
        min_prominence=min_prominence,
        min_distance=min_distance,
    )
    match = match_peaks_order_preserving(clean_peaks, defect_peaks)
    mean_d, mean_abs, rms = _shift_stats(match.matched)
    return ResonanceTrack(
        mode=mode,
        position=position,
        parity=parity,
        valid=True,
        clean_peaks=tuple(clean_peaks),
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
    min_distance: int = DEFAULT_MIN_DISTANCE,
) -> Dict[str, object]:
    "Track resonance for every position of one mode"
    t = np.asarray(transmission, dtype=np.float64)
    t_clean = np.asarray(t_clean, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    parity = position_parity(geometry, mode)
    valid = position_valid_mask(geometry, mode, sweep_valid=sweep_valid)
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
                min_distance=min_distance,
            )
        )
    return {
        "mode": mode,
        "generation": geometry.generation,
        "energies": energies,
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
        n_matched = 0
        n_unmatched_clean = 0
        n_unmatched_defect = 0
        for track in group:
            n_matched += track.match.n_matched
            n_unmatched_clean += len(track.match.unmatched_clean)
            n_unmatched_defect += len(track.match.unmatched_defect)
            deltas.extend(m.delta_E for m in track.match.matched)
        if deltas:
            arr = np.asarray(deltas, dtype=np.float64)
            stats = {
                "n_positions": float(len(group)),
                "n_matched": float(n_matched),
                "n_unmatched_clean": float(n_unmatched_clean),
                "n_unmatched_defect": float(n_unmatched_defect),
                "mean_delta_E": float(np.mean(arr)),
                "mean_abs_delta_E": float(np.mean(np.abs(arr))),
                "rms_delta_E": float(np.sqrt(np.mean(arr ** 2))),
            }
        else:
            stats = {
                "n_positions": float(len(group)),
                "n_matched": float(n_matched),
                "n_unmatched_clean": float(n_unmatched_clean),
                "n_unmatched_defect": float(n_unmatched_defect),
                "mean_delta_E": float("nan"),
                "mean_abs_delta_E": float("nan"),
                "rms_delta_E": float("nan"),
            }
        by_parity["even" if parity_value == 0 else "odd"] = stats

    return {"by_position": by_position, "by_parity": by_parity}

def analyze_generation_resonances(
    raw_bundle: Dict[str, object],
    geometry: Optional[FibonacciGeometry] = None,
    *,
    modes: Optional[Sequence[str]] = None,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    min_distance: int = DEFAULT_MIN_DISTANCE,
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
            min_distance=min_distance,
        )
    return {
        "generation": generation,
        "energies": energies,
        "geometry_metadata": geom.to_metadata(),
        "modes": mode_results,
    }