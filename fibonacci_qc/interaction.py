"Non additivity remainder I_p(E)=∆T_sub(p,E)-∆T_pot(p,E)-∆T_width(p,E) and one similar for logs. "

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence
import numpy as np

from fibonacci_qc import config as cfg

@dataclass(frozen=True)
class InteractionRemainder:
    "Interaction remainder array on the site x energy grid"
    I: np.ndarray
    I_log: np.ndarray
    energies: np.ndarray
    positions: np.ndarray
    valid: np.ndarray
    delta_T_sub: np.ndarray
    delta_T_pot: np.ndarray
    delta_T_width: np.ndarray
    delta_log_T_sub: np.ndarray
    delta_log_T_pot: np.ndarray
    delta_log_T_width: np.ndarray

def delta_log_t(
    log_t: np.ndarray,
    log_t_clean: np.ndarray,
    *,
    usable: Optional[np.ndarray] = None,
) -> np.ndarray:
    "Give log(T_m) - log(T_clean), broadcasting on the energy axis."
    log_t = np.asarray(log_t, dtype=np.float64)
    log_t_clean = np.asarray(log_t_clean, dtype=np.float64)
    if log_t.ndim == 1:
        out = log_t - log_t_clean
    else:
        out = log_t - log_t_clean[np.newaxis, :]
    if usable is not None:
        out = np.where(np.asarray(usable, dtype=bool), out, np.nan)
    out = np.where(np.isfinite(out), out, np.nan)
    return out

def interaction_remainder(
    delta_t_sub: np.ndarray,
    delta_t_pot: np.ndarray,
    delta_t_width: np.ndarray,
) -> np.ndarray:
    "I = ΔT_sub-ΔT_pot-ΔT_width."
    return np.asarray(delta_t_sub, dtype=np.float64)-np.asarray(delta_t_pot, dtype=np.float64)-np.asarray(delta_t_width, dtype=np.float64)

def interaction_remainder_log(
    delta_log_t_sub: np.ndarray,
    delta_log_t_pot: np.ndarray,
    delta_log_t_width: np.ndarray,
) -> np.ndarray:
    "I_log = Δlog(T_sub) - Δlog(T_pot) - Δlog(T_width). Separate multiplicative null diagnose"
    out = np.asarray(delta_log_t_sub, dtype=np.float64)-np.asarray(delta_log_t_pot, dtype=np.float64)-np.asarray(delta_log_t_width, dtype=np.float64)
    return np.where(np.isfinite(out), out, np.nan)

def _require_mode(
    modes: Mapping[str, Mapping[str, np.ndarray]],
    mode: str,
) -> Mapping[str, np.ndarray]:
    if mode not in modes:
        raise KeyError(f"mode {mode!r} required for interaction remainder; available: {tuple(modes)}")
    return modes[mode]

def _status_log_usable(status: np.ndarray) -> np.ndarray:
    "True when log_T is trusted"
    status = np.asarray(status, dtype=object)
    return status == "ok"

def _broadcast_clean_usable(
    clean_usable: np.ndarray,
    n_pos: int,
) -> np.ndarray:
    clean_usable = np.asarray(clean_usable, dtype = bool)
    if clean_usable.ndim == 1:
        return np.broadcast_to(clean_usable[np.newaxis, :], (n_pos, clean_usable.size)).copy
    return clean_usable

def _cell_ok_from_status(status: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if status is None:
        return None
    return np.asarray(status, dtype=object) == "ok"



def _log_usable_mask(
    *,
    log_t_sub: np.ndarray,
    log_t_pot: np.ndarray,
    log_t_width: np.ndarray,
    log_t_clean: np.ndarray,
    status_sub: Optional[np.ndarray] = None,
    status_pot: Optional[np.ndarray] = None,
    status_width: Optional[np.ndarray] = None,
    status_clean: Optional[np.ndarray] = None,
 ) -> np.ndarray:
    "Cell wise mask for logs, prefer status ok when given, or req finite logs, prefer status ok or require finite logs"
    log_t_sub = np.asarray(log_t_sub, dtype=np.float64)
    log_t_pot = np.asarray(log_t_pot, dtype=np.float64)
    log_t_width = np.asarray(log_t_width, dtype=np.float64)
    log_t_clean = np.asarray(log_t_clean, dtype=np.float64)
    n_pos = log_t_sub.shape[0]
    usable = np.isfinite(log_t_sub) & np.isfinite(log_t_pot) & np.isfinite(log_t_width) & np.isfinite(log_t_clean)[np.newaxis, :]
    if status_sub is not None:
        usable = usable & _status_log_usable(status_sub)
    if status_pot is not None:
        usable = usable & _status_log_usable(status_pot)
    if status_width is not None:
        usable = usable & _status_log_usable(status_width)
    if status_clean is not None:
        usable = usable & _broadcast_clean_usable(_status_log_usable(status_clean), n_pos)
    return usable

def interaction_from_mode_arrays(
    *,
    t_sub: np.ndarray,
    t_pot: np.ndarray,
    t_width: np.ndarray,
    log_t_sub: np.ndarray,
    log_t_pot: np.ndarray,
    log_t_width: np.ndarray,
    t_clean: np.ndarray,
    log_t_clean: np.ndarray,
    energies: np.ndarray,
    valid: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    status_sub: Optional[np.ndarray] = None,
    status_pot: Optional[np.ndarray] = None,
    status_width: Optional[np.ndarray] = None,
    status_clean: Optional[np.ndarray] = None,
) -> InteractionRemainder:
    "Build interaction remainder from explicit transmission arrays."
    t_clean = np.asarray(t_clean, dtype=np.float64)
    log_t_clean = np.asarray(log_t_clean, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    t_sub_a = np.asarray(t_sub, dtype=np.float64)
    t_pot_a = np.asarray(t_pot, dtype=np.float64)
    t_width_a = np.asarray(t_width, dtype=np.float64)

    cell_ok = np.isfinite(t_sub_a) & np.isfinite(t_pot_a) & np.isfinite(t_width_a) & np.isfinite(t_clean)[np.newaxis, :]

    for s in (status_sub, status_pot, status_width):
        ok = _cell_ok_from_status(s)
        if ok is not None:
            cell_ok = cell_ok & ok
    if status_clean is not None:
        cell_ok = cell_ok & _broadcast_clean_usable(_status_log_usable(status_clean), t_sub_a.shape[0])
    delta_sub = t_sub_a - t_clean[np.newaxis, :]
    delta_pot = t_pot_a - t_clean[np.newaxis, :]
    delta_width = t_width_a - t_clean[np.newaxis, :]
    delta_sub = np.where(cell_ok, delta_sub, np.nan)
    delta_pot = np.where(cell_ok, delta_pot, np.nan)
    delta_width = np.where(cell_ok, delta_width, np.nan)
    usable = _log_usable_mask(
        log_t_sub=log_t_sub,
        log_t_pot=log_t_pot,
        log_t_width=log_t_width,
        log_t_clean=log_t_clean,
        status_sub=status_sub,
        status_pot=status_pot,
        status_width=status_width,
        status_clean=status_clean,
    )
    dlog_sub = delta_log_t(log_t_sub, log_t_clean)
    dlog_pot = delta_log_t(log_t_pot, log_t_clean)
    dlog_width = delta_log_t(log_t_width, log_t_clean)
    n_pos = delta_sub.shape[0]
    if positions is None:
        positions = np.arange(n_pos, dtype=np.int64)
    else:
        positions = np.asarray(positions, dtype=np.int64)

    if valid is None:
        valid = np.ones(n_pos, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    return InteractionRemainder(
        I=interaction_remainder(delta_sub, delta_pot, delta_width),
        I_log=interaction_remainder_log(dlog_sub, dlog_pot, dlog_width),
        energies=energies,
        positions=positions,
        valid=valid,
        delta_T_sub=delta_sub,
        delta_T_pot=delta_pot,
        delta_T_width=delta_width,
        delta_log_T_sub=dlog_sub,
        delta_log_T_pot=dlog_pot,
        delta_log_T_width=dlog_width,
    )

def interaction_from_sweeps(
    modes: Mapping[str, Mapping[str, np.ndarray]],
    *,
    t_clean: np.ndarray,
    log_t_clean: np.ndarray,
    energies: np.ndarray,
    status_clean: Optional[np.ndarray] = None,
) -> InteractionRemainder:
    "Build I_p(E) from multi-mode sweep dictionary."
    sub = _require_mode(modes, cfg.MODE_SUBSTITUTION)
    pot = _require_mode(modes, cfg.MODE_POTENTIAL_ONLY)
    width = _require_mode(modes, cfg.MODE_WIDTH_ONLY)
    valid = np.asarray(sub["valid"], dtype=bool)&np.asarray(pot["valid"], dtype=bool)&np.asarray(width["valid"], dtype=bool)
    positions = np.asarray(sub.get("positions", np.arange(valid.size)),dtype=np.int64,)
    return interaction_from_mode_arrays(
        t_sub=sub["T"],
        t_pot=pot["T"],
        t_width=width["T"],
        log_t_sub=sub["log_T"],
        log_t_pot=pot["log_T"],
        log_t_width=width["log_T"],
        t_clean=t_clean,
        log_t_clean=log_t_clean,
        energies=energies,
        valid=valid,
        positions=positions,
        status_sub=sub.get("status"),
        status_pot=pot.get("status"),
        status_width=width.get("status"),
        status_clean=status_clean,
    )

def interaction_summary(
    remainder: InteractionRemainder,
    *,
    windows: Optional[Mapping[str, Sequence[float]]] = None,
) -> Dict[str, Dict[str, float]]:
    "RMS / mean absolute summaries of I over locked energy windows."
    windows = windows or cfg.ENERGY_WINDOWS
    energies = remainder.energies
    valid = remainder.valid
    out: Dict[str, Dict[str, float]] = {}
    for name, (e_min, e_max) in windows.items():
        sel = (energies >= e_min) & (energies <= e_max)
        if not np.any(sel) or not np.any(valid):
            out[name] = {
                "rms_I": float("nan"),
                "mean_abs_I": float("nan"),
                "rms_I_log": float("nan"),
                "mean_abs_I_log": float("nan"),
            }
            continue
        i_win = remainder.I[valid][:, sel]
        i_log_win = remainder.I_log[valid][:, sel]
        out[name] = {
            "rms_I": float(np.sqrt(np.nanmean(i_win ** 2))),
            "mean_abs_I": float(np.nanmean(np.abs(i_win))),
            "rms_I_log": float(np.sqrt(np.nanmean(i_log_win ** 2))),
            "mean_abs_I_log": float(np.nanmean(np.abs(i_log_win))),
        }
    return out
