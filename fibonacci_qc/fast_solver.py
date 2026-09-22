"Prefix / suffix cache for local defects"

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.defects import apply_defect, apply_phason, validate_defect_position
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry
from fibonacci_qc.scattering import (
    SMatrix,
    TransmissionResult,
    abeles_matrix,
    interface_s_matrix,
    propagation_s_matrix,
    q_number,
    redheffer_star,
    transfer_to_s_matrix,
    transmission_from_s,
    vacuum_passthrough_s_matrix,
)

def _site_coupling(
    site_index: int,
    q_lead: complex,
    q_layers: np.ndarray,
) -> SMatrix:
    if site_index == 0 :
        return interface_s_matrix(q_lead, q_layers[0])
    return interface_s_matrix(q_layers[site_index - 1], q_layers[site_index])

def _site_propagation(
    site_index: int,
    q_layers:np.ndarray,
    width_nm: float,
) -> SMatrix:
    return propagation_s_matrix(q_layers[site_index], width_nm)

def _stack_elements(
    q_lead:complex,
    q_layers:np.ndarray,
    widths_nm:np.ndarray,
    n_sites:int,
    *,
    start_site: int=0,
    end_site: Optional[int] = None,
    include_exit: bool = True,
) -> Tuple[SMatrix, ...]:
    "Return redheffer factors for site [start_site, end_site)"
    if end_site is None:
        end_site = n_sites
    elements: list[SMatrix] = []
    for idx in range(start_site, end_site):
        elements.append(_site_coupling(idx, q_lead, q_layers))
        elements.append(_site_propagation(idx,q_layers, float(widths_nm[idx])))
    if include_exit and end_site == n_sites and n_sites > 0:
        elements.append(interface_s_matrix(q_layers[n_sites - 1], q_lead))
    return tuple(elements)

def _compose_elements(elements: Tuple[SMatrix, ...]) -> SMatrix:
    current = vacuum_passthrough_s_matrix()
    for element in elements:
        current = redheffer_star(current, element)
    return current

@dataclass
class StackCache:
    "Prefix products for a clean stack"
    energy_ev: float
    generation: int
    n_sites: int
    q_lead: complex
    q_layers: np.ndarray
    widths_nm: np.ndarray
    potentials_ev: np.ndarray
    prefix_M: np.ndarray #Shape (N+1, 2, 2)
    suffix_M: np.ndarray

    @classmethod
    def build(
        cls,
        geometry: FibonacciGeometry,
        energy_ev: float,
        *,
        v_lead_ev: float = cfg.V_LEAD_EV,
    ) -> "StackCache":
        widths = np.asarray(geometry.widths_nm, dtype = np.float64)
        pots = np.asarray(geometry.potentials_ev, dtype=np.float64)
        n_sites = int(widths.size)
        q0 = q_number(energy_ev, v_lead_ev).item()
        q_layers = q_number(energy_ev, pots)

        q0=q_number(energy_ev,v_lead_ev).item()
        q_layers = q_number(energy_ev, pots)

        prefix_M = np.empty((n_sites+1,2,2), dtype=np.complex128)
        suffix_M = np.empty((n_sites + 1, 2, 2), dtype=np.complex128)
        prefix_M[0] = np.eye(2, dtype = np.complex128)
        for idx in range(n_sites):
            prefix_M[idx+1] = abeles_matrix(q_layers[idx], float(widths[idx])) @ prefix_M[idx]
        suffix_M[n_sites] = np.eye(2, dtype=np.complex128)
        for idx in range(n_sites -1, -1, -1):
            suffix_M[idx] = suffix_M[idx+1] @ abeles_matrix(q_layers[idx], float(widths[i]))
        
        return cls(energy_ev=float(energy_ev), generation = geometry.generation,
                   n_sites=n_sites, q_lead=q0, q_layers=q_layers, widths_nm=widths,
                   potentials_ev = pots, prefix_M=prefix_M, suffix_M=suffix_M)

    def clean_s_matrix(self) -> SMatrix:
        if self.n_sites == 0:
            return vacuum_passthrough_s_matrix()
        return transfer_to_s_matrix(self.prefix_M[self.n_sites], self.q_lead)

    def clean_transmission(self) -> TransmissionResult:
        return transmission_from_s(self.clean_s_matrix(), self.q_lead)

    def _compose_one_site_defect(cache, geometry, mode, position):
        ok, reason = validate_defect_position(mode, position, geometry)
        if not ok:
            return cache.clean_s_matrix(), False, reason
        defected = apply_defect(mode, geometry, position)
        p = position
        q_p = q_number(cache.energy_ev, float(defected.potentials_ev[p])).item()
        w_p = float(defected.widths_nm[p])
        M = cache.suffix_M[p + 1] @ abeles_matrix(q_p, w_p) @ cache.prefix_M[p]
        return transfer_to_s_matrix(M, cache.q_lead), True, ""
    
def _compose_phason_defect(cache, geometry, position):
    ok, reason = validate_defect_position(cfg.MODE_PHASON, position, geometry)
    if not ok:
        return cache.clean_s_matrix(), False, reason
    p = position
    defected = apply_phason(geometry, p)
    q_p = q_number(cache.energy_ev, float(defected.potentials_ev[p])).item()
    q_p1 = q_number(cache.energy_ev, float(defected.potentials_ev[p + 1])).item()
    w_p = float(defected.widths_nm[p])
    w_p1 = float(defected.widths_nm[p + 1])
    middle = abeles_matrix(q_p1, w_p1) @ abeles_matrix(q_p, w_p)
    M = cache.suffix_M[p + 2] @ middle @ cache.prefix_M[p]
    return transfer_to_s_matrix(M, cache.q_lead), True, ""

def _compose_one_site_defect(
    cache: StackCache,
    geometry: FibonacciGeometry,
    mode:str,
    position:int
) -> Tuple[SMatrix, bool, str]:
    ok, reason = validate_defect_position(mode, position, geometry)
    if not ok:
        return cache.clean_s_matrix(), False, reason
    defected = apply_defect(mode,geometry, position)
    p = position
    q_p = q_number(cache.energy_ev, float(defected.potentials_ev[p])).item()
    w_p = float(defected.widths_nm[p])
    M = cache.suffix_M[p + 1] @ abeles_matrix(q_p, w_p) @ cache.prefix_M[p]
    return transfer_to_s_matrix(M, cache.q_lead), True, ""

def _compose_phason_defect(
    cache: StackCache,
    geometry: FibonacciGeometry,
    position: int,
) -> Tuple[SMatrix, bool, str]:
    ok, reason = validate_defect_position(cfg.MODE_PHASON, position, geometry)
    if not ok: 
        return cache.clean_s_matrix(), False, reason

    p=position
    defected = apply_phason(geometry, p)
    q_p = q_number(cache.energy_ev, float(defected.potentials_ev[p])).item()
    q_p1 = q_number(cache.energy_ev, float(defected.potentials_ev[p + 1])).item()
    w_p = float(defected.widths_nm[p])
    w_p1 = float(defected.widths_nm[p + 1])
    middle = abeles_matrix(q_p1, w_p1) @ abeles_matrix(q_p, w_p)
    M = cache.suffix_M[p + 2] @ middle @ cache.prefix_M[p]
    return transfer_to_s_matrix(M, cache.q_lead), True, ""

def fast_s_matrix(
    mode:str,
    position:int,
    cache:StackCache,
    geometry:FibonacciGeometry,
) -> Tuple[SMatrix, bool, str]:
    if mode == cfg.MODE_PHASON:
        return _compose_phason_defect(cache, geometry, position)
    if mode in cfg.ONE_SITE_MODES:
        return _compose_one_site_defect(cache, geometry, mode, position)
    raise ValueError(f"unknown defect mode {mode!r}")

def fast_transmission(mode: str, position: int, cache: StackCache, geometry: FibonacciGeometry,) ->TransmissionResult:
    s_mat, ok, _ = fast_s_matrix(mode, position, cache, geometry)
    if not ok:
        return TransmissionResult(
            T= float("nan"),
            log_T=float("nan"),
            status="invalid",
            S21=None,
        )
    return transmission_from_s(s_mat, cache.q_lead)

def fast_sweep(mode:str, energies_ev: np.ndarray, *, geometry: Optional[FibonacciGeometry] = None,generation: Optional[int]=None, v_lead_ev:float = cfg.V_LEAD_EV) -> dict[str, np.ndarray]:
    geom = _resolve_geometry(geometry, generation)
    energies = np.asarray(energies_ev, dtype =np.float64)
    n_pos = geom.n_bonds if mode == cfg.MODE_PHASON else geom.n_sites
    positions = np.arange(n_pos, dtype=np.int64)
    valid_mask = np.zeros(n_pos, dtype=bool)
    for p in positions:
        ok, _ = validate_defect_position(mode, int(p), geom)
        valid_mask[p] = ok
    t_arr = np.full((n_pos, energies.size), np.nan, dtype=np.float64)
    log_arr = np.full((n_pos, energies.size), np.nan, dtype=np.float64)
    status_arr = np.empty((n_pos, energies.size), dtype=object)

    t_clean=np.empty(energies.size, dtype = np.float64)
    log_clean = np.empty(energies.size, dtype=np.float64)

    for e_idx, energy in enumerate(energies):
        cache = StackCache.build(geom, float(energy), v_lead_ev=v_lead_ev)
        clean = cache.clean_transmission()
        t_clean[e_idx] = clean.T
        log_clean[e_idx] = clean.log_T
        for p in positions:
            if not valid_mask[p]:
                status_arr[p,e_idx] = "invalid"
                continue
            result = fast_transmission(mode, int(p), cache, geom)
            t_arr[p, e_idx] = result.T
            log_arr[p, e_idx] = result.log_T
            status_arr[p, e_idx] = result.status

    return {
        "mode": np.array(mode), 
        "generation": np.array(geom.generation, dtype=np.int64),
        "positions": positions,
        "energies": energies,
        "T": t_arr,
        "log_T": log_arr,
        "delta_T": t_arr - t_clean[np.newaxis, :],
        "T_clean": t_clean,
        "log_T_clean": log_clean,
        "valid": valid_mask,
        "status": status_arr,
    }

def validate_against_reference(
    geometry: FibonacciGeometry,
    energies_ev: Sequence[float],
    *,
    modes: Optional[Sequence[str]] = None,
    positions_per_mode: Optional[dict[str, Sequence[int]]] = None,
    atol: float = cfg.TOLERANCES.solver_crosscheck_atol,
    rtol: float = cfg.TOLERANCES.solver_crosscheck_rtol,
    v_lead_ev: float = cfg.V_LEAD_EV
) -> List[dict[str, object]]:
    from fibonacci_qc.reference_solver import reference_transmission

    modes = list(modes or cfg.DEFECT_MODES)
    mismatches: list[dict[str, object]] = []

    for energy in energies_ev:
        cache = StackCache.build(geometry, float(energy), v_lead_ev=v_lead_ev)
        for mode in modes: 
            if positions_per_mode and mode in positions_per_mode:
                positions = positions_per_mode[mode]
            elif mode == cfg.MODE_PHASON:
                positions = range(geometry.n_bonds)
            else:
                positions = range(geometry.n_sites)

            for p in positions:
                fast = fast_transmission(mode, int(p), cache, geometry)
                ref, defected = reference_transmission(mode, int(p), float(energy), geometry = geometry, v_lead_ev=v_lead_ev)
                if defected.valid != (fast.status != "invalid"):
                    mismatches.append(
                        {"mode":mode,
                        "position": int(p),
                        "energy": float(energy),
                        "field": "valid",
                        "fast": fast.status,
                        "reference": defected.valid,}
                    )
                    continue
                if not defected.valid:
                    continue
                if not np.isclose(fast.T, ref.T, rtol=rtol, atol=atol):
                    mismatches.append(
                        {"mode":mode,
                        "position": int(p),
                        "energy": float(energy),
                        "field": "T",
                        "fast": fast.T,
                        "reference": ref.T,}
                    )
    return mismatches

def _resolve_geometry(geometry:Optional[FibonacciGeometry], generation: Optional[int]) -> FibonacciGeometry:
    if geometry is not None:
        if generation is not None and geometry.generation != generation:
            raise ValueError(f"geometry.generation={geometry.generation}"
                             f"does not match generation = {generation}"
            )
        return geometry
    if generation is None:
        raise ValueError('either geometry or generation is needed')
    return build_geometry(generation)