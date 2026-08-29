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
    interface_s_matrix,
    propagation_s_matrix,
    q_number,
    redheffer_star,
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
    prefix: np.ndarray #Shape (N+1, 2, 2)

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
        n_sites = widths.size

        q0=q_number(energy_ev,v_lead_ev).item()
        q_layers = q_number(energy_ev, pots)

        prefix = np.empty((n_sites+1,2,2), dtype=np.complex128)
        prefix[0] = vacuum_passthrough_s_matrix()
        current = prefix[0]
        for idx in range(n_sites):
            for element in _stack_elements(q0, q_layers, widths, n_sites, start_site=idx, end_site = idx+1):
                current = redheffer_star(current, element)
            prefix[idx+1] = current

        return cls(energy_ev=float(energy_ev), generation = geometry.generation,
                   n_sites=n_sites, q_lead=q0, q_layers=q_layers, widths_nm=widths,
                   potentials_ev = pots, prefix=prefix,)

    def clean_s_matrix(self) -> SMatrix:
        if self.n_sites == 0:
            return vacuum_passthrough_s_matrix()
        elements = _stack_elements(self.q_lead, self.q_layers, self.widths_nm, self.n_sites)
        return _compose_elements(elements)

    def clean_transmission(self) -> TransmissionResult:
        return transmission_from_s(self.clean_s_matrix(), self.q_lead)

    def suffix(self, start_site:int, *, q_layers: Optional[np.ndarray]=None, widths_nm: Optional[np.ndarray]=None,) -> SMatrix:
        "Elements through start site through exit interface"
        if start_site >= self.n_sites:
            return vacuum_passthrough_s_matrix()
        q_use = self.q_layers if q_layers is None else q_layers
        w_use = self.widths_nm if widths_nm is None else widths_nm
        elements = _stack_elements(self.q_lead, q_use, w_use,self.n_sites,start_site=start_site,)
        return _compose_elements(elements)

def _modified_q_layers(cache: StackCache, geometry:FibonacciGeometry, mode: str, position: int) -> np.ndarray:
    defected = apply_defect(mode, geometry, position)
    q_layers = np.array(cache.q_layers, copy=True)
    q_layers[position] = q_number(cache.energy_ev, float(defected.potentials_ev[position])).item()
    return q_layers

def _one_site_block(cache:StackCache, defected, position: int, q_layers: np.ndarray,) -> Tuple[SMatrix,...]:
    p=position
    widths = np.array(cache.widths_nm, copy=True)
    widths[p] = float(defected.widths_nm[p])
    return(_site_coupling(p,cache.q_lead, q_layers), _site_propagation(p,q_layers, widths[p]))

def _phason_block(cache:StackCache, defected, position:int, q_layers:np.ndarray)->Tuple[SMatrix,...]:
    p = position
    widths = np.array(cache.widths_nm, copy=True)
    widths[p] = float(defected.widths_nm[p])
    widths[p+1] = float(defected.widths_nm[p+1])
    return(_site_coupling(p,cache.q_lead, q_layers), _site_propagation(p,q_layers,widths[p]), interface_s_matrix(q_layers[p],q_layers[p+1]), _site_propagation(p+1, q_layers, widths[p+1]))

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
    q_modified = _modified_q_layers(cache, geometry, mode, p)
    middle = _compose_elements(_one_site_block(cache, defected, p, q_modified))

    s_total = redheffer_star(cache.prefix[p], middle)
    if p+1<cache.n_sites:
        s_total = redheffer_star(
            s_total,
            cache.suffix(p+1, q_layers= q_modified, widths_nm=defected.widths_nm),
        )
    else:
        s_total = redheffer_star(s_total, interface_s_matrix(q_modified[p], cache.q_lead))
    return s_total, True, ""

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
    q_modified= np.array(cache.q_layers, copy=True)
    q_modified[p] = q_number(cache.energy_ev, float(defected.potentials_ev[p])).item()
    q_modified[p+1] = q_number(cache.energy_ev, float(defected.potentials_ev[p+1])).item()

    middle = _compose_elements(_phason_block(cache, defected, p, q_modified))
    s_total = redheffer_star(cache.prefix[p], middle)
    if p+2 < cache.n_sites:
        s_total = redheffer_star(
            s_total, cache.suffix(p+2, q_layers=q_modified, widths_nm=defected.widths_nm),
        )
    else:
        s_total = redheffer_star(s_total, interface_s_matrix(q_modified[p+1], cache.q_lead))

    return s_total, True, ""

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
    log_arr = np.full((np_pos, energies.size), np.nan, dtype=np.float64)
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
                positions = rnage(geometry.n_sites)

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