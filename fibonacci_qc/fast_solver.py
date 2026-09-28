"Prefix / suffix cache for local defects"

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

from fibonacci_qc import config as cfg
from fibonacci_qc.defects import apply_defect, apply_phason, validate_defect_position
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry
from fibonacci_qc.scattering import (
    SMatrix,
    STATUS_TO_CODE,
    TransmissionResult,
    abeles_matrix,
    interface_s_matrix,
    propagation_s_matrix,
    q_number,
    redheffer_star,
    transfer_to_s_matrix,
    transmission_from_s,
    transmission_from_scaled,
    scaled_to_dense,
    passthrough_scaled,
    layer_block_scaled,
    vacuum_passthrough_s_matrix,
    trusted_transmission_mask,
    redheffer_star_scaled,
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
    blocks: tuple
    prefix: tuple
    suffix: tuple

    @classmethod
    def build(
        cls,
        geometry: FibonacciGeometry,
        energy_ev: float,
        *,
        v_lead_ev: float = cfg.V_LEAD_EV,
    ) -> "StackCache":
        widths = np.asarray(geometry.widths_nm, dtype = np.float64, copy=True)
        pots = np.asarray(geometry.potentials_ev, dtype=np.float64, copy=True)
        qs = q_number(energy_ev, pots)
        n_sites = int(widths.size)
        q0 = complex(q_number(energy_ev, v_lead_ev).item())
        q_layers = np.array(q_number(energy_ev, pots), copy=True)

        blocks = tuple(layer_block_scaled(q0, q, float(w)) for q, w in zip(qs, widths))
        prefix = [passthrough_scaled()]
        for block in blocks:
            prefix.append(redheffer_star_scaled(prefix[-1], block))
        suffix = [passthrough_scaled()] * (len(blocks)+1)
        for i in range(len(blocks)-1, -1, -1):
            suffix[i] = redheffer_star_scaled(blocks[i], suffix[i+1])
        for arr in (widths, pots, qs):
            arr.setflags(write=False)
        return cls(float(energy_ev), geometry.generation, len(blocks), q0, qs,
                   widths, pots, blocks, tuple(prefix), tuple(suffix))

    def clean_scaled(self):
        return self.prefix[-1]

    def clean_s_matrix(self) -> SMatrix:
        return scaled_to_dense(self.clean_scaled())

    def clean_transmission(self) -> TransmissionResult:
        return transmission_from_scaled(self.clean_scaled(), self.q_lead)

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

def _local_parameters(cache, geometry, mode, p):
    other = 'B' if str(geometry.letters[p]) == "A" else "A"
    width, pot = float(cache.widths_nm[p]), float(cache.potentials_ev[p])
    if mode in (cfg.MODE_SUBSTITUTION, cfg.MODE_WIDTH_ONLY):
        width = cfg.LAYER_WIDTH_NM[other]
    if mode in (cfg.MODE_SUBSTITUTION, cfg.MODE_POTENTIAL_ONLY):
        pot = cfg.LAYER_POTENTIAL_EV[other]
    return width, pot

def fast_scaled_s_matrix(mode, position, cache, geometry):
    if mode not in cfg.DEFECT_MODES:
        raise ValueError(f"unknown defect mode {mode!r}")
    if cache.generation != geometry.generation or cache.n_sites != geometry.n_sites:
        raise ValueError("cache and geometry do not match")
    p = int(position)
    ok, reason = validate_defect_position(mode, p, geometry)
    if not ok:
        return cache.clean_scaled(), False, reason
    if mode == cfg.MODE_PHASON:
        middle = redheffer_star_scaled(cache.blocks[p+1], cache.blocks[p])
        end = p+2
    else:
        width, pot = _local_parameters(cache, geometry, mode, p)
        middle = layer_block_scaled(cache.q_lead, q_number(cache.energy_ev, pot).item(), width)
        end = p+1
    state = redheffer_star_scaled(redheffer_star_scaled(cache.prefix[p], middle), cache.suffix[end])
    return state, True, ""
    
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
    state, valid, reason = fast_scaled_s_matrix(mode, position, cache, geometry)
    return scaled_to_dense(state), valid, reason

def fast_transmission(mode: str, position: int, cache: StackCache, geometry: FibonacciGeometry,) ->TransmissionResult:
    try:
        state, valid, _ = fast_scaled_s_matrix(mode, position, cache, geometry)
        if valid:
            return transmission_from_scaled(state, cache.q_lead)
    except (FloatingPointError, OverflowError):
        pass
    return TransmissionResult(float('nan'), float('nan'), 'invalid')

from fibonacci_qc.parity_analysis import numerical_cell_mask

def _allocate(shape, dtype, fill, spill_dir, name):
    if spill_dir is None:
        arr = np.empty(shape, dtype=dtype)
    else:
        folder = Path(spill_dir)
        folder.mkdir(parents=True, exist_ok=True)
        arr = np.memmap(folder / (name+'.dat'), mode='w+', dtype=dtype, shape=shape)
    arr.fill(fill)
    return arr

def fast_sweep(mode:str, energies_ev: np.ndarray, *, geometry: Optional[FibonacciGeometry] = None, generation: Optional[int]=None, v_lead_ev:float = cfg.V_LEAD_EV, energy_chunk: int = 64, spill_dir: Optional[object]=None) -> dict[str, np.ndarray]:
    geom = _resolve_geometry(geometry, generation)
    if mode not in cfg.DEFECT_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    
    energies = np.asarray(energies_ev, dtype =np.float64)
    if energies.ndim != 1 or not energies.size or not np.all(np.isfinite(energies)):
        raise ValueError('energies must be a nonempty finite 1D array')
    if energies.size > 1 and not np.all(np.diff(energies) > 0):
        raise ValueError("energies must be strictly increasing")
    if np.any(energies <= v_lead_ev):
        raise ValueError("propagating exterior leads required")
    
    n_pos = geom.n_bonds if mode == cfg.MODE_PHASON else geom.n_sites
    positions = np.arange(n_pos, dtype=np.int64)

    valid_mask = np.zeros(n_pos, dtype=bool)
    for p in positions:
        ok, _ = validate_defect_position(mode, int(p), geom)
        valid_mask[p] = ok
    valid = np.array([validate_defect_position(mode, int(p), geom)[0] for p in positions])
    shape = (n_pos, energies.size)
    step = max(int(energy_chunk), 1)
    invalid = np.uint8(STATUS_TO_CODE["invalid"])
    t_arr = _allocate(shape, np.float64, np.nan, spill_dir, "T")
    log_arr = _allocate(shape, np.float64, np.nan, spill_dir, "log_T")
    status_arr = _allocate(shape, np.uint8, invalid, spill_dir, "status")
    t_clean = np.full(energies.size, np.nan)
    log_clean = np.full(energies.size, np.nan)
    clean_status = np.full(energies.size, STATUS_TO_CODE["invalid"], dtype=np.uint8)
    step = max(1, int(energy_chunk))
    for start in range(0, energies.size, step):
        for j in range(start, min(energies.size, start+step)):
            try:
                cache = StackCache.build(geom, float(energies[j]), v_lead_ev=v_lead_ev)
            except (FloatingPointError, OverflowError):
                continue
            result = cache.clean_transmission()
            t_clean[j], log_clean[j], clean_status[j] = result.T, result.log_T, STATUS_TO_CODE[result.status]
            for p in positions[valid]:
                result = fast_transmission(mode, int(p), cache, geom)
                t_arr[p,j], log_arr[p,j], status_arr[p,j] = result.T, result.log_T, STATUS_TO_CODE[result.status]
        for arr in (t_arr, log_arr, status_arr):
            if isinstance(arr, np.memmap): arr.flush()
    delta = _allocate(shape, np.float64, np.nan, spill_dir, 'delta_T')
    clean_ok = trusted_transmission_mask(clean_status, t_clean, log_clean)
    for start in range(0, energies.size, step):
        sl = slice(start, start+step)
        mask = trusted_transmission_mask(status_arr[:,sl], t_arr[:,sl], log_arr[:,sl]) & clean_ok[sl]
        delta[:,sl] = np.where(mask, t_arr[:,sl]-t_clean[sl], np.nan)
    if isinstance(delta, np.memmap): delta.flush()
    cell_ok = numerical_cell_mask(status_arr)
    return dict(mode=np.array(mode), 
                generation=np.array(geom.generation), 
                positions=positions,
                energies=energies, 
                T=t_arr, 
                log_T=log_arr, 
                delta_T=delta, 
                T_clean=t_clean,
                log_T_clean=log_clean, 
                clean_status=clean_status, 
                valid=valid, 
                status=status_arr,
                cell_ok=trusted_transmission_mask(status_arr, t_arr, log_arr))

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
    errors = []
    for energy in energies_ev:
        cache = StackCache.build(geometry, float(energy), v_lead_ev=v_lead_ev)
        for mode in (cfg.DEFECT_MODES if modes is None else modes): 
            positions = (positions_per_mode or {}).get(mode)
            if positions is None:
                positions = range(geometry.n_bonds if mode == cfg.MODE_PHASON else geometry.n_sites)

            for p in positions:
                fast = fast_transmission(mode, int(p), cache, geometry)
                ref, defected = reference_transmission(mode, int(p), float(energy), geometry = geometry, v_lead_ev=v_lead_ev)
                if not defected.valid:
                    if fast.status != "invalid": errors.append(dict(mode=mode, position=int(p), field="valid"))
                    continue
                if not (fast.is_valid and ref.is_valid):
                    errors.append(dict(mode=mode, position=int(p), energy=float(energy), field="status", fast=fast.status, reference=ref.status))
                    continue
                for field, aa, bb in [("T", fast.T, ref.T), ("log_T", fast.log_T, ref.log_T)]:
                    tol = atol if field == "T" else 1e-7
                    rel = rtol if field == "T" else 1e-12
                    if not np.isclose(aa, bb, atol=tol, rtol=rel):
                        errors.append(dict(mode=mode, position=int(p), energy=float(energy), field=field, fast=aa, reference=bb))
    return errors

def _resolve_geometry(geometry:Optional[FibonacciGeometry], generation: Optional[int]) -> FibonacciGeometry:
    if geometry is not None:
        if generation is not None and geometry.generation != generation: raise ValueError("geometry and generation don't match")
        return geometry
    if generation is None:
        raise ValueError("either geometry or generation required")
    return build_geometry(generation)