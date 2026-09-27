"""
Scattering matrix and clean transmission. 

Layer parameters and stacks from config and geometry respectively

Layers composed with Redheffer star product

For energy E and potential V def complex wave number Q so Q**2=(E-V)/HBAR2_OVER_2M in nm

Propagating regions have a real and positive Q; below a layer threshold have Q=i*kappa with kappa>0
This is so slab thickness is not just a phase

Scattering matrix S connects a_left and a_right inputs to outgoing b_left, b_right.

b_left = S11 a_left + S12 a_right
b_right = S21 a_left + S22 a_right

When entering the quasicrystal from the left vacuum (i.e. a_right = 0), the transmission is T=|S21|**2.

Defect modes width_only and potential_only still change evanescenet attenuation when E< V_layer and propagation phase when E> V_layer respectively
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Union
import numpy as np
import math

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry

ArrayLike = Union[float, np.ndarray]
SMatrix = np.ndarray
TransferStatus = Literal["ok", "underflow", "overflow", "invalid", "non_unitary"]
STATUS_TO_CODE = {"ok": 0, "invalid": 1, "underflow": 2, "overflow": 3, "non_unitary": 4}
CODE_TO_STATUS = {v: k for k, v in STATUS_TO_CODE.items()}
STAR_DENOM_RTOL = 32.0 * np.finfo(np.float64).eps
UNITARITY_ATOL = 1e-8
SOLVER_REVISION = "scaled-redheffer-v1"

@dataclass(frozen=True)
class TransmissionResult:
    "Transmission value"

    T:float
    log_T: float
    status: TransferStatus
    S21: complex | None = None

    @property
    def is_valid(self) -> bool:
        return self.status in {"ok", "underflow"}

@dataclass(frozen=True)
class ScaledS:
    s11: complex
    s22: complex
    log_abs_s21: float
    phase_s21: complex
    log_abs_s12: float
    phase_s12: complex


def encode_status(status) -> np.ndarray:
    "Canonical uint8 codes"
    arr = np.asarray(status)
    if arr.dtype.kind in "iu":
        return np.where((arr >= 0) & (arr <= 4), arr, 1).astype(np.uint8)
    out = np.full(arr.shape, STATUS_TO_CODE["invalid"], dtype=np.uint8)
    for idx, value in np.ndenumerate(arr):
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            out[idx] = int(value) if int(value) in CODE_TO_STATUS else 1
        elif isinstance(value, (str, np.str_)):
            out[idx] = STATUS_TO_CODE.get(str(value), 1)
    return out

def decode_status(status) -> np.ndarray:
    return np.asarray(tuple(CODE_TO_STATUS[i] for i in range(5)), dtype="<U12")[encode_status(status)]

def trusted_transmission_mask(status, t=None, log_t=None) -> np.ndarray:
    "Certified underflow is ZERO"
    code = encode_status(status)
    ok = (code == 0) | (code == 2)
    if t is not None:
        values = np.asarray(t, dtype=np.float64)
        ok = ok & np.isfinite(values) & (values >= 0) & (values <= 1 + cfg.TOLERANCES.t_bounds_atol)
        ok &= (code != 2) | (values == 0)
    if log_t is not None:
        ok &= np.isfinite(np.asarray(log_t, dtype=np.float64))
    return ok

def trusted_log_mask(status, log_t=None) -> np.ndarray:
    code = encode_status(status)
    ok = (code == 0) | (code == 2)
    if log_t is not None:
        values = np.asarray(log_t, dtype=np.float64)
        ok &= np.isfinite(values) & (values <= math.log1p(cfg.TOLERANCES.t_bounds_atol))
    return ok

def status_is_ok(status: np.ndarray) -> np.ndarray:
    return trusted_transmission_mask(status)

def q_number(energy_ev: ArrayLike, potential_ev: ArrayLike) -> np.ndarray:
    """
    Complex wave num Q(E,V)

    Propagating: Q = sqrt((E-V)*pref)
    Evanescent: Q = i*sqrt((V-E)*pref)
    Width enter transport as Q*d (not phase only below a threshold)
    Potential enter through Q(E,V) so changes phase above threshold
    """
    e, v = np.broadcast_arrays(np.asarray(energy_ev, dtype=float), np.asarray(potential_ev, dtype=float))
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(v))):
        raise ValueError("energies and potentials must be finite")
    return np.sqrt(((e - v) * cfg.KAPPA2_PREFACTOR_PER_EV_NM2).astype(np.complex128))

def _finite_complex(z):
    return math.isfinite(complex(z).real) and math.isfinite(complex(z).imag)

def _unit_phase(z):
    z = complex(z)
    mag = abs(z)
    if not math.isfinite(mag) or mag == 0:
        raise FloatingPointError("nonfinite or zero phase")
    return z / mag

def _log_abs(z):
    mag = abs(complex(z))
    if not math.isfinite(mag):
        raise FloatingPointError("nonfinite amplitude")
    return math.log(mag) if mag else -math.inf

def _materialize_amp(log_abs, phase):
    if log_abs == -math.inf:
        return 0j
    if not math.isfinite(log_abs) or not _finite_complex(phase):
        raise FloatingPointError("invalid logarithmic amplitude")
    return complex(phase) * math.exp(log_abs)

def _scaled_from_dense(s):
    a = np.asarray(s, dtype=np.complex128)
    if a.shape != (2, 2) or not np.all(np.isfinite(a)):
        raise FloatingPointError("S must be a finite 2x2 matrix")
    return ScaledS(complex(a[0, 0]), complex(a[1, 1]), _log_abs(a[1, 0]), _unit_phase(a[1, 0]) if a[1, 0] else 1+0j, _log_abs(a[0, 1]), _unit_phase(a[0, 1]) if a[0, 1] else 1+0j)

def passthrough_scaled():
    return ScaledS(0j, 0j, 0., 1+0j, 0., 1+0j)

def scaled_to_dense(s):
    return np.array([[s.s11, _materialize_amp(s.log_abs_s12, s.phase_s12)], [_materialize_amp(s.log_abs_s21, s.phase_s21), s.s22]], dtype=complex)

def _as_complex_scalar(value: complex | float) -> complex:
    return complex(value)

_Q_ABS_TOL = 1e-14
def _q_near_zero(q: complex, *, tol: float = _Q_ABS_TOL) -> bool:
    return abs(q) < tol

def _sin_qd_over_q(q: complex, width_nm: float) -> complex:
    "Stable sin(Q d)/Q tends to d as Q goes to 0"
    beta = q*width_nm
    if abs(q) < 1e-14:
        return width_nm * (1.0+0.0j)
    return np.sin(beta) / q

def redheffer_star_scaled(left: ScaledS, right: ScaledS) -> ScaledS:
    product = right.s11 * left.s22
    den = 1.0 - product
    if not _finite_complex(den) or abs(den) <= STAR_DENOM_RTOL * max(1., abs(product)):
        raise FloatingPointError("unresolved cancellation in Redheffer denominator")
    ld, pd = math.log(abs(den)), _unit_phase(den)
    def transmission(la, pa, lb, pb):
        if la == -math.inf or lb == -math.inf:
            return -math.inf, 1+0j
        log = math.fsum((la, lb, -ld))
        if not math.isfinite(log):
            raise FloatingPointError("nonfinite retained transmission logarithm")
        return log, _unit_phase(pa * pb / pd)
    def reflection(base, la, pa, bridge, lb, pb):
        if bridge == 0 or la == -math.inf or lb == -math.inf:
            return base
        # Combine logs BEFORE exponentiating
        lc = math.fsum((la, lb, math.log(abs(bridge)), -ld))
        correction = _materialize_amp(lc, _unit_phase(pa * pb * _unit_phase(bridge) / pd))
        result = base + correction
        if not _finite_complex(result):
            raise FloatingPointError("nonfinite reflection")
        return result
    l21, p21 = transmission(right.log_abs_s21, right.phase_s21, left.log_abs_s21, left.phase_s21)
    l12, p12 = transmission(left.log_abs_s12, left.phase_s12, right.log_abs_s12, right.phase_s12)
    r1 = reflection(left.s11, left.log_abs_s12, left.phase_s12, right.s11, left.log_abs_s21, left.phase_s21)
    r2 = reflection(right.s22, right.log_abs_s21, right.phase_s21, left.s22, right.log_abs_s12, right.phase_s12)
    return ScaledS(r1, r2, l21, p21, l12, p12)

def redheffer_star(s_left: SMatrix, s_right: SMatrix):
    return scaled_to_dense(redheffer_star_scaled(_scaled_from_dense(s_left), _scaled_from_dense(s_right)))

def interface_s_matrix(q_left, q_right):
    a, b = complex(q_left), complex(q_right)
    if not (_finite_complex(a) and _finite_complex(b)) or abs(a+b) == 0:
        raise FloatingPointError("zero/nonfinite interface port; use finite threshold slab")
    r = (a-b)/(a+b)
    return np.array([[r, 2*b/(a+b)], [2*a/(a+b), -r]], dtype=complex)

def propagation_scaled(q, width_nm):
    q, d = complex(q), float(width_nm)
    if not _finite_complex(q) or not math.isfinite(d) or d < 0 or q.imag < 0:
        raise ValueError("finite passive q and nonnegative width required")
    beta = q*d
    return ScaledS(0j, 0j, -beta.imag, complex(np.exp(1j*beta.real)), -beta.imag, complex(np.exp(1j*beta.real)))

def layer_s_matrix(
        q_left: complex,
        q_layer: complex,
        q_right: complex,
        width_nm: float,
) -> SMatrix:
    "Finite slab between leads q_left and q_right, use Fabry-Perot for single layer of wave number and width q_layer and width_nm"

    a, b, q = complex(q_left), complex(q_right), complex(q_layer)
    if abs(a-b) <= 1e-14 and a.real > 0 and abs(a.imag) <= 1e-14:
        return scaled_to_dense(layer_block_scaled(a, q, width_nm))
    # Bridge unequal ports
    k = max(1., abs(a), abs(b))
    left = _scaled_from_dense(interface_s_matrix(a, k))
    block = layer_block_scaled(k, q, width_nm)
    right = _scaled_from_dense(interface_s_matrix(k, b))
    return scaled_to_dense(redheffer_star_scaled(redheffer_star_scaled(left, block), right))

def interface_s_matrix(q_left: complex, q_right: complex) -> SMatrix:
    "Zero thickness interface from q_left to q_right. Implements constant mass continuity matching (psi, psi'), potential jumps change Q so also reflection / transmission"
    q_left = _as_complex_scalar(q_left)
    q_right = _as_complex_scalar(q_right)
    if _q_near_zero(q_left) and _q_near_zero(q_right):
        return vacuum_passthrough_s_matrix()
    denom = q_left + q_right
    r = (q_left - q_right) / denom
    t_lr = 2.0 * q_left / denom
    t_rl = 2.0 * q_right / denom
    return np.array([[r, t_rl], [t_lr, -r]], dtype=np.complex128)
    

def propagation_s_matrix(q, width_nm):
    return scaled_to_dense(propagation_scaled(q, width_nm))

def transfer_to_s_matrix(transfer:np.ndarray, q_lead: complex) -> SMatrix:
    "Stack transfer matrix to s matrix"
    k = _lead(q_lead)
    m = np.asarray(transfer, dtype=complex)
    if m.shape != (2, 2) or not np.all(np.isfinite(m)) or np.linalg.cond(m) > 1e10:
        raise FloatingPointError("ill-conditioned characteristic matrix")
    b = np.array([[1., 1.], [k, -k]], dtype=complex)
    w = np.linalg.solve(b, m @ b)
    if abs(w[1, 1]) <= STAR_DENOM_RTOL:
        raise FloatingPointError("singular characteristic conversion")
    return np.array([[-w[1, 0], 1.], [1., w[0, 1]]], dtype=complex)/w[1, 1]


def _stack_s_matrix_redheffer(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
    v_lead_ev: float,
) -> SMatrix:
    "Build stack S with per layer Abeles product + one transfer -> S."
    return scaled_to_dense(stack_scaled_s_matrix(widths_nm, potentials_ev, energy_ev, v_lead_ev))

def _stack_inputs(widths_nm, potentials_ev, energy_ev, v_lead_ev):
    w, v = np.asarray(widths_nm, dtype=float), np.asarray(potentials_ev, dtype=float)
    if w.ndim != 1 or w.shape != v.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("widths/potentials must be matching finite 1D arrays; widths >= 0")
    k = _lead(q_number(energy_ev, v_lead_ev).item())
    return w, q_number(energy_ev, v), k

def vacuum_passthrough_s_matrix():
    return scaled_to_dense(passthrough_scaled())

def _sin_qd_over_q(q, width_nm):
    z = complex(q)*width_nm
    if abs(z) < 1e-4:
        return width_nm * (1 - z*z/6 + z**4/120 - z**6/5040)
    return np.sin(z)/q

def _lead(q):
    k = complex(q)
    if not _finite_complex(k) or k.real <= 0 or abs(k.imag) > 1e-14:
        raise ValueError("positive propagating exterior lead required")
    return float(k.real)

def slab_s_matrix(
    energy_ev: float,
    potential_ev: float,
    width_nm: float,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> SMatrix:
    "S matrix for single layer embedded in vacuum leads"
    return scaled_to_dense(layer_block_scaled(q_number(energy_ev, v_lead_ev).item(), q_number(energy_ev, potential_ev).item(), width_nm))

def stack_s_matrix(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> SMatrix:
    return _stack_s_matrix_redheffer(widths_nm, potentials_ev, energy_ev, v_lead_ev)

def abeles_matrix(q: complex, width_nm: float) -> np.ndarray:
    "2x2 Abeles characteristic matrix for one uniform layer"
    z = complex(q)*width_nm
    return np.array([[np.cos(z), 1j*_sin_qd_over_q(q, width_nm)], [1j*q*np.sin(z), np.cos(z)]], dtype=complex)

def layer_block_scaled(q_lead, q_layer, width_nm):
    "Equal, nonzero reference ports"
    k, q, d = _lead(q_lead), complex(q_layer), float(width_nm)
    if not _finite_complex(q) or q.imag < 0 or not math.isfinite(d) or d < 0:
        raise ValueError("finite passive layer and nonnegative width required")
    if d == 0:
        return passthrough_scaled()
    beta = q*d
    if abs(beta) < 1e-4:
        sinc = _sin_qd_over_q(q, d)
        den = np.cos(beta) - 1j*(k*k+q*q)*sinc/(2*k)
        r = -1j*(k*k-q*q)*sinc/(2*k)/den
        tlog, phase = -math.log(abs(den)), _unit_phase(1/den)
    else:
        r0 = (k-q)/(k+q)
        logphase = -beta.imag
        z = complex(np.exp(2j*beta.real)) * math.exp(2*logphase)
        den = 1-r0*r0*z
        if abs(den) <= STAR_DENOM_RTOL * max(1., abs(r0*r0*z)):
            raise FloatingPointError("unresolved single-slab denominator")
        coeff = 4*k*q/(k+q)**2
        r = r0*(1-z)/den
        tlog = math.fsum((math.log(abs(coeff)), logphase, -math.log(abs(den))))
        phase = _unit_phase(coeff * complex(np.exp(1j*beta.real)) / den)
    return ScaledS(complex(r), complex(r), tlog, phase, tlog, phase)

def transfer_matrix_stack(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
) -> np.ndarray:
    "Product of Abeles matrices"
    w, qs, _ = _stack_inputs(widths_nm, potentials_ev, energy_ev, cfg.V_LEAD_EV)
    m = np.eye(2, dtype=complex)
    for q, d in zip(qs, w):
        m = abeles_matrix(q, d) @ m
        if not np.all(np.isfinite(m)) or np.linalg.cond(m) > 1e10:
            raise FloatingPointError("transfer product is too ill-conditioned for this diagnostic")
    return m

def stack_scaled_s_matrix(widths_nm, potentials_ev, energy_ev, v_lead_ev=cfg.V_LEAD_EV):
    "Full reference recomposition"
    widths, qs, k = _stack_inputs(widths_nm, potentials_ev, energy_ev, v_lead_ev)
    current, prev = passthrough_scaled(), complex(k)
    for q, width in zip(qs, widths):
        q, width = complex(q), float(width)
        if abs(q)*max(width, 1.) < 1e-4:
            # Do not expose q=0 as a scattering port
            current = redheffer_star_scaled(current, _scaled_from_dense(interface_s_matrix(prev, k)))
            current = redheffer_star_scaled(current, layer_block_scaled(k, q, width))
            prev = complex(k)
        else:
            current = redheffer_star_scaled(current, _scaled_from_dense(interface_s_matrix(prev, q)))
            current = redheffer_star_scaled(current, propagation_scaled(q, width))
            prev = q
    return redheffer_star_scaled(current, _scaled_from_dense(interface_s_matrix(prev, k)))

def transmission_from_transfer(transfer: np.ndarray, q_lead: complex) -> float:
    return transmission_from_s(transfer_to_s_matrix(transfer, q_lead), q_lead).T

def safe_log_t(
    t: float,
    *,
    floor: float = cfg.TOLERANCES.log_t_floor,
) -> Tuple[float, TransferStatus]:
    "Return log(T)"
    if not np.isfinite(t) or t <= 0:
        return float("nan"), "invalid"
    return math.log_t, "ok"

def exterior_unitarity_residuals(s_matrix: SMatrix, q_lead: complex) -> dict[str, float]:
    "For propagating exterior leads (Im q roughly 0)"
    _lead(q_lead)
    s = np.asarray(s_matrix, dtype=complex)
    u = s.conj().T @ s
    return {
        "applicable": 1., 
        "R_plus_T_left": float(abs(s[0,0])**2+abs(s[1,0])**2),
        "R_plus_T_right": float(abs(s[1,1])**2+abs(s[0,1])**2),
        "reciprocity": float(abs(s[0,1]-s[1,0])),
        "unitarity_frob": float(np.linalg.norm(u-np.eye(2)))
    }

def _classify_transmission(t: float, s_matrix: SMatrix = None, q_lead = None) -> TransferStatus:
    if not math.isfinite(t) or (s_matrix is not None and not np.all(np.isfinite(s_matrix))):
        return "invalid"
    if t < 0 or t > 1+cfg.TOLERANCES.t_bounds_atol:
        return "non_unitary"
    if s_matrix is not None:
        r = exterior_unitarity_residuals(s_matrix, 1. if q_lead is None else q_lead)
        if max(abs(r["R_plus_T_left"]-1), abs(r["R_plus_T_right"]-1),
               r["unitarity_frob"], r["reciprocity"]) > UNITARITY_ATOL:
            return "non_unitary"
    return "ok"

def transmission_from_scaled(scaled, q_lead=None, dense=None):
    try:
        if not (math.isfinite(scaled.log_abs_s21) and math.isfinite(scaled.log_abs_s12)):
            return TransmissionResult(math.nan, math.nan, "invalid")
        log_t = 2*scaled.log_abs_s21
        if log_t > math.log1p(cfg.TOLERANCES.t_bounds_atol):
            return TransmissionResult(math.nan, math.nan, "non_unitary")
        t = math.exp(log_t)
        matrix = scaled_to_dense(scaled) if dense is None else np.asarray(dense, dtype=complex)
        status = _classify_transmission(t, matrix, q_lead)
        if status != "ok":
            return TransmissionResult(math.nan, math.nan, status)
        if abs(scaled.log_abs_s21-scaled.log_abs_s12) > 1e-8 + 1e-12*abs(log_t):
            return TransmissionResult(math.nan, math.nan, "non_unitary")
        return TransmissionResult(t, log_t, "underflow" if t == 0 else 'ok', complex(matrix[1,0]))
    except (FloatingPointError, OverflowError, ValueError):
        return TransmissionResult(math.nan, math.nan, "invalid")

def transmission_from_s(
        s_matrix: SMatrix,
        q_lead: complex | None = None,
) -> TransmissionResult:
    "Extract T=|S21|^2"
    try:
        return transmission_from_scaled(_scaled_from_dense(s_matrix), q_lead, dense=s_matrix)
    except (FloatingPointError, ValueError):
        return TransmissionResult(math.nan, math.nan, "invalid")

def stack_transmission(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> TransmissionResult:
    "Transmission through layer stack at one energy"
    _stack_inputs(widths_nm, potentials_ev, energy_ev, v_lead_ev)
    try:
        state = stack_scaled_s_matrix(widths_nm, potentials_ev, energy_ev, v_lead_ev)
        return transmission_from_scaled(state, q_number(energy_ev, v_lead_ev).item())
    except (FloatingPointError, OverflowError):
        return TransmissionResult(math.nan, math.nan, 'invalid')

def clean_transmission(
    generation: int,
    energy_ev: float,
    *,
    geometry: FibonacciGeometry | None = None,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> TransmissionResult:
    "T_clean(n, E) for Fib generation n"
    geom = build_geometry(generation) if geometry is not None else geometry
    if geom.generation != generation:
        raise ValueError(f"geometry generation {geom.generation} doesn't match generation {generation}")
    return stack_transmission(geom.widths_nm, geom.potentials_ev, energy_ev, v_lead_ev)

def clean_transmission_grid(
    generation: int,
    energy_ev: np.ndarray,
    *,
    geometry: FibonacciGeometry | None = None,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize T_clean(n, E) over energy grid
    Returns T, log(T), status arrays aligned with energies_ev
    """
    energies = np.asarray(energy_ev, dtype=float)
    geom = build_geometry(generation) if geometry is None else geometry
    ts, logs, statuses = np.empty(energies.shape), np.empty(energies.shape), np.empty(energies.shape, dtype=np.uint8)
    for idx, e in np.ndenumerate(energies):
        r = clean_transmission(generation, float(e), geometry=geom, v_lead_ev=v_lead_ev)
        ts[idx], logs[idx], statuses[idx] = r.T, r.log_T, STATUS_TO_CODE[r.status]
    return ts, logs, statuses

def rectangular_barrier_transmission(
    energy_ev: float,
    barrier_height_ev: float,
    width_nm: float,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> float:
    "Analytical transmission through a single rectangular barrier"
    e, v, d = float(energy_ev), float(barrier_height_ev), float(width_nm)
    if v_lead_ev != 0 or e <= 0 or d < 0:
        raise ValueError('positive E, vacuum leads, and nonnegative width required')
    h = cfg.HBAR2_OVER_2M_EV_NM2
    if abs(e-v) < 1e-14:
        return 1/(1+v*v*d*d/(4*e*h))
    if e > v:
        return 1/(1+v*v*np.sin(np.sqrt((e-v)/h)*d)**2/(4*e*(e-v)))
    x = math.sqrt((v-e)/h)*d
    if x > 350:
        return math.exp(-2*x)*16*e*(v-e)/(v*v)
    return 1/(1+v*v*math.sinh(x)**2/(4*e*(v-e)))