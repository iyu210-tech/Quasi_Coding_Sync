"""Scattering matrix and clean transmission. 

Layer parameters and stacks from config and geometry respectively

Layers composed with Redheffer star product

For energy E and potential V def complex wave number Q so Q**2=(E-V)/HBAR2_OVER_2M in nm\\ :sup:'-1'

Propagating regions have a real and positive Q, while evanescent regions have Q=i*kappa with kappa>0


Scattering matrix S connects a_left and a_right inputs to outgoing b_left, b_right.

b_left = S11 a_left + S12 a_right
b_right = S21 a_left + S22 a_right

When entering the quasicrystal from the left vacuum (i.e. a_right = 0), the transmission probability is T=|S21|**2.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Union
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import FibonacciGeometry, build_geometry

ArrayLike = Union[float, np.ndarray]
SMatrix = np.ndarray
TransferStatus = Literal["ok", "underflow", "overflow", "invalid", "non_unitary"]

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

def q_number(energy_ev: ArrayLike, potential_ev: ArrayLike) -> np.ndarray:
    "Return Q with stable E approx. V"
    e = np.asarray(energy_ev, dtype=npfloat64)
    v = np.asarray(potential_ev, dtype = np.float64)
    arg = (e-v) * cfg.KAPPA2_PREFACTOR_PER_EV_NM2
    out = np.zeros(np.broadcast_shapes(e.shape, v.shape), dtype = np.complex128)
    propagating = arg >= 0.0
    out[propagating] = np.sqrt(arg[propagating].astype(np.complex128))
    evanescent = ~propagating
    out[evanescent] = 1j * np.sqrt((-arg[evanescent]).astype(np.float64))
    return out

def _as_complex_scalar(value: complex | float) -> complex:
    return complex(value)

def _sin_qd_over_q(q: complex, width_nm: float) -> complex:
    "Stable sin(Q d)/Q tends to d as Q goes to 0"
    beta = q*width_nm
    if abs(q) < 1e-14:
        return width_nm * (1.0+0.0j)
    return np.sin(beta) / q

def redheffer_star(s_left: SMatrix, s_right: SMatrix) -> SMatrix:
    "Redheffer star product"
    s_left = np.asarray(s_left, dtype = np.complex128)
    s_right = np.asarray(s_right, dtype=np.complex128)
    if s_left.shape != (2,2) or s_right.shape != (2,2):
        raise ValueError("S matrices must be 2x2")

    s_a11, s_a12 = s_left[0,0], s_left[0,1]
    s_a21, s_a22 = s_left[1,0], s_left[1,1]
    s_b11, s_b12 = s_right[0,0], s_right[0,1]
    s_b21, s_b22 = s_right[1,0], s_right[1,1]

    denom = 1.0 - s_b11 * s_a22
    if abs(denom) < 1e-300:
        raise FloatingPointError("Redheffer denominator numerically singular")

    inv = 1.0 / denom
    return np.array(
        [
            [s_a11 + s_a12 * s_b11 * s_a21 * inv, s_a12 * s_b12 * inv],
            [s_b21 * s_a21 * inv, s_b22 + s_b21 * s_a22 * s_b12 * inv],
        ],
        dtype=np.complex128,
    )

def layer_s_matrix(
        q_left: complex,
        q_layer: complex,
        q_right: complex,
        width_nm: float,
) -> SMatrix:
    "Finite slab between leads q_left and q_right, use Fabry-Perot for single layer of wave number and width q_layer and width_nm"

    if width_nm < 0.0:
        raise ValueError("width_nm cannot be negative.")

    q_left = _as_complex_scalar(q_left)
    q_layer = _as_complex_scalar(q_layer)
    q_right = _as_complex_scalar(q_right)

    if width_nm == 0.0:
        return interface_s_matrix(q_left, q_right)

    r01 = (q_left - q_layer) / (q_left + q_layer)
    r12 = (q_layer - q_right) / (q_layer + q_right)
    t01 = 2.0 * q_left / (q_left + q_layer)
    t12 = 2.0 * q_layer / (q_layer + q_right)

    phase = np.exp(1j*q_layer*width_nm)
    phase2 = np.exp(2j*q_layer*width_nm)
    denom = 1.0+r01*r12*phase2

    s11 = (r01+r12 * phase2) / denom
    s21 = (t01*t12*phase)/denom
    s22 = (r12 + r01 * phase2) / denom
    t21 = 2.0 * q_right / (q_layer + q_right)
    t10 = 2.0* q_layer / (q_layer + q_left)
    s12 = (t21*t10*phase) / denom

    return np.array([[s11,s12],[s21,s22]], dtype = np.complex128)

def interface_s_matrix(q_left: complex, q_right: complex) -> SMatrix:
    "Zero thickness interface from q_left to q_right"
    q_left = _as_complex_scalar(q_left)
    q_right = _as_complex_scalar(q_right)
    r = (q_left - q_right) / (q_left + q_right)
    t_lr = 2.0 * q_left / (q_left + q_right)
    t_rl = 2.0 * q_right / (q_left + q_right)
    return np.array([[r, t_rl],[t_lr, -r]], dtype=np.complex128)
    

def propagation_s_matrix(q: complex, width_nm: float) ->SMatrix:
    if width_nm < 0.0:
        raise ValueError("width_nm cannot be negative")
    q = _as_complex_scalar(q)
    phase = np.exp(1j * q * width_nm)
    return np.array([[0.0, phase],[phase, 0.0]], dtype = np.complex128)

def transfer_to_s_matrix(transfer:np.ndarray, q_lead: complex) -> SMatrix:
    "Stack transfer matrix to s matrix"
    q_lead = _as_complex_scalar(q_lead)
    m11, m12 = transfer[0,0], transfer[0,1]
    m21, m22 = transfer[1,0], transfer[1,1]
    denom = m11 * q_lead + m12 * q_lead**2 + m21 + m22 * q_lead
    if abs(denom) < 1e-300:
        raise FloatingPointError("Transfer to S denominator numerically singular")
    t = 2.0 * q_lead / denom
    r = (m11 * q_lead + m12 * q_lead**2 - m21 - m22 * q_lead) / denom
    return np.array([[r,t],[t,r]], dtype=np.complex128)

def _stack_s_matrix_redheffer(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
    v_lead_ev: float,
) -> SMatrix:
    "Build stack S matrix from Redheffer propagation and interface"
    q0 = q_number(energy_ev, v_lead_ev).item()
    q_layers = q_number(energy_ev, potentials_ev)
    s_total = interface_s_matrix(q0, q_layers[0])
    s_total = redheffer_star(s_total, [propagation_s_matrix(q_layers[0], widths_nm[0])])

    for idx in range(1, widths_nm.size):
        s_total = redheffer_star(
            s_total, interface_s_matrix(q_layers[idx-1], q_layers[idx])
        )
        s_total = redheffer_star(
            s_total, propagation_s_matrix(q_layers[idx], widths_nm[idx])
        )

    s_total = redheffer_star(s_total, interface_s_matrix(q_layers[-1], q0))
    return s_total

def vacuum_passthrough_s_matrix() -> SMatrix:
    "Connection between vacuum leads"
    return np.array([[0.0,1.0], [1.0,0.0]], dtype = np.complex128)

def slab_s_matrix(
    energy_ev: float,
    potential_ev: float,
    width_nm: float,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> SMatrix:
    "S matrix for single layer embedded in vacuum leads"
    q_0 = q_number(energy_ev, v_lead_ev).item()
    q_layer = q_number(energy_ev, potential_ev).item()
    return layer_s_matrix(q0, q_layer, q0, width_nm)

def stack_s_matrix(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> SMatrix:
    "Stack S matrix composing with Redheffer star products, at edges where E roughly equals V, go back to transfer matrix conversion"
    widths = np.asarray(widths_nm, dtype=np.float64)
    pots = np.asarray(potentials_ev, dtype=np.float64)
    if widths.shape != pots.shape:
        raise ValueError("widths_nm and potentials_ev must have the same shape")
    if widths.ndim != 1:
        raise ValueError("widths_nm and potentials_ev must be 1D")
    if widths.size == 0:
        return vacuum_passthrough_s_matrix()

    try:
        return _stack_s_matrix_redheffer(widths, pots, energy_ev, v_lead_ev) 
    except FloatingPointError:
        q0 = q_number(energy_ev, v_lead_ev).item()
        transfer = transfer_matrix_stack(widths, pots, energy_ev)
        return transfer_to_s_matrix(transfer, q0)

def abeles_matrix(q: complex, width_nm: float) -> np.ndarray:
    "2x2 Abeles characteristic matrix for one uniform layer"
    q = _as_complex_scalar(q)
    beta = q * width_nm
    cos_b = np.cos(beta)
    sin_over_q = _sin_qd_over_q(q, width_nm)
    return np.array(
        [[cos_b,1j * sin_over_q],[1j*q*np.sin(beta),cos_b],], dtype=np.complex128,
    )

def transfer_matrix_stack(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
) -> np.ndarray:
    "Product of Abeles matrices"
    widths = np.asarray(widths_nm, dtype = np.float64)
    pots = np.asarray(potentials_ev, dtype=np.float64)
    m_total = np.eye(2, dtype=np.complex128)
    for width, pot in zip(widths, pots):
        q=q_number(energy_ev, pot).item()
        m_total = abeles_matrix(q_width) @ m_total
    return m_total

def transmission_from_transfer(
    transfer: np.ndarray, q_lead: complex
) -> float:
    "Transmission from Abeles transfer matrix & lead wave number"
    q_lead = _as_complex_scalar(q_lead)
    m11, m12 = transfer[0,0], transfer[0,1]
    m21, m22 = transfer[1,0], transfer[1,1]
    denom = m11 * q_lead + m12 * q_lead**2 + m21 + m22 * q_lead
    t_amp = 2.0 * q_lead / denom
    return float(np.abs(t_amp)**2)

def safe_log_t(
    t: float,
    *,
    floor: float = cfg.cfg.TOLERANCES.log_t_floor,
) -> Tuple[float, TransferStatus]:
    "Return log(T)"
    if not np.isfinite(t):
        return float("nan"), "invalid"
    if t < 0.0:
        return float("nan"), "invalid"
    if t == 0.0:
        return floor, "underflow"
    log_t = float(np.log(t))
    if not np.isfinite(log_t):
        return floor, "overflow"
    if log_t < floor:
        return floor, "underflow"
    return log_t, "ok"

def _classify_transmission(t: float) -> TransferStatus:
    tol = cfg.TOLERANCES.t_bounds_atol
    if not np.isfinite(t):
        return "invalid"
    if t < -tol or t > 1.0 + tol:
        return "non_unitary"
    return "ok"

def transmission_from_s(
        s_matrix: SMatrix,
        q_lead: complex | None = None,
) -> TransmissionResult:
    "Extract T=|S21|^2"
    s_matrix = np.asarray(s_matrix, dtype=np.complex128)
    s21 = complex(s_matrix[1,0])
    t=float(np.abs(s21)**2)
    status = _classify_transmission(t)
    log_t, log_status = safe_log_t(t)
    if status == "ok" and log_status != "ok":
        status = log_status
    return TransmissionResult(T=t, log_T=log_t, status=status, S21=s21)

def stack_transmission(
    widths_nm: np.ndarray,
    potentials_ev: np.ndarray,
    energy_ev: float,
    v_lead_ev: float = cfg.V_LEAD_EV,
) -> TransmissionResult:
    "Transmission through layer stack at one energy"
    s_mat = stack_s_matrix(widths_nm, potentials_ev, energy_ev, v_lead_ev)
    q0 = q_number(energy_ev, v_lead_ev).item()
    return transmission_from_s(s_mat, q0)

def clean_transmission(
    generation: int,
    energy_ev: float,
    