from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

import numpy as np

#Units
LENGTH_UNIT = "nm"
ENERGY_UNIT = "eV"

#Terms for derivation of hbar^2/(2m_e) in (nm, eV)
_ELECTRON_MASS_KG = 9.1093837139e-31
_HBAR_J_S = 1.054571817e-34
_EV_IN_J = 1.602176634e-19
_NM_IN_M = 1.0e-9

HBAR2_OVER_2M_EV_NM2 = ((_HBAR_J_S**2) / (2.0* _ELECTRON_MASS_KG) / _EV_IN_J / (_NM_IN_M ** 2))
KAPPA2_PREFACTOR_PER_EV_NM2 = 1.0/HBAR2_OVER_2M_EV_NM2

#Fib Word Generation
WORD_W1 = "B"
WORD_W2 = "A"
SUBSTITUTION = MappingProxyType({"A": "AB", "B": "A"})

#Lead and Layer Parameters
V_LEAD_EV = 0.0
D_A_NM = 1.0
V_A_EV = 0.01
D_B_NM = 2.0
V_B_EV = 5.0

LAYER_WIDTH_NM: Mapping[str, float] = MappingProxyType({"A": D_A_NM, "B": D_B_NM})
LAYER_POTENTIAL_EV: Mapping[str, float] = MappingProxyType({"A": V_A_EV, "B": V_B_EV})

#Boundary tokens
LEAD_LEFT_TOKEN = "L"
LEAD_RIGHT_TOKEN = "R"

#Energy Grid
E_MIN_EV = 0.1
E_MAX_EV = 10.0
N_ENERGY = 2000

N_GENERATION_MIN = 10
N_GENERATION_MAX = 20
GENERATIONS: Tuple[int, ...] = tuple(range(N_GENERATION_MIN, N_GENERATION_MAX + 1))

#Defect Modes
MODE_SUBSTITUTION = "substitution"
MODE_PHASON = "phason"
MODE_POTENTIAL_ONLY = "potential_only"
MODE_WIDTH_ONLY = "width_only"
DEFECT_MODES: Tuple[str, ...] = (MODE_SUBSTITUTION, MODE_PHASON, MODE_POTENTIAL_ONLY, MODE_WIDTH_ONLY)

#One site modes work on one singular layer, while two site modes (phason) work on two
ONE_SITE_MODES: Tuple[str, ...] = (MODE_SUBSTITUTION, MODE_POTENTIAL_ONLY, MODE_WIDTH_ONLY)
TWO_SITE_MODES: Tuple[str, ...] = (MODE_PHASON,)

#Motif families
SITE_MOTIF_RADIUS: Mapping[str, int] = MappingProxyType({"site_letter": 0, "site_3": 1, "site_5": 2})

BOND_MOTIF_RADIUS: Mapping[str, int] = MappingProxyType({"bond_pair": 0, "bond_4": 1, "bond_6": 2})

SITE_MOTIF_FAMILIES: Tuple[str, ...] = tuple(SITE_MOTIF_RADIUS)
BOND_MOTIF_FAMILIES: Tuple[str, ...] = tuple(BOND_MOTIF_RADIUS)
MOTIF_FAMILIES: Tuple[str, ...] = SITE_MOTIF_FAMILIES + BOND_MOTIF_FAMILIES

#Strata names
GEOMETRY_STRATA: Tuple[str, ...] = ("site_letter", "site_3", "site_5", "bond_pair", "bond_4")

#Analysis parameters
PERMUTATION_REPLICATES = 500
ALPHA_FLAT_THRESHOLD = 0.05
FIT_GENERATION_WINDOWS: Tuple[Tuple[int, int], ...] = ((10,15), (10,17), (10,20))

SCALING_D_FLOOR = 1e-12
SCALING_MIN_USABLE_GENERATIONS = 3
SCALING_LOO_ALPHA_TOL = 0.05
SCALING_RMSE_MAX = 0.5

# "high" = [5,10] eV = [V_B, 2*V_B]. Conclusions should be for finite windows, or have to define separate extension
ENERGY_WINDOWS: Mapping[str, Tuple[float, float]] = MappingProxyType({
    "full": (E_MIN_EV, E_MAX_EV),
    "low": (0.1,1.0),
    "mid": (1.0,5.0),
    "high": (5.0,10.0)
})

#Tolerances
@dataclass(frozen=True)
class Tolerances:
    """Numerical tolerances for calculations."""
    t_bounds_atol: float = 1e-10
    transmission_atol: float = 1e-12
    transmission_rtol: float = 1e-9
    solver_crosscheck_atol: float = 1e-10
    solver_crosscheck_rtol: float = 1e-8
    log_t_floor: float = -700.0
    word_length_match: int = 0
    #|W_n| has to equal F_n

TOLERANCES = Tolerances()

#Output directories
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
OUTPUT_RAW = OUTPUT_ROOT / "raw"
OUTPUT_ANALYSIS = OUTPUT_ROOT / "analysis"
OUTPUT_FIGURES = OUTPUT_ROOT / "figures"
OUTPUT_METADATA = OUTPUT_ROOT / "metadata"

OUTPUT_DIRS: Tuple[Path, ...] = (OUTPUT_RAW, OUTPUT_ANALYSIS, OUTPUT_FIGURES, OUTPUT_METADATA)

def ensure_output_dirs() -> None:
    """Make directories for output if they do not exist."""
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

def energy_grid(e_min: float = E_MIN_EV, e_max: float = E_MAX_EV, n_points: int = N_ENERGY) -> np.ndarray:
    """Inclusive linear energy grid in eV."""
    if n_points < 2:
        raise ValueError("Energy grid must have at least 2 points.")
    if e_max <= e_min:
        raise ValueError("Energy grid max must be greater than min.")
    return np.linspace(e_min, e_max, n_points, dtype=np.float64)

@dataclass(frozen=True)
class SimulationConfig:
    "Frozen snapshot for each locked value"
    length_unit: str = LENGTH_UNIT
    energy_unit: str = ENERGY_UNIT
    hbar2_over_2m_ev_nm2: float = HBAR2_OVER_2M_EV_NM2
    v_lead_ev: float = V_LEAD_EV
    d_a_nm: float = D_A_NM
    v_a_ev: float = V_A_EV
    d_b_nm: float = D_B_NM
    v_b_ev: float = V_B_EV
    e_min_ev: float = E_MIN_EV
    e_max_ev: float = E_MAX_EV
    n_energy: int = N_ENERGY
    n_generation_min: int = N_GENERATION_MIN
    n_generation_max: int = N_GENERATION_MAX
    generations: Tuple[int, ...] = GENERATIONS
    word_w1:str = WORD_W1
    word_w2:str = WORD_W2
    defect_modes: Tuple[str, ...] = DEFECT_MODES
    motif_families: Tuple[str, ...] = MOTIF_FAMILIES
    geometry_strata: Tuple[str, ...] = GEOMETRY_STRATA
    permutation_replicates: int = PERMUTATION_REPLICATES
    alpha_flat_threshold: float = ALPHA_FLAT_THRESHOLD
    scaling_d_floor: float = SCALING_D_FLOOR
    scaling_min_usable_generations: int = SCALING_MIN_USABLE_GENERATIONS
    scaling_loo_alpha_tol: float = SCALING_LOO_ALPHA_TOL
    scaling_rmse_max: float = SCALING_RMSE_MAX
    fit_generation_windows: Tuple[Tuple[int, int], ...] = FIT_GENERATION_WINDOWS
    energy_windows: Mapping[str, Tuple[float, float]] = ENERGY_WINDOWS
    tolerances: Tolerances = TOLERANCES
    output_root: str = str(OUTPUT_ROOT)

    def as_dict(self) -> Dict[str, object]:
        return {
            "length_unit": self.length_unit,
            "energy_unit": self.energy_unit,
            "hbar2_over_2m_ev_nm2": self.hbar2_over_2m_ev_nm2,
            "v_lead_ev": self.v_lead_ev,
            "d_a_nm": self.d_a_nm,
            "v_a_ev": self.v_a_ev,
            "d_b_nm": self.d_b_nm,
            "v_b_ev": self.v_b_ev,
            "e_min_ev": self.e_min_ev,
            "e_max_ev": self.e_max_ev,
            "n_energy": self.n_energy,
            "n_generation_min": self.n_generation_min,
            "n_generation_max": self.n_generation_max,
            "generations": list(self.generations),
            "word_w1": self.word_w1,
            "word_w2": self.word_w2,
            "defect_modes": list(self.defect_modes),
            "motif_families": list(self.motif_families),
            "geometry_strata": list(self.geometry_strata),
            "permutation_replicates": self.permutation_replicates,
            "alpha_flat_threshold": self.alpha_flat_threshold,
            "scaling_d_floor": self.scaling_d_floor,
            "scaling_min_usable_generations": self.scaling_min_usable_generations,
            "scaling_loo_alpha_tol": self.scaling_loo_alpha_tol,
            "scaling_rmse_max": self.scaling_rmse_max,
            "fit_generation_windows": [list(w) for w in self.fit_generation_windows],
            "energy_windows": {k: list(v) for k, v in self.energy_windows.items()},
            "tolerances": asdict(self.tolerances),
            "output_root": self.output_root,
            "layer_width_nm": dict(LAYER_WIDTH_NM),
            "layer_potential_ev": dict(LAYER_POTENTIAL_EV)
        }

    def grid(self) -> np.ndarray:
        return energy_grid(self.e_min_ev, self.e_max_ev, self.n_energy)

CONFIG = SimulationConfig()