"testing"

from __future__ import annotations

import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.config import CONFIG, SimulationConfig, energy_grid, ensure_output_dirs

def test_units_and_layer_parameters_are_locked():
    assert cfg.LENGTH_UNIT == "nm"
    assert cfg.ENERGY_UNIT == "eV"
    assert cfg.V_LEAD_EV == 0.0
    assert cfg.D_A_NM == 1.0
    assert cfg.V_A_EV == 0.01
    assert cfg.D_B_NM == 2.0
    assert cfg.V_B_EV == 5.0
    assert cfg.LAYER_WIDTH_NM["A"] == cfg.D_A_NM
    assert cfg.LAYER_WIDTH_NM["B"] == cfg.D_B_NM
    assert cfg.LAYER_POTENTIAL_EV["A"] == cfg.V_A_EV
    assert cfg.LAYER_POTENTIAL_EV["B"] == cfg.V_B_EV

def test_energy_grid_default():
    grid = energy_grid()
    assert grid.dtype == np.float64
    assert grid.shape == (2000,)
    assert grid[0] == pytest.approx(0.1)
    assert grid[-1] == pytest.approx(10.0)
    assert np.all(np.diff(grid)>0)
    np.testing.assert_allclose(grid, CONFIG.grid())

def test_generation_range():
    assert cfg.N_GENERATION_MIN == 10
    assert cfg.N_GENERATION_MAX == 20
    assert cfg.GENERATIONS == tuple(range(10,21))
    assert len(cfg.GENERATIONS) == 11

def test_hbar2_over_2m_is_electron_scale():
    assert 0.037 < cfg.HBAR2_OVER_2M_EV_NM2 < 0.039
    assert cfg.KAPPA2_PREFACTOR_PER_EV_NM2 == pytest.approx(
        1.0/cfg.HBAR2_OVER_2M_EV_NM2
    )

def test_config_is_frozen():
    with pytest.raises(Exception):
        CONFIG.v_lead_ev = 1.0
    clone = SimulationConfig()
    assert clone.as_dict()["d_a_nm"] == 1.0
    assert clone.as_dict()["tolerances"]["t_bounds_atol"] == cfg.TOLERANCES.t_bounds.atol["tolerances"]["t_bounds_atol"]

def test_output_paths_and_ensure_dirs():
    assert cfg.OUTPUT_ROOT == cfg.PROJECT_ROOT / "output"
    assert cfg.OUTPUT_RAW.parent == cfg.OUTPUT_ROOT
    ensure_output_dirs()
    for path in cfg.OUTPUT_DIRS:
        assert path.is_dir()

def test_word_convention_and_motif_vocabulary():
    assert cfg.WORD_W1 == "B"
    assert cfg.WORD_W2 == "A"
    assert dict(cfg.SUBSTITUTION) == {"A":"AB", "B":"A"}
    assert "site_letter" in cfg.SITE_MOTIF_FAMILIES
    assert "bond_pair" in cfg.BOND_MOTIF_FAMILIES
    for family in cfg.GEOMETRY_STRATA:
        assert family in cfg.MOTIF_FAMILIES
    with pytest.raises(TypeError):
        cfg.LAYER_WIDTH_NM["A"] = 3.0