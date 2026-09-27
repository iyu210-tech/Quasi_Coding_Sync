"Tests the fast solver"

from __future__ import annotations

import json
import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import (
    StackCache, fast_sweep, fast_transmission, validate_against_reference,
)
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.raw_data import (
    load_generation_raw,
    produce_raw_dataset,
    save_generation_raw,
    sweep_generation,
)
from fibonacci_qc.reference_solver import delta_t
from fibonacci_qc.scattering import stack_transmission

def test_prefix_suffix_reproduces_clean_transmission():
    geom = build_geometry(5)
    energy = 1.8
    cache = StackCache.build(geom, energy)
    cached = cache.clean_transmission()
    direct = stack_transmission(geom.widths_nm, geom.potentials_ev, energy)
    assert cached.T == pytest.approx(direct.T, rel = 1e-12, abs = 1e-14)

@pytest.mark.parametrize("mode", list(cfg.ONE_SITE_MODES))
def test_fast_one_site_matches_reference(mode):
    geom = build_geometry(6)
    energies = [0.4,1.7,4.2]
    for e in energies:
        cache = StackCache.build(geom, e)
        for p in range(geom.n_sites):
            fast = fast_transmission(mode, p, cache, geom)
            ref = delta_t(mode, p, e, geometry=geom)
            assert ref.valid
            assert fast.T == pytest.approx(ref.T, rel=1e-10, abs=1e-12)
            if np.isfinite(ref.log_T):
                assert fast.log_T == pytest.approx(ref.log_T, rel = 1e-8, abs = 1e-6)

def test_fast_phason_matches_reference():
    geom = build_geometry(7)
    energies = [0.6,2.5,7.0]
    for e in energies:
        cache = StackCache.build(geom, e)
        for p in range(geom.n_bonds):
            fast = fast_transmission(cfg.MODE_PHASON, p, cache, geom)
            ref = delta_t(cfg.MODE_PHASON, p, e, geometry = geom)
            if not ref.valid:
                assert fast.status == "invalid"
                continue
            assert fast.T == pytest.approx(ref.T, rel = 1e-10, abs = 1e-12)

def test_validate_against_reference_random_subset():
    geom = build_geometry(5)
    energies = np.linspace(0.3,6.0,8)
    mismatches = validate_against_reference(geom, energies)
    assert mismatches ==[]

def test_fast_sweep_shapes_and_uncompacted_phason_axis():
    geom = build_geometry(5)
    energies = np.linspace(0.5,2.0,6)
    sweep = fast_sweep(cfg.MODE_PHASON, energies, geometry=geom)
    assert sweep["T"].shape == (geom.n_bonds, energies.size)
    assert sweep["positions"].shape == (geom.n_bonds,)
    assert sweep["valid"].shape == (geom.n_bonds,)
    assert sweep["valid"][2] is np.False_ or sweep["valid"][2] == False
    assert np.isnan(sweep["T"][2,0])
    assert np.isfinite(sweep["T"][0,0])

def test_sweep_generation_contains_clean_and_modes():
    geom=build_geometry(4)
    energies = np.linspace(0.4,3.0,5)
    bundle = sweep_generation(4, energies, geometry = geom, modes = [cfg.MODE_SUBSTITUTION])
    assert bundle["T_clean"].shape == (5,)
    assert cfg.MODE_SUBSTITUTION in bundle["modes"]
    assert bundle["modes"][cfg.MODE_SUBSTITUTION]["T"].shape == (geom.n_sites, 5)

def test_save_and_load_generation_raw(tmp_path):
    geom = build_geometry(4)
    energies = np.linspace(0.5,2.5,4)
    bundle = sweep_generation(
        4,
        energies,
        geometry = geom,
        modes = [cfg.MODE_WIDTH_ONLY],
    )
    path = save_generation_raw(bundle, tmp_path / "n04" / "width_only.npz")
    loaded = load_generation_raw(path)
    np.testing.assert_allclose(loaded["T_clean"], bundle["T_clean"])
    mode = loaded["modes"][cfg.MODE_WIDTH_ONLY]
    np.testing.assert_allclose(
        mode["T"],
        bundle["modes"][cfg.MODE_WIDTH_ONLY]["T"],
    )

def test_produce_raw_dataset_small_subset(tmp_path):
    manifest = produce_raw_dataset(generations=[4,5], energies_ev=np.linspace(0.5, 2.0, 4),output_dir = tmp_path/"mini_study", modes = [cfg.MODE_POTENTIAL_ONLY, cfg.MODE_PHASON], validate = True,)
    assert manifest["files"]
    manifest_path = manifest["manifest_path"]
    assert manifest_path
    with open(manifest_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["run_metadata"]["n_energy"] == 4