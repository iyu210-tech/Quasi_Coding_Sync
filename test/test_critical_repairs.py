"Check the repairs"

from __future__ import annotations

import json
import copy
from pathlib import Path
import numpy as np
import pytest

from fibonacci_qc import config as cfg
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.scattering import (
    ScaledS, layer_block_scaled, q_number, stack_transmission, stack_s_matrix,
    clean_transmission, transmission_from_scaled, transmission_from_s,
    trusted_transmission_mask, trusted_log_mask, encode_status, decode_status,
    exterior_unitarity_residuals, rectangular_barrier_transmission,
)
from fibonacci_qc.fast_solver import StackCache,fast_transmission,fast_sweep,validate_against_reference
from fibonacci_qc.parity_analysis import parity_contrast,analyze_generation_parity
from fibonacci_qc.interaction import interaction_from_sweeps
from fibonacci_qc.raw_data import sweep_generation,save_generation_raw,load_generation_bundle,load_generation_raw,assert_same_run
from fibonacci_qc.resonance import detect_peaks,Peak,match_peaks_order_preserving
from fibonacci_qc.robustness import (
    compare_parity_contrast,classify_contrast_robustness,compare_resonance_spectra,
    compare_endpoint_summaries,nested_fine_grid,
)
from fibonacci_qc.scaling import fit_scaling_curve,paired_alpha_on_shared_generations
from fibonacci_qc.regression import assert_matches_regression_fixture

@pytest.mark.parametrize("dtype",[np.uint8,np.int64,object])
def test_integer_status_roundtrip_decoder(dtype):
    code=np.array([0,1,2,3,4],dtype=dtype)
    assert decode_status(code).tolist()==["ok","invalid","underflow","overflow","non_unitary"]
    assert trusted_transmission_mask(code).tolist()==[True,False,True,False,False]
    assert trusted_log_mask(code).tolist()==[True,False,True,False,False]

def test_underflow_keeps_mean_denominator():
    t=np.array([[0.],[.8],[.2],[.8]])
    status=np.array([[2],[0],[0],[0]],dtype=np.uint8)
    r=parity_contrast(t,np.array([0,1,0,1]),np.ones(4,dtype=bool),cell_valid=trusted_transmission_mask(status,t))
    assert r.D[0]==pytest.approx(-.7)
    assert r.coverage==1
    assert r.n_even_by_energy[0]==2

def test_subnormal_not_prematurely_flushed():
    log=-360.
    state=ScaledS(1+0j,1+0j,log,1j,log,1j)
    r=transmission_from_scaled(state,1.)
    assert r.status=="ok" and r.T>0 and r.log_T==-720.

def test_finite_unknown_or_invalid_matrix_not_trusted():
    bad=np.array([[.6,.8],[.8,.6]])
    assert transmission_from_s(bad,1.).status=="non_unitary"
    bad[0,0]=np.nan
    assert transmission_from_s(bad,1.).status=="invalid"

@pytest.mark.parametrize("n,energy,expected",[(5,.2,-92.20080849515435),(10,.5,-931.294305099148)])
def test_previously_corrupted_cases(n,energy,expected):
    g=build_geometry(n)
    for r in (clean_transmission(n,energy),StackCache.build(g,energy).clean_transmission()):
        assert r.is_valid and np.isfinite(r.log_T)
        assert r.log_T==pytest.approx(expected,abs=1e-8)
        assert 0<=r.T<=1+cfg.TOLERANCES.t_bounds_atol

@pytest.mark.parametrize("energy",[5.-1e-10,5.,5.+1e-10])
def test_threshold_both_paths_and_all_modes(energy):
    g=build_geometry(6)
    assert validate_against_reference(g,[energy])==[]
    r=stack_transmission(np.array([2.,2.]),np.array([5.,5.]),energy)
    assert r.T==pytest.approx(rectangular_barrier_transmission(energy,5.,4.),rel=1e-8,abs=1e-12)

def test_deep_log_interaction_and_linear_zero():
    b=sweep_generation(10,np.array([.5,.6]),modes=cfg.ONE_SITE_MODES)
    rem=interaction_from_sweeps(b["modes"],t_clean=b["T_clean"],log_t_clean=b["log_T_clean"],
        status_clean=b["clean_status"],energies=b["energies"])
    assert rem.I[0,0]==0
    assert rem.I_log[0,0]==pytest.approx(-21.7277845154,abs=1e-7)
    b["modes"]["substitution"]["status"][0,0]=1
    rem=interaction_from_sweeps(b["modes"],t_clean=b["T_clean"],log_t_clean=b["log_T_clean"],
        status_clean=b["clean_status"],energies=b["energies"])
    assert np.isnan(rem.I[0,0]) and np.isnan(rem.I_log[0,0])

def test_all_nan_has_no_coverage():
    r=parity_contrast(np.full((4,2),np.nan),np.array([0,1,0,1]),np.ones(4,dtype=bool))
    assert r.coverage==0 and np.array_equal(r.coverage_by_energy,[0.,0.])

def test_saved_clean_status_and_geometry_roundtrip(tmp_path):
    b=sweep_generation(10,np.array([.5,5.,7.]))
    for mode,s in b["modes"].items():
        save_generation_raw({**b,"modes":{mode:s}},tmp_path/"n10"/(mode+".npz"))
    loaded=load_generation_bundle(tmp_path/"n10",require_modes=cfg.DEFECT_MODES)
    assert loaded["clean_status"].dtype==np.uint8
    assert loaded["geometry_metadata"]==build_geometry(10).to_metadata()
    result=analyze_generation_parity(loaded)
    assert not result["numerical_failure"]
    assert result["modes"]["substitution"]["unconditional"].D[0]==0
    assert "response" in result["modes"]["substitution"]

def test_cross_generation_identity_does_not_compare_position_lengths():
    a=sweep_generation(4,np.array([.5,5.,7.]))
    b=sweep_generation(5,np.array([.5,5.,7.]))
    assert_same_run(a,b,where="test",same_generation=False)
    bad=copy.deepcopy(b);bad["run_metadata"]["config_sha256"]="wrong"
    with pytest.raises(ValueError,match="config hash"):assert_same_run(a,bad,where="bad",same_generation=False)

def test_clean_log_mismatch_not_merged(tmp_path):
    b=sweep_generation(4,np.array([.5,5.,7.]),modes=["substitution","width_only"])
    save_generation_raw({**b,"modes":{"substitution":b["modes"]["substitution"]}},tmp_path/"n04"/"substitution.npz")
    other=copy.deepcopy(b);other["log_T_clean"][0]-=1
    save_generation_raw({**other,"modes":{"width_only":other["modes"]["width_only"]}},tmp_path/"n04"/"width_only.npz")
    with pytest.raises(ValueError,match="log_T_clean"):load_generation_bundle(tmp_path/"n04")

def test_raw_overwrite_is_not_implicit(tmp_path):
    b=sweep_generation(4,np.array([.5,5.,7.]),modes=["substitution"])
    path=tmp_path/"substitution.npz"
    save_generation_raw(b,path)
    with pytest.raises(FileExistsError):save_generation_raw(b,path)

def test_new_nodes_cannot_falsely_pass():
    r=compare_parity_contrast(np.zeros(3),np.array([.1,.6,1.1]),
                             np.array([0.,1.,0.,1.,0.]),np.array([.1,.35,.6,.85,1.1]))
    assert r.max_abs_D==0
    assert r.max_abs_new_nodes==1
    assert classify_contrast_robustness(r)=="fail"

def test_response_gate_detects_parity_cancelling_change():
    b={"full":dict(mean_abs=.2,rms=.2,coverage=1.)}
    f={"full":dict(mean_abs=.38,rms=.4,coverage=1.)}
    assert compare_endpoint_summaries(b,f,kind="response")["gate"]=="fail"

def test_log_interaction_has_its_own_gate():
    b={"full":dict(mean_abs_I=0.,rms_I=0.,mean_abs_I_log=1.,rms_I_log=1.,coverage_I=1.,coverage_I_log=1.)}
    f=copy.deepcopy(b);f["full"]["mean_abs_I_log"]=2
    assert compare_endpoint_summaries(b,f,kind="interaction")["gate"]=="fail"

def test_absent_peak_pairs_are_not_false_numerical_failure():
    eb=np.linspace(.1,10.,21);ef=nested_fine_grid(eb)
    r=compare_resonance_spectra(np.ones(21)*.2,eb,np.ones(41)*.2,ef,
        t_clean_baseline=np.ones(21)*.3,t_clean_fine=np.ones(41)*.3,position=0)
    assert r.gate=="not_applicable" and np.isnan(r.delta_E_summary_diff)

def test_genuine_invalid_spectrum_is_unresolved():
    eb=np.linspace(.1,10.,21);ef=nested_fine_grid(eb)
    r=compare_resonance_spectra(np.full(21,np.nan),eb,np.ones(41)*.2,ef,
        t_clean_baseline=np.ones(21)*.3,t_clean_fine=np.ones(41)*.3,position=0)
    assert r.gate=="unresolved"

def test_prominence_is_not_neighbor_step_size():
    for n in [1001,2001]:
        e=np.linspace(0.,10.,n);y=np.exp(-((e-5)/1.)**2)
        assert len(detect_peaks(y,e))==1

def test_default_match_gate_retains_unmatched_resonances():
    r=match_peaks_order_preserving([Peak(0,1.,1.,1.)],[Peak(0,8.,1.,1.)])
    assert r.n_matched==0 and len(r.unmatched_clean)==1 and np.isnan(r.total_displacement)

def test_loo_needs_every_relevant_deletion():
    sizes=np.array([5.,8.,13.,21.,34.,55.]);d=np.zeros((6,1));d[:3,0]=sizes[:3]**-.2
    r=fit_scaling_curve(sizes,d,np.array([1.]),tuple(range(5,11)),mode="substitution",window=(5,10))
    assert r.support[0] and not r.stable[0] and r.indeterminate[0]

def test_shared_generation_pairing():
    sizes=np.array([5.,8.,13.,21.,34.]);a=sizes[:,None]**-.3;b=sizes[:,None]**-.1
    b[0]=0
    r=paired_alpha_on_shared_generations(sizes,a,b,np.array([1.]))
    assert r["shared_n_usable"][0]==4
    assert r["delta_alpha_shared_gens"][0]==pytest.approx(.2,abs=1e-10)
    assert r["support_stable_shared_gens"][0]

def test_fixture_is_checked_without_regeneration():
    assert_matches_regression_fixture()

def test_injected_failure_flag_and_response_are_saved(tmp_path):
    from fibonacci_qc.analysis_io import save_parity_analysis
    b=sweep_generation(4,np.array([.5,5.,7.]),modes=["substitution"])
    b["modes"]["substitution"]["status"][0,0]=1
    a=analyze_generation_parity(b)
    assert a["numerical_failure"]
    path=save_parity_analysis(a,tmp_path/"parity.npz")
    m=json.loads(path.with_suffix(".meta.json").read_text())
    assert m["numerical_failure"] and "substitution" in m["response"]

def test_full_four_mode_multi_generation_persistence(tmp_path):
    from fibonacci_qc.raw_data import produce_raw_dataset
    from fibonacci_qc.drivers import run_analyze_driver,run_figures_driver
    from fibonacci_qc.scaling_io import load_scaling_analysis
    raw=tmp_path/"complete_four_mode_case"
    manifest=produce_raw_dataset(generations=[4,5,6,7],energies_ev=np.array([.1,.5,1.,3.,5.,7.,10.]),
                                output_dir=raw,modes=cfg.DEFECT_MODES,validate=True)
    analysis=run_analyze_driver(raw_dir=str(raw),generations=[4,5,6,7],modes=cfg.DEFECT_MODES,n_replicates=4)
    out=Path(analysis["analyze_manifest_path"]).parent
    assert len(manifest["files"])==16
    for gen in [4,5,6,7]:
        meta=json.loads((out/"resonance"/f"n{gen:02d}"/"resonance.meta.json").read_text())
        assert set(meta["modes"])==set(cfg.DEFECT_MODES)
        for v in meta["modes"].values(): assert set(v["summary"]["by_parity"])=={"even","odd"}
        assert (out/"resonance"/f"n{gen:02d}"/"interaction.npz").exists()
        checks=json.loads((out/"robustness"/f"n{gen:02d}"/"robustness.meta.json").read_text())
        assert "primary_gate" in checks and "interaction_gate" in checks
    scal=load_scaling_analysis(out/"scaling"/"scaling.npz")
    assert "global_support" in scal["arrays"]
    figs=run_figures_driver(raw_dir=str(raw),generation=4,output_dir=str(tmp_path/"figures"))
    assert "resonance_phason" in figs["files"]
    assert analysis["robustness"]["primary_gate"]!="pass"  # deliberately coarse, not validated science

@pytest.mark.parametrize("profile,support,gate,expected",[("smoke",5,"pass",False),("production",0,"pass",True)])
def test_numerical_acceptance_is_not_secondary_scaling(profile,support,gate,expected,tmp_path,monkeypatch):
    import fibonacci_qc.pipeline as p
    monkeypatch.setattr(p,"run_raw_driver",lambda **kw:{"files":[]})
    monkeypatch.setattr(p,"run_figures_driver",lambda **kw:{"files":{}})
    monkeypatch.setattr(p,"run_analyze_driver",lambda **kw:{"scaling":{"summary":[{"n_support":support}]},
        "robustness":{"primary_gate":gate},"parity":{"numerical_failure":False},"analyze_manifest_path":"unit-test-only"})
    result=p.run_pipeline(profile=profile)
    assert result["numerically_accepted"]==expected
    assert result["scientific_claim"]==expected

def test_failed_primary_gate_cannot_be_overridden_by_scaling(tmp_path,monkeypatch):
    import fibonacci_qc.pipeline as p
    monkeypatch.setattr(p,"run_raw_driver",lambda **kw:{"files":[]})
    monkeypatch.setattr(p,"run_figures_driver",lambda **kw:{"files":{}})
    monkeypatch.setattr(p,"run_analyze_driver",lambda **kw:{"scaling":{"summary":[{"n_support":500}]},
        "robustness":{"primary_gate":"fail"},"parity":{"numerical_failure":False},"analyze_manifest_path":"unit-test-only"})
    with pytest.raises(RuntimeError,match="acceptance blocked"):p.run_pipeline(profile="production")
    assert list(cfg.OUTPUT_METADATA.glob("pipeline_production*.json"))