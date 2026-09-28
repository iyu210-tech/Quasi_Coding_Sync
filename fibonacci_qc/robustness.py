"Recompute key results with a finer energy grid and make sure they still match (parity difference, summary numbers, transmission peaks and how they line up)"

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from fibonacci_qc import config as cfg
from fibonacci_qc.fast_solver import fast_sweep
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.parity_analysis import integrate_contrast_windows, parity_contrast, position_parity, position_valid_mask, summarize_delta_t_windows
from fibonacci_qc.resonance import detect_peaks, match_peaks_order_preserving, track_resonances
from fibonacci_qc.scattering import trusted_transmission_mask
from fibonacci_qc.interaction import interaction_from_sweeps, interaction_summary

ROBUSTNESS_N_ENERGY = 2*cfg.N_ENERGY-1
ROBUSTNESS_PASS_MAX_ABS_D = cfg.GATE_TOLERANCES["parity_pointwise"]
ROBUSTNESS_PASS_RMSE_D = ROBUSTNESS_PASS_MAX_ABS_D
ROBUSTNESS_INDETERMINATE_MAX_ABS_D = 1e-2
ROBUSTNESS_INDETERMINATE_RMSE_D = 1e-2

@dataclass(frozen=True)
class ContrastRobustness:
    max_abs_D: float
    rmse_D: float
    integrated_baseline: dict
    integrated_fine: dict
    integrated_abs_diff: dict
    max_abs_new_nodes: float = float("nan")
    rmse_new_nodes: float = float("nan")
    numerical_complete: bool = False
    window_coverage: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ResonanceRobustness:
    position: int
    n_peaks_baseline: int
    n_peaks_fine: int
    n_matched: int
    mean_abs_peak_shift: float
    max_abs_peak_shift: float
    mean_abs_delta_E_baseline: float
    mean_abs_delta_E_fine: float
    delta_E_summary_diff: float
    gate: str = "unresolved"
    reason: str = ""
    n_clean_baseline: int = 0
    n_clean_fine: int = 0
    max_clean_peak_shift: float = float("nan")
    max_individual_shift_change: float = float("nan")

def worst_gate(gates):
    values = list(gates)
    for gate in ("fail","unresolved","indeterminate"):
        if gate in values: return gate
    return "pass" if "pass" in values else "not_applicable"

def finer_energy_grid(e_min=cfg.E_MIN_EV, e_max=cfg.E_MAX_EV, n_points=ROBUSTNESS_N_ENERGY):
    return cfg.energy_grid(e_min,e_max,n_points)

def nested_fine_grid(energies):
    e = np.asarray(energies,dtype=float)
    if e.ndim != 1 or e.size < 2 or not np.all(np.diff(e)>0):
        raise ValueError("strictly increasing baseline with >=2 points required")
    out = np.empty(2*e.size-1);out[::2]=e;out[1::2]=e[:-1]+np.diff(e)/2
    if not np.all(np.diff(out)>0): raise ValueError("grid cannot be bisected at float64 precision")
    return out

def offset_energy_grid(e_min=cfg.E_MIN_EV,e_max=cfg.E_MAX_EV,n_points=cfg.N_ENERGY):
    base = cfg.energy_grid(e_min,e_max,n_points)
    return base[:-1]+np.diff(base)/2

def adaptive_energy_grid_near_peaks(baseline_energies,peak_energies,*,half_window_ev,n_per_window=21):
    e=np.asarray(baseline_energies,dtype=float)
    if half_window_ev <= 0 or n_per_window<3: raise ValueError("positive window and >=3 points required")
    return np.unique(np.concatenate([e]+[np.linspace(max(e[0],p-half_window_ev),min(e[-1],p+half_window_ev),n_per_window) for p in peak_energies]))

def interpolate_to_grid(values,from_energies,to_energies):
    return np.interp(np.asarray(to_energies,dtype=float),np.asarray(from_energies,dtype=float),np.asarray(values,dtype=float))

def _error(values):
    x=np.asarray(values,dtype=float)
    if not x.size or not np.all(np.isfinite(x)): return float("nan"),float("nan")
    return float(np.max(np.abs(x))),float(np.sqrt(np.mean(x*x)))

def compare_parity_contrast(d_baseline,energies_baseline,d_fine,energies_fine):
    db,eb,df,ef=map(lambda x:np.asarray(x,dtype=float),(d_baseline,energies_baseline,d_fine,energies_fine))
    old_max,old_rms=_error(interpolate_to_grid(df,ef,eb)-db)
    new_max,new_rms=_error(df-interpolate_to_grid(db,eb,ef))
    bm,fm=integrate_contrast_windows(db,eb),integrate_contrast_windows(df,ef)
    b={w:{k:getattr(v,k) for k in ("signed","abs","rms")} for w,v in bm.items()}
    f={w:{k:getattr(v,k) for k in ("signed","abs","rms")} for w,v in fm.items()}
    dif={w:{k:abs(f[w][k]-b[w][k]) for k in b[w]} for w in b}
    cov={w: min(bm[w].coverage,fm[w].coverage) for w in bm}
    same_span=bool(eb[0]==ef[0] and eb[-1]==ef[-1])
    return ContrastRobustness(old_max,old_rms,b,f,dif,new_max,new_rms, bool(np.all(np.isfinite(db)) and np.all(np.isfinite(df)) and same_span),cov)

def classify_contrast_robustness(contrast):
    if not contrast.numerical_complete: return "unresolved"
    if not np.isfinite(contrast.max_abs_new_nodes) or not np.isfinite(contrast.rmse_new_nodes): return "unresolved"
    gates=[]
    gates.append("pass" if max(contrast.max_abs_new_nodes,contrast.rmse_new_nodes)<=cfg.GATE_TOLERANCES["parity_pointwise"] else "fail")
    for window,values in contrast.integrated_abs_diff.items():
        if contrast.window_coverage.get(window,0)<1-1e-12 or not all(np.isfinite(x) for x in values.values()):
            gates.append("unresolved");continue
        for key,delta in values.items():
            tol=cfg.GATE_TOLERANCES["parity_rms" if key=="rms" else "parity_integral_ev"]
            gates.append("pass" if delta<=tol else "fail")
    return worst_gate(gates)

def compare_resonance_spectra(spectrum_baseline,energies_baseline,spectrum_fine,energies_fine,*,
        t_clean_baseline,t_clean_fine,position,mode="robustness",parity=0,
        clean_peaks_baseline=None,clean_peaks_fine=None):
    sb,sf,cb,cf=map(lambda x:np.asarray(x,dtype=float),(spectrum_baseline,spectrum_fine,t_clean_baseline,t_clean_fine))
    empty=ResonanceRobustness(position,0,0,0,np.nan,np.nan,np.nan,np.nan,np.nan, gate="unresolved",reason="untrusted or nonfinite spectrum")
    if not all(np.all(np.isfinite(x)) for x in (sb,sf,cb,cf)): return empty
    pb,pf=detect_peaks(sb,energies_baseline),detect_peaks(sf,energies_fine)
    qb=detect_peaks(cb,energies_baseline) if clean_peaks_baseline is None else clean_peaks_baseline
    qf=detect_peaks(cf,energies_fine) if clean_peaks_fine is None else clean_peaks_fine
    tol=cfg.GATE_TOLERANCES["peak_energy_ev"]
    dm=match_peaks_order_preserving(pb,pf,max_delta_e=tol)
    cm=match_peaks_order_preserving(qb,qf,max_delta_e=tol)
    maxdef=max((abs(m.delta_E) for m in dm.matched),default=float("nan"))
    meandef=float(np.mean([abs(m.delta_E) for m in dm.matched])) if dm.matched else float("nan")
    maxclean=max((abs(m.delta_E) for m in cm.matched),default=float("nan"))
    tb=track_resonances(cb,sb,energies_baseline,mode=mode,position=position,parity=parity,clean_peaks=qb)
    tf=track_resonances(cf,sf,energies_fine,mode=mode,position=position,parity=parity,clean_peaks=qf)
    diff=abs(tf.mean_abs_delta_E-tb.mean_abs_delta_E) if tb.match.n_matched and tf.match.n_matched else np.nan
    gate,reason="pass","peak identities and shifts agree on sampled grids"
    individual=np.nan
    if len(pb)!=len(pf) or dm.n_matched!=len(pb) or len(qb)!=len(qf) or cm.n_matched!=len(qb):
        gate,reason="fail","clean or defect peaks changed count/location under refinement"
    else:
        dmap={m.clean_index:m.defect_index for m in dm.matched}
        cmap={m.clean_index:m.defect_index for m in cm.matched}
        expected={(cmap[m.clean_index],dmap[m.defect_index]):m for m in tb.match.matched}
        actual={(m.clean_index,m.defect_index):m for m in tf.match.matched}
        if set(expected)!=set(actual):
            gate,reason="fail","clean-defect pairing changes across numerical grids"
        elif not expected:
            gate,reason="not_applicable","stable absence of admissible clean-defect matches; no shift estimate"
        else:
            individual=max(abs(actual[k].delta_E-v.delta_E) for k,v in expected.items())
            if max(individual,diff)>cfg.GATE_TOLERANCES["peak_shift_ev"]:
                gate,reason="fail","finite matched shifts have not converged"
    return ResonanceRobustness(position,len(pb),len(pf),dm.n_matched,meandef,maxdef, tb.mean_abs_delta_E,tf.mean_abs_delta_E,float(diff),gate,reason,len(qb),len(qf),maxclean,float(individual))

def compare_endpoint_summaries(baseline,fine,*,kind):
    "Gate only estimands"
    changes={};gates=[]
    metrics=("mean_abs","rms") if kind=="response" else ("mean_abs_I","rms_I","mean_abs_I_log","rms_I_log")
    for window in baseline:
        b=asdict(baseline[window]) if hasattr(baseline[window],"__dataclass_fields__") else baseline[window]
        f=asdict(fine[window]) if hasattr(fine[window],"__dataclass_fields__") else fine[window]
        changes[window]={}
        coverage_keys=("coverage",) if kind=="response" else ("coverage_I","coverage_I_log")
        if any(not np.isfinite(item.get(k,np.nan)) or item.get(k,0)<cfg.MIN_NUMERICAL_COVERAGE for item in (b,f) for k in coverage_keys):
            gates.append("unresolved")
        for key in metrics:
            a,z=b.get(key,np.nan),f.get(key,np.nan)
            if not(np.isfinite(a) and np.isfinite(z)):
                changes[window][key]=np.nan;gates.append("unresolved");continue
            diff=abs(a-z);changes[window][key]=float(diff)
            if key.endswith("_log"):
                tol=cfg.GATE_TOLERANCES["interaction_log_atol"]+cfg.GATE_TOLERANCES["interaction_log_rtol"]*max(abs(a),abs(z))
            else:tol=cfg.GATE_TOLERANCES[kind]
            gates.append("pass" if diff<=tol else "fail")
    return {"gate":worst_gate(gates),"differences":changes}

def classify_robustness(contrast,resonance_rows,*,response_gate="unresolved",numerical_gate="pass"):
    return worst_gate([classify_contrast_robustness(contrast),response_gate,numerical_gate]+[r.gate for r in resonance_rows])

def _trusted_sweep(sweep):
    cell=trusted_transmission_mask(sweep["status"],sweep["T"],sweep["log_T"])
    clean=trusted_transmission_mask(sweep["clean_status"],sweep["T_clean"],sweep["log_T_clean"])
    return cell,clean

def robustness_check_mode(geometry,mode,*,energies_baseline=None,energies_fine=None,positions=None,
                          baseline_sweep=None,fine_sweep=None):
    eb=np.asarray(cfg.energy_grid() if energies_baseline is None else energies_baseline,dtype=float)
    ef=nested_fine_grid(eb) if energies_fine is None else np.asarray(energies_fine,dtype=float)
    if ef.size<=eb.size or ef[0]!=eb[0] or ef[-1]!=eb[-1]:
        raise ValueError("fine grid must add samples over the identical energy span")
    b=fast_sweep(mode,eb,geometry=geometry) if baseline_sweep is None else baseline_sweep
    f=fast_sweep(mode,ef,geometry=geometry) if fine_sweep is None else fine_sweep
    if not np.array_equal(b["energies"],eb) or not np.array_equal(f["energies"],ef):
        raise ValueError("robustness sweep coordinates do not match supplied grids")
    if not np.array_equal(b["valid"],f["valid"]): raise ValueError("geometric validity changes between grids")
    valid=position_valid_mask(geometry,mode,sweep_valid=b["valid"]);parity=position_parity(geometry,mode)
    bc,bclean=_trusted_sweep(b);fc,fclean=_trusted_sweep(f)
    db=parity_contrast(b["T"],parity,valid,energies=eb,cell_valid=bc).D
    df=parity_contrast(f["T"],parity,valid,energies=ef,cell_valid=fc).D
    complete=bool(np.all(bc[valid]) and np.all(fc[valid]) and np.all(bclean) and np.all(fclean))
    contrast=compare_parity_contrast(db,eb,df,ef)
    rb=summarize_delta_t_windows(np.asarray(b["T"])-b["T_clean"],eb,valid,cell_valid=bc & bclean)
    rf=summarize_delta_t_windows(np.asarray(f["T"])-f["T_clean"],ef,valid,cell_valid=fc & fclean)
    rc=compare_endpoint_summaries(rb,rf,kind="response")
    pb=detect_peaks(b["T_clean"],eb) if np.all(bclean) else []
    pf=detect_peaks(f["T_clean"],ef) if np.all(fclean) else []
    pos=np.flatnonzero(valid) if positions is None else np.asarray(positions,dtype=int)
    rows=[]
    for p in pos:
        if p<0 or p>=len(valid): raise ValueError("robustness position out of range")
        if not valid[p]:continue
        rows.append(compare_resonance_spectra(np.where(bc[p],b["T"][p],np.nan),eb,
            np.where(fc[p],f["T"][p],np.nan),ef,t_clean_baseline=np.where(bclean,b["T_clean"],np.nan),
            t_clean_fine=np.where(fclean,f["T_clean"],np.nan),position=int(p),mode=mode,parity=int(parity[p]),
            clean_peaks_baseline=pb,clean_peaks_fine=pf))
    full_positions=set(int(p) for p in pos if valid[p])==set(np.flatnonzero(valid))
    numerical="pass" if complete else "unresolved"
    gate=classify_robustness(contrast,rows,response_gate=rc["gate"],numerical_gate=numerical)
    if not full_positions:gate=worst_gate([gate,"unresolved"])
    return dict(
        mode=mode,
        generation=geometry.generation,
        n_energy_baseline=len(eb),
        n_energy_fine=len(ef),
        contrast=contrast,
        resonance=rows,
        D_baseline=db,
        D_fine=df,
        energies_baseline=eb,
        energies_fine=ef,
        contrast_gate=classify_contrast_robustness(contrast),
        primary_gate=gate,
        numerical_gate=numerical,
        response_baseline={w:asdict(v) for w,v in rb.items()},
        response_fine={w:asdict(v) for w,v in rf.items()},
        response_comparison=rc,
        all_valid_resonance_positions=full_positions,
        workflow="actual baseline -> nested midpoint refinement -> parity, response, peak identities/shifts")

def analyze_generation_robustness(generation,*,geometry=None,modes=None,energies_baseline=None, energies_fine=None,baseline_bundle=None):
    geom=build_geometry(generation) if geometry is None else geometry
    wanted=list(cfg.DEFECT_MODES if modes is None else modes)
    eb=np.asarray(baseline_bundle["energies"] if baseline_bundle is not None else (cfg.energy_grid() if energies_baseline is None else energies_baseline),dtype=float)
    if energies_baseline is not None and not np.array_equal(eb,energies_baseline):
        raise ValueError("saved baseline differs from requested robustness grid")
    ef=nested_fine_grid(eb) if energies_fine is None else np.asarray(energies_fine,dtype=float)
    results={};base={};fine={}
    with TemporaryDirectory(prefix="fibonacci_robustness_") as tmp:
        for mode in wanted:
            if baseline_bundle is None:
                base[mode]=fast_sweep(mode,eb,geometry=geom,spill_dir=Path(tmp)/"base"/mode)
            else:
                original=baseline_bundle["modes"][mode]
                base[mode]={
                    **original,
                    "T_clean":baseline_bundle["T_clean"],
                    "log_T_clean":baseline_bundle["log_T_clean"],
                    "clean_status":baseline_bundle["clean_status"],
                    "energies":eb
                }
            fine[mode]=fast_sweep(mode,ef,geometry=geom,spill_dir=Path(tmp)/"fine"/mode)
            results[mode]=robustness_check_mode(geom,mode,energies_baseline=eb,energies_fine=ef, baseline_sweep=base[mode],fine_sweep=fine[mode])
        interaction_gate="not_applicable";interaction={}
        if all(m in wanted for m in cfg.ONE_SITE_MODES):
            summaries=[]
            for sweeps,energies in ((base,eb),(fine,ef)):
                first=next(iter(sweeps.values()))
                rem=interaction_from_sweeps(sweeps,t_clean=first["T_clean"],log_t_clean=first["log_T_clean"],
                    status_clean=first["clean_status"],energies=energies)
                summaries.append(interaction_summary(rem));del rem
            comparison=compare_endpoint_summaries(*summaries,kind="interaction")
            interaction_gate=comparison["gate"]
            interaction=dict(baseline=summaries[0],fine=summaries[1],**comparison)
        primary=worst_gate([v["primary_gate"] for v in results.values()]+[interaction_gate])
    first_mode = wanted[0]
    bb, ff = base[first_mode], fine[first_mode]
    cb = trusted_transmission_mask(bb["clean_status"], bb["T_clean"], bb["log_T_clean"])
    cf = trusted_transmission_mask(ff["clean_status"], ff["T_clean"], ff["log_T_clean"])
    clean_complete = bool(np.all(cb) and np.all(cf))
    clean_base = [asdict(p) for p in detect_peaks(bb["T_clean"], eb)] if np.all(cb) else []
    clean_fine = [asdict(p) for p in detect_peaks(ff["T_clean"], ef)] if np.all(cf) else []
    return dict(
        clean_peaks_baseline=clean_base,
        clean_peaks_fine=clean_fine,
        clean_peaks_complete=clean_complete,
        generation=geom.generation,
        modes=results,
        geometry_metadata=geom.to_metadata(),
        interaction=interaction,
        interaction_gate=interaction_gate,
        primary_gate=primary,
        gate_tolerances=dict(cfg.GATE_TOLERANCES)
    )