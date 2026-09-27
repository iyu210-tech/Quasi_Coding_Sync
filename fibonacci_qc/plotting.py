"""
Plotting helpers

Non-interactive Matplotlib lets figures be written in batch jobs & tests w/o display
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union
import numpy as np

from fibonacci_qc import config as cfg

PathLike = Union[str, Path]

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _resolve_figure_path(path: PathLike) -> Path:
    cfg.ensure_output_dirs()
    out = Path(path)
    if not out.is_absolute():
        out = cfg.OUTPUT_FIGURES / out
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def save_figure(fig, path: PathLike, *, dpi: int = 150) -> Path:
    "Save and close Matplotlib figure"
    out = _resolve_figure_path(path)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    _plt().close(fig)
    return out

def plot_clean_transmission(
    energies: np.ndarray,
    t_clean: np.ndarray,
    *,
    path: Optional[PathLike] = None,
    title: str = "Clean transmission",
) -> object:
    "Line plot of T_clean(E)"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(energies, t_clean, color="#1f4e79", lw=1.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$T_{\mathrm{clean}}$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_defect_transmission(
    energies: np.ndarray,
    t_defect: np.ndarray,
    *,
    t_clean: Optional[np.ndarray] = None,
    path: Optional[PathLike] = None,
    title: str = "Defect transmission",
) -> object:
    "Line plot of defected T(E)"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    if t_clean is not None:
        ax.plot(energies, t_clean, color="#9aa0a6", lw=1.2, label="clean")
    ax.plot(energies, t_defect, color="#c45c26", lw=1.5, label="defect")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$T$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_transmission_heatmap(
    energies: np.ndarray,
    positions: np.ndarray,
    transmission: np.ndarray,
    *,
    path: Optional[PathLike] = None,
    title: str = r"$T(p, E)$",
    log: bool = True,
) -> object:
    "Heatmap of transmission over position and energy"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = np.asarray(transmission, dtype=np.float64)
    plot_data = np.log10(np.clip(data, 1e-300, None)) if log else data
    extent = [
        float(energies[0]),
        float(energies[-1]),
        float(positions[0]) - 0.5,
        float(positions[-1]) + 0.5,
    ]
    im = ax.imshow(plot_data, aspect="auto", origin="lower", extent=extent, cmap="magma")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10} T$" if log else r"$T$")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Position $p$")
    ax.set_title(title)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_parity_curve(
    energies: np.ndarray,
    contrast: np.ndarray,
    *,
    path: Optional[PathLike] = None,
    title: str = r"Parity contrast $D_n(E)$",
) -> object:
    "Line plot of D(E)"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.plot(energies, contrast, color="#2a6f6f", lw=1.4)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$D_n(E)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_integrated_metrics(
    labels: Sequence[str],
    signed: Sequence[float],
    abs_vals: Sequence[float],
    rms: Sequence[float],
    *,
    path: Optional[PathLike] = None,
    title: str = "Integrated parity metrics",
) -> object:
    "Grouped bar chart of signed/abs/RMS integrals"
    plt = _plt()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.35
    ax0.bar(x - width / 2, signed, width, label="signed", color="#1f4e79")
    ax0.bar(x + width / 2, abs_vals, width, label="abs", color="#c45c26")
    ax0.set_xticks(x)
    ax0.set_xticklabels(list(labels), rotation=20, ha="right")
    ax0.set_ylabel(r"Integral (eV)")
    ax0.set_title("Signed / absolute")
    ax0.legend(frameon=False)
    ax0.grid(True, axis="y", alpha=0.3)
    ax1.bar(x, rms, 0.5, label="rms", color="#2a6f6f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(labels), rotation=20, ha="right")
    ax1.set_ylabel("RMS (dimensionless)")
    ax1.set_title("RMS")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_resonance_shifts(
    positions: np.ndarray,
    mean_abs_delta_e: np.ndarray,
    *,
    path: Optional[PathLike] = None,
    title: str = r"Mean $|\Delta E|$ by position",
) -> object:
    "Bar or stem plot of mean absolute resonance shifts"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(positions, mean_abs_delta_e, basefmt=" ")
    ax.set_xlabel("Position $p$")
    ax.set_ylabel(r"Mean $|\Delta E|$ (eV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_interaction_remainder(
    energies: np.ndarray,
    positions: np.ndarray,
    remainder: np.ndarray,
    *,
    path: Optional[PathLike] = None,
    title: str = r"Interaction remainder $I_p(E)$",
) -> object:
    "Heatmap of non-additivity remainder"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    extent = [
        float(energies[0]),
        float(energies[-1]),
        float(positions[0]) - 0.5,
        float(positions[-1]) + 0.5,
    ]
    vmax = np.nanmax(np.abs(remainder))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(remainder, aspect="auto", origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$I$")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Position $p$")
    ax.set_title(title)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_null_histogram(
    null_samples: np.ndarray,
    observed: float,
    *,
    path: Optional[PathLike] = None,
    title: str = r"Permutation null for $S$",
    xlabel: str = "S",
) -> object:
    "Histogram of null replicates"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    samples = np.asarray(null_samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    ax.hist(samples, bins=min(30, max(10, samples.size // 5)), color="#9aa0a6", edgecolor="white")
    ax.axvline(observed, color="#c45c26", lw=2.0, label="observed")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_scaling_exponents(
    energies: np.ndarray,
    alphas: Mapping[str, np.ndarray],
    *,
    support: Optional[np.ndarray] = None,
    stable: Optional[np.ndarray] = None,
    path: Optional[PathLike] = None,
    title: str = r"Effective exponents $\alpha_{\mathrm{eff}}(E)$",
) -> object:
    "Overlay alpha_eff(E) curves for at least one defect mode"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.axhline(cfg.ALPHA_FLAT_THRESHOLD, color="#bbbbbb", lw=0.8, ls="--")
    ax.axhline(-cfg.ALPHA_FLAT_THRESHOLD, color="#bbbbbb", lw=0.8, ls="--")
    for mode, alpha in alphas.items():
        y = np.asarray(alpha, dtype=np.float64).copy()
        if support is not None and stable is not None:
            y_feas = np.where(support, y, np.nan)
            y_stab = np.where(stable, y, np.nan)
            ax.plot(energies, y_feas, lw=1.0, ls="--", alpha=0.5, label=f"{mode} feasible")
            ax.plot(energies, y_stab, lw=1.5, label=f"{mode} stable")
        elif support is not None:
            ax.plot(energies, np.where(support, y, np.nan), lw=1.3, label=mode)
        else:
            ax.plot(energies, y, lw=1.3, label=mode)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\alpha_{\mathrm{eff}}$")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    if path is not None:
        save_figure(fig, path)
    return fig

def plot_paired_deltas(energies, deltas, *, supports=None, path=None, title=r"Paired $\Delta\alpha(E)$"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="#888888", lw=0.8)
    for name, delta in deltas.items():
        y = np.asarray(delta, dtype=np.float64)
        support = None if supports is None else supports.get(name)
        if support is not None:
            y = np.where(np.asarray(support, dtype=bool), y, np.nan)
        ax.plot(energies, y, lw=1.3, label=name)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\Delta\alpha$")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    if path is not None:
        save_figure(fig, path)
    return fig

def _figure_name(stem, *, run_id, generation, mode) -> str:
    parts = [stem]
    if run_id:
        parts.append(str(run_id))
    if generation is not None:
        parts.append(f"n{int(generation):02d}")
    if mode:
        parts.append(str(mode))
    return "_".join(parts) + ".png"

def _first_valid_row(sweep) -> Optional[int]:
    transmission = np.asarray(sweep["T"])
    valid = np.asarray(sweep.get("valid", np.ones(transmission.shape[0], dtype=bool)), dtype=bool)
    usable = np.flatnonzero(valid)
    if usable.size == 0:
        return None
    return int(usable[0])

def _iter_null_results(nulls):
    if hasattr(nulls, "null_S_max"):
        yield "null", nulls
        return
    for key, value in nulls.items():
        if hasattr(value, "null_S_max"):
            yield str(key), value
        elif isinstance(value, dict):
            for family, result in value.items():
                if hasattr(result, "null_S_max"):
                    yield f"{key}_{family}", result

def produce_figures(
    *,
    raw_bundle: Optional[Mapping[str, object]] = None,
    parity: Optional[Mapping[str, object]] = None,
    interaction: Optional[Mapping[str, object]] = None,
    resonance: Optional[Mapping[str, object]] = None,
    nulls: Optional[Mapping[str, object]] = None,
    scaling: Optional[Mapping[str, object]] = None,
    output_dir: PathLike = "study",
    mode: str = cfg.MODE_SUBSTITUTION,
    run_id: Optional[str] = None,
    generation: Optional[int] = None,
) -> Dict[str, str]:
    "Write standard figure set from available analysis products"
    out_root = Path(output_dir)
    if not out_root.is_absolute():
        out_root = cfg.OUTPUT_FIGURES / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    if raw_bundle is not None and generation is None and "generation" in raw_bundle:
        generation = int(raw_bundle["generation"])
    if raw_bundle is not None:
        energies = np.asarray(raw_bundle["energies"], dtype=np.float64)
        t_clean = np.asarray(raw_bundle["T_clean"], dtype=np.float64)
        written["clean"] = str(save_figure(plot_clean_transmission(energies, t_clean), out_root / _figure_name("clean_transmission", run_id=run_id, generation=generation, mode=None)))
        for mode_name, sweep in dict(raw_bundle.get("modes", {})).items():
            transmission = np.asarray(sweep["T"], dtype=np.float64)
            positions = np.asarray(sweep.get("positions", np.arange(transmission.shape[0])), dtype=np.float64)
            heat_key = "heatmap" if mode_name == mode else f"heatmap_{mode_name}"
            written[heat_key] = str(save_figure(plot_transmission_heatmap(energies, positions, transmission), out_root / _figure_name("heatmap", run_id=run_id, generation=generation, mode=mode_name)))
            row = _first_valid_row(sweep)
            if row is None:
                continue
            defect_key = "defect" if mode_name == mode else f"defect_{mode_name}"
            written[defect_key] = str(save_figure(plot_defect_transmission(energies, transmission[row], t_clean=t_clean, title=f"{mode_name} p={row}"), out_root / _figure_name("defect", run_id=run_id, generation=generation, mode=f"{mode_name}_p{row}")))
    if parity is not None:
        mode_payloads = parity["modes"] if isinstance(parity, dict) and "modes" in parity else {mode: parity}
        for mode_name, payload in mode_payloads.items():
            if isinstance(payload, dict) and "unconditional_D" in payload:
                curve = np.asarray(payload["unconditional_D"], dtype=np.float64)
                curve_energies = np.asarray(payload.get("energies", parity.get("energies")), dtype=np.float64)
            elif isinstance(payload, dict) and "unconditional" in payload:
                curve = payload["unconditional"].D
                curve_energies = np.asarray(payload.get("energies"), dtype=np.float64)
            else:
                continue
            written[f"parity_{mode_name}"] = str(save_figure(plot_parity_curve(curve_energies, curve), out_root / _figure_name("parity_contrast", run_id=run_id, generation=generation, mode=mode_name)))
    if interaction is not None:
        arrays = interaction["arrays"] if isinstance(interaction, dict) and "arrays" in interaction else None
        remainder = energies = positions = None
        if arrays is not None and "I" in arrays:
            remainder, energies, positions = arrays["I"], arrays["energies"], arrays["positions"]
        elif hasattr(interaction, "I"):
            remainder, energies, positions = interaction.I, interaction.energies, interaction.positions
        if remainder is not None:
            written["interaction"] = str(save_figure(plot_interaction_remainder(energies, positions, remainder), out_root / _figure_name("interaction_remainder", run_id=run_id, generation=generation, mode=None)))
    if resonance is not None and isinstance(resonance, dict) and "metadata" in resonance:
        for mode_name, payload in resonance["metadata"].get("modes", {}).items():
            by_pos = payload["summary"]["by_position"]
            positions = np.array([r["position"] for r in by_pos], dtype=float)
            shifts = np.array([r["mean_abs_delta_E"] for r in by_pos], dtype=float)
            written[f"resonance_{mode_name}"] = str(save_figure(plot_resonance_shifts(positions, shifts), out_root / _figure_name("resonance_shifts", run_id=run_id, generation=generation, mode=mode_name)))
    if resonance is not None and isinstance(resonance, dict) and "summary" in resonance:
        by_pos = resonance["summary"]["by_position"]
        positions = np.array([row["position"] for row in by_pos], dtype=np.float64)
        shifts = np.array([row["mean_abs_delta_E"] for row in by_pos], dtype=np.float64)
        written["resonance"] = str(save_figure(plot_resonance_shifts(positions, shifts), out_root / _figure_name("resonance_shifts", run_id=run_id, generation=generation, mode=mode)))
    if nulls:
        for label, result in _iter_null_results(nulls):
            written[f"nulls_{label}"] = str(save_figure(plot_null_histogram(result.null_S_max, result.observed.S_max, title=f"Permutation null for S_max ({label})", xlabel=r"$S_{\max}$"), out_root / _figure_name("null_S_max", run_id=run_id, generation=generation, mode=label)))
    if scaling is not None and "windows" in scaling:
        for key, payload in scaling["windows"].items():
            energies = np.asarray(payload["energies"], dtype=np.float64)
            mode_fits = payload["modes"]
            alphas = {mode_name: fit.alpha for mode_name, fit in mode_fits.items()}
            stables = [np.asarray(fit.stable, dtype=bool) for fit in mode_fits.values() if getattr(fit, "stable", None) is not None]
            stable_common = None
            if stables:
                stable_common = stables[0].copy()
                for item in stables[1:]:
                    stable_common &= item
            written[f"scaling_{key}"] = str(save_figure(plot_scaling_exponents(energies, alphas, support=payload.get("common_support"), stable=stable_common, title=f"Scaling exponents ({key})"), out_root / _figure_name("scaling_exponents", run_id=run_id, generation=None, mode=key)))
            paired = payload.get("paired") or {}
            deltas = {name: pair["delta_alpha"] for name, pair in paired.items()}
            if deltas:
                supports = {name: pair.get("support") for name, pair in paired.items()}
                written[f"paired_{key}"] = str(save_figure(plot_paired_deltas(energies, deltas, supports=supports, title=f"Paired delta alpha ({key})"), out_root / _figure_name("paired_delta_alpha", run_id=run_id, generation=None, mode=key)))
    return written