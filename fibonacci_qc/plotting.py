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
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.25
    ax.bar(x - width, signed, width, label="signed", color="#1f4e79")
    ax.bar(x, abs_vals, width, label="abs", color="#c45c26")
    ax.bar(x + width, rms, width, label="rms", color="#2a6f6f")
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=20, ha="right")
    ax.set_ylabel("Integrated value")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
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
        if support is not None:
            y = np.where(support, y, np.nan)
        ax.plot(energies, y, lw=1.3, label=mode)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\alpha_{\mathrm{eff}}$")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    if path is not None:
        save_figure(fig, path)
    return fig

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
) -> Dict[str, str]:
    "Write standard figure set from available analysis products"
    out_root = Path(output_dir)
    if not out_root.is_absolute():
        out_root = cfg.OUTPUT_FIGURES / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    if raw_bundle is not None:
        energies = np.asarray(raw_bundle["energies"], dtype=np.float64)
        t_clean = np.asarray(raw_bundle["T_clean"], dtype=np.float64)
        written["clean"] = str(save_figure(plot_clean_transmission(energies, t_clean), out_root / "clean_transmission.png"))
        modes = raw_bundle.get("modes", {})
        if mode in modes:
            sweep = modes[mode]
            t = np.asarray(sweep["T"], dtype=np.float64)
            positions = np.asarray(sweep.get("positions", np.arange(t.shape[0])), dtype=np.float64)
            written["heatmap"] = str(save_figure(plot_transmission_heatmap(energies, positions, t), out_root / f"heatmap_{mode}.png"))
            mid = t.shape[0] // 2
            written["defect"] = str(save_figure(plot_defect_transmission(energies, t[mid], t_clean=t_clean, title=f"{mode} p={mid}"), out_root / f"defect_{mode}_p{mid}.png"))
    if parity is not None and "unconditional" in parity:
        uncond = parity["unconditional"]
        energies = np.asarray(parity.get("energies", uncond.energies), dtype=np.float64)
        written["parity"] = str(save_figure(plot_parity_curve(energies, uncond.D), out_root / "parity_contrast.png"))
        integrated = parity.get("integrated", {}).get("unconditional", {})
        if integrated:
            labels = list(integrated.keys())
            written["integrated"] = str(save_figure(plot_integrated_metrics(labels, [integrated[k].signed for k in labels], [integrated[k].abs for k in labels], [integrated[k].rms for k in labels]), out_root / "integrated_parity.png"))

    if interaction is not None:
        rem = interaction
        if hasattr(rem, "I"):
            written["interaction"] = str(save_figure(plot_interaction_remainder(rem.energies, rem.positions, rem.I), out_root / "interaction_remainder.png"))

    if resonance is not None and "summary" in resonance:
        by_pos = resonance["summary"]["by_position"]
        positions = np.array([row["position"] for row in by_pos], dtype=np.float64)
        shifts = np.array([row["mean_abs_delta_E"] for row in by_pos], dtype=np.float64)
        written["resonance"] = str(save_figure(plot_resonance_shifts(positions, shifts), out_root / "resonance_shifts.png"))

    if nulls is not None:
        first = next(iter(nulls.values())) if isinstance(nulls, dict) else nulls
        if hasattr(first, "null_S_max"):
            written["nulls"] = str(save_figure(plot_null_histogram(first.null_S_max, first.observed.S_max, title="Permutation null for S_max", xlabel=r"$S_{\max}$"), out_root / "null_S_max.png"))

    if scaling is not None and "windows" in scaling:
        windows = scaling["windows"]
        key = next(iter(windows))
        payload = windows[key]
        energies = np.asarray(payload["energies"], dtype=np.float64)
        alphas = {mode_name: fit.alpha for mode_name, fit in payload["modes"].items()}
        written["scaling"] = str(save_figure(plot_scaling_exponents(energies, alphas, support=payload.get("common_support"), title=f"Scaling exponents ({key})"), out_root / "scaling_exponents.png"))

    return written