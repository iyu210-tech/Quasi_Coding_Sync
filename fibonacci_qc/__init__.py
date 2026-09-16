"Configuration"

from pkgutil import extend_path

__path__=extend_path(__path__, __name__)

from fibonacci_qc.config import CONFIG, energy_grid, ensure_output_dirs
from fibonacci_qc.geometry import(
    FibonacciGeometry,
    build_geometry,
    fibonacci_number,
    fibonacci_word,
    iter_study_geometries,
)

from fibonacci_qc.scattering import (
    TransmissionResult,
    clean_transmission,
    clean_transmission_grid,
    interface_s_matrix,
    layer_s_matrix,
    propagation_s_matrix,
    q_number,
    redheffer_star,
    safe_log_t,
    slab_s_matrix,
    stack_s_matrix,
    transfer_to_s_matrix,
    transmission_from_s,
    transmission_from_transfer,
)

from fibonacci_qc.defects import(
    DefectedStack,
    allowed_positions,
    apply_defect,
    apply_phason,
    apply_potential_only,
    apply_substitution,
    apply_width_only,
    validate_defect_position,
)

from fibonacci_qc.reference_solver import (
    DefectTransmissionResult,
    delta_t,
    load_delta_t,
    reference_sweep,
    reference_transmission,
    save_delta_t,
)

from fibonacci_qc.fast_solver import (
    StackCache, fast_sweep, fast_transmission, validate_against_reference
)

from fibonacci_qc.raw_data import (
    load_generation_raw, produce_raw_dataset, run_metadata, save_generation_raw, sweep_generation,
)

from fibonacci_qc.analysis_io import (
    analyze_raw_file, load_parity_analysis, produce_parity_analysis, save_parity_analysis
)

from fibonacci_qc.parity_analysis import (
    IntegratedContrast,
    ParityCheckResult,
    ParityContrastResult,
    analyze_generation_parity,
    analyze_mode_parity,
    conditional_parity_contrasts,
    integrate_contrast,
    integrate_contrast_windows,
    integrated_summary,
    origin_shift_check,
    parity_checks,
    parity_contrast,
    reversal_check,
)

from fibonacci_qc.interaction import (
    InteractionRemainder,
    interaction_from_sweeps,
    interaction_remainder,
    interaction_remainder_log,
    interaction_summary,
)

from fibonacci_qc.resonance import (
    MatchResult,
    Peak,
    PeakMatch,
    ResonanceTrack,
    analyze_generation_resonances,
    analyze_mode_resonances,
    detect_peaks,
    match_peaks_order_preserving,
    summarize_resonance_tracks,
    track_resonances,
)

from fibonacci_qc.resonance_io import (
    analyze_raw_interaction_and_resonance,
    load_interaction_remainder,
    load_resonance_analysis,
    produce_resonance_analysis,
    save_interaction_remainder,
    save_resonance_analysis,
)

from fibonacci_qc.geometry_nulls import (
    GlobalParityStats,
    PermutationNullResult,
    analyze_generation_nulls,
    applicable_strata,
    global_parity_stats,
    null_summary_table,
    permute_parity_within_strata,
    run_geometry_nulls,
    stratified_permutation_null,
)

from fibonacci_qc.nulls_io import (
    load_nulls_analysis,
    load_robustness_analysis,
    produce_nulls_analysis,
    produce_robustness_analysis,
    save_nulls_analysis,
    save_robustness_analysis,
)

from fibonacci_qc.robustness import (
    ContrastRobustness,
    ResonanceRobustness,
    analyze_generation_robustness,
    compare_parity_contrast,
    compare_resonance_spectra,
    finer_energy_grid,
    robustness_check_mode,
)

from fibonacci_qc.scaling import (
    ModeScalingResult,
    ScalingFitPoint,
    analyze_mode_scaling,
    analyze_scaling,
    classify_direction,
    common_support_mask,
    fit_loglog_scaling,
    fit_scaling_curve,
    paired_alpha_contrast,
    scaling_summary,
)

from fibonacci_qc.scaling_io import (
    load_scaling_analysis,
    produce_scaling_analysis,
    save_scaling_analysis,
)

__all__=[
    "CONFIG",
    "FibonacciGeometry",
    "build_geometry",
    "energy_grid",
    "ensure_output_dirs",
    "fibonacci_number",
    "fibonacci_word",
    "iter_study_geometries",
    "TransmissionResult",
    "clean_transmission",
    "clean_transmission_grid",
    "interface_s_matrix",
    "layer_s_matrix",
    "propagation_s_matrix",
    "q_number",
    "redheffer_star",
    "safe_log_t",
    "slab_s_matrix",
    "stack_s_matrix",
    "transfer_to_s_matrix",
    "transmission_from_s",
    "transmission_from_transfer",
    "DefectedStack",
    "DefectTransmissionResult",
    "allowed_positions",
    "apply_defect"
    "apply_phason",
    "apply_potential_only",
    "apply_substitution",
    "apply_width_only",
    "delta_t",
    "load_delta_t"
    "reference_sweep"
    "reference_transmission",
    "save_delta_t",
    "validate_defect_position",
    "StackCache",
    "fast_sweep",
    "fast_transmission",
    "validate_against_reference",
    "load_generation_raw",
    "produce_raw_dataset",
    "run_metadata",
    "save_generation_raw",
    "sweep_generation",
    "IntegratedContrast",
    "ParityCheckResult",
    "ParityContrastResult",
    "analyze_generation_parity",
    "analyze_mode_parity",
    "analyze_raw_file",
    "conditional_parity_contrasts",
    "integrate_contrast",
    "integrate_contrast_windows",
    "integrated_summary",
    "load_parity_analysis",
    "origin_shift_check",
    "parity_checks",
    "parity_contrast",
    "produce_parity_analysis",
    "reversal_check",
    "save_parity_analysis",
    "InteractionRemainder",
    "MatchResult",
    "Peak",
    "PeakMatch",
    "ResonanceTrack",
    "analyze_generation_resonances",
    "analyze_mode_resonances",
    "analyze_raw_interaction_and_resonance",
    "detect_peaks",
    "interaction_from_sweeps",
    "interaction_remainder",
    "interaction_remainder_log",
    "interaction_summary",
    "load_interaction_remainder",
    "load_resonance_analysis",
    "match_peaks_order_preserving",
    "produce_section6_analysis",
    "save_interaction_remainder",
    "save_resonance_analysis",
    "summarize_resonance_tracks",
    "track_resonances",
    "GlobalParityStats",
    "PermutationNullResult",
    "ContrastRobustness",
    "ResonanceRobustness",
    "analyze_generation_nulls",
    "analyze_generation_robustness",
    "applicable_strata",
    "compare_parity_contrast",
    "compare_resonance_spectra",
    "finer_energy_grid",
    "global_parity_stats",
    "load_nulls_analysis",
    "load_robustness_analysis",
    "null_summary_table",
    "permute_parity_within_strata",
    "produce_nulls_analysis",
    "produce_robustness_analysis",
    "robustness_check_mode",
    "run_geometry_nulls",
    "save_nulls_analysis",
    "save_robustness_analysis",
    "stratified_permutation_null",
    "ModeScalingResult",
    "ScalingFitPoint",
    "analyze_mode_scaling",
    "analyze_scaling",
    "classify_direction",
    "common_support_mask",
    "fit_loglog_scaling",
    "fit_scaling_curve",
    "load_scaling_analysis",
    "paired_alpha_contrast",
    "produce_scaling_analysis",
    "save_scaling_analysis",
    "scaling_summary",
]