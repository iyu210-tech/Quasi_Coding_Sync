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
]