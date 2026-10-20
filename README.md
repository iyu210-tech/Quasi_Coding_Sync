How to run:

1. Install Python 3.13

2. python3.13 -m venv .venv-reproduce

3. source .venv-reproduce/bin/activate

4. pip install -r requirements.lock

5. pip install -e ".[dev]"

6. python -m pytest

7. python tools/validate_release.py --output validation_report.json

8. 
python -u - <<'PY'
import os

for key in (
    'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
):
    os.environ[key] = '1'

import json
from pathlib import Path
import numpy as np
from fibonacci_qc import config as cfg
from fibonacci_qc.robustness import nested_fine_grid
from fibonacci_qc.raw_data import (
    produce_raw_dataset,
    load_generation_raw,
    save_generation_raw,
)
from fibonacci_qc.geometry import build_geometry
from fibonacci_qc.defects import apply_defect
from fibonacci_qc.independent_reference import checked_high_precision
from fibonacci_qc.scattering import (
    STATUS_TO_CODE,
    trusted_transmission_mask,
    exterior_unitarity_residuals,
    q_number,
)

# Existing output directories are not overwritten; no automatic resume.
root = Path('output/reproduced_raw').resolve()
root.mkdir(parents=True, exist_ok=False)
energies = cfg.energy_grid(n_points=2000)

while len(energies) < 31985:
    energies = nested_fine_grid(energies)
    if len(energies) < 7997:
        continue

    for n in range(10, 21):
        job = root / 'original' / f'n{n}_e{len(energies):05d}'
        print(f'Generating n={n}, energies={len(energies)}', flush=True)
        produce_raw_dataset(
            generations=[n],
            modes=cfg.DEFECT_MODES,
            energies_ev=energies,
            output_dir=job,
            validate=True,
        )
        geometry = build_geometry(n)

        for mode in cfg.DEFECT_MODES:
            original = job / f'n{n:02d}' / f'{mode}.npz'
            raw = load_generation_raw(original)
            sw = raw['modes'][mode]

            if not trusted_transmission_mask(
                raw['clean_status'], raw['T_clean'], raw['log_T_clean']
            ).all():
                raise RuntimeError(f'Unusable clean reference: n{n}')

            repairs = []
            for p in np.flatnonzero(sw['valid']):
                bad = np.flatnonzero(~trusted_transmission_mask(
                    sw['status'][p], sw['T'][p], sw['log_T'][p],
                ))

                for j in bad:
                    if (
                        n < 17
                        or energies[j] <= 5
                        or sw['status'][p, j] != STATUS_TO_CODE['non_unitary']
                    ):
                        raise RuntimeError(
                            f'Unexpected rejection: n{n}, {mode}, '
                            f'p={p}, E={energies[j]}'
                        )

                    if len(repairs) >= (50 if n <= 18 else 250):
                        raise RuntimeError('Fallback safety cap exceeded.')

                    defect = apply_defect(mode, geometry, int(p))
                    ref = checked_high_precision(
                        defect.widths_nm,
                        defect.potentials_ev,
                        float(energies[j]),
                        dps_low=80,
                        dps_high=120,
                    )

                    r = exterior_unitarity_residuals(
                        ref['S'],
                        q_number(energies[j], cfg.V_LEAD_EV).item(),
                    )
                    errors = np.array([
                        r['R_plus_T_left'] - 1,
                        r['R_plus_T_right'] - 1,
                        r['unitarity_frob'],
                        r['reciprocity'],
                    ])
                    code = STATUS_TO_CODE[ref['status']]

                    if (
                        not np.isfinite(errors).all()
                        or np.max(np.abs(errors)) > 1e-8
                        or not trusted_transmission_mask(
                            code, ref['T'], ref['log_T']
                        ).item()
                    ):
                        raise RuntimeError(
                            'Independent fallback failed validation.'
                        )

                    sw['T'][p, j] = ref['T']
                    sw['log_T'][p, j] = ref['log_T']
                    sw['status'][p, j] = code

                    if 'cell_ok' in sw:
                        sw['cell_ok'][p, j] = True

                    if 'delta_T' in sw:
                        sw['delta_T'][p, j] = ref['T'] - raw['T_clean'][j]

                    repairs.append(dict(
                        position=int(p),
                        energy_index=int(j),
                        energy_ev=float(energies[j]),
                        old_status='non_unitary',
                        new_status=ref['status'],
                        new_T=ref['T'],
                        new_log_T=ref['log_T'],
                    ))

            selected = original
            if repairs:
                meta = json.loads(
                    original.with_suffix('.meta.json').read_text()
                )
                selected = save_generation_raw(
                    raw,
                    root / 'corrected' / f'n{n}'
                    / f'e{len(energies):05d}' / f'{mode}.npz',
                    metadata=dict(
                        derived_data=True,
                        source_raw_file=str(original),
                        source_data_sha256=meta['data_sha256'],
                        repaired_cells=repairs,
                        independent_reference_precisions=[80, 120],
                        fallback_revision=(
                            'independent-high-precision-cell-fallback-v1'
                        ),
                    ),
                )

            destination = (
                root / 'selected' / f'n{n}' / f'e{len(energies):05d}'
            )
            destination.mkdir(parents=True, exist_ok=True)

            for source in (selected, selected.with_suffix('.meta.json')):
                (destination / source.name).symlink_to(
                    os.path.relpath(source, destination)
                )

            print(f'  {mode}: {len(repairs)} repaired cells', flush=True)
            del raw, sw

print(f'Raw data complete: {root / "selected"}')
PY
