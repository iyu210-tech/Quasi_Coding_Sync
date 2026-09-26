# Fibonacci quasicrystal transport

Finite size transmission through a 1D Fibonacci stack with four local defects (`substitution`, `phason`, `potential_only`, and `width_only`)

## Profiles

`run_pipeline(profile="smoke")` uses generations 4-7, is only substitution and width only, 12 energies, 12 permuation replicates. Writes new timestamped directory.

`run_pipeline(profile="production")` uses generations 10-20, is all four defect modes, 2000 energies, 500 permutation replicates. `verify_reproducibility` checks the two runs agree.

## Commands

Console scripts:

`fibonacci-qc-raw` writes `output/raw/<run>/nXX/<mode>.npz` and a manifest.
`fibonacci-qc-analyze` reads a raw run, writing parity, resonance, nulls, scaling under `output/analysis/`.
`fibonacci-qc-figures` reads that run's products, checking `source_raw_dir`

## Frozen check

`fibonacci_qc/fixtures/regression_n04_substitution.npz` is generation 4 substitution reference, tests load it. Regenerate using `write_regression_fixture()` call, recommit the `.npz` and `.meta.json`.

## Environment

Install from `requirements.lock`. Saved runs record `config_sha256` and a freeze tag (hash, package version, git commit, sha256).