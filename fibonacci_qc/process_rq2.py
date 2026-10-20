#!/usr/bin/env python3
"Read-only, bounded-memory RQ2 postprocessing of nXX/eNNNNN mode NPZ files."

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

for _name in ('OPENBLAS_NUM_THREADS',
              'OMP_NUM_THREADS',
              'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS',
              'BLIS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'
              ):
    os.environ[_name] = '1'
try:
    import numpy as np
except ImportError as exc:
    raise SystemExit('NumPy is required. Run with the existing .venv-qc313/bin/python.') from exc

VERSION = 'rq2-streaming-1.0.0'
MODES = ('substitution', 'potential_only', 'width_only')
SHORT = dict(zip(MODES, ('sub', 'pot', 'width')))
GRIDS = (7997, 15993, 31985)
WINDOWS = {'full': (0.1, 10.0), 'low': (0.1, 1.0), 'mid': (1.0, 5.0), 'high': (5.0, 10.0)}
MODEL_KEYS = ('length_unit', 'energy_unit', 'hbar2_over_2m_ev_nm2', 'v_lead_ev', 'd_a_nm', 'd_b_nm', 'v_a_ev', 'v_b_ev', 'word_w1', 'word_w2', 'substitution', 'layer_width_nm', 'layer_potential_ev', 'defect_modes', 'site_motif_radius', 'bond_motif_radius', 'lead_left_token', 'lead_right_token')
EXPECTED_SOLVER = 'scaled-redheffer-v1'
PRIMARY = ('mean_abs_I', 'rms_I', 'mean_abs_I_log', 'rms_I_log')
LINEAR_QUANTITIES = ('I', 'delta_sub', 'delta_pot', 'delta_width', 'additive_prediction')
LOG_QUANTITIES = ('I_log', 'dlog_sub', 'dlog_pot', 'dlog_width', 'additive_log_prediction')
GATED_METRICS = tuple(f'{m}_{q}' for q in LINEAR_QUANTITIES + LOG_QUANTITIES for m in ('mean_abs', 'rms'))
RATIOS = {'mean_abs_I_over_sub_response': ('mean_abs_I', ('mean_abs_delta_sub',)), 'rms_I_over_sub_response': ('rms_I', ('rms_delta_sub',)), 'mean_abs_I_over_sum_component_responses': ('mean_abs_I', ('mean_abs_delta_pot', 'mean_abs_delta_width')), 'rms_I_over_sum_component_rms': ('rms_I', ('rms_delta_pot', 'rms_delta_width'))}
STUDY_TOLS = {'interaction': 0.001, 'response': 0.001, 'interaction_log_atol': 0.001, 'interaction_log_rtol': 0.001}

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def load_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)

def canonical(value) -> str:
    return json.dumps(clean_json(value), sort_keys=True, separators=(',', ':'), allow_nan=False)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(clean_json(value), f, indent=2, allow_nan=False)
        f.write('\n')
    tmp.replace(path)

def atomic_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (canonical(v) if isinstance(v, (dict, list, tuple)) else clean_json(v)) for k, v in row.items()})
    tmp.replace(path)

def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def array_hash(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def stamp(path):
    s = Path(path).stat()
    return (s.st_size, s.st_mtime_ns, s.st_ino)

def same_array(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def model_hash(meta):
    rm = meta['run_metadata']
    payload = {k: rm['config'][k] for k in MODEL_KEYS}
    payload['solver_revision'] = rm['solver_revision']
    return hashlib.sha256(canonical(payload).encode()).hexdigest()

def analysis_settings(meta):
    "Read the recorded criteria"
    config = meta['run_metadata']['config']
    win = {k: tuple(float(x) for x in v) for k, v in config['energy_windows'].items()}
    if win != WINDOWS:
        raise ValueError('Unexpected energy windows; this processor implements the four recorded study windows.')
    tols = {k: float(config['gate_tolerances'][k]) for k in STUDY_TOLS}
    if tols != STUDY_TOLS or float(config['numerical_min_coverage']) != 1.0:
        raise ValueError('Recorded coverage/tolerances differ from the original study.')
    bound = float(config['tolerances']['t_bounds_atol'])
    if not math.isfinite(bound) or bound < 0 or bound > 1e-8:
        raise ValueError('Missing or invalid probability-bound allowance.')
    return {'windows': win, 'tolerances': tols, 'min_coverage': 1.0, 't_bound_atol': bound}

def check_zip_names(z):
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError(f'Duplicate archive members in {z.filename}')

def read_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f'Unsupported NPY version {version}; expected numeric NPY 1/2.')
    if dtype.hasobject or dtype.fields or dtype.kind not in 'fiub':
        raise ValueError('Only plain numeric arrays are supported.')
    if any(not isinstance(x, int) or x < 0 for x in shape):
        raise ValueError('Invalid array shape.')
    return shape, order, dtype

def small_array(z, key):
    member = z.getinfo(key + '.npy')
    if member.file_size > 4 * 1024**2:
        raise ValueError(f'{key} is too large for an energy/reference/position vector.')
    with z.open(member, 'r') as f:
        shape, order, dtype = read_header(f)
        n = math.prod(shape)
        expected = n * dtype.itemsize
        if f.tell() + expected != member.file_size:
            raise ValueError(f'{key}: header/data size mismatch.')
        data = f.read(expected)
        if len(data) != expected or f.read(1):
            raise ValueError(f'{key}: truncated or overlong NPY member.')
        return np.frombuffer(data, dtype=dtype).reshape(shape, order='F' if order else 'C').copy()

class RowStream:
    "One sequential numeric C-order NPY member"
    def __init__(self, z, key, shape=None):
        self.member = z.getinfo(key + '.npy')
        self.f = z.open(self.member, 'r')
        try:
            self.shape, order, self.dtype = read_header(self.f)
            if len(self.shape) != 2 or order:
                raise ValueError(f'{key}: expected a C-order two-dimensional matrix.')
            if shape is not None and self.shape != shape:
                raise ValueError(f'{key}: shape {self.shape} does not match {shape}.')
            if self.f.tell() + math.prod(self.shape) * self.dtype.itemsize != self.member.file_size:
                raise ValueError(f'{key}: declared dimensions do not match stored member size.')
            self.position = 0
            self.finished = False
        except BaseException:
            self.f.close()
            raise

    def read_rows(self, count):
        count = min(count, self.shape[0] - self.position)
        size = count * self.shape[1] * self.dtype.itemsize
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError('Truncated raw NPZ matrix.')
        self.position += count
        if self.position == self.shape[0]:
            if self.f.read(1):
                raise ValueError('Unexpected bytes following matrix.')
            self.finished = True
        return np.frombuffer(data, dtype=self.dtype).reshape(count, self.shape[1])

    def close(self):
        self.f.close()

def usable_cells(code, t, logt, bound, *, where):
    "Same usable statuses as the repaired solver."
    if code.dtype.kind not in 'iu' or np.any((code < 0) | (code > 4)):
        raise ValueError(f'{where}: unknown status code. Expected integer codes 0--4.')
    certified = (code == 0) | (code == 2)
    good = (certified & np.isfinite(t) & np.isfinite(logt) & (t >= 0)
            & (t <= 1 + bound) & (logt <= math.log1p(bound))
            & ((code != 2) | (t == 0)))
    if np.any(certified & ~good):
        raise ValueError(f'{where}: an ok/underflow status contradicts its T/log_T.')
    return good

def metadata_record(path, generation, size, mode):
    path = Path(path).resolve()
    sidecar = path.with_suffix('.meta.json')
    meta = load_json(sidecar)
    rm = meta['run_metadata']
    if int(meta['generation']) != generation or int(rm['n_energy']) != size:
        raise ValueError(f'Folder and metadata generation/grid disagree: {path}')
    if rm.get('schema_version') != 2 or rm.get('solver_revision') != EXPECTED_SOLVER:
        raise ValueError(f'Unsupported raw schema/solver in {sidecar}')
    expected = meta.get('data_sha256', '')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError(f'Raw SHA-256 is missing/invalid in {sidecar}')
    if model_hash(meta) != rm.get('config_sha256'):
        raise ValueError(f'Config snapshot/fingerprint mismatch: {sidecar}')
    settings = analysis_settings(meta)
    geom = meta.get('geometry_metadata', {})
    if (geom.get('generation') != generation or geom.get('n_sites') != fibonacci(generation)
            or geom.get('n_bonds') != fibonacci(generation)-1):
        raise ValueError(f'Geometry metadata disagree with the stated Fibonacci generation: {path}')
    if (rm['config']['word_w1'], rm['config']['word_w2']) != ('B', 'A'):
        raise ValueError('Unexpected Fibonacci generation convention.')
    with zipfile.ZipFile(path, 'r') as z:
        check_zip_names(z)
        if small_array(z, 'generation').item() != generation:
            raise ValueError(f'NPZ generation differs from its folder: {path}')
        e = small_array(z, 'energies')
        if (e.shape != (size,) or e.dtype.kind != 'f' or e.dtype.itemsize != 8
                or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0)
                or e[0] != 0.1 or e[-1] != 10.0):
            raise ValueError(f'Invalid or unexpected energy grid: {path}')
        if array_hash(e) != rm['energy_sha256']:
            raise ValueError(f'Energy fingerprint mismatch: {path}')
        p = small_array(z, mode + '__positions')
        valid = small_array(z, mode + '__valid')
        n = fibonacci(generation)
        if (p.shape != (n,) or p.dtype.kind not in 'iu'
                or not np.array_equal(p, np.arange(n)) or valid.shape != p.shape
                or not np.all(valid == 1)):
            raise ValueError(f'One-site modes must retain all original layer indices: {path}')
        refs = {k: small_array(z, k) for k in ('T_clean', 'log_T_clean', 'clean_status')}
        if any(v.shape != e.shape for v in refs.values()):
            raise ValueError(f'Clean-reference vectors have incorrect axes: {path}')
        for k in ('T_clean', 'log_T_clean'):
            if refs[k].dtype.kind != 'f' or refs[k].dtype.itemsize != 8:
                raise ValueError(f'{k}: expected float64 reference data.')
        usable_cells(refs['clean_status'], refs['T_clean'], refs['log_T_clean'], settings['t_bound_atol'], where=str(path) + '/clean')
        for key, value in {'energies': e, **refs}.items():
            redundant = mode + '__' + key
            if redundant + '.npy' in z.namelist():
                if not same_array(small_array(z, redundant), value):
                    raise ValueError(f'Redundant {redundant} differs from the authoritative reference: {path}')
        for key in ('T', 'log_T', 'status'):
            stream = RowStream(z, mode + '__' + key, shape=(n, size))
            try:
                if key != 'status' and (stream.dtype.kind != 'f' or stream.dtype.itemsize != 8):
                    raise ValueError(f'{path}:{key} must be float64.')
                if key == 'status' and stream.dtype.kind not in 'iu':
                    raise ValueError(f'{path}:status must contain encoded integers.')
            finally:
                stream.close()
    repaired_cells = rm.get('repaired_cells', rm.get('repair_cells', []))
    repaired = bool(rm.get('repair_revision') or rm.get('fallback_revision') or repaired_cells)
    rec = {'generation': generation,
           'n_energy': size,
           'mode': mode,
           'path': str(path),
           'meta_path': str(sidecar),
           'data_sha256': expected.lower(),
           'meta_sha256': file_hash(sidecar),
           'bytes': path.stat().st_size,
           'run_id': rm.get('run_id'),
           'config_sha256': rm['config_sha256'],
           'schema_version': rm['schema_version'],
           'solver_revision': rm['solver_revision'],
           'energy_sha256': array_hash(e),
           'geometry_sha256': hashlib.sha256(canonical(geom).encode()).hexdigest(),
           'uses_corrected_raw': repaired,
           'recorded_repaired_cell_count': len(repaired_cells),
           'repair_revision': rm.get('repair_revision', rm.get('fallback_revision')),
           'source_parent_file': rm.get('source_raw_file', rm.get('repair_source_raw_file')),
           'source_parent_sha256': rm.get('source_data_sha256', rm.get('repair_source_raw_sha256')),
           'settings': settings,
           'geometry_metadata': geom
           }
    return rec, e, refs

def preflight(root, generations, grids):
    "Read headers/small arrays only. Fail before heavy I/O on missing or incompatible pairs."
    missing = []
    for g in generations:
        for size in grids:
            for mode in MODES:
                for suffix in ('.npz', '.meta.json'):
                    path = root / f'n{g}' / f'e{size:05d}' / (mode + suffix)
                    if not path.is_file():
                        missing.append(str(path))
    if missing:
        raise FileNotFoundError('Missing input files (no simulation is needed):\n' + '\n'.join(missing))
    jobs = []
    model_ids, settings_ids, global_grids = set(), set(), {}
    for g in generations:
        per_gen_grids = []
        for size in grids:
            sources = []
            anchor_e = anchor_refs = anchor_geom = None
            for mode in MODES:
                rec, e, refs = metadata_record(root / f'n{g}' / f'e{size:05d}' / (mode + '.npz'), g, size, mode)
                model_ids.add(rec['config_sha256'])
                settings_ids.add(canonical(rec['settings']))
                if anchor_e is not None:
                    if not same_array(e, anchor_e) or rec['geometry_metadata'] != anchor_geom:
                        raise ValueError(f'Mode energy/geometry mismatch at n{g}/e{size}.')
                    for key in refs:
                        if not same_array(refs[key], anchor_refs[key]):
                            raise ValueError(f'Mode {key} mismatch at n{g}/e{size}.')
                else:
                    anchor_e, anchor_refs, anchor_geom = e, refs, rec['geometry_metadata']
                sources.append(rec)
            if size in global_grids and not same_array(global_grids[size], anchor_e):
                raise ValueError(f'Generation n{g} uses different energy coordinates at e{size}.')
            global_grids[size] = anchor_e.copy()
            per_gen_grids.append(anchor_e)
            jobs.append({
                        'generation': g,
                        'n_energy': size,
                        'sources': sources,
                        'energy_sha256': array_hash(anchor_e),
                        'energies': anchor_e,
                        'refs': anchor_refs,
                        'settings': sources[0]['settings']
                        })
        for low, fine in zip(per_gen_grids, per_gen_grids[1:]):
            if not same_array(low, fine[::2]):
                raise ValueError(f'n{g}: successive grids are not EXACTLY nested. Original coordinates preserved.')
    if len(model_ids) != 1 or len(settings_ids) != 1:
        raise ValueError('Different physical models or numerical criteria in the input collection.')
    return jobs

@dataclass
class Moments:
    "Streaming mean/mean-absolute/scaled RMS."
    count: int = 0
    sum_abs: float = 0.0
    sum_signed: float = 0.0
    comp_abs: float = 0.0
    comp_signed: float = 0.0
    scale: float = 0.0
    scaled_squares: float = 0.0
    naive_squares: float = 0.0

    def add(self, values):
        x = np.asarray(values, dtype=np.float64)
        if not x.size:
            return
        if not np.all(np.isfinite(x)):
            raise ValueError('Nonfinite values reached the streaming accumulator.')
        a = np.abs(x)
        for field, comp, value in (
            ('sum_abs', 'comp_abs', float(np.sum(a, dtype=np.float64))),
            ('sum_signed', 'comp_signed', float(np.sum(x, dtype=np.float64)))):
            old = getattr(self, field)
            new = old + value
            correction = (old - new) + value if abs(old) >= abs(value) else (value - new) + old
            setattr(self, comp, getattr(self, comp) + correction)
            setattr(self, field, new)
        m = float(np.max(a))
        if m > 0:
            q = float(np.sum(np.square(x / m), dtype=np.float64))
            if self.scale == 0:
                self.scale, self.scaled_squares = m, q
            elif m > self.scale:
                self.scaled_squares = self.scaled_squares * (self.scale / m)**2 + q
                self.scale = m
            else:
                self.scaled_squares += q * (m / self.scale)**2
        with np.errstate(under='ignore', over='ignore'):
            self.naive_squares += float(np.sum(np.square(x), dtype=np.float64))
        self.count += x.size

    def values(self):
        if not self.count:
            return {'mean_abs': None, 'mean': None, 'rms': None, 'legacy_rms': None}
        return {'mean_abs': (self.sum_abs + self.comp_abs) / self.count,
                'mean': (self.sum_signed + self.comp_signed) / self.count,
                'rms': self.scale * math.sqrt(self.scaled_squares / self.count) if self.scale else 0.0,
                'legacy_rms': math.sqrt(self.naive_squares / self.count)}

def process_job(job, chunk_rows):
    g, size = job['generation'], job['n_energy']
    e, refs, bound = job['energies'], job['refs'], job['settings']['t_bound_atol']
    n = fibonacci(g)
    tc, lc, sc = (refs[k] for k in ('T_clean', 'log_T_clean', 'clean_status'))
    clean_good = usable_cells(sc, tc, lc, bound, where=f'n{g}/clean')
    indices = {w: np.flatnonzero((e >= lo) & (e <= hi)) for w, (lo, hi) in WINDOWS.items()}
    stats = {w: {q: Moments() for q in LINEAR_QUANTITIES + LOG_QUANTITIES} for w in WINDOWS}
    counts = {mode: {w: {'ok_cells': 0, 'underflow_cells': 0, 'invalid_cells': 0, 'overflow_cells': 0, 'non_unitary_cells': 0} for w in WINDOWS} for mode in MODES}
    start = time.monotonic()
    before = {r['path']: stamp(r['path']) for r in job['sources']}
    with ExitStack() as stack:
        streams = {}
        for rec in job['sources']:
            z = stack.enter_context(zipfile.ZipFile(rec['path'], 'r'))
            check_zip_names(z)
            members = []
            for key in ('T', 'log_T', 'status'):
                s = RowStream(z, rec['mode'] + '__' + key, shape=(n, size))
                stack.callback(s.close)
                members.append(s)
            streams[rec['mode']] = members
        next_print = 0.0
        for first in range(0, n, chunk_rows):
            arrays, goods = {}, []
            for mode in MODES:
                t, l, code = [s.read_rows(chunk_rows) for s in streams[mode]]
                good = usable_cells(code, t, l, bound, where=f'n{g}/e{size}/{mode}/rows{first}')
                arrays[mode] = (t, l)
                goods.append(good)
                for w, idx in indices.items():
                    codes = code[:, idx]
                    for value, name in enumerate(('ok_cells', 'invalid_cells', 'underflow_cells', 'overflow_cells', 'non_unitary_cells')):
                        counts[mode][w][name] += int(np.count_nonzero(codes == value))
            joint = clean_good[None, :] & goods[0] & goods[1] & goods[2]
            with np.errstate(invalid='ignore', over='ignore', under='ignore'):
                ds, dp, dw = (arrays[m][0] - tc[None, :] for m in MODES)
                ls, lp, lw = (arrays[m][1] - lc[None, :] for m in MODES)
                i = ds - dp - dw
                ilog = ls - lp - lw
                pred, predlog = dp + dw, lp + lw
            linear = joint & np.isfinite(i) & np.isfinite(pred)
            logmask = joint & np.isfinite(ls) & np.isfinite(lp) & np.isfinite(lw) & np.isfinite(ilog) & np.isfinite(predlog)
            for w, idx in indices.items():
                ml = linear[:, idx]
                mg = logmask[:, idx]
                for name, a in zip(LINEAR_QUANTITIES, (i, ds, dp, dw, pred)):
                    stats[w][name].add(a[:, idx][ml])
                for name, a in zip(LOG_QUANTITIES, (ilog, ls, lp, lw, predlog)):
                    stats[w][name].add(a[:, idx][mg])
            elapsed = time.monotonic() - start
            if elapsed >= next_print or first + chunk_rows >= n:
                print(f'  processed {min(n, first+chunk_rows):,}/{n:,} positions ({elapsed/60:.1f} min)', flush=True)
                next_print = elapsed + 30
        if any(not s.finished for ss in streams.values() for s in ss):
            raise ValueError('Some matrix streams were not completely consumed.')
    for rec in job['sources']:
        if stamp(rec['path']) != before[rec['path']] or file_hash(rec['meta_path']) != rec['meta_sha256']:
            raise ValueError(f'Source changed during processing: {rec["path"]}')
    rows, coverage = [], []
    for w, idx in indices.items():
        row = {
            'generation': g,
            'n_layers': n,
            'n_energy': size,
            'window': w,
            'energy_min_ev': WINDOWS[w][0],
            'energy_max_ev': WINDOWS[w][1],
            'energy_samples_in_window': len(idx),
            'n_intended': n * len(idx),
            'n_cells_I': stats[w]['I'].count,
            'n_cells_I_log': stats[w]['I_log'].count,
            'coverage_I': stats[w]['I'].count / (n*len(idx)) if len(idx) else None,
            'coverage_I_log': stats[w]['I_log'].count / (n*len(idx)) if len(idx) else None,
            'energy_sha256': job['energy_sha256'],
            'config_sha256': job['sources'][0]['config_sha256'],
            'uses_any_corrected_raw': any(r['uses_corrected_raw'] for r in job['sources']),
            'original_combined_gate': 'not_reassessed',
            'averaging': 'unweighted_jointly_usable_position_energy_cells'}
        for q, s in stats[w].items():
            v = s.values()
            row[f'mean_abs_{q}'], row[f'rms_{q}'] = v['mean_abs'], v['rms']
            if q in ('I', 'I_log'):
                row[f'mean_{q}'] = v['mean']
                row[f'rms_{q}_legacy_direct_square_diagnostic'] = v['legacy_rms']
                row[f'rms_{q}_legacy_underflow_to_zero'] = bool(v['rms'] is not None and v['rms'] > 0 and v['legacy_rms'] == 0)
        for name, (num, den) in RATIOS.items():
            d = sum(row[k] for k in den) if all(row[k] is not None for k in den) else None
            row[name + '_denominator'] = d
            row[name] = row[num]/d if d is not None and d > 0 and row[num] is not None else None
            row[name + '_small_denominator_warning'] = d is None or d <= 0.001
        for rec in job['sources']:
            mode = rec['mode']
            row[SHORT[mode] + '_data_sha256'] = rec['data_sha256']
            row[SHORT[mode] + '_run_id'] = rec['run_id']
            coverage.append({
                            'generation': g,
                            'n_energy': size,
                            'mode': mode,
                            'window': w,
                            'intended_cells': n*len(idx),
                            **counts[mode][w],
                            'clean_rejected_energy_cells': int(np.count_nonzero(~clean_good[idx])),
                            'clean_underflow_energy_cells': int(np.count_nonzero(sc[idx] == 2)),
                            'joint_usable_cells_linear': stats[w]['I'].count,
                            'joint_usable_cells_log': stats[w]['I_log'].count
                            })
        rows.append(clean_json(row))
    return {
        'generation': g,
        'n_energy': size,
        'rows': rows,
        'coverage_rows': coverage,
        'elapsed_processing_seconds': time.monotonic() - start}

def is_log_metric(metric):
    return any(metric.endswith('_' + q) for q in LOG_QUANTITIES)

def metric_tolerance(metric, a, b, settings):
    t = settings['tolerances']
    if is_log_metric(metric):
        return t['interaction_log_atol'] + t['interaction_log_rtol'] * max(abs(a), abs(b))
    return t['interaction'] if metric in ('mean_abs_I', 'rms_I') else t['response']

def refinement_tables(jobs, processed, grids):
    all_rows = [row for cp in processed for row in cp['rows']]
    lookup = {(r['generation'], r['n_energy'], r['window']): r for r in all_rows}
    jobs_by = {(j['generation'], j['n_energy']): j for j in jobs}
    final, comparisons = [], []
    for g in sorted({j['generation'] for j in jobs}):
        for window in WINDOWS:
            rr = [lookup.get((g, size, window)) for size in grids]
            if rr[-1] is None:
                continue
            f = dict(rr[-1])
            f['three_grids_available'] = all(r is not None for r in rr)
            for metric in GATED_METRICS:
                statuses, changes, tolerances = [], [], []
                coverage_key = 'coverage_I_log' if is_log_metric(metric) else 'coverage_I'
                for k in range(2):
                    low, high = rr[k:k+2]
                    a = low.get(metric) if low else None
                    b = high.get(metric) if high else None
                    present = a is not None and b is not None
                    change = abs(b-a) if present else None
                    tol = metric_tolerance(metric, a, b, jobs_by[(g, grids[k])]['settings']) if present else None
                    full = bool(low and high and low.get(coverage_key) == 1.0 and high.get(coverage_key) == 1.0)
                    status = ('missing' if not present else 'unresolved_coverage' if not full else 'pass' if change <= tol else 'fail_change')
                    row = {
                        'generation': g,
                        'window': window,
                        'metric': metric,
                        'n_energy_baseline': grids[k],
                        'n_energy_fine': grids[k+1],
                        'baseline': a,
                        'fine': b,
                        'abs_change': change,
                        'tolerance': tol,
                        'coverage_baseline': low.get(coverage_key) if low else None,
                        'coverage_fine': high.get(coverage_key) if high else None,
                        'complete_joint_coverage': full,
                        'status': status,
                        'units': 'dimensionless',
                        'scope': 'RQ2-only_postprocessing_check',
                        'original_RQ2_metric': metric in PRIMARY}
                    comparisons.append(row)
                    statuses.append(status); changes.append(change); tolerances.append(tol)
                f[metric + '_previous_change'], f[metric + '_latest_change'] = changes
                f[metric + '_previous_tolerance'], f[metric + '_latest_tolerance'] = tolerances
                f[metric + '_previous_status'], f[metric + '_latest_status'] = statuses
                good = statuses == ['pass', 'pass']
                f[metric + '_passes_both_RQ2_checks'] = good
                d = max(changes) if all(x is not None for x in changes) else None
                f[metric + '_max_observed_change'] = d
                val = f[metric]
                f[metric + '_max_change_over_value'] = d/abs(val) if d is not None and val is not None and val != 0 else None
            f['all_four_primary_RQ2_summaries_pass'] = all(f[k + '_passes_both_RQ2_checks'] for k in PRIMARY)
            for name in RATIOS:
                vals = [r.get(name) if r else None for r in rr]
                d = max(abs(vals[1]-vals[0]), abs(vals[2]-vals[1])) if all(v is not None for v in vals) else None
                f[name + '_max_observed_change'] = d
                f[name + '_max_change_over_value'] = d/abs(vals[-1]) if d is not None and vals[-1] != 0 else None
                f[name + '_inference'] = 'descriptive_ratio_no_preset_ratio_acceptance_threshold'
            f['interpretation'] = ('Same recorded scalar tolerances and full joint coverage')
            final.append(f)
    return all_rows, final, comparisons

def make_plots(final, out, generations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = out / 'figures'
    plots.mkdir(exist_ok=True)
    made = []

    def save(fig, name, note):
        fig.set_size_inches(10.8, 6.6)
        fig.subplots_adjust(left=.11, right=.96, top=.87, bottom=.24)
        fig.text(.11, .055, note, fontsize=9, va='bottom')
        path = plots / name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        made.append(str(path.relative_to(out)))

    high = {r['generation']: r for r in final if r['window'] == 'high'}
    for measure, num, title in [('mean_abs', '01', 'Mean absolute additive-approximation error'), ('rms', '02', 'RMS additive-approximation error')]:
        fig, ax = plt.subplots()
        series = [(f'{measure}_I', 'Remainder I'),
                  (f'{measure}_delta_sub', 'Substitution response'),
                  (f'{measure}_delta_pot', 'Potential-only response'),
                  (f'{measure}_delta_width', 'Width-only response')]
        flags = []
        for metric, label in series:
            vals = [high.get(g, {}).get(metric) for g in generations]
            y = [float(v) if v is not None else np.nan for v in vals]
            good = [k for k, g in enumerate(generations) if high.get(g, {}).get(metric + '_passes_both_RQ2_checks')]
            bad = [k for k in range(len(generations)) if k not in good and math.isfinite(y[k])]
            ax.plot(generations, y, marker='o', markevery=good, label=label)
            flags.append((y, bad))
        ax.set_prop_cycle(None)
        for y, bad in flags:
            ax.plot([generations[k] for k in bad], [y[k] for k in bad], linestyle='none', marker='o', fillstyle='none', markersize=7)
        ax.set(title=title + ' | High window, 5–10 eV', xlabel='Fibonacci generation n', ylabel='Dimensionless magnitude')
        ax.set_xticks(generations); ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.legend(frameon=False)
        save(fig, f'{num}_{measure}_high.png', 'Finest saved grid; component responses use the same joint cells as the remainder.\n' 'Filled: this scalar passed both RQ2/companion checks. Open: check failed or unresolved.\n')

    fig, ax = plt.subplots()
    for metric, label in [('mean_abs_I_over_sub_response', 'Mean |I| / mean |substitution response|'), ('mean_abs_I_over_sum_component_responses', 'Mean |I| / sum of component mean magnitudes')]:
        vals = [high.get(g, {}).get(metric) for g in generations]
        ax.plot(generations, [v if v is not None else np.nan for v in vals], marker='o', label=label)
    ax.set(title='Relative size of the additive-approximation error | High window',
           xlabel='Fibonacci generation n', ylabel='Dimensionless descriptive ratio')
    ax.set_xticks(generations); ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
    save(fig, '03_relative_error_high.png', 'Ratios are calculated AFTER paired cell-level remainders. Denominators are reported in the CSV.\n Missing/zero denominators are unavailable')

    fig, ax = plt.subplots()
    flagged = []
    for window in ('low', 'mid', 'high', 'full'):
        bygen = {r['generation']: r for r in final if r['window'] == window}
        metric = 'mean_abs_I_log'
        y = [bygen.get(g, {}).get(metric) for g in generations]
        y = [v if v is not None else np.nan for v in y]
        good = [k for k,g in enumerate(generations) if bygen.get(g, {}).get(metric + '_passes_both_RQ2_checks')]
        bad = [k for k in range(len(generations)) if k not in good and math.isfinite(y[k])]
        ax.plot(generations, y, marker='o', markevery=good, label=window.capitalize())
        flagged.append((y, bad))
    ax.set_prop_cycle(None)
    for y, bad in flagged:
        ax.plot([generations[k] for k in bad], [y[k] for k in bad], linestyle='none', marker='o', fillstyle='none')
    ax.set(title='Mean absolute logarithmic remainder by energy window', xlabel='Fibonacci generation n',
           ylabel='Mean |I_log| (natural-log units; dimensionless)')
    ax.set_xticks(generations); ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.legend(frameon=False)
    save(fig, '04_log_remainder_by_window.png',
         'Uses the retained natural logarithms, including usable linear-T underflow.\n'
         'This measures departure from multiplicativity.\n'
         'Filled/open symbols apply the original log absolute-plus-relative tolerance to both comparisons.')
    return made

README = ''

def final_exports(out, jobs, processed, generations, grids, plots=True):
    all_rows, final, comparisons = refinement_tables(jobs, processed, grids)
    atomic_csv(out/'rq2_all_grids.csv', all_rows)
    atomic_csv(out/'rq2_finest_with_refinement.csv', final)
    main_fields = ('generation',
                   'n_layers',
                   'window',
                   'n_energy',
                   'mean_abs_I',
                   'rms_I',
                   'mean_abs_I_log',
                   'rms_I_log',
                   'mean_abs_delta_sub',
                   'mean_abs_delta_pot',
                   'mean_abs_delta_width',
                   'rms_delta_sub',
                   'rms_delta_pot',
                   'rms_delta_width',
                   'mean_abs_I_over_sub_response',
                   'mean_abs_I_over_sum_component_responses',
                   'coverage_I',
                   'coverage_I_log',
                   'uses_any_corrected_raw')
    main_fields += tuple(k + '_passes_both_RQ2_checks' for k in PRIMARY)
    atomic_csv(out/'rq2_main_table.csv', [{k:r[k] for k in main_fields} for r in final])
    atomic_csv(out/'rq2_refinement_checks.csv', comparisons)
    atomic_csv(out/'rq2_coverage.csv', [r for cp in processed for r in cp['coverage_rows']])
    files = []
    for cp in processed:
        for rec in cp['sources']:
            files.append({k:v for k,v in rec.items() if k not in ('settings','geometry_metadata')})
    atomic_csv(out/'rq2_source_files.csv', files)
    (out/'README.md').write_text(README, encoding='utf-8')
    figures = make_plots(final, out, generations) if plots and final else []
    esc = html.escape
    captions = ['Linear mean absolute error versus response magnitudes',
                'Linear RMS error versus response magnitudes',
                'Descriptive relative approximation error',
                'Logarithmic remainder across windows']
    gallery = ''.join(f'<h2>{esc(t)}</h2><img src="{esc(p)}" style="max-width:100%">'
                      for p,t in zip(figures,captions))
    body = f'''<!doctype html><meta charset="utf-8"><title>RQ2 writing outputs</title>
<style>body{{max-width:1050px;margin:35px auto;padding:0 20px;font:17px/1.5 system-ui}}code{{word-break:break-word}}</style>
<h1>RQ2: cross-generation writing outputs</h1>
<p>{len(processed)} of {len(jobs)} requested generation/grid groups processed.</p>
<p><a href="rq2_main_table.csv">Start with the compact main table</a> |
<a href="rq2_finest_with_refinement.csv">Detailed finest-grid table and refinement flags</a> |
<a href="rq2_all_grids.csv">All three grids</a> | <a href="rq2_refinement_checks.csv">Individual checks</a> |
<a href="README.md">Definitions and limitations</a></p>
<p>The remainder is calculated at matching positions and energies before taking absolute values.
Full joint coverage and both refinements are required for the RQ2 scalar flags.</p>{gallery}'''
    (out/'START_HERE.html').write_text(body, encoding='utf-8')
    return len(all_rows), len(final), len(comparisons), figures

def package(out):
    dest = out/'rq2_writing_outputs.zip'
    temp = out/'rq2_writing_outputs.tmp.zip'
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            rel = p.relative_to(out)
            if not p.is_file() or p in (dest,temp) or p.name.endswith('.tmp') or p.name == '.lock':
                continue
            if rel.parts[0] == 'checkpoints':
                continue
            z.write(p, rel)
    temp.replace(dest)
    return dest

def parse_generations(text):
    values = []
    for x in text.split(','):
        if '-' in x:
            a,b = map(int,x.split('-'))
            if b < a:
                raise ValueError('Reversed generation range.')
            values.extend(range(a,b+1))
        else:
            values.append(int(x))
    if not values or len(values) != len(set(values)) or any(not 1 <= g <= 25 for g in values):
        raise ValueError('Generations must be unique positive indices, no larger than 25.')
    return sorted(values)

def run(args):
    root = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else root.parent/'output'/'rq2_analysis'
    if not root.is_dir():
        raise FileNotFoundError(f'Input root not found: {root}. Select the folder containing n10, n11, ... .')
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('Output must be separate from (and not a parent of) the input dataset.')
    if not 1 <= args.chunk_rows <= 32:
        raise ValueError('--chunk-rows must be 1--32; default 4 is suitable for a laptop.')
    generations = parse_generations(args.generations)
    grids = tuple(int(x) for x in args.grids.split(','))
    if len(grids) != 3 or grids[0] < 3 or any(b != 2*a-1 for a,b in zip(grids,grids[1:])):
        raise ValueError('--grids requires three nested point counts, e.g. 7997,15993,31985.')
    if not args.no_plots:
        import matplotlib
    capacity_path = out
    while not capacity_path.exists():
        capacity_path = capacity_path.parent
    if shutil.disk_usage(capacity_path).free < 256*1024**2:
        raise ValueError('At least 256 MiB free working space is required')
    print('INPUT:',root,flush=True)
    print('OUTPUT:',out,flush=True)
    print('Preflight: metadata, small arrays, exact grids and three-mode alignment...',flush=True)
    jobs = preflight(root, generations, grids)
    print(f'Found {len(jobs)*3} NPZ + metadata pairs; {sum(r["bytes"] for j in jobs for r in j["sources"])/1024**3:.2f} GiB compressed inputs.',flush=True)
    if args.preflight_only:
        print('PREFLIGHT PASSED.')
        return 0
    script_sha = file_hash(__file__)
    inputs = [{k:r[k] for k in ('path','meta_path','data_sha256','meta_sha256')}
              for j in jobs for r in j['sources']]
    identity = {'version':VERSION,
                'script_sha256':script_sha,
                'input_root':str(root),
                'generations':generations,
                'grids':list(grids),
                'chunk_rows':args.chunk_rows,
                'numpy_version':np.__version__,
                'input_inventory':inputs,
                'analysis_settings':jobs[0]['settings']}
    identity_sha = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if out.exists() and any(out.iterdir()) and not (out/'rq2_run.json').is_file():
        raise FileExistsError(f'{out} contains unrelated files; choose a new --out directory.')
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('This output folder is already being processed.')
    try:
        mf = out/'rq2_run.json'
        if mf.exists():
            manifest = load_json(mf)
            if manifest.get('identity_sha256') != identity_sha:
                raise ValueError('Input/version/settings changed since this output was created. Use a NEW --out folder.')
        else:
            manifest = {'created_utc':now(), 'identity_sha256':identity_sha, 'identity':identity, 'source_files_modified':False, 'simulations_run':False, 'full_study_accepted':False}
            shutil.copyfile(__file__,out/'process_rq2_snapshot.py')
        manifest.update(status='running',started_or_resumed_utc=now(),completed_groups=0, requested_groups=len(jobs),error=None)
        atomic_json(mf,manifest)
        (out/'checkpoints').mkdir(exist_ok=True)
        (out/'input_metadata').mkdir(exist_ok=True)
        processed=[]
        start=time.monotonic()
        status='completed'
        try:
            for i,job in enumerate(jobs,1):
                g,size=job['generation'],job['n_energy']
                label=f'n{g}_e{size:05d}'
                print(f'\n[{i}/{len(jobs)}] {label}: verifying the three raw-file SHA-256 checksums...',flush=True)
                for rec in job['sources']:
                    before=stamp(rec['path'])
                    if file_hash(rec['path']) != rec['data_sha256'] or before != stamp(rec['path']):
                        raise ValueError(f'Raw checksum failed or file changed: {rec["path"]}. Preserve it for inspection.')
                    if file_hash(rec['meta_path']) != rec['meta_sha256']:
                        raise ValueError(f'Metadata changed after preflight: {rec["meta_path"]}')
                    target=out/'input_metadata'/f'{label}_{rec["mode"]}.meta.json'
                    if target.exists():
                        if file_hash(target) != rec['meta_sha256']:
                            raise ValueError(f'Output metadata copy changed: {target}')
                    else:
                        shutil.copyfile(rec['meta_path'],target)
                cp_path=out/'checkpoints'/f'{label}.json'
                cp_hash=cp_path.with_suffix('.sha256')
                if cp_path.exists() and cp_hash.exists():
                    if file_hash(cp_path) != cp_hash.read_text().strip():
                        raise ValueError(f'Checkpoint checksum mismatch: {cp_path}')
                    cp=load_json(cp_path)
                    if cp.get('identity_sha256') != identity_sha or (cp['generation'],cp['n_energy']) != (g,size):
                        raise ValueError(f'Incompatible completed checkpoint: {cp_path}')
                    print('  resumed: completed paired calculation already verified.',flush=True)
                else:
                    print(f'  scanning paired matrices, at most {args.chunk_rows} positions per chunk...',flush=True)
                    cp=process_job(job,args.chunk_rows)
                    cp.update(identity_sha256=identity_sha,sources=job['sources'],completed_utc=now())
                    atomic_json(cp_path,cp)
                    tmp=cp_hash.with_name(cp_hash.name+'.tmp');tmp.write_text(file_hash(cp_path)+'\n');tmp.replace(cp_hash)
                processed.append(cp)
                manifest['completed_groups']=len(processed);atomic_json(mf,manifest)
            counts=final_exports(out,jobs,processed,generations,grids,plots=not args.no_plots)
            manifest.update(all_grid_rows=counts[0],finest_rows=counts[1],refinement_check_rows=counts[2],figures=counts[3])
        except KeyboardInterrupt:
            status='interrupted'
            manifest['error']='KeyboardInterrupt: rerun the same command to resume completed groups.'
        except Exception as exc:
            status='error'
            manifest['error']=f'{type(exc).__name__}: {exc}'
        finally:
            if status != 'completed':
                try:
                    final_exports(out,jobs,processed,generations,grids,plots=False)
                except Exception as exc:
                    manifest['partial_export_error']=f'{type(exc).__name__}: {exc}'
            manifest.update(status=status,finished_utc=now(),elapsed_this_invocation_seconds=time.monotonic()-start)
            atomic_json(mf,manifest)
        print('\nSTATUS:',status,flush=True)
        if manifest.get('error'):
            print(manifest['error'],file=sys.stderr,flush=True)
        archive=package(out)
        print('WRITING TABLE:',out/'rq2_main_table.csv',flush=True)
        print('OPEN:',out/'START_HERE.html',flush=True)
        print('SMALL UPLOAD:',archive,flush=True)
        print('Original NPZ and metadata files were not modified.',flush=True)
        return 0 if status=='completed' else 130 if status=='interrupted' else 1
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
        lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',default='RQ2_input_data',help='Folder containing n10/, n11/, ... .')
    p.add_argument('--out',help='Separate output folder. Same command safely resumes verified completed groups.')
    p.add_argument('--generations',default='10-20',help='Default 10-20; a subset such as 15 or 10,15 is allowed.')
    p.add_argument('--grids',default='7997,15993,31985',help='Three exactly nested grids. Defaults match the study.')
    p.add_argument('--chunk-rows',type=int,default=4,help='Rows read per matrix at once. Default 4, maximum 32.')
    p.add_argument('--no-plots',action='store_true',help='Write tables/provenance only.')
    p.add_argument('--preflight-only',action='store_true',help='Check file layout, metadata and small axes.')
    args=p.parse_args()
    try:
        return run(args)
    except (ValueError,OSError,KeyError,ImportError,zipfile.BadZipFile,RuntimeError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 1

if __name__=='__main__':
    raise SystemExit(main())