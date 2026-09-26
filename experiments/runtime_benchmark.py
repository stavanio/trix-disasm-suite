#!/usr/bin/env python3
"""Recorded filter-call latency, without training, rollouts or provider calls.

The protocol is docs/runtime_benchmark_protocol.md. Raw NPZ files contain only
numeric/string arrays, loaded with allow_pickle=False. `check` recomputes every
summary from saved calls; `table` emits the manuscript table from that summary.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Set before any numerical imports, including when invoked as a worker.
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for _name in THREAD_VARS:
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from benchmark.registry import ENVS, FILTERS
from baselines import screw_solvers as SS, screw_filters as SF
from envs import snap_env_v2 as SN, crank_env_v2 as CR, pry_env_v2 as PR

TASKS = ('SCREW', 'PCB', 'SNAP', 'CRANK', 'BATTERY', 'PRY')
N, REPEATS, WARMUP, BLOCK, SEED = 2048, 5, 256, 64, 20260925
AGREEMENT_TOL, FEASIBILITY_TOL = 1e-5, 1e-7
STRATA = ('uniform_cube', 'scaled_projection', 'perturbed_projection')
METHODS = [(task, arm) for task in TASKS for arm in FILTERS[task] if arm != 'none']
METHODS += [('SCREW', 'numpy_polygon'), ('SCREW', 'osqp_reused')]
PACKAGES = ('numpy', 'scipy', 'osqp', 'quadprog', 'gymnasium')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    paths = [Path(__file__), ROOT/'docs/runtime_benchmark_protocol.md',
             ROOT/'requirements-runtime.txt']
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_relative_to(ROOT) and p.suffix == '.py' and p.is_file():
                paths.append(p)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def bank(path):
    """Encode specified environment states; no env.step or policy is called."""
    rng = np.random.default_rng(SEED)
    obs = np.empty((len(TASKS), N, 10), np.float32)
    actions = np.empty((len(TASKS), N, 3), np.float32)
    strata = np.tile(np.repeat(np.array([0, 1, 2], np.int8),
                              [N//2, N//4, N//4]), (len(TASKS), 1))
    for ti, task in enumerate(TASKS):
        env = ENVS[task]()
        env.reset(seed=SEED + ti)
        primary = FILTERS[task]['trix']
        for i in range(N):
            if task == 'SNAP':
                env.delta = SN.DELTA_DISENGAGE * (1.25 if i % 2 else 0.5)
            elif task == 'CRANK':
                th = 2 * np.pi * CR.TARGET_ROTATIONS
                env.theta = ((th + rng.uniform(0.01, 2*np.pi)) if i % 2
                             else rng.uniform(-th + 0.01, th - 0.01))
            elif task == 'PRY':
                env.state.position[0] = rng.uniform(0, PR.MAX_INSERTION)
            obs[ti, i] = env._get_obs()
            a = rng.uniform(-1, 1, 3).astype(np.float32)
            if strata[ti, i]:
                projected = primary(obs[ti, i], a)[0].astype(np.float64)
                if strata[ti, i] == 1:
                    a = projected * rng.uniform(0.1, 0.9)
                else:
                    delta = a - projected
                    norm = np.linalg.norm(delta)
                    a = projected + (1e-4 * delta/norm if norm > 0 else 0)
            actions[ti, i] = np.clip(a, -1, 1)
    assert np.all(np.isfinite(obs)) and np.all(np.abs(actions) <= 1)
    np.savez_compressed(path, observations=obs, actions=actions, strata=strata,
                        tasks=np.array(TASKS), stratum_names=np.array(STRATA))


def host(cpu):
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    cpuinfo = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    model = next((line.split(':', 1)[1].strip() for line in cpuinfo.splitlines()
                  if line.startswith('model name')), platform.processor())
    freq = {}
    for name in ('scaling_governor', 'scaling_driver', 'scaling_min_freq',
                 'scaling_max_freq', 'scaling_cur_freq'):
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}')
        freq[name] = p.read_text().strip() if p.exists() else None
    return dict(cpu_model=model, platform=platform.platform(),
                python=sys.version, python_implementation=platform.python_implementation(),
                packages={p: importlib.metadata.version(p) for p in PACKAGES},
                affinity=sorted(os.sched_getaffinity(0)), frequency=freq,
                thread_environment={v: os.environ[v] for v in THREAD_VARS},
                observed_process_threads=len(list(Path('/proc/self/task').iterdir())),
                numpy_configuration=config.getvalue(),
                clock=vars(time.get_clock_info('perf_counter')))


def worker(out, repeat, cpu):
    os.sched_setaffinity(0, {cpu})
    started = now()
    with np.load(out/'inputs.npz', allow_pickle=False) as f:
        observations, actions = f['observations'], f['actions']
    observations.flags.writeable = actions.flags.writeable = False
    functions = []
    construction = {}
    for task, arm in METHODS:
        if arm == 'osqp_reused':
            t0 = time.perf_counter_ns()
            fn = SS.OSQPFilter()
            construction['osqp_reused_ns'] = time.perf_counter_ns() - t0
        elif arm == 'numpy_polygon':
            fn = SF.trix_project
        else:
            fn = FILTERS[task][arm]
        functions.append(fn)
    settings = {key: getattr(functions[-1].prob.settings, key)
                for key in ('eps_abs', 'eps_rel', 'max_iter', 'polishing',
                            'polish_refine_iter', 'warm_starting', 'adaptive_rho')}
    rng = np.random.default_rng(SEED + repeat + 1)
    orders = np.stack([rng.permutation(N) for _ in TASKS])
    gc.collect()
    for mi, (task, _) in enumerate(METHODS):
        ti = TASKS.index(task)
        for idx in orders[ti, :WARMUP]:
            functions[mi](observations[ti, idx], actions[ti, idx])
    durations = np.zeros((len(METHODS), N), np.int64)
    outputs = np.zeros((len(METHODS), N, 3), np.float32)
    flags = np.zeros((len(METHODS), N), np.uint8)
    overhead = np.empty(4096, np.int64)
    schedule = []
    gc.disable()
    try:
        for i in range(len(overhead)):
            t0 = time.perf_counter_ns()
            overhead[i] = time.perf_counter_ns() - t0
        for block in range(N//BLOCK):
            for ti in rng.permutation(len(TASKS)):
                method_ids = [j for j, (task, _) in enumerate(METHODS) if task == TASKS[ti]]
                for mi in rng.permutation(method_ids):
                    schedule.append((int(mi), block))
                    fn = functions[mi]
                    for idx in orders[ti, block*BLOCK:(block+1)*BLOCK]:
                        o, a = observations[ti, idx], actions[ti, idx]
                        t0 = time.perf_counter_ns()
                        result = fn(o, a)
                        elapsed = time.perf_counter_ns() - t0
                        durations[mi, idx] = elapsed
                        y, info = result
                        outputs[mi, idx] = y
                        flags[mi, idx] = (int(bool(info.get('infeasible', False)))
                                         + 2*int(bool(info.get('recovery', False))))
    finally:
        gc.enable()
    assert np.all(durations > 0)
    np.savez_compressed(out/f'repeat_{repeat}.npz', latency_ns=durations,
                        outputs=outputs, flags=flags, input_order=orders,
                        schedule=np.array(schedule, np.int16), timer_pair_ns=overhead,
                        methods=np.array([f'{t}/{a}' for t, a in METHODS]))
    write_json(out/f'repeat_{repeat}.json', dict(started_utc=started, finished_utc=now(),
               repeat=repeat, input_sha256=sha(out/'inputs.npz'),
               host=host(cpu), osqp_settings=settings, construction=construction,
               warmup_calls_per_arm=WARMUP, gc_disabled_during_measurement=True,
               source_sha256=sources()))


def stats(ns):
    values = np.asarray(ns, np.float64).ravel()/1000
    return dict(n=int(values.size), median_us=float(np.median(values)),
                p95_us=float(np.quantile(values, 0.95)),
                p99_us=float(np.quantile(values, 0.99)), max_us=float(np.max(values)))


def summarize(out):
    manifest = json.loads((out/'manifest.json').read_text())
    for filename, expected in manifest['raw_sha256'].items():
        assert sha(out/filename) == expected, filename
    for filename, expected in manifest['source_sha256'].items():
        assert sha(ROOT/filename) == expected, filename
    with np.load(out/'inputs.npz', allow_pickle=False) as f:
        actions, strata, observations = f['actions'], f['strata'], f['observations']
    times, outputs, flags, overhead = [], [], [], []
    repetitions = []
    for rep in range(REPEATS):
        meta = json.loads((out/f'repeat_{rep}.json').read_text())
        assert meta['input_sha256'] == sha(out/'inputs.npz')
        assert meta['source_sha256'] == manifest['source_sha256']
        assert meta['host']['affinity'] == [manifest['cpu']]
        repetitions.append(meta)
        with np.load(out/f'repeat_{rep}.npz', allow_pickle=False) as f:
            assert f['latency_ns'].shape == (len(METHODS), N)
            assert f['outputs'].shape == (len(METHODS), N, 3)
            assert np.all(f['latency_ns'] > 0)
            assert np.array_equal(f['methods'], [f'{t}/{a}' for t, a in METHODS])
            assert sorted(map(tuple, f['schedule'])) == [(a, b) for a in range(len(METHODS)) for b in range(N//BLOCK)]
            assert all(np.array_equal(np.sort(row), np.arange(N)) for row in f['input_order'])
            times.append(f['latency_ns']); outputs.append(f['outputs'])
            flags.append(f['flags']); overhead.append(f['timer_pair_ns'])
    times, outputs, flags = np.array(times), np.array(outputs), np.array(flags)
    assert np.all(np.isfinite(outputs))
    rows = []
    for mi, (task, arm) in enumerate(METHODS):
        ti = TASKS.index(task)
        rows.append(dict(task=task, arm=arm, **stats(times[:, mi]),
            per_process=[stats(times[r, mi]) for r in range(REPEATS)],
            strata={name: stats(times[:, mi, strata[ti] == si]) for si, name in enumerate(STRATA)},
            phase={str(phase): stats(times[:, mi, observations[ti, :, 7] == phase])
                   for phase in (0, 1)} if task in ('SNAP', 'CRANK') else {},
            returned_infeasible=int(np.count_nonzero(flags[:, mi] & 1)),
            returned_recovery=int(np.count_nonzero(flags[:, mi] & 2)),
            changed_fraction=float(np.mean(np.max(np.abs(outputs[:, mi] - actions[ti]), axis=2) > 1e-6))))
    ref = outputs[:, METHODS.index(('SCREW', 'trix'))].astype(np.float64)
    validation = {}
    for arm in ('trix', 'qp_matched', 'numpy_polygon', 'osqp_reused'):
        mi = METHODS.index(('SCREW', arm)); x = outputs[:, mi].astype(np.float64)
        slab = np.maximum(0, np.abs(SF.A_V*x[..., 0]-SF.A_W*x[..., 1]) - (SF.EPS-SF.INTERIOR))
        box = np.maximum(0, np.max(np.abs(x[..., :2]), axis=2)-1)
        radial = np.maximum(0, np.abs(x[..., 2])-SF.RAD_LIM_N)
        error = np.max(np.abs(x-ref), axis=2)
        feasibility = np.maximum.reduce([slab, box, radial])
        validation[arm] = dict(max_abs_action_error=float(np.max(error)),
            max_constraint_excess=float(np.max(feasibility)),
            agreement_failures=int(np.count_nonzero(error > AGREEMENT_TOL)),
            feasibility_failures=int(np.count_nonzero(feasibility > FEASIBILITY_TOL)),
            recoveries=int(np.count_nonzero(flags[:, mi] & 2)))
    return dict(schema=1, scope='Complete filter callable; single-host implementation benchmark',
        seed=SEED, inputs_per_task=N, fresh_processes=REPEATS, arms=len(METHODS),
        measured_calls=int(times.size), warmup_calls_per_arm_per_process=WARMUP,
        timer_pair=stats(overhead), host=repetitions[0]['host'],
        osqp_settings=repetitions[0]['osqp_settings'],
        osqp_construction_ns=[m['construction']['osqp_reused_ns'] for m in repetitions],
        agreement_tolerance=AGREEMENT_TOL, feasibility_tolerance=FEASIBILITY_TOL,
        matched_screw_validation=validation,
        matched_screw_pass=all(v['agreement_failures'] == v['feasibility_failures'] == v['recoveries'] == 0
                               for v in validation.values()), rows=rows)


def table(data):
    labels = {'trix': 'TRiX', 'oracle_tangent': 'Oracle tangent',
              'qp_matched': 'Active-set QP', 'numpy_polygon': 'NumPy polygon',
              'osqp_reused': 'Reused OSQP', 'box_clip': 'Box clip',
              'static_clip': 'Static clip', 'trix_preventive': 'TRiX preventive',
              'box_clip_preventive': 'Box preventive'}
    rows = sorted(data['rows'], key=lambda r: (TASKS.index(r['task']),
                  list(labels).index(r['arm'])))
    lines = [r'% Generated from saved measurements by experiments/runtime_benchmark.py table.',
        r'\begin{longtable}{llrrrr}',
        r'\caption{Filter-call latency on the recorded host. Times are in $\mu$s.',
        r'Each row contains 10,240 calls across five fresh processes. Median and',
        r'95th percentile pool the retained calls; the range is the minimum and',
        r'maximum of the five process medians. Controls outside the matched SCREW',
        r'comparison implement different restrictions.}\label{tab:runtime-latency}\\',
        r'\toprule Task & Implementation & Median & 95th & \multicolumn{2}{c}{Process medians}\\',
        r' & & & percentile & Min. & Max.\\\midrule\endfirsthead',
        r'\toprule Task & Implementation & Median & 95th & Min. & Max.\\\midrule\endhead',
        r'\bottomrule\endfoot']
    for row in rows:
        meds = [r['median_us'] for r in row['per_process']]
        lines.append(f"{row['task']} & {labels[row['arm']]} & {row['median_us']:.2f} & {row['p95_us']:.2f} & {min(meds):.2f} & {max(meds):.2f}"+r'\\')
    lines += [r'\end{longtable}', '']
    return '\n'.join(lines)


def run(out, cpu):
    out.mkdir(parents=True, exist_ok=False)
    assert cpu in os.sched_getaffinity(0)
    bank(out/'inputs.npz')
    for rep in range(REPEATS):
        print(f'Fresh process {rep + 1}/{REPEATS}', flush=True)
        with (out/f'repeat_{rep}.log').open('w') as log:
            subprocess.run([sys.executable, '-B', __file__, 'worker', '--out', str(out),
                            '--repeat', str(rep), '--cpu', str(cpu)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    source_sha = json.loads((out/'repeat_0.json').read_text())['source_sha256']
    manifest = dict(schema=1, completed_utc=now(), cpu=cpu,
                    source_sha256=source_sha,
                    raw_sha256={p.name: sha(p) for p in sorted(out.iterdir())})
    write_json(out/'manifest.json', manifest)
    data = summarize(out)
    write_json(out/'summary.json', data)
    print(json.dumps(dict(measured_calls=data['measured_calls'],
                         matched_screw_pass=data['matched_screw_pass']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker', 'check', 'table'))
    parser.add_argument('--out', type=Path, default=ROOT/'results/runtime')
    parser.add_argument('--cpu', type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args(); args.out = args.out.resolve()
    if args.mode == 'run':
        run(args.out, args.cpu)
    elif args.mode == 'worker':
        worker(args.out, args.repeat, args.cpu)
    else:
        data = summarize(args.out)
        assert data == json.loads((args.out/'summary.json').read_text()), 'Summary differs from raw records'
        if args.mode == 'table':
            (ROOT/'manuscript/tables/runtime_latency.tex').write_text(table(data))
        print(json.dumps(dict(measured_calls=data['measured_calls'], arms=data['arms'],
              matched_screw_pass=data['matched_screw_pass'], hashes_verified=True), indent=2))


if __name__ == '__main__':
    main()
