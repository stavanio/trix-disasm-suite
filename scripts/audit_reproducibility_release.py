#!/usr/bin/env python3
"""Audit a relocated evidence package without training, rendering or actuation.

Reads JSON/CSV/logs and plain SB3 ZIP metadata only. Never unpickles a policy.
Requires NumPy for the existing archived-command/statistics audit utilities.
Outputs are written outside the evidence package by default.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

sys.dont_write_bytecode = True


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def policy_records(root):
    linkage, exceptions, fingerprints = [], [], Counter()
    exact_groups = defaultdict(list)
    selected_models, all_checkpoints = {}, set()
    metadata_files = 0
    for stage, fn, nominal in [('stage1', 'stage1_records.json', True),
                               ('stage2', 'stage2_records.json', False),
                               ('bayonet', 'stage1_records.json', True),
                               ('bayonet2', 'stage2_records.json', False)]:
        folder = root / 'results' / stage
        shards = defaultdict(list)
        for p in (folder / 'shards').glob('*test*.json'):
            # Strip the numeric batch index and hash, retaining the arm name.
            key = p.name.rsplit('__', 1)[0].rsplit('_', 1)[0]
            shards[key].append(p)
        for i, r in enumerate(read(folder / fn)['records']):
            task, algo, seed, step, arm = (r[k] for k in
                ('task', 'algorithm', 'seed', 'selected_step', 'evaluation_arm'))
            tag = ('test_' if nominal else 'test__') + arm
            key = f'{task}_{algo}_{seed}_{step}_{tag}'
            matched = sorted(shards[key])
            assert matched, (stage, key)
            totals, seeds, hashes = Counter(), [], Counter()
            for p in matched:
                d = read(p)
                summary = d['summary']
                assert (summary['task'], summary['method'], summary['seed']) == (
                    task, f'{algo}+{arm}', seed)
                assert len(d['episode_seeds']) == summary['counts']['episodes']
                totals.update(summary['counts'])
                seeds.extend(d['episode_seeds'])
                hashes[summary['constraint_hash']] += 1
            # Reproduce the historical prefix-based gather separately. It also
            # matched longer arm names (trix_preventive / box_clip_preventive).
            historical = sorted(p for k, ps in shards.items()
                                if k == key or k.startswith(key + '_') for p in ps)
            pooled, pooled_seeds, methods = Counter(), [], Counter()
            for p in historical:
                d = read(p)
                pooled.update(d['summary']['counts'])
                pooled_seeds.extend(d['episode_seeds'])
                methods[d['summary']['method']] += 1
            assert dict(pooled) == r['counts'] == r['summary']['counts'], (stage, key)
            assert abs(100 * pooled['safe_completions'] / pooled['episodes']
                       - r['summary']['safe_completion_rate']) < 1e-10
            assert totals['episodes'] == len(seeds) == len(set(seeds)) == 100
            exact_rate = 100 * totals['safe_completions'] / totals['episodes']
            exact_groups[f'{stage}/{task}/{algo}/{arm}'].append(exact_rate)
            prefix = f'{task}__{algo}__{("none" if nominal else arm)}__seed{seed}'
            cp = folder / 'checkpoints' / f'{prefix}_{step}.zip'
            meta = read(folder / 'checkpoints' / f'{prefix}_meta.json')
            assert cp.is_file() and any(Path(p).name == cp.name for _, p in meta['checkpoints'])
            assert meta['train_steps'] == r['train_steps']
            rel = str(cp.relative_to(root))
            if rel not in selected_models:
                with ZipFile(cp) as z:
                    b = z.read('data')
                    model = json.loads(b)
                assert model['seed'] == seed
                selected_models[rel] = dict(seed=seed, selected_step=step,
                    actual_steps=model['num_timesteps'], metadata_sha256=hashlib.sha256(b).hexdigest())
            item = dict(aggregate=f'results/{stage}/{fn}', row_index_zero_based=i,
                task=task, algorithm=algo, arm=arm, seed=seed, selected_step=step,
                checkpoint=rel, shards=[str(p.relative_to(root)) for p in matched],
                episodes=totals['episodes'], unique_episode_seeds=len(set(seeds)),
                exact_arm_counts=dict(totals), exact_arm_safe_completion_rate=exact_rate,
                historical_episodes=pooled['episodes'], historical_unique_episode_seeds=len(set(pooled_seeds)),
                historical_shard_count=len(historical), historical_shard_methods=dict(methods),
                historical_safe_completion_rate=r['summary']['safe_completion_rate'],
                shard_constraint_hash_counts=dict(hashes), aggregate_constraint_hash=r['constraint_hash'],
                protocol_freeze_hash=r['protocol_freeze_hash'], recorded_commit=r['provenance']['git_commit'])
            linkage.append(item)
            if totals != pooled or set(hashes) != {r['constraint_hash']}:
                exceptions.append({k: v for k, v in item.items() if k != 'shards'})
        for p in sorted((folder / 'checkpoints').glob('*_meta.json')):
            d = read(p)
            fingerprints[json.dumps(d['fingerprint'], sort_keys=True)] += 1
            metadata_files += 1
            for step, name in d['checkpoints']:
                cp = folder / 'checkpoints' / Path(name).name
                assert cp.is_file(), cp
                all_checkpoints.add(str(cp.relative_to(root)))
    assert len(linkage) == 1020 and len(selected_models) == metadata_files == 520
    return dict(records_checked=len(linkage), selected_unique_checkpoints=len(selected_models),
        training_metadata_files=metadata_files, candidate_and_final_checkpoints=len(all_checkpoints),
        matched_test_shards=sum(len(r['shards']) for r in linkage),
        episodes_in_aggregate_records=sum(r['historical_episodes'] for r in linkage),
        records_with_repeated_episode_seed_ids=sum(r['historical_unique_episode_seeds'] < r['historical_episodes'] for r in linkage),
        exact_arm_group_means={k: sum(v)/len(v) for k, v in sorted(exact_groups.items())},
        exact_arm_group_seed_rates=dict(exact_groups),
        historical_exceptions=exceptions, selected_models=selected_models, records=linkage,
        fingerprints=[dict(fingerprint=json.loads(k), files=v) for k, v in fingerprints.items()])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--verify-hashes', action='store_true', help='Also verify every file in SHA256SUMS.jsonl')
    args = ap.parse_args()
    package = args.package_root.resolve()
    out = (args.out_dir or package.parent / 'TRIX_reproducibility_audit').resolve()
    assert not out.is_relative_to(package), 'Keep generated audit reports outside the evidence package.'
    out.mkdir(parents=True, exist_ok=True)
    root, hw = package / 'research', package / 'hardware'
    provenance = read(package / 'SOURCE_PROVENANCE.json')
    assert sha(root / 'manuscript/TRIX_REVISION.tex') == provenance['manuscript_sha256']
    assert all(sha(root / n) == h for n, h in provenance['frozen_files_verified'].items())
    for x in provenance['snapshots']:
        assert sha(package / x['archive']) == x['sha256']
    frames = read(root / 'assets/workspaces/frozen_frames.json')['states']
    for task, frame in frames.items():
        p = root / 'results/workspace_states' / frame['trace_file']
        assert sha(p) == frame['trace_sha256']
        trace = read(p)
        assert any(row == frame['frame'] for row in trace['trace']), task
    policy = policy_records(root)
    groups = defaultdict(dict)
    for row in policy['records']:
        if row['aggregate'] == 'results/stage1/stage1_records.json':
            groups[(row['task'], row['algorithm'], row['arm'])][row['seed']] = row['exact_arm_safe_completion_rate']
    oracle = read(root / 'results/audits/pcb_oracle_seed_fix/pcb_oracle_seed0_corrected.json')
    for row in oracle['records']:
        groups[(row['task'], row['algorithm'], row['arm'])][row['training_seed']] = row['summary']['safe_completion_rate']
    tex = (root / 'manuscript/TRIX_REVISION.tex').read_text()
    table = tex.split(r'\label{tab:submitted-4}', 1)[1].split(r'\end{table}', 1)[0]
    table_cells = []
    for line in table.splitlines():
        if not re.match(r'^(SCREW|PCB|SNAP|CRANK|BATTERY|PRY) & (SAC|PPO) &', line):
            continue
        cells = [s.strip().rstrip('\\').strip() for s in line.split('&')]
        task, algo = cells[:2]
        for arm, shown in zip(['none', 'box_clip', 'static_clip', 'oracle_tangent',
                               'qp_matched', 'trix', 'trix_preventive'], cells[2:]):
            if shown == '--':
                continue
            values = groups[(task, algo.lower(), arm)]
            mean = sum(values.values()) / len(values)
            assert shown == f'{mean:.1f}', (task, algo, arm, shown, mean)
            table_cells.append([task, algo, arm, shown, mean])
    assert len(table_cells) == 50
    (out / 'table4_checked.json').write_text(json.dumps(table_cells, indent=2) + '\n')
    (out / 'policy_record_linkage.json').write_text(json.dumps(policy, indent=2) + '\n')
    print(f"Checked {policy['records_checked']} policy records and {policy['selected_unique_checkpoints']} selected checkpoints.", flush=True)
    for script, arguments in [
        ('audit_revision_claims.py', ['--research-root', str(root), '--hardware-root', str(hw),
                                      '--out', str(out / 'claims.json')]),
        ('audit_vlm_reconciliation.py', ['--archive-root', str(root), '--out', str(out / 'vlm.json')])]:
        subprocess.run([sys.executable, '-B', str(root / 'scripts' / script), *arguments], check=True)
    stats = module('archived_statistics', root / 'scripts/build_revision_seed_statistics.py')
    data = stats.audit_archive(root)
    assert data == read(root / 'manuscript/data/seed_statistics_input.json')
    contrasts, _ = stats.compute(data)
    assert contrasts == read(root / 'manuscript/data/seed_statistics_summary.json')['contrasts']
    # Verify the intervention package's root-relative manifest separately from
    # the six confirmation runs and four media items checked above.
    cf = hw / 'experiments/hardware/2026-08-27_trix_physical_safety_v1/SHA256SUMS.txt'
    intervention = []
    for line in cf.read_text().splitlines():
        h, name = line.split(maxsplit=1)
        p = (hw / name.lstrip('*')).resolve()
        assert p.is_relative_to(hw) and sha(p) == h, p
        intervention.append(name)
    # Ensure the relocation fix cannot accidentally fall back to old-machine paths.
    claims = module('claims_checker', root / 'scripts/audit_revision_claims.py')
    example_cf = hw / 'results/hardware/v4_confirmation/20260914T023526Z/SHA256SUMS.txt'
    recorded = '/home/stavanio/rebot_control/results/hardware/v4_confirmation/20260914T023526Z/can_raw.log'
    assert claims.hardware_checksum_path(recorded, example_cf, hw) == example_cf.parent / 'can_raw.log'
    for invalid in ['/etc/passwd', '../../../../../../etc/passwd']:
        try:
            claims.hardware_checksum_path(invalid, example_cf, hw)
        except ValueError:
            pass
        else:
            raise AssertionError('Checksum path escaped the supplied package')
    assets = read(root / 'assets/workspaces/b601_assets.json')
    for n, h in assets['sha256'].items():
        assert sha(package / 'dependencies/b601_description' / n) == h
    fonts = read(root / 'assets/workspaces/reference_environment.json')['fonts_sha256']
    for n, h in fonts.items():
        assert sha(package / 'dependencies/fonts' / n) == h
    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'tests.test_exact_arm_aggregation'],
                   cwd=root, check=True)
    verified_files, verified_bytes = 0, 0
    if args.verify_hashes:
        seen = set()
        with (package / 'SHA256SUMS.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                assert row['path'] not in seen, row['path']
                seen.add(row['path'])
                p = (package / row['path']).resolve()
                assert p.is_relative_to(package) and p.is_file()
                assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
                verified_files += 1
                verified_bytes += p.stat().st_size
        actual_paths = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
        assert actual_paths == seen | {'SHA256SUMS.jsonl'}, actual_paths ^ (seen | {'SHA256SUMS.jsonl'})
    summary = dict(schema=1, audit_date=date.today().isoformat(), passed=True,
        source_snapshots=len(provenance['snapshots']), frozen_files_unchanged=61,
        policy_records=1020, selected_checkpoints=520, table4_cells=50,
        candidate_and_final_checkpoints=policy['candidate_and_final_checkpoints'],
        test_shards=policy['matched_test_shards'], frozen_traces=len(frames),
        historical_exception_records=len(policy['historical_exceptions']),
        repeated_episode_seed_records=policy['records_with_repeated_episode_seed_ids'],
        seed_statistic_records=len(data['records']), figure4_points=data['figure4_points_checked'],
        seed_contrasts=len(contrasts), vlm_decisions=273, vlm_bank_images=91,
        confirmation_and_media_checksums=len(read(out / 'claims.json')['hardware']['checksum_verified_files']),
        intervention_checksums=len(intervention), b601_assets=len(assets['sha256']), reference_fonts=len(fonts),
        package_files_verified=verified_files, package_bytes_verified=verified_bytes,
        manuscript_sha256=provenance['manuscript_sha256'],
        scope='Archived-data reconciliation and file integrity; no fresh experiment, renderer, model load, API call or hardware actuation.',
        limitations=['Recorded Git commits do not establish historical dirty-worktree state.',
                     'Dependency fingerprints are partial; no historical full transitive lockfile was recovered.',
                     'Historical BATTERY filename-prefix aggregation pooled distinct arm names; exact-arm counts are reported separately in policy_record_linkage.json.',
                     'VLM resolved backend identities, API access times and per-trial retry provenance remain unavailable.'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
