#!/usr/bin/env python3
"""Reconcile saved VLM decisions without API calls, rendering, or simulation.

Requires NumPy for the archived primitive parser. Writes a summary and a
273-row JSON ledger, including explicit nulls for unavailable model/retry IDs.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    hashes = {}

    def source(name):
        p = args.archive_root / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    def read(name):
        return json.loads(source(name).read_text())

    def literal(name, key):
        tree = ast.parse(source(name).read_text())
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in n.targets):
                return ast.literal_eval(n.value)
        raise ValueError((name, key))

    man = read('results/vlm_bank/manifest.json')
    bank = {e['image']: e for e in man['images']}
    assert len(bank) == len(man['images']) == 91
    spec = importlib.util.spec_from_file_location('saved_vlm_parser', source('experiments/vlm_primitives.py'))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    pools = literal('experiments/vlm_client.py', 'POOLS')
    source('experiments/vlm_trial.py')
    ledger, summary, failure_sets, examples = [], {}, {}, {}
    expected = {'anthropic': (0, 88, 3), 'openai': (16, 66, 9), 'google': (0, 65, 26)}
    for provider in expected:
        rel = f'results/vlm/vlm_{provider}.json'
        d = read(rel)
        rows = d['rows']
        assert len(rows) == len({r['image'] for r in rows}) == 91
        assert {r['image'] for r in rows} == set(bank)
        counts, arm_counts = Counter(), {'none': Counter(), 'trix': Counter()}
        failures, prevented, introduced = set(), 0, 0
        for i, r in enumerate(rows):
            e = bank[r['image']]
            assert 'error' not in r and r['raw']
            assert (r['state'], r['state_id'], r['bank_hash']) == (e['state'], e['state_id'], man['bank_hash'])
            action, mag, cls = parser.parse(r['raw'])
            assert (action, mag, cls) == (r['action'], r['magnitude'], r['class'])
            category = 'refused' if cls == 'refused' else 'ungroundable'
            counts_by_arm = {}
            pre, new = False, False
            if cls == 'grounded':
                assert parser.ground(action, mag).tolist() == r['command']
                category = 'admissible' if r['admissible'] else 'inadmissible'
                assert r['failure_class'] == ('executable' if r['admissible'] else 'inadmissible')
                for arm in arm_counts:
                    c = r[arm]['counts']
                    assert c['episodes'] == 1 and c['total_steps'] == 40
                    assert c['safe_completions'] + c['unsafe_completions_intact'] + c['destructive_completions'] + c['mechanical_failures'] + c['timeouts'] == 1
                    assert r[arm]['violation'] == 100 * c['episodes_with_violation']
                    arm_counts[arm].update(c)
                    counts_by_arm[arm] = c
                a, b = counts_by_arm['none'], counts_by_arm['trix']
                pre = bool(a['episodes_with_violation'] and not b['episodes_with_violation'])
                new = bool(b['episodes_with_violation'] and not a['episodes_with_violation'])
                # Here the historical harm indicator is entirely its violation term.
                assert a['destructive_completions'] == b['destructive_completions'] == 0
                assert pre == r['projection_prevented_harm']
                assert new == r['projection_introduced_harm']
                prevented += pre
                introduced += new
            else:
                assert 'none' not in r and 'trix' not in r
            counts[category] += 1
            if category == 'inadmissible':
                failures.add(r['image'])
            item = {'source_file': rel, 'row_index_zero_based': i,
                    'json_pointer': f'/rows/{i}', 'provider': provider,
                    'configured_model_alias': r['model'],
                    'resolved_model_id': r.get('resolved_model'),
                    'api_access_timestamp': r.get('timestamp'),
                    'retry_count': r.get('retry_count'),
                    'fallback_used': r.get('fallback_used'),
                    'backend_provenance': 'not recorded',
                    'image': r['image'], 'state_id': r['state_id'],
                    'stratum': r['stratum'], 'raw': r['raw'], 'category': category,
                    'command': r.get('command'), 'replays': counts_by_arm,
                    'prevented_constraint_violation_case': pre,
                    'introduced_constraint_violation_case': new}
            ledger.append(item)
            if (provider, r['image']) in [('openai', 's06_occluded.png'), ('anthropic', 's00_misleading.png'), ('google', 's01_occluded.png')]:
                examples[provider + '/' + r['image']] = item
        actual = tuple(counts[k] for k in ('refused', 'admissible', 'inadmissible'))
        assert actual == expected[provider] and counts['ungroundable'] == 0
        assert all(r['model'] == d['model'] == pools[provider][0] for r in rows)
        summary[provider] = {'responses': len(rows), 'counts': dict(counts),
                             'ungroundable': counts['ungroundable'],
                             'grounded': counts['admissible'] + counts['inadmissible'],
                             'configured_model_alias': d['model'],
                             'configured_fallback_pool': pools[provider],
                             'outcomes': {k: dict(v) for k, v in arm_counts.items()},
                             'prevented_constraint_violation_cases': prevented,
                             'introduced_constraint_violation_cases': introduced,
                             'inadmissible_by_stratum': dict(Counter(r['stratum'] for r in rows if r['failure_class'] == 'inadmissible'))}
        failure_sets[provider] = failures

    e = examples['openai/s06_occluded.png']
    assert e['command'] == [0.0, 0.0, 0.8]
    assert e['replays']['none']['violating_steps'] == e['replays']['none']['total_steps'] == 40
    assert e['replays']['trix']['violating_steps'] == 0 and e['replays']['trix']['total_steps'] == 40
    # Reconstruct the original algebra, without importing an environment or renderer.
    pc = 'envs/pcb_env_v2.py'
    st, kb = literal(pc, 'S_TAU'), literal(pc, 'K_BEND')
    tau, sig = literal(pc, 'TAU_TILT_MAX'), literal(pc, 'SIGMA_TAU_PCB')
    delta = literal('benchmark/margin.py', 'DEFAULT_DELTA')
    horizon = literal('benchmark/margin.py', 'DEFAULT_HORIZON')
    z = NormalDist().inv_cdf((1 - delta) ** (1 / horizon))
    interior = literal('baselines/pcb_filters.py', 'INTERIOR')
    bound = (tau - z * sig) / st * (1 - interior)
    visual_max = literal('experiments/vlm_bank.py', 'TILT_RANGE')[1]
    source('envs/render.py')
    configs = {e['state_id']: e for e in bank.values()}
    main_configs = [e for e in configs.values() if e['stratum'] != 'partial']
    partial = [e for e in configs.values() if e['stratum'] == 'partial']
    assert len(main_configs) == 21 and len(partial) == 7
    # Compare angles within their original encoding groups, never pooled.
    strata = {}
    for name in ('interior', 'mid', 'near_boundary', 'partial'):
        entries = [e for e in bank.values() if e['stratum'] == name]
        physical = [abs(e['state']['tilt_norm'] * st / kb) for e in entries]
        displayed = [math.degrees(abs(e['state']['tilt'])) for e in entries]
        strata[name] = {'images': len(entries),
                        'base_configurations': len({e['state_id'] for e in entries}),
                        'abs_replay_theta_x_rad_range': [min(physical), max(physical)],
                        'abs_display_tilt_deg_range': [min(displayed), max(displayed)]}
    assert [strata[n]['images'] for n in strata] == [28, 28, 28, 7]
    fracture = literal(pc, 'THETA_FRAC')
    assert strata['near_boundary']['abs_display_tilt_deg_range'][0] > math.degrees(fracture)
    by_margin = sorted(main_configs, key=lambda e: e['margin'])
    # Tighter generator margin implies larger curvature magnitude and display tilt.
    for key in ('tilt_norm', 'tilt'):
        vals = [abs(e['state'][key]) for e in by_margin]
        assert all(a > b for a, b in zip(vals, vals[1:]))
    for e in main_configs:
        assert math.isclose(abs(e['state']['tilt']), visual_max * (1 - e['margin']), abs_tol=1e-4)
    totals = {k: sum(s['counts'].get(k, 0) for s in summary.values()) for k in ('refused', 'ungroundable', 'admissible', 'inadmissible')}
    totals['responses'] = len(ledger)
    totals['grounded'] = sum(s['grounded'] for s in summary.values())
    totals['prevented_constraint_violation_cases'] = sum(s['prevented_constraint_violation_cases'] for s in summary.values())
    totals['introduced_constraint_violation_cases'] = sum(s['introduced_constraint_violation_cases'] for s in summary.values())
    assert totals == dict(refused=16, ungroundable=0, admissible=219, inadmissible=38, responses=273, grounded=257, prevented_constraint_violation_cases=31, introduced_constraint_violation_cases=0)
    # An empty intersection of these two runs precludes a three-way overlap
    # for every possible Gemini failure set, independent of its provenance.
    claude_gpt_images = failure_sets['anthropic'] & failure_sets['openai']
    claude_gpt_states = ({bank[n]['state_id'] for n in failure_sets['anthropic']}
                         & {bank[n]['state_id'] for n in failure_sets['openai']})
    assert not claude_gpt_images and not claude_gpt_states
    observed_google = summary['google']['counts']['inadmissible']
    change_budget = 4
    sensitivity = [max(0, observed_google - change_budget),
                   min(91, observed_google + change_budget)]
    assert sensitivity == [22, 30]
    result = {'schema': 2, 'new_api_calls': 0, 'new_simulation_trials': 0,
              'sources_sha256': hashes, 'bank_spec': man['spec'],
              'per_provider': summary, 'totals': totals,
              'tilt_mapping': {'robust_normalized_tilt_bound': bound,
                  'display_to_replay_multiplier_main': visual_max / (bound * st / kb),
                  'display_to_replay_multiplier_partial': visual_max / (st / kb),
                  'main_to_partial_display_scale_ratio': 1 / bound,
                  'main_comparison_images': 84,
                  'partial_images_excluded_from_margin_comparisons': 7,
                  'stratum_ranges': strata,
                  'fracture_threshold_rad': fracture,
                  'fracture_threshold_deg': math.degrees(fracture),
                  'display_interpretation': 'Visibility-scaled cue, not a physically realizable intact-board pose under the benchmark bending model.',
                  'main_strata_ordering_checked': True,
                  'formula_applies_before_stored_rounding': True,
                  'example': {'image': 's00_normal.png', 'display_tilt_rad': bank['s00_normal.png']['state']['tilt'],
                              'replay_theta_x_rad': bank['s00_normal.png']['state']['tilt_norm'] * st / kb}},
              'failure_image_union': sorted(set.union(*failure_sets.values())),
              'failure_image_three_way_intersection': sorted(set.intersection(*failure_sets.values())),
              'inadmissible_images_by_provider': {p: sorted(v) for p, v in failure_sets.items()},
              'claude_gpt_overlap': {'images': sorted(claude_gpt_images),
                  'base_state_ids': sorted(claude_gpt_states),
                  'three_way_empty_for_any_gemini_failure_set': True,
                  'condition': 'Claude and GPT saved classifications held fixed; no Gemini retry or backend assumption required.'},
              'gemini_four_response_sensitivity': {
                  'observed_inadmissible': observed_google, 'denominator': 91,
                  'maximum_changed_response_classifications_assumed': change_budget,
                  'counterfactual_inadmissible_range': sensitivity,
                  'lower_bound_exceeds_fixed_claude_and_gpt_counts': sensitivity[0] > max(summary[p]['counts']['inadmissible'] for p in ('anthropic', 'openai')),
                  'four_response_change_budget_established_by_records': False,
                  'interpretation': 'Conditional replacement sensitivity, not a bound on the number of unknown backend variants. All 91 Gemini rows lack resolved backend IDs; fallback is not restricted to the four historically reported timeouts.'},
              'verified_examples': examples,
              'retry_provenance': {'final_error_rows': 0,
                  'attempt_level_log_available': False,
                  'four_historical_retry_claim': 'Reported in earlier notes; trial IDs and successful backends not recovered.',
                  'can_identify_retried_images_or_changed_variants': False},
              'google_summary_correction': {'old_admissible_count': 62, 'verified_admissible_count': 65,
                  'explanation': 'Final archive has 91 unique successful rows, 65 admissible and 26 inadmissible. No missing decisions. The earlier aggregate has no row-level partition with which to identify three particular omitted trials.'}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'vlm_reconciliation.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'vlm_trial_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'totals': totals, 'main_tilt_multiplier': result['tilt_mapping']['display_to_replay_multiplier_main'], 'example_verified': True}))


if __name__ == '__main__':
    main()
