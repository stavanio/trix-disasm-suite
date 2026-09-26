#!/usr/bin/env python3
"""Regenerate Figure 4 from retained, audited seed data, without new experiments.

This is a replacement generator, not the unavailable historical plotting script.
Use --check-only for a standard-library-only audit of every plotted value.
Plotting dependencies are pinned in requirements-figures.txt.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CSV = 'manuscript/figures/figure4_seed_data.csv'
AUDIT = 'manuscript/data/seed_statistics_input.json'
PANELS = [
    ('a', 'PCB: same restriction across regimes', [
        ('PCB post-hoc box', 'stage1', 'PCB', 'box_clip', 'Post-hoc\nbox'),
        ('PCB filter-aware box', 'stage2', 'PCB', 'box_clip', 'Filter-aware\nbox')]),
    ('b', 'SNAP: filter-aware governance', [
        ('SNAP static', 'stage2', 'SNAP', 'static_clip', 'Static'),
        ('SNAP TRiX', 'stage2', 'SNAP', 'trix', 'TRiX')]),
    ('c', 'CRANK: filter-aware governance', [
        ('CRANK static', 'stage2', 'CRANK', 'static_clip', 'Static'),
        ('CRANK TRiX', 'stage2', 'CRANK', 'trix', 'TRiX')]),
    ('d', 'PRY: simpler restriction is sufficient', [
        ('PRY static', 'stage2', 'PRY', 'static_clip', 'Static'),
        ('PRY TRiX', 'stage2', 'PRY', 'trix', 'TRiX')]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_points():
    audit = json.loads((ROOT / AUDIT).read_text())
    for path, expected in audit['source_file_sha256'].items():
        assert sha(ROOT / path) == expected, f'Changed audited input: {path}'
    with (ROOT / CSV).open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 80
    assert len({(r['condition'], int(r['seed'])) for r in rows}) == 80
    points = {}
    for panel, _, conditions in PANELS:
        for condition, stage, task, arm, _ in conditions:
            selected = sorted((r for r in rows if r['condition'] == condition),
                              key=lambda r: int(r['seed']))
            assert [int(r['seed']) for r in selected] == list(range(10)), condition
            assert all(r['panel'] == panel for r in selected), condition
            values = []
            for row in selected:
                record = audit['records'][f'{stage}/{task}/sac/{arm}/{row["seed"]}']
                counts = record['counts']
                assert counts['episodes'] == 100
                value = 100 * counts['safe_completions'] / counts['episodes']
                assert value == float(row['safe_completion_rate']) == record['safe_completion_rate']
                values.append(value)
            points[condition] = values
    assert len(points) == 8
    return points


def build(output):
    points = checked_points()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/trix-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 10, 'axes.labelsize': 10,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.95))
    colors = plt.get_cmap('tab10').colors
    for ax, (panel, title, conditions) in zip(axes.flat, PANELS):
        data = [points[c[0]] for c in conditions]
        ax.boxplot(data, positions=[1, 2], widths=.45, whis=1.5,
                   showfliers=False, showmeans=True,
                   meanprops={'marker': '^', 'markerfacecolor': '#2ca02c',
                              'markeredgecolor': '#2ca02c', 'markersize': 7},
                   medianprops={'color': '#ff7f0e', 'linewidth': 1.1})
        for x, values in enumerate(data, 1):
            # Only horizontal offsets separate seeds; all measured y values are exact.
            ax.scatter([x - .14 + seed * .28 / 9 for seed in range(10)], values,
                       c=colors, s=24, zorder=3)
        ax.set(ylim=(-5, 105), xlim=(.5, 2.5), yticks=list(range(0, 101, 20)),
               xticks=[1, 2], xticklabels=[c[4] for c in conditions],
               ylabel='Safe completion (%)', title=title)
        ax.grid(axis='y', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(.02, .96, f'({panel})', transform=ax.transAxes,
                va='top', fontweight='bold')
    fig.suptitle('Seed-level safe completion from the frozen production experiments', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / 'figure4_seed_distributions.pdf'
    fig.savefig(pdf, metadata={'Title': 'TRiX Figure 4: seed-level safe completion',
                              'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    metadata = dict(schema=1, generator='scripts/build_revision_figure4.py',
        provenance='Replacement generator from retained seed-level data; historical generator unavailable.',
        historical_pdf_sha256='399ed017a0b0b4d55509cf7782c0396e67dc132649e033fac0255eba6ce73e79',
        generator_sha256=sha(Path(__file__)),
        inputs={p: sha(ROOT / p) for p in [CSV, AUDIT]},
        software={p: importlib.metadata.version(p) for p in ['matplotlib', 'numpy']},
        points_checked=80, training_seeds_per_arm=10, held_out_episodes_per_seed=100,
        means={condition: statistics.mean(values) for condition, values in points.items()},
        plotting=dict(box_whiskers='1.5 IQR', triangles='arithmetic means',
                      seed_offsets='fixed horizontal offsets only', seed_colors='tab10, seed order 0-9'),
        pdf_sha256=sha(pdf), new_experiments=False)
    (output / 'figure4_rebuild.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'manuscript/figures')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        points = checked_points()
        print(json.dumps(dict(points_checked=sum(map(len, points.values())), conditions=len(points))))
    else:
        print(json.dumps(build(args.out_dir), indent=2))


if __name__ == '__main__':
    main()
