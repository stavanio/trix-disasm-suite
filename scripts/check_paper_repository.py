#!/usr/bin/env python3
"""Check the curated Git payload without importing experiments or running them."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import statistics

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {'baselines', 'benchmark', 'envs', 'experiments', 'scripts', 'training', 'tests'}


def policy_summary(tex):
    """Reconcile the condensed main table with recorded selected-policy counts."""
    start = tex.index(r'\label{tab:policy-summary}')
    table = tex[start:tex.index(r'\end{table}', start)]
    rows = re.findall(
        r'^(\w+) & (SAC|PPO) & Stage ([12]) & ([^&]+) & ([\d.]+) & ([\d.]+)',
        table, re.M)
    assert len(rows) == 8
    checked = 0
    for task, algorithm, stage, pair, first, second in rows:
        records = json.loads((ROOT / f'results/stage{stage}/stage{stage}_records.json').read_text())['records']
        names = {'Matched QP': 'qp_matched', 'Box': 'box_clip',
                 'Static': 'static_clip', 'Preventive box': 'box_clip_preventive',
                 'TRiX': 'trix_preventive' if task == 'BATTERY' else 'trix'}
        arms = [names[name.strip()] for name in pair.split('/')]
        for arm, printed in zip(arms, (first, second)):
            selected = [r for r in records if r['task'] == task
                        and r['algorithm'] == algorithm.lower()
                        and r['evaluation_arm'] == arm]
            assert sorted(r['seed'] for r in selected) == list(range(10))
            assert all(r['counts']['episodes'] == 100 for r in selected)
            mean = statistics.mean(100*r['counts']['safe_completions']/r['counts']['episodes']
                                   for r in selected)
            assert abs(mean-float(printed)) < 0.000001, (task, algorithm, stage, arm, mean, printed)
            checked += 1
    return checked


def imports(paths):
    modules = {p[:-3].replace('/', '.').removesuffix('.__init__'): p for p in paths if p.endswith('.py')}
    edges = {p: set() for p in paths if p.endswith('.py')}
    for p in edges:
        for node in ast.walk(ast.parse((ROOT / p).read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ''
                if node.level:
                    name = (p.split('/')[0] + '.' + name).rstrip('.')
                names = [name] + [name + '.' + a.name for a in node.names]
                if name.split('.')[0] in PACKAGES and name not in modules:
                    raise AssertionError(f'{p}: missing local import {name}')
            for name in names:
                if name in modules:
                    edges[p].add(modules[name])
                elif name.split('.')[0] in PACKAGES and isinstance(node, ast.Import):
                    raise AssertionError(f'{p}: missing local import {name}')
                if name.split('.')[0] in modules:
                    edges[p].add(modules[name.split('.')[0]])
                if 'scripts.' + name in modules:
                    edges[p].add(modules['scripts.' + name])
    return edges


def main():
    manifest = json.loads((ROOT / 'docs/repository_manifest.json').read_text())
    files = manifest['files']
    candidates = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    actual = {p for p in candidates if p and (ROOT / p).is_file()}
    assert actual == set(files), {'unmapped': sorted(actual - set(files)), 'missing': sorted(set(files) - actual)}
    for path, entry in files.items():
        assert entry['role'] and entry['references'], f'Unmapped purpose: {path}'
        assert not (ROOT / path).is_symlink()
        assert (ROOT / path).stat().st_size < 5_000_000, f'Large artifact belongs in evidence archive: {path}'
    code = {p for p in files if p.endswith('.py')}
    edges = imports(code)
    for caller, deps in manifest['explicit_dependencies'].items():
        assert caller in code and set(deps) <= code
        edges[caller].update(deps)
    roots = set(manifest['entrypoints']) | set(manifest['tests']) | {'setup.py'}
    reached = set(roots)
    todo = list(roots)
    while todo:
        for dependency in edges.get(todo.pop(), set()):
            if dependency not in reached:
                reached.add(dependency)
                todo.append(dependency)
    initializers = {p for p in code if p.endswith('/__init__.py')}
    assert not code - reached - initializers, f'Orphan code: {sorted(code - reached - initializers)}'
    for test in manifest['tests']:
        assert edges[test] - initializers, f'Test has no retained source dependency: {test}'
    def expand(path):
        source = (ROOT / path).read_text()
        return source + ''.join('\n' + expand(name)
            for name in re.findall(r'\\input\{([^}]+)\}', source))
    tex = '\n'.join(expand(path) for path in
        ['manuscript/TRIX_REVISION.tex', 'manuscript/TRIX_SUPPLEMENT.tex'])
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    table_cells = policy_summary(tex)
    refs = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
    covered = set()
    for item in manifest['paper_items']:
        assert item['label'] in labels, item['label']
        assert set(item['citations']) <= refs, item
        assert set(item['files']) <= set(files), item
        assert item['command_or_check']
        covered.add(item['label'])
    assert {x for x in labels if x.startswith(('fig:', 'tab:'))} == covered
    for source in re.findall(r'\\(?:includegraphics(?:\[[^]]*\])?|input)\{([^}]+)\}', tex):
        assert source in files, source
    for p in [ROOT/'README.md', ROOT/'docs/PAPER_MAP.md', ROOT/'docs/REPRODUCING.md', ROOT/'docs/REVIEWER_MAP.md']:
        for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' not in link and not link.startswith('#'):
                target = (p.parent / link.split('#')[0]).resolve()
                assert target.is_file(), (p, link)
    for name, h in manifest['locked_renderer_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h, name
    archive = json.loads((ROOT / 'assets/evidence/archive_manifest.json').read_text())
    assert archive['remote_uri'] is None  # No fabricated public archive/DOI.
    assert re.fullmatch('[0-9a-f]{64}', archive['sha256_manifest'])
    sizes = sum((ROOT/p).stat().st_size for p in files)
    print(json.dumps(dict(files=len(files), bytes=sizes, code_files=len(code),
        paper_figures_and_tables=len(covered), tests=len(manifest['tests']),
        policy_summary_cells_verified=table_cells,
        orphan_code=0, missing_local_dependencies=0, broken_map_links=0,
        locked_renderer_files=len(manifest['locked_renderer_files'])), indent=2))


if __name__ == '__main__':
    main()
