#!/usr/bin/env python3
"""Check the curated Git payload without importing experiments or running them."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {'baselines', 'benchmark', 'envs', 'experiments', 'scripts', 'training', 'tests'}


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
    tex = (ROOT / 'manuscript/TRIX_REVISION.tex').read_text()
    for included in re.findall(r'\\input\{([^}]+)\}', tex):
        tex += '\n' + (ROOT / included).read_text()
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
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
        orphan_code=0, missing_local_dependencies=0, broken_map_links=0,
        locked_renderer_files=len(manifest['locked_renderer_files'])), indent=2))


if __name__ == '__main__':
    main()
