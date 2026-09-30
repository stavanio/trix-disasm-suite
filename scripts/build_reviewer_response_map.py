#!/usr/bin/env python3
"""Map the response draft to manuscript pages, source ranges and saved evidence.

Build after `make paper`; --check verifies the saved map without a TeX runtime.
No training, simulation, provider calls or hardware execution are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = 'manuscript/TRIX_REVISION.tex'
SUPPLEMENT = 'manuscript/TRIX_SUPPLEMENT.tex'
CROSS = 'manuscript/data/cross_document_refs.tex'
LETTER = 'manuscript/TRIX_RESPONSE.tex'
MAP = 'manuscript/data/reviewer_response_map.json'
LOCATIONS = 'manuscript/data/response_locations.tex'
GUIDE = 'docs/REVIEWER_MAP.md'
COMMENT_SOURCE = 'manuscript/data/reviewer_comment_source.json'
EXPECTED = ([f'E.{i}' for i in range(1, 7)]
            + [f'R1.{i}' for i in range(1, 10)]
            + [f'R2.{i}' for i in range(1, 8)]
            + [f'R4.{i}' for i in range(1, 14)])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(text, start):
    assert text[start] == '{'
    depth = 1
    i = start + 1
    while depth:
        if text[i] in '{}':
            escapes = 0
            j = i - 1
            while j >= 0 and text[j] == '\\':
                escapes += 1
                j -= 1
            if escapes % 2 == 0:
                depth += 1 if text[i] == '{' else -1
        i += 1
    return text[start + 1:i - 1], i


def norm(text):
    return ' '.join(text.split())


def quote_tex(text):
    """Escape verified review text, preserving words and original spelling."""
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
               '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '^': r'\textasciicircum{}', '~': r'\textasciitilde{}',
               '\u2011': '-', '\u2010': '-', '\u00a0': ' '}
    parts = re.split(r'(https?://\S+)', text)
    return ''.join(r'\url{' + part + '}' if part.startswith(('https://', 'http://'))
                   else ''.join(escapes.get(c, c) for c in part) for part in parts)


def response_entries(text):
    originals = json.loads((ROOT / COMMENT_SOURCE).read_text())['comments']
    starts = list(re.finditer(r'^\\response\{', text, re.M))
    entries = []
    for n, match in enumerate(starts):
        cursor = match.end() - 1
        values = []
        for _ in range(3):
            value, cursor = group(text, cursor)
            values.append(norm(value))
        end = starts[n + 1].start() if n + 1 < len(starts) else text.index('\\end{document}')
        body = text[cursor:end]
        paths = re.findall(r'\\evidence\{([^}]+)\}', body)
        original = originals[values[0]]
        assert values[2] == norm(quote_tex(original['original_excerpt'])), values[0]
        entries.append(dict(id=values[0], title=values[1], comment_quote=original['original_excerpt'],
            response_source_lines=[text.count('\n', 0, match.start()) + 1,
                                   text.count('\n', 0, end)],
            sections=list(dict.fromkeys(re.findall(r'\\mssec\{([^}]+)\}', body))),
            items=list(dict.fromkeys(re.findall(r'\\msitem\{([^}]+)\}', body))),
            citations=list(dict.fromkeys(re.findall(r'\\mscite\{([^}]+)\}', body))),
            evidence_paths=[x for v in paths for x in v.split('; ')],
            status=('timing_benchmark_reported' if values[0] in {'E.4', 'R1.6'}
                    else 'matched_command_QP_scope' if values[0] == 'R4.2'
                    else 'response_drafted'),
            original_comment_verified=original['original_comment_verified'],
            original_source_pages=original['source_pdf_pages'],
            coverage_status=original['coverage_status']))
    assert [e['id'] for e in entries] == EXPECTED
    return entries


def expanded_tex(path):
    text = (ROOT / path).read_text()
    def include(match):
        name = match.group(1)
        return '' if name == CROSS else expanded_tex(name)
    return re.sub(r'\\input\{([^}]+)\}', include, text)


def cross_items():
    items = {}
    for source, prefix in [(PAPER, ''), (SUPPLEMENT, 'S')]:
        counters = {'fig:': 0, 'tab:': 0}
        for label in re.findall(r'\\label\{([^}]+)\}', expanded_tex(source)):
            family = next((k for k in counters if label.startswith(k)), None)
            if family:
                assert label not in items, label
                counters[family] += 1
                items[label] = dict(source=source, number=prefix+str(counters[family]),
                    kind='Figure' if family == 'fig:' else 'Table')
    return items


def cross_tex():
    lines = ['% Generated from display-item order; checked against both compiled documents.',
        r'\newcommand{\suppref}[1]{\csname supitem#1\endcsname}',
        r'\newcommand{\mainref}[1]{\csname mainitem#1\endcsname}']
    for label, item in cross_items().items():
        is_si = item['source'] == SUPPLEMENT
        macro = 'supitem' if is_si else 'mainitem'
        prefix = 'Supplementary ' if is_si else 'main '
        lines.append(f"\\expandafter\\def\\csname {macro}{label}\\endcsname{{{prefix}{item['kind']}~{item['number']}}}")
    for document, prefix, macro in [(PAPER, '', 'mainsection'), (SUPPLEMENT, 'S', 'suppsection')]:
        counts = [0, 0, 0]
        levels = {'section': 0, 'subsection': 1, 'subsubsection': 2}
        lines.append('\\newcommand{\\' + macro + r'}[1]{\csname sectiontitle#1\endcsname}')
        for match in re.finditer(r'^\\(section|subsection|subsubsection)\{([^}]+)\}', (ROOT/document).read_text(), re.M):
            level = levels[match[1]]
            counts[level] += 1
            for j in range(level+1,3): counts[j] = 0
            key = prefix + '.'.join(str(n) for n in counts[:level+1])
            owner = 'Supplementary Information' if prefix else 'main text'
            lines.append(r'\expandafter\def\csname sectiontitle'+key+r'\endcsname{'+owner+", ``"+match[2]+"''}")
    return '\n'.join(lines)+'\n'


def paper_locations(text, aux, document=PAPER):
    source = []
    pattern = r'^\\(section|subsection|subsubsection)\{'
    for m in re.finditer(pattern, text, re.M):
        title, _ = group(text, m.end() - 1)
        source.append(dict(level=m.group(1), title=norm(title),
                           line=text.count('\n', 0, m.start()) + 1))
    toc = []
    for m in re.finditer(r'\\contentsline \{(section|subsection|subsubsection)\}\{\\numberline \{([^}]+)\}', aux):
        # Titles in this manuscript contain no nested braces.
        tail = re.match(r'(.*?)\}\{(\d+)\}\{', aux[m.end():])
        assert tail, m.group(2)
        toc.append(dict(number=m.group(2), title=norm(tail.group(1)), page=int(tail.group(2))))
    assert len(source) == len(toc), (len(source), len(toc))
    levels = {'section': 1, 'subsection': 2, 'subsubsection': 3}
    sections = {}
    for i, (s, t) in enumerate(zip(source, toc)):
        assert s['title'] == t['title'], (s, t)
        end = len(text.splitlines())
        for later in source[i + 1:]:
            if levels[later['level']] <= levels[s['level']]:
                end = later['line'] - 1
                break
        sections[t['number']] = dict(title=t['title'], page=t['page'],
            source=document, source_lines=[s['line'], end], heading_level=s['level'])
    labels = {}
    for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}', aux):
        name = m.group(1)
        owner = document
        raw = text
        if f'\\label{{{name}}}' not in raw:
            for included in re.findall(r'\\input\{([^}]+)\}', text):
                included_text = (ROOT / included).read_text()
                if f'\\label{{{name}}}' in included_text:
                    owner, raw = included, included_text
                    break
        literal = f'\\label{{{name}}}'
        assert literal in raw, name
        labels[name] = dict(number=m.group(2), page=int(m.group(3)), source=owner, document=document,
                            source_line=raw[:raw.index(literal)].count('\n') + 1)
    citations = dict(re.findall(r'\\bibcite\{([^}]+)\}\{([^}]+)\}', aux))
    return sections, labels, citations


def location_tex(data):
    lines = ['% Generated by scripts/build_reviewer_response_map.py; do not hand-edit.',
             '% Manuscript SHA-256: ' + data['manuscript_sha256']]
    used_sections = {k for e in data['responses'] for k in e['sections']}
    used_items = {k for e in data['responses'] for k in e['items']}
    used_cites = {k for e in data['responses'] for k in e['citations']}
    for k in sorted(used_sections):
        value = data['sections'][k]
        prefix = 'Supplementary ' if value['source'] == SUPPLEMENT else ''
        lines.append(f"\\expandafter\\def\\csname mssection{k}\\endcsname{{{prefix}``{value['title']}'' (p.~{value['page']})}}")
    for k in sorted(used_items):
        value = data['labels'][k]
        kind = 'Figure' if k.startswith('fig:') else 'Table' if k.startswith('tab:') else 'Section'
        if kind == 'Section':
            section = data['sections'][value['number']]
            prefix = 'Supplementary ' if value['document'] == SUPPLEMENT else ''
            lines.append(f"\\expandafter\\def\\csname msitem{k}\\endcsname{{{prefix}``{section['title']}'' (p.~{value['page']})}}")
            continue
        if value['document'] == SUPPLEMENT:
            kind = 'Supplementary ' + kind
        lines.append(f"\\expandafter\\def\\csname msitem{k}\\endcsname{{{kind}~{value['number']} (p.~{value['page']})}}")
    for k in sorted(used_cites):
        lines.append(f"\\expandafter\\def\\csname mscite{k}\\endcsname{{{data['citations'][k]}}}")
    return '\n'.join(lines) + '\n'


def check_reviewer_package(package_root, article_bundle, archive_root=None):
    """Resolve every evidence line against the delivered ZIP/TAR members."""
    letter = (ROOT / LETTER).read_text()
    paths = sorted({path.strip()
                    for block in re.findall(r'\\evidence\{([^}]+)\}', letter)
                    for path in block.split(';')})
    resolved = {}
    source_name = 'TRIX_manuscript_source.zip'
    with zipfile.ZipFile(package_root / source_name) as source:
        members = set(source.namelist())
        for path in paths:
            if path.startswith('archive/'):
                continue
            assert path in members, f'Missing reviewer source member: {path}'
            assert source.read(path) == (ROOT / path).read_bytes(), f'Stale reviewer source: {path}'
            resolved[path] = dict(container=source_name, member=path, kind='file')
        aliases = {PAPER: 'TRIX_MAIN.tex', SUPPLEMENT: 'TRIX_SUPPLEMENT.tex'}
        with zipfile.ZipFile(article_bundle) as article:
            for path, alias in aliases.items():
                assert source.read(path) == article.read(alias), f'Article/source mismatch: {alias}'
    archive_name = 'TRIX_reproducibility_release.tar'
    archive_paths = [p for p in paths if p.startswith('archive/')]
    matches = {p: [] for p in archive_paths}
    with tarfile.open(package_root / archive_name) as archive:
        for member in archive:
            for path in archive_paths:
                target = 'TRIX_reproducibility_release/' + path[len('archive/'):]
                if member.name == target or member.name.startswith(target + '/'):
                    matches[path].append(member.name)
                    if member.name == target and member.isfile() and archive_root:
                        with archive.extractfile(member) as stream:
                            assert stream.read() == (archive_root / path[len('archive/'):]).read_bytes(), path
        for path, members in matches.items():
            assert members, f'Missing reviewer archive member: {path}'
            target = 'TRIX_reproducibility_release/' + path[len('archive/'):]
            resolved[path] = dict(container=archive_name, member=target,
                kind='file' if members == [target] else 'directory', matched_members=len(members))
    assert len(resolved) == len(paths)
    return dict(status='passed', evidence_paths=len(paths), source_files=len(paths)-len(archive_paths),
        archive_paths=len(archive_paths), article_aliases=aliases, resolved=resolved,
        remote_reviewer_access_tested=False)


def check(data, archive_root=None, reviewer_package=None, article_bundle=None):
    assert digest(ROOT / PAPER) == data['manuscript_sha256'], 'Manuscript changed; rebuild response map.'
    assert digest(ROOT / SUPPLEMENT) == data['supplement_sha256'], 'Supplement changed; rebuild response map.'
    assert digest(ROOT / LETTER) == data['response_sha256'], 'Response changed; rebuild response map.'
    assert response_entries((ROOT / LETTER).read_text()) == data['responses']
    assert (ROOT / LOCATIONS).read_text() == location_tex(data)
    originals = json.loads((ROOT / COMMENT_SOURCE).read_text())
    assert data['comment_source']['original_reports_available'] is True
    assert digest(ROOT / COMMENT_SOURCE) == data['comment_source']['excerpts_sha256']
    assert set(originals['comments']) == set(EXPECTED)
    assert originals['r3']['verified'] and not originals['r3']['separate_substantive_requests']
    assert all(e['original_comment_verified'] for e in data['responses'])
    assert originals['r4_general']['original_comment_verified']
    assert quote_tex(originals['r4_general']['original_excerpt']) in (ROOT / LETTER).read_text()
    for s in data['sections'].values():
        paper_lines = (ROOT / s['source']).read_text().splitlines()
        assert s['title'] in norm(paper_lines[s['source_lines'][0] - 1])
    assert (ROOT / CROSS).read_text() == cross_tex()
    manifest = json.loads((ROOT/'docs/repository_manifest.json').read_text())
    mapped_items = {item['label']: item for item in manifest['paper_items']}
    for label, item in cross_items().items():
        assert data['labels'][label]['number'] == item['number'], label
        assert mapped_items[label]['number'] == item['number'], label
        assert mapped_items[label]['document'] == item['source'], label
    for path in (PAPER, SUPPLEMENT):
        for macro, label in re.findall(r'\\(suppref|mainref)\{([^}]+)\}', (ROOT/path).read_text()):
            target = SUPPLEMENT if macro == 'suppref' else PAPER
            assert data['labels'][label]['document'] == target, label
    for label, info in data['labels'].items():
        line = (ROOT / info['source']).read_text().splitlines()[info['source_line'] - 1]
        assert f'\\label{{{label}}}' in line
    for e in data['responses']:
        assert set(e['sections']) <= data['sections'].keys()
        assert set(e['items']) <= data['labels'].keys()
        assert set(e['citations']) <= data['citations'].keys()
        assert e['evidence_paths']
    for path, sha in data['local_evidence_sha256'].items():
        assert digest(ROOT / path) == sha, path
    timing = data['timing_evidence']
    assert digest(ROOT / timing['summary']) == timing['summary_sha256']
    summary = json.loads((ROOT / timing['summary']).read_text())
    assert summary['matched_screw_pass']
    assert summary['measured_calls'] == summary['inputs_per_task'] * summary['fresh_processes'] * summary['arms']
    manifest = json.loads((ROOT / timing['manifest']).read_text())
    for name, expected in manifest['raw_sha256'].items():
        assert digest(ROOT / 'results/runtime' / name) == expected, name
    for name, expected in manifest['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    if archive_root:
        assert digest(archive_root/'SHA256SUMS.jsonl') == data['archive_manifest_sha256']
        for path in data['archive_evidence_paths']:
            assert (archive_root/path).exists(), path
    packaged = check_reviewer_package(reviewer_package, article_bundle, archive_root) if reviewer_package else None
    print(json.dumps(dict(responses=len(data['responses']),
        original_comment_verification_pending=sum(not e['original_comment_verified'] for e in data['responses']),
        editorial_requirements=len(originals['editorial_requirements']), timing_requests_open=[],
        timing_calls_verified=summary['measured_calls'],
        matched_QP_scope_explicit=True, local_evidence_files=len(data['local_evidence_sha256']),
        missing_evidence_paths=0, manuscript_source_references_verified=True,
        final_journal_line_numbers=False, reviewer_package=packaged), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--cross-references', action='store_true')
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--reviewer-package', type=Path,
                        help='Directory containing the prepared reviewer source ZIP and evidence TAR.')
    parser.add_argument('--article-bundle', type=Path,
                        help='Article LaTeX ZIP; required with --reviewer-package to verify export aliases.')
    args = parser.parse_args()
    if bool(args.reviewer_package) != bool(args.article_bundle):
        parser.error('--reviewer-package and --article-bundle must be supplied together')
    if args.cross_references:
        (ROOT / CROSS).write_text(cross_tex())
        return
    if args.check:
        check(json.loads((ROOT/MAP).read_text()), args.archive_root,
              args.reviewer_package, args.article_bundle)
        return
    text = (ROOT/PAPER).read_text()
    letter = (ROOT/LETTER).read_text()
    aux = (ROOT/'manuscript/build/TRIX_REVISION.aux').read_text()
    sections, labels, citations = paper_locations(text, aux)
    si_sections, si_labels, _ = paper_locations((ROOT/SUPPLEMENT).read_text(),
        (ROOT/'manuscript/build/TRIX_SUPPLEMENT.aux').read_text(), SUPPLEMENT)
    assert not sections.keys() & si_sections.keys()
    assert not labels.keys() & si_labels.keys()
    sections.update(si_sections)
    labels.update(si_labels)
    entries = response_entries(letter)
    paths = {p for e in entries for p in e['evidence_paths']}
    local_paths = sorted(p for p in paths if not p.startswith('archive/'))
    archive = json.loads((ROOT/'assets/evidence/archive_manifest.json').read_text())
    originals = json.loads((ROOT/COMMENT_SOURCE).read_text())
    data = dict(schema=2, stage='author_review_draft',
        manuscript_sha256=digest(ROOT/PAPER), response_sha256=digest(ROOT/LETTER),
        supplement_sha256=digest(ROOT/SUPPLEMENT),
        manuscript_pdf_sha256_at_build=digest(ROOT/'manuscript/build/TRIX_REVISION.pdf'),
        supplement_pdf_sha256_at_build=digest(ROOT/'manuscript/build/TRIX_SUPPLEMENT.pdf'),
        archive_manifest_sha256=archive['sha256_manifest'],
        comment_source=dict(original_reports_available=True,
            kind='verbatim excerpts from the original decision letter; whitespace and typographic hyphens normalized',
            excerpts=COMMENT_SOURCE, excerpts_sha256=digest(ROOT/COMMENT_SOURCE),
            original_pdf=originals['source'], r3=originals['r3'],
            numbering_note=originals['numbering_note'],
            current_evidence_overrides_historical_summary=True),
        editorial_requirements=originals['editorial_requirements'],
        page_reference_basis='Separate main/supplement builds; named headings and S-prefixed supplementary display items; section IDs are internal mapping keys; source lines are not typeset line numbers',
        sections=sections, labels=labels, citations=citations, responses=entries,
        local_evidence_sha256={p:digest(ROOT/p) for p in local_paths},
        archive_evidence_paths=sorted(p[8:] for p in paths if p.startswith('archive/')),
        timing_evidence=dict(summary='results/runtime/summary.json',
            summary_sha256=digest(ROOT/'results/runtime/summary.json'),
            manifest='results/runtime/manifest.json',
            scope='Measured complete-filter calls across six tasks; matched SCREW solvers'),
        submission_gates=['Complete the actual linked editorial requirements table',
            'Complete NSF award details and corresponding-author ORCID confirmation',
            'Release and test public code at resubmission; private evidence reviewer access is verified', 'Author review and clean/marked submission',
            'Final journal page/line references'])
    (ROOT/MAP).write_text(json.dumps(data, indent=2)+'\n')
    (ROOT/LOCATIONS).write_text(location_tex(data))
    guide = ['# Reviewer response map', '',
        'Author-review draft: all 35 requests are mapped to the original decision letter',
        'dated 20 July 2026 and the current manuscript. Comments reproduce verified quotations.',
        'The earlier handoff omitted editor bullet 5. It is now E.5; generality is E.6.',
        'R3 is a confirmed co-review acknowledgement with no separate substantive requests.', '',
        '[Response LaTeX](../manuscript/TRIX_RESPONSE.tex) ·',
        '[Exact source ranges and evidence hashes](../manuscript/data/reviewer_response_map.json)',
        '[Original excerpts and requirement mapping](../manuscript/data/reviewer_comment_source.json)',
        '', 'Build: `make response`. Check without TeX: `make check`.', '',
        'Check every evidence line against the actual prepared delivery containers:', '',
        '```sh',
        'python3 scripts/build_reviewer_response_map.py --check \\',
        '  --reviewer-package /path/to/TRIX_reviewer_files \\',
        '  --article-bundle /path/to/01_Article_LaTeX.zip',
        '```', '',
        'Unprefixed evidence paths resolve inside `TRIX_manuscript_source.zip`.',
        '`archive/` resolves inside `TRIX_reproducibility_release.tar`, under',
        '`TRIX_reproducibility_release/`. The article bundle exports the same main',
        'and SI sources as `TRIX_MAIN.tex` and `TRIX_SUPPLEMENT.tex`; the check',
        'compares their bytes. Container checks do not establish remote reviewer access.', '',
        'Section/page references come from separate main and supplement builds. S-prefixed',
        'locations belong to Supplementary Information. Source ranges',
        'are TeX file lines, not journal margin line numbers.', '',
        'R4.1-R4.13 are response identifiers for the unnumbered original report.',
        'Verified coverage is not a claim of reviewer acceptance: R4.2 supplies a',
        'matched command-set QP, and R4.6 withdraws independent physical-validation',
        'claims. Those scope choices are explicit in the replies.', '',
        '| ID | Request | Draft status | Manuscript sections |', '|---|---|---|---|']
    for e in entries:
        guide.append('| '+e['id']+' | '+e['title']+' | '+e['status'].replace('_',' ')+' | '+', '.join(e['sections'])+' |')
    guide += ['', '## Submission gates', ''] + ['- '+g+'.' for g in data['submission_gates']]
    guide += ['', '## Original editorial requirements', '']
    for requirement in originals['editorial_requirements']:
        guide.append('- '+requirement['request']+' Status: '+requirement['status']+'.')
    guide += ['', '## Evidence and privacy', '',
        'Local evidence paths and SHA-256 values are in the JSON map. Hardware files',
        'and the complete record-to-shard map remain in the separate frozen archive',
        'identified by [archive_manifest.json](../assets/evidence/archive_manifest.json).',
        'The evidence is deposited in unpublished Zenodo draft 23031294. Its confidential',
        'read-only preview link is supplied in the editor correspondence and is excluded',
        'from this repository. Access and representative downloads were tested without',
        'account authentication. No public archive URI or DOI is claimed.', '',
        'For transfer reliability, the main tar is supplied as 24 ordered byte parts.',
        '`ARCHIVE_PARTS.json` and `reassemble_archive.py` verify and reconstruct the',
        'original tar before extraction; internal evidence paths are unchanged.', '',
        'The earlier R4.9 fragment outside this repository is superseded by the complete',
        'response source. Historical scientific records are retained. The reviewer',
        'archive documents its setup-photo privacy derivative and hash mapping in',
        '`PRIVACY_REDACTIONS.json`.', '']
    (ROOT/GUIDE).write_text('\n'.join(guide))
    check(data, args.archive_root, args.reviewer_package, args.article_bundle)


if __name__ == '__main__':
    main()
