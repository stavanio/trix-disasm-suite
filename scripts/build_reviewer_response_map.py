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

ROOT = Path(__file__).resolve().parents[1]
PAPER = 'manuscript/TRIX_REVISION.tex'
SUPPLEMENT = 'manuscript/TRIX_SUPPLEMENT.tex'
CROSS = 'manuscript/data/cross_document_refs.tex'
LETTER = 'manuscript/TRIX_RESPONSE.tex'
MAP = 'manuscript/data/reviewer_response_map.json'
LOCATIONS = 'manuscript/data/response_locations.tex'
GUIDE = 'docs/REVIEWER_MAP.md'
EXPECTED = ([f'E.{i}' for i in range(1, 6)]
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


def response_entries(text):
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
        entries.append(dict(id=values[0], title=values[1], comment_summary=values[2],
            response_source_lines=[text.count('\n', 0, match.start()) + 1,
                                   text.count('\n', 0, end)],
            sections=list(dict.fromkeys(re.findall(r'\\mssec\{([^}]+)\}', body))),
            items=list(dict.fromkeys(re.findall(r'\\msitem\{([^}]+)\}', body))),
            citations=list(dict.fromkeys(re.findall(r'\\mscite\{([^}]+)\}', body))),
            evidence_paths=[x for v in paths for x in v.split('; ')],
            status=('timing_benchmark_reported' if values[0] in {'E.4', 'R1.6'}
                    else 'matched_command_QP_scope' if values[0] == 'R4.2'
                    else 'response_drafted'),
            original_comment_verified=False))
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
        lines.append(f"\\expandafter\\def\\csname mssection{k}\\endcsname{{{prefix}\\S\\,{k} (p.~{value['page']})}}")
    for k in sorted(used_items):
        value = data['labels'][k]
        kind = 'Figure' if k.startswith('fig:') else 'Table' if k.startswith('tab:') else 'Section'
        if value['document'] == SUPPLEMENT:
            kind = 'Supplementary ' + kind
        lines.append(f"\\expandafter\\def\\csname msitem{k}\\endcsname{{{kind}~{value['number']} (p.~{value['page']})}}")
    for k in sorted(used_cites):
        lines.append(f"\\expandafter\\def\\csname mscite{k}\\endcsname{{{data['citations'][k]}}}")
    return '\n'.join(lines) + '\n'


def check(data, archive_root=None):
    assert digest(ROOT / PAPER) == data['manuscript_sha256'], 'Manuscript changed; rebuild response map.'
    assert digest(ROOT / SUPPLEMENT) == data['supplement_sha256'], 'Supplement changed; rebuild response map.'
    assert digest(ROOT / LETTER) == data['response_sha256'], 'Response changed; rebuild response map.'
    assert response_entries((ROOT / LETTER).read_text()) == data['responses']
    assert (ROOT / LOCATIONS).read_text() == location_tex(data)
    assert data['comment_source']['original_reports_available'] is False
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
    print(json.dumps(dict(responses=len(data['responses']),
        original_comment_verification_pending=34, timing_requests_open=[],
        timing_calls_verified=summary['measured_calls'],
        matched_QP_scope_explicit=True, local_evidence_files=len(data['local_evidence_sha256']),
        missing_evidence_paths=0, manuscript_source_references_verified=True,
        final_journal_line_numbers=False), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--cross-references', action='store_true')
    parser.add_argument('--archive-root', type=Path)
    args = parser.parse_args()
    if args.cross_references:
        (ROOT / CROSS).write_text(cross_tex())
        return
    if args.check:
        check(json.loads((ROOT/MAP).read_text()), args.archive_root)
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
    data = dict(schema=2, stage='author_review_draft',
        manuscript_sha256=digest(ROOT/PAPER), response_sha256=digest(ROOT/LETTER),
        supplement_sha256=digest(ROOT/SUPPLEMENT),
        manuscript_pdf_sha256_at_build=digest(ROOT/'manuscript/build/TRIX_REVISION.pdf'),
        supplement_pdf_sha256_at_build=digest(ROOT/'manuscript/build/TRIX_SUPPLEMENT.pdf'),
        archive_manifest_sha256=archive['sha256_manifest'],
        comment_source=dict(original_reports_available=False,
            kind='paraphrases of historical revision map, not reviewer quotations',
            conversation_title='Trix v2.0', conversation_id='6a8e8a3e-0758-83e8-b35d-daf5c8c0e92f',
            turn_id='69543ecc-f4f8-4fa0-87fd-7c4face44268',
            current_evidence_overrides_historical_summary=True),
        page_reference_basis='Separate main/supplement builds; S-prefixed sections/items belong to the supplement; source lines are not typeset line numbers',
        sections=sections, labels=labels, citations=citations, responses=entries,
        local_evidence_sha256={p:digest(ROOT/p) for p in local_paths},
        archive_evidence_paths=sorted(p[8:] for p in paths if p.startswith('archive/')),
        timing_evidence=dict(summary='results/runtime/summary.json',
            summary_sha256=digest(ROOT/'results/runtime/summary.json'),
            manifest='results/runtime/manifest.json',
            scope='Measured complete-filter calls across six tasks; matched SCREW solvers'),
        submission_gates=['Check original reports and exact editorial requirements',
            'Review whether matched command-QP scope addresses original R4.2 wording',
            'Arrange private reviewer access', 'Author review and clean/marked submission',
            'Final journal page/line references'])
    (ROOT/MAP).write_text(json.dumps(data, indent=2)+'\n')
    (ROOT/LOCATIONS).write_text(location_tex(data))
    guide = ['# Reviewer response map', '',
        'Author-review draft: all 34 recovered points have responses. Original reports',
        'are not available in the checked local material; summaries are paraphrases.',
        'Completeness against the actual decision letter is not yet verified.', '',
        '[Response LaTeX](../manuscript/TRIX_RESPONSE.tex) ·',
        '[Exact source ranges and evidence hashes](../manuscript/data/reviewer_response_map.json)',
        '', 'Build: `make response`. Check without TeX: `make check`.', '',
        'Section/page references come from separate main and supplement builds. S-prefixed',
        'locations belong to Supplementary Information. Source ranges',
        'are TeX file lines, not journal margin line numbers.', '',
        '| ID | Request | Draft status | Manuscript sections |', '|---|---|---|---|']
    for e in entries:
        guide.append('| '+e['id']+' | '+e['title']+' | '+e['status'].replace('_',' ')+' | '+', '.join(e['sections'])+' |')
    guide += ['', '## Submission gates', ''] + ['- '+g+'.' for g in data['submission_gates']]
    guide += ['', '## Evidence and privacy', '',
        'Local evidence paths and SHA-256 values are in the JSON map. Hardware files',
        'and the complete record-to-shard map remain in the separate frozen archive',
        'identified by [archive_manifest.json](../assets/evidence/archive_manifest.json).',
        'The archive has no remote URI. No push or visibility change is part of this build.', '',
        'The earlier R4.9 fragment outside this repository is superseded by the complete',
        'response source. The dated evidence archive itself remains unchanged.', '']
    (ROOT/GUIDE).write_text('\n'.join(guide))
    check(data, args.archive_root)


if __name__ == '__main__':
    main()
