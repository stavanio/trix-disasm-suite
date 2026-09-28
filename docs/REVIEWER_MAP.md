# Reviewer response map

Author-review draft: all 35 requests are mapped to the original decision letter
dated 20 July 2026 and the current manuscript. Comments reproduce verified quotations.
The earlier handoff omitted editor bullet 5. It is now E.5; generality is E.6.
R3 is a confirmed co-review acknowledgement with no separate substantive requests.

[Response LaTeX](../manuscript/TRIX_RESPONSE.tex) ·
[Exact source ranges and evidence hashes](../manuscript/data/reviewer_response_map.json)
[Original excerpts and requirement mapping](../manuscript/data/reviewer_comment_source.json)

Build: `make response`. Check without TeX: `make check`.

Check every evidence line against the actual prepared delivery containers:

```sh
python3 scripts/build_reviewer_response_map.py --check \
  --reviewer-package /path/to/TRIX_reviewer_files \
  --article-bundle /path/to/01_Article_LaTeX.zip
```

Unprefixed evidence paths resolve inside `TRIX_manuscript_source.zip`.
`archive/` resolves inside `TRIX_reproducibility_release.tar`, under
`TRIX_reproducibility_release/`. The article bundle exports the same main
and SI sources as `TRIX_MAIN.tex` and `TRIX_SUPPLEMENT.tex`; the check
compares their bytes. Container checks do not establish remote reviewer access.

Section/page references come from separate main and supplement builds. S-prefixed
locations belong to Supplementary Information. Source ranges
are TeX file lines, not journal margin line numbers.

R4.1-R4.13 are response identifiers for the unnumbered original report.
Verified coverage is not a claim of reviewer acceptance: R4.2 supplies a
matched command-set QP, and R4.6 withdraws independent physical-validation
claims. Those scope choices are explicit in the replies.

| ID | Request | Draft status | Manuscript sections |
|---|---|---|---|
| E.1 | Positioning among existing safety methods | response drafted | 1, 2.1, 4.3 |
| E.2 | Introduce the architecture figure in context | response drafted | 2.1, 1 |
| E.3 | Safety, damage and task quality | response drafted | 1, 3, 4.1, S1, S4.4, S13 |
| E.4 | Quantitative latency | timing benchmark reported | S8, 2.7, 4.8 |
| E.5 | Position TRiX against the reviewed approaches | response drafted | 2.2, 4.3, 1, 2.3 |
| E.6 | Generality beyond task examples | response drafted | 2.1, 2.4, 3, 4.4, S10 |
| R1.1 | References in comparison tables | response drafted | 4.3 |
| R1.2 | Accessibility of the theoretical framework | response drafted | S1, S2, S3, 1, 2.1 |
| R1.3 | Simulated workspace and interaction | response drafted | 2.2, 4.1, S4 |
| R1.4 | Concrete model proposals and corrections | response drafted | 2.5, 4.5, S11 |
| R1.5 | Adding or changing constraints | response drafted | 2.1, 2.4, 3, 4.4, S10 |
| R1.6 | Computational time | timing benchmark reported | S8, 2.7, 4.8 |
| R1.7 | Distributions and uncertainty | response drafted | 2.3, 4.6, S6 |
| R1.8 | Figure references and Figure~4 quality | response drafted | 1, 2.1, 2.2, 2.3, 2.6 |
| R1.9 | Foundation-model robotics literature | response drafted | 1, 2.5 |
| R2.1 | PCB variables and physical interpretation | response drafted | S4, S2.2, 2.2, 4.1, S4.1, S4.2 |
| R2.2 | Observation dimensionality | response drafted | 4.1, S4.1 |
| R2.3 | Reward symbols and outcome notation | response drafted | 4.1, 4.2, S4.3, S4.4 |
| R2.4 | Domain randomisation appendix | response drafted | S4.6, 4.6, S14.4, S7, 2.3, 3, 4.1, S14 |
| R2.5 | Placement of runtime discussion | response drafted | S8, 2.1, 2.7, 4.8 |
| R2.6 | Orphaned Figure~2 reference | response drafted | 2.1 |
| R2.7 | Narrative structure | response drafted | 1, 2.1, 2.3, 2.4, 3 |
| R4.1 | Trained policies and experimental provenance | response drafted | 4.2, 4.6, S5, S6, S9 |
| R4.2 | Matched analytical comparator | matched command QP scope | 2.1, 2.3, 4.3, S2.1, S8 |
| R4.3 | CBF capabilities | response drafted | 1, 4.3 |
| R4.4 | ISS theorem assumptions | response drafted | 2.1, 3, S3 |
| R4.5 | Non-expansiveness of projection | response drafted | 2.1, S1, S2, S3 |
| R4.6 | Circular validation | response drafted | S8, 2.1, 3, 4.4, S4.4, S7 |
| R4.7 | Role of PyBullet | response drafted | 2.2, 4.1, S4 |
| R4.8 | Physical robot evidence | response drafted | 2.6, 3, 4.7, S12 |
| R4.9 | Actual VLM evaluation | response drafted | 2.5, 4.5, 4.6, S11 |
| R4.10 | BATTERY threshold and thermal interpretation | response drafted | 2.3, 3, 4.3, S2.3, S7.1, S9.4, S13.1 |
| R4.11 | Simulation and control rates | response drafted | S4.2, 4.1, 4.7, 4.8, S8 |
| R4.12 | Figure~4 and Table~4 use different experiments | response drafted | 2.3, S9 |
| R4.13 | PCB residual violations and representation analysis | response drafted | 2.3, 2.4, 4.3, 4.4, S9.3, S10.2, S10.3 |

## Submission gates

- Complete the actual linked editorial requirements table.
- Confirm funding, acknowledgements and corresponding-author ORCID.
- Test public code access and private evidence reviewer access.
- Author review and clean/marked submission.
- Final journal page/line references.

## Original editorial requirements

- Complete the linked editorial requirements table, describing revisions and relevant notes in its right-hand column. Status: open; actual online table must be completed.
- Supply a point-by-point response; explain any requests that cannot be addressed or are considered invalid. Status: draft mapped to all original requests; author review pending.
- Supply a clean revised manuscript without markup. Status: final submission version pending.
- Supply a marked manuscript with all changes highlighted in a different colour. Status: open.
- Aim to return the revision within twelve weeks and notify the editor if substantially more time is needed. Status: scheduling instruction; no message sent.

## Evidence and privacy

Local evidence paths and SHA-256 values are in the JSON map. Hardware files
and the complete record-to-shard map remain in the separate frozen archive
identified by [archive_manifest.json](../assets/evidence/archive_manifest.json).
The archive has no remote URI. No push or visibility change is part of this build.

The earlier R4.9 fragment outside this repository is superseded by the complete
response source. The dated evidence archive itself remains unchanged.
