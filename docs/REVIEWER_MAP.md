# Reviewer response map

All 35 requests are mapped to the original decision letter dated 20 July 2026
and the current manuscript. Comments reproduce verified quotations.
R3 is a confirmed co-review acknowledgment with no separate substantive requests.

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

| ID | Request | Status | Manuscript sections |
|---|---|---|---|
| E.1 | Positioning among existing safety methods | response final | 1, 2.1, 2.2.2 |
| E.2 | Introduce the architecture figure in context | response final | 1, 2.1 |
| E.3 | Safety, damage and task quality | response final | 1, 4, 2.2.1, S1, S4.4, S13 |
| E.4 | Quantitative latency | timing benchmark reported | 3.4, 2.5, S8 |
| E.5 | Position TRiX against the reviewed approaches | response final | 1, 3.1.1, 3.1.2, 3.1.3, 2.2.2 |
| E.6 | Generality beyond task examples | response final | 3.1.4, 4, 2.1, 2.2.3, S10 |
| R1.1 | References in comparison tables | response final | 2.2.2 |
| R1.2 | Accessibility of the theoretical framework | response final | 1, 2.1, S1, S2, S3 |
| R1.3 | Simulated workspace and interaction | response final | 2.2.1, S4 |
| R1.4 | Concrete model proposals and corrections | response final | 3.2, 2.3, S11 |
| R1.5 | Adding or changing constraints | response final | 3.1.4, 4, 2.1, 2.2.3, S10 |
| R1.6 | Computational time | timing benchmark reported | 3.4, 2.5, S8 |
| R1.7 | Distributions and uncertainty | response final | 3.1.1, 3.1.2, 3.1.3, 2.5, S6 |
| R1.8 | Figure references and Figure~4 quality | response final | 1, 3.1.1, 3.3, 2.1, 2.2.1 |
| R1.9 | Foundation-model robotics literature | response final | 1, 3.2 |
| R2.1 | PCB variables and physical interpretation | response final | 2.2.1, S2.2, S4.1, S4.2 |
| R2.2 | Observation dimensionality | response final | 2.2.1, S4.1 |
| R2.3 | Reward symbols and outcome notation | response final | 2.2.1, 2.2.2, S4.3, S4.4 |
| R2.4 | Domain randomization appendix | response final | S7, 3.1.2, 3.1.5, 4, 2.2.1, 2.2.3, 2.5, S4.6, S14, S14.4 |
| R2.5 | Placement of runtime discussion | response final | 3.4, 2.1, 2.5, S8 |
| R2.6 | Orphaned Figure~2 reference | response final | 2.1 |
| R2.7 | Narrative structure | response final | 1, 3.1.1, 3.1.2, 3.1.4, 4, 2.1 |
| R4.1 | Trained policies and experimental provenance | response final | S9.4, 2.2.2, 2.5, S5, S6, S9 |
| R4.2 | Matched analytical comparator | matched command QP scope | S9.5, 3.1.2, 2.1, 2.2.2, 2.2.3, S2.1, S8 |
| R4.3 | CBF capabilities | response final | 1, 2.2.2 |
| R4.4 | ISS theorem assumptions | response final | 4, 2.1, S3 |
| R4.5 | Non-expansiveness of projection | response final | 2.1, S1, S2, S3 |
| R4.6 | Circular validation | response final | 4, 2.1, 2.2.3, S4.4, S7, S8 |
| R4.7 | Role of PyBullet | response final | 2.2.1, S4 |
| R4.8 | Physical robot evidence | response final | 3.3, 4, 2.4, S12 |
| R4.9 | Actual VLM evaluation | response final | S11, 3.2, 2.3, 2.5 |
| R4.10 | BATTERY threshold and thermal interpretation | response final | 3.1.3, 4, 2.2.2, S2.3, S7.1, S9.4, S13.1 |
| R4.11 | Simulation and control rates | response final | S4.2, 2.2.1, 2.4, 2.5 |
| R4.12 | Figure~4 and Table~4 use different experiments | response final | 3.1.1, S9 |
| R4.13 | PCB residual violations and representation analysis | response final | 3.1.1, 3.1.3, 3.1.4, 2.2.2, 2.2.3, S9.3, S10.2, S10.3 |

## Original editorial requirements

- Complete the linked editorial requirements table, describing revisions and relevant notes in its right-hand column. Status: completed in the online editorial form.
- Supply a point-by-point response; explain any requests that cannot be addressed or are considered invalid. Status: point-by-point response to all original requests.
- Supply a clean revised manuscript without markup. Status: clean revised manuscript supplied.
- Supply a marked manuscript with all changes highlighted in a different colour. Status: marked manuscript supplied.
- Aim to return the revision within twelve weeks and notify the editor if substantially more time is needed. Status: revision returned within the twelve-week window.

## Evidence and privacy

Local evidence paths and SHA-256 values are in the JSON map. Hardware files
and the complete record-to-shard map remain in the separate frozen archive
identified by [archive_manifest.json](../assets/evidence/archive_manifest.json).
The evidence is deposited in Zenodo (DOI 10.5281/zenodo.23031294), private during
peer review and public on publication. Its confidential read-only link is supplied
in the editor correspondence and is excluded from this repository. Access and
representative downloads were tested without account authentication.

For transfer reliability, the main tar is supplied as 24 ordered byte parts.
`ARCHIVE_PARTS.json` and `reassemble_archive.py` verify and reconstruct the
original tar before extraction; internal evidence paths are unchanged.

Historical scientific records are retained. The reviewer
archive documents its setup-photo privacy derivative and hash mapping in
`PRIVACY_REDACTIONS.json`.
