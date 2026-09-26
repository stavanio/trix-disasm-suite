# Reviewer response map

Author-review draft: all 34 recovered points have responses. Original reports
are not available in the checked local material; summaries are paraphrases.
Completeness against the actual decision letter is not yet verified.

[Response LaTeX](../manuscript/TRIX_RESPONSE.tex) ·
[Exact source ranges and evidence hashes](../manuscript/data/reviewer_response_map.json)

Build: `make response`. Check without TeX: `make check`.

Section/page references are generated from the manuscript build. Source ranges
are TeX file lines, not journal margin line numbers.

| ID | Request | Draft status | Manuscript sections |
|---|---|---|---|
| E.1 | Positioning among existing safety methods | response drafted | 1.2, 2.1, 2.3, 2.6 |
| E.2 | Introduce the architecture figure in context | response drafted | 1.1 |
| E.3 | Safety, damage and task quality | response drafted | 3.1, 4.2, 5.5.1, A |
| E.4 | Quantitative latency | timing benchmark reported | 3.5, 4.9, H |
| E.5 | Generality beyond six task implementations | response drafted | 3.1, 3.4, 4.10, 5.3.2, 5.5.5 |
| R1.1 | References in comparison tables | response drafted | 4.6 |
| R1.2 | Accessibility of the theoretical framework | response drafted | 1.1, 3.1, 3.4, 3.7 |
| R1.3 | Simulated workspace and interaction | response drafted | 4.1, 4.3 |
| R1.4 | Concrete model proposals and corrections | response drafted | 3.8, 4.12, 5.4.1 |
| R1.5 | Adding or changing constraints | response drafted | 3.1, 3.4, 4.9, 4.10, 5.3, 6.2 |
| R1.6 | Computational time | timing benchmark reported | 4.9, H |
| R1.7 | Distributions and uncertainty | response drafted | 4.13, F |
| R1.8 | Figure references and Figure~4 quality | response drafted | 1.1, 3.2, 4.3, 5.2, 5.6 |
| R1.9 | Foundation-model robotics literature | response drafted | 1.1, 2.4 |
| R2.1 | PCB variables and physical interpretation | response drafted | 3.3, B.2, D.1, D.2 |
| R2.2 | Observation dimensionality | response drafted | 4.4, D.1 |
| R2.3 | Reward symbols and outcome notation | response drafted | 4.2, 4.4, 4.7, D.3, D.4 |
| R2.4 | Domain randomisation appendix | response drafted | D.6, 4.9, 4.11 |
| R2.5 | Placement of runtime discussion | response drafted | 3.2, 4.9, H |
| R2.6 | Orphaned Figure~2 reference | response drafted | 1.3, 3.2 |
| R2.7 | Narrative structure | response drafted | 1.2, 1.3, 3.2, 3.4, 5.2, 5.3, 6.1 |
| R4.1 | Trained policies and experimental provenance | response drafted | 4.4, 4.7, 4.13, 4.15, E, F |
| R4.2 | Matched analytical comparator | matched command QP scope | 2.3, 3.5, 4.6, 4.9, 5.1.1, B.1 |
| R4.3 | CBF capabilities | response drafted | 2.3 |
| R4.4 | ISS theorem assumptions | response drafted | 3.4, 3.6, 3.7, C |
| R4.5 | Non-expansiveness of projection | response drafted | 3.1, 3.5, 3.7, B, C |
| R4.6 | Circular validation | response drafted | 3.2, 4.1, 4.9, 5.5.3, 6.2.1, D.4, G, H |
| R4.7 | Role of PyBullet | response drafted | 4.1, 4.3 |
| R4.8 | Physical robot evidence | response drafted | 4.14, 5.6, 6.2.1 |
| R4.9 | Actual VLM evaluation | response drafted | 4.12, 4.13, 5.4 |
| R4.10 | BATTERY threshold and thermal interpretation | response drafted | 4.8, 5.5.1, 5.5.3, B.3 |
| R4.11 | Simulation and control rates | response drafted | 4.14, H |
| R4.12 | Figure~4 and Table~4 use different experiments | response drafted | 5.1, 5.2 |
| R4.13 | PCB residual violations and representation analysis | response drafted | 4.8, 5.2.3, 5.3.1, 5.3.2 |

## Submission gates

- Check original reports and exact editorial requirements.
- Review whether matched command-QP scope addresses original R4.2 wording.
- Arrange private reviewer access.
- Author review and clean/marked submission.
- Final journal page/line references.

## Evidence and privacy

Local evidence paths and SHA-256 values are in the JSON map. Hardware files
and the complete record-to-shard map remain in the separate frozen archive
identified by [archive_manifest.json](../assets/evidence/archive_manifest.json).
The archive has no remote URI. No push or visibility change is part of this build.

The earlier R4.9 fragment outside this repository is superseded by the complete
response source. The dated evidence archive itself remains unchanged.
