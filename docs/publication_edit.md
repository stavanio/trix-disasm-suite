# Publication edit: main paper and Supplementary Information

Date: 25 September 2026. Source before this edit: `44e0e76`.

The previous 72-page combined manuscript has been edited into a concise main
paper and a separately compiled supplement. Internal XeLaTeX builds produce
21 main-paper pages, including Methods and references, and 43 supplementary
pages. The remapped response letter has 19 pages. This is an editorial reorganization
of the recorded evidence. Results, environment definitions and frozen artwork
retain their existing provenance and qualifications.

## Content destinations

| Material | Main manuscript | Supplementary Information |
|---|---|---|
| Contribution, related methods and execution contract | Introduction, Results 2.1, Tables 1 and 3 | S1 definitions and assumptions |
| Projection equations and four representation results | Results 2.1, Figures 1 and 2 | S2 operators and S3 proofs |
| Six environments, interfaces and rendered states | Results 2.2, Table 2, Figure 3, Methods 4.1 | S4 complete specifications, observations, scales, outcome taxonomy, margins and reset distributions |
| Policy adaptation and specification controls | Results 2.3, Table 4, Figure 4, Methods 4.2 and 4.3 | S5 training, S6 seed statistics, S9 extended results |
| Representation and prospective generalization | Results 2.4, Discussion, Methods 4.4 | S10 protocols, learned-model decomposition, procedural table and inconclusive BAYONET test |
| VLM decisions and prevented constraint violations | Abstract, Results 2.5, Table 5, Methods 4.5 and 4.6 | S11 full visual encoding, grounding, overlap, strata and model-provenance account |
| Physical deployment | Results 2.6, Figure 5, Methods 4.7 | S12 implementation details and trial records |
| Computational cost | Results 2.7, Methods 4.8 | S8 full measured workload and latency table |
| Model/specification and reactive-governance failures | Discussion, Methods 4.4 | S7 sensitivity/implementation analysis and S13 failure mechanisms |

The main manuscript has five figures and five tables. The supplement has eleven
tables. Existing stable labels are preserved when material moves; two compact
new tables give the main policy contrasts and complete physical action scales.
[The figure/table map](PAPER_MAP.md) records current numbers, files and checks.

## Evidence preservation and numerical checks

- All 55 pinned renderer, environment and artwork files retain their hashes.
- `make check` recomputes all 16 numeric cells in the new main policy table from
  the recorded selected-policy counts, using ten training seeds and 100 test
  episodes per policy/arm. It matches exact `evaluation_arm` names. BATTERY's
  main summary uses Stage 2 `box_clip_preventive` and `trix_preventive`.
- Supplementary action scales match the six frozen environment definitions:
  SCREW `(0.05,60,40)`, PCB `(0.5,0.5,60)`, SNAP `(25,40,20)`, CRANK `(30,30,40)`,
  BATTERY `(40,30,30)` and PRY `(2.5,80,40)`, with component units in the table.
- The full policy tables, 13 seed contrasts, procedural table, VLM accounting,
  hardware table and generated latency table retain their numeric contents.
  BATTERY's corrected Stage 1 SAC/TRiX rate remains 20.0%.
- VLM accounting remains 273 decisions: 16 refusals, 219 admissible and
  38 inadmissible commands. The 257 paired command replays contain 31 prevented
  constraint-violation cases. Display scaling, nonphysical displayed tilts,
  separate partial-view strata and unavailable backend/retry IDs remain explicit.
- All 225,280 timing calls, their raw hashes and experiment source hashes remain
  unchanged. The private 24.5 GB evidence archive is also unchanged.
- All 34 response entries now refer to main or S-prefixed supplementary sections
  with separate page numbering. The map records both source hashes and rejects
  stale references. The original reviewer reports still require verification.

## Rebuild and package

`make response` compiles both manuscripts to convergence, derives current display
numbers and reviewer locations, then compiles the response. `make check` verifies
file inventory, code dependencies, display mappings, numerical cells, frozen
hashes and response references. `make overleaf` packages both TeX entrypoints,
all included figures/tables, cross-document numbers and supplied text/math fonts.

The Overleaf entrypoints are `TRIX_MAIN.tex` and `TRIX_SUPPLEMENT.tex`; choose
XeLaTeX and select the document to compile. They are independent builds, with
no dependency on the other document's `.aux` file. The response is exported
separately with its generated location macros embedded.

PDFs are internal layout checks; delivery is editable TeX and source ZIPs.
The edit uses 12-point body text and line numbers. It removes repeated exposition
and moves detail to the supplement. It does not shrink body type to reduce pages.
No em dashes are used in the three TeX documents.

The journal's [formatting guidance](https://www.nature.com/commseng/submit/formatting-guidelines)
was checked for abstract length, manuscript organization, main-text length and
display-item count. Meeting those checks is not a declaration that submission
preparation is complete: original decision-letter verification, author review,
reviewer access and final clean/marked submission files remain open.
