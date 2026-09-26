# Publication edit: main paper and Supplementary Information

Date: 25 September 2026. Source before this edit: `44e0e76`.

The previous 72-page combined manuscript has been edited into a concise main
paper and a separately compiled supplement. Internal XeLaTeX builds produce
22 main-paper pages after the editorial review below, including Methods and references, and 43 supplementary
pages. The remapped response letter has 20 pages after the submission follow-up below. This is an editorial reorganization
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

## Editorial review follow-up, 25 September 2026

The review of the condensed manuscript was checked against the frozen records
before editing. The title is now **TRiX: Task-conditioned execution governance
for robotic disassembly**, synchronized across the main paper, supplement,
response letter, README and citation metadata.

- The abstract specifies post-hoc application to a trained policy and restores
  the observed absence of an inadmissible image shared across all three runs.
- Results 2.5 places the visibility-scaled image qualification beside the 31
  prevented constraint-violation cases. Methods 4.5, S11.4 and R4.9 distinguish
  descriptive stratum counts from an interpretation of model behavior that
  would require an unscaled-image control. The seven partial views remain separate.
- Introduction and R1.9 now cite Rrapi et al., DOI
  [10.1016/j.rcim.2026.103269](https://doi.org/10.1016/j.rcim.2026.103269),
  corresponding to supplied PII S0736584526000487, for its survey of reasoning,
  planning and interaction in collaborative robotics.
- Methods 4.1 and R4.7 explicitly explain the withdrawn PyBullet pitch-validation
  argument. The archived prototype imposed the analytical pitch through a gear
  constraint. Its location is `research/experiments/pybullet_helix.py` in the
  frozen evidence archive, SHA-256
  `4b4e1f2aec3d71e96aa070444a458787478a11893d4b1271fce622247c360e6e`.
  No saved result is used to claim independent validation or a measured outcome
  from that prototype. The response evidence map identifies the archived source.
- Discussion explicitly limits the latency advantage to the scalar
  implementation and saved workload. It names the sensitivity exceptions as
  six PCB, one CRANK and two BATTERY cases; the BATTERY cases have zero
  destructive completions in both arms.

The verified PCB decomposition remains 2.9%, 78.0% and 92.3% on 2,000 held-out
proposals. Gemini remains 65 admissible and 26 inadmissible decisions from its
existing 91-row artifact. E.4 and R1.6 retain the recorded 12.578-microsecond
active-set QP median. The 96/105 sensitivity result is not recast as a 96/96
guarantee. No experiment, figure, renderer or result artifact was changed.

This update adds one main-paper page at the existing 12-point body size.
Reviewer access and verification against the original decision letter remain
submission tasks; no public archive or access arrangement is claimed.


## Submission follow-up: identity, audit, randomisation and Figure 4

Manuscript COMMSENG-26-0216-T is identified in the response cover and the
new editor email draft. Both that email and the Reviewer 4 introduction name
the submitted and revised titles and state that no autonomous agent loop was
evaluated. The email is a draft for author review, not a sent message.

The BATTERY correction is explicitly linked from R4.1 and R4.10 to
`docs/battery_aggregation_correction.md` and the archive's
`verification/policy_record_linkage.json`. The latter records
`exact_arm_group_means["stage1/BATTERY/sac/trix"] = 20.0`, from ten seed rates
`[0,0,0,100,0,0,0,0,0,100]`. The historical pooled files remain intact.

Supplementary S4.6 already contained the six-task reset table and separate
per-step noise account. Table S4 now includes an evaluation-use column:
training, validation and held-out tests share the listed ranges; a widened
OOD reset distribution was not evaluated. The six constraint identifiers are
preserved beneath the table. Main Methods 4.1 and R2.4 point to this table.

Figure 4 was regenerated from the unchanged 80-point CSV using the newly
supplied `scripts/build_revision_figure4.py`. This is a replacement generator,
not a recovered historical script. It checks each value against the retained
audited seed counts before plotting and preserves the four panels, distributions
and means. Its pinned dependencies and build manifest support reproduction.
Two builds under NumPy 1.26.4 and Matplotlib 3.6.3 yielded identical PDF bytes;
the vector output embeds TrueType fonts. No new experiment was run.

This authorized regeneration supersedes the earlier Figure 4 artwork hash.
The previous PDF remains in Git history and the frozen archive. The other
54 pinned files retain their prior hashes, including all six workspace renders.
Main and Supplement remain 22 and 43 pages; the response is 20 pages. The
source inventory now has 203 files and 90 mapped Python files, with no orphan code.


## R2.4 and reviewer-access follow-up

R2.4 now states directly that no parameters were randomised under a separate
OOD reset distribution. It identifies the existing perturbation evidence as the
105-case sweep of 21 constants at five multipliers, with eight scripted episodes
per arm/case and a recorded 96/105 ordering result. It explicitly distinguishes
this scripted sensitivity study from OOD generalisation of trained policies and
links to S7, the experiment code and the saved outcomes. The compiled S4.6 and
Table S4 were rechecked on SI page 17; the remaining noise/evaluation details
are on page 18. Title declarations remain in the editor email and R4 introduction.

The source remains an author-review draft. The original decision/reviewer reports
and an operational reviewer-access route are still required before submission.
The availability text has not been rewritten to claim unestablished access.
