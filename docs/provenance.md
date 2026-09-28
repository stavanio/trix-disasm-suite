# Provenance and corrections

## 1. Submitted headline results and their withdrawal

The pre-revision implementation retained in Git at
`8b1a746a806907b548066cfe30fff829f5192330`,
`benchmark/disasm_bench.py`, generates uniform random action proposals in
`run_benchmark`. `Algorithms.get_action` returns fixed transformations named
PPO, SAC, PPO-Lag, CPO, Lambda, SafeLayer and TRiX. It does not train policies.
The submitted Table 4 (p. 15) assigns 86.9% SCREW violations to SafeLayer,
43.9% PCB violations (approximately 44%) to TRiX, and 0.0% PRY violations to
TRiX. These assignments were checked against the author-supplied submitted
PDF (SHA-256 `df186311c2f39b78069572b9f7da69aa902c7b67c20c78dfe30f09bb47ab83fc`).
They must not be described as trained-policy results. The 84.5-percentage-point reduction and claimed
9-million-step evaluation claim are withdrawn. The synthetic grounding
exercise also does not establish the submitted over-90% model-grounding claim.
The revision's trained-policy and actual-model studies replace these claims.
This is a provenance correction, not a retrospective change to historical code.

## 2. Task-description mismatches in the retained benchmark

| Task | Retained implementation defect | Revision treatment |
|---|---|---|
| SCREW | The state update imposes helical motion. The stated realized-velocity criterion is therefore not independently violable by that motion; the benchmark instead flags load limits and an ad hoc command ratio. | Independent actuation and the declared realized-motion/load criteria in SCREW v3. |
| PRY | Bond strength is 150 N and lever length 0.15 m, but admissible torque is at most 1.5 N m, giving at most 10 N for bond release. | The submitted TRiX value of 0.0% violations does not demonstrate safe task completion: the filter could not command the force required for release. Revised crack/bond dynamics provide a reachable task. |
| SNAP | A spring and generic friction/release flag replace the described separate latch-deflection and pull state. | Explicit latch state, release and pull dynamics. |
| CRANK | A fixed axial limit is used, without the stated rotation-complete gating of extraction. | Separate structural and phase-gated axial constraints. |
| BATTERY | Force flags are counted although submitted Table 8 states only a thermal criterion. No event-resolved historical trace establishes that the reported aggregate was a thermal-event rate. | Separate force, deformation, internal-short and diagnostic temperature definitions. |
| PCB | Actions represent lift/lateral forces, so the stated tilt-torque constraint cannot be expressed in that action interface. Force checks and fixed per-axis clips do not implement it. | Explicit coupled tilt-torque and lift interface. |

These statements concern the benchmark runner, not interchangeable versions of
similarly named classes in `envs/`. In particular, "no latch state" means no
separate latch-and-pull state for the submitted task; the old benchmark did
contain a generic spring/release flag. "Unviolable" refers to the imposed
helical kinematics, not every force or torque flag. Gaussian noise is unbounded;
the PRY statement concerns release under admissible torque.

## 3. Submitted thermal table

The submitted Table 5 claims continued temperature rise after power cutoff.
For its stated first-order model, zero input power implies cooling toward
ambient temperature, so that claimed overshoot does not follow from the model.
The table and its temperature/time values are withdrawn. Historical
per-event temperature traces have not been recovered; a claim that the
thermal counter was exactly zero for every submitted run is therefore not
independently attested here. The retained benchmark combines force and
temperature flags with a logical OR and saves aggregate `vio` counts, without
separate thermal-event counters or temperature traces in its result output.
The aggregate therefore cannot establish that the thermal criterion never
fired in every historical run or version. The revised deformation/internal-short model is
new evidence, not validation of the old table.

## 4. Corrections during revision development

These are internal development corrections, not versions circulated to the
editor or reviewers. They belong in provenance and must not be described in
correspondence as corrections to an "intermediate draft" seen by the journal.

- BATTERY Stage 1 SAC/TRiX: an arm-prefix collision produced 59.75% (rounded
  59.8%). Exact-arm shard aggregation gives 20.0%. The detailed audit is
  `docs/battery_aggregation_correction.md`, with the full linkage in the
  evidence archive at `verification/policy_record_linkage.json`. Selected
  checkpoints were preserved; this does not establish that repeating the
  historical selection would choose the same checkpoint.
- Gemini: 62 admissible decisions in an internal summary omitted three saved
  decisions. The current single-artifact accounting is 65 admissible plus
  26 inadmissible, with no refusals, for 91. Across providers there are 219
  admissible, 38 inadmissible and 16 refusals. This is a correction to internal
  accounting, not evidence that three new calls were made.


## 5. What the SNAP and CRANK comparisons vary

SNAP's `trix` arm is a phase-aware componentwise clip. The
phase-conditioned admissible set is axis-aligned, so given the phase a
box expresses it exactly. `snap_phase_clip` delegates directly to
`snap_project`. The SNAP comparison therefore varies phase awareness,
not set shape, and must not be presented as geometry evidence.

CRANK's `static_clip` calls `crank_bounds(obs, sigma)` and so uses the
rotation-complete phase flag for its axial bound. It does not condition
its transverse bounds on crank angle: it applies `min(ft, fr)` to world
x and y. `crank_project` additionally reads the angle, rotates into
tangent and radial coordinates, clips there, and rotates back.

The CRANK comparison therefore varies two things together, angle
conditioning and task-frame representation, and isolates neither.

Two further points, both from the repository itself:

The implemented static box is not safe at every orientation.
`tests/test_phase_filters.py` expects fewer than 75% of sampled static
outputs to be feasible at 45 degrees. The implemented half-width there is
about 0.3228 against a largest safe world-axis half-width of about
0.2283.

Nor does the static box necessarily exclude every task-essential
tangential command. At 45 degrees a maximum admissible pure-tangent
command has components near (-0.2283, 0.2283), inside the implemented
square. Stage 2 stalling under `static_clip` is an observed learning
outcome, not a proof of mathematical exclusion.

## 6. Volume matching

"Volume-matched" is established for PRY only, where `static_clip` and
`trix` each admit 10.96% of the action box at zero insertion, measured
identically over 200,000 uniform samples. The phrase must not be applied
to SNAP or CRANK, whose arms differ in admitted volume as well as in
form.

## Margin routing audit

An implementation audit distinguished the margin values used by archived
executable filters from the later per-family margin-policy registry. The
historical measurements are preserved exactly as executed; no filter is being
changed and no historical run record is being rewritten.

| Task | Archived executable routing | Later registry | Interpretation |
|---|---|---|---|
| SCREW | Radial restriction uses 4.3932477 sigma | The same value is registered under `realized_helix`; no separate radial family is exposed | Numerical execution is consistent; family bookkeeping is incomplete |
| PCB | Filter module uses common default 4.3043128 sigma; tilt target is approximately 0.087087 N m | Planarity and lift declare 4.1482643 sigma | Stage 1 and Stage 2 PCB arm comparisons are not geometry-only comparisons |
| SNAP | TRiX arm uses common default 4.3043128 sigma; static arm uses nominal phase-blind bounds | Families declare 4.4266298 sigma | The 5.2% versus 96.5% result is an implemented-governor comparison, not a phase-only ablation |
| CRANK | Static and TRiX bounds both use common default 4.3043128 sigma | Families declare 4.4883184 sigma; tightening-status metadata reads the registry | Static versus TRiX remains margin-matched as executed |
| BATTERY | Box and projection arms use common default 4.3043128 sigma | Enforced families declare 4.5029968 sigma | Matched preventive box versus TRiX remains margin-matched as executed |
| PRY | Static, projection, and oracle paths use common default 4.3043128 sigma | Families declare 4.5415771 sigma | Static versus TRiX remains margin-matched as executed |
| BAYONET | Filters use 4.6103391 sigma | Registry declares 4.6103391 sigma | Routing is consistent |

For PCB, the manuscript reports the physical executed tilt target where the
margin asymmetry affects interpretation. For SNAP, the result is scoped to the
combined implemented governor because phase conditioning and margin
specification differ simultaneously. For CRANK, BATTERY, and PRY, both primary
paired arms use the same executable margin construction, so the pairwise
comparisons are not confounded by unequal margin routing.

The margin-policy hash remains useful provenance for the declared experimental
specification, but for these archived runs it is not treated as a byte-level
fingerprint of the filter's executable tightening. The executable source and
archived records remain the authority for what was run.
