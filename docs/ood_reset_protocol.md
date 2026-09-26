# R2.4 prospective reset-distribution evaluation amendment

This new experiment is added during revision to answer R2.4's request for
training versus OOD evaluation use in Table S4. It was not part of the original
evaluation. The machine-readable declaration is [ood_reset_protocol.json](ood_reset_protocol.json).
Commit this declaration and its implementation before any OOD policy rollout.
The run records that commit and the declaration's SHA-256. The original protocol,
all historical records and all previously reported in-range values remain frozen.

## Scope and exclusions

Use only the selected checkpoints for the seven Table 4 comparisons involving
SCREW (SAC and PPO Stage 1), PCB (SAC Stages 1 and 2), SNAP, CRANK and BATTERY
(SAC Stage 2). Keep both arms and all ten training seeds in every comparison,
including arms with 0% in-range safe completion. Changed dynamics can improve
completion, and damage/timeout outcomes remain informative. For BATTERY use
the matched preventive box and preventive TRiX arms.

PRY is **not applicable** to this range-widening experiment: its physical reset
is deterministic. No bond-strength randomisation or new PRY parameter is added.
There is no training, checkpoint screening, reselection, governor adaptation,
new comparator, task or noise experiment. Use the existing endpoint contract.

## Distributions

For a native uniform range with centre c and half-width h, sample uniform
proposals from [c - k h, c + k h], at k = 1.5 and k = 2. The third condition
uses the k = 2 rectangle conditioned on being outside the original training
rectangle. For one parameter it is the two outer intervals with their uniform
length weighting. For SCREW it is the joint rectangle minus the original
rectangle: at least one friction coefficient must be outside training support.
The shell condition is the primary pure-OOD endpoint; the two full widened
distributions report the effect of increasing the sampling range.

| Task | Instance field | Training range | 1.5 half-width | 2 half-width |
|---|---|---|---|---|
| SCREW | mu_s | 0.549 to 0.671 | 0.5185 to 0.7015 | 0.488 to 0.732 |
| SCREW | mu_k | 0.423 to 0.517 | 0.3995 to 0.5405 | 0.376 to 0.564 |
| PCB | f_clip, N | 21.25 to 28.75 | 19.375 to 30.625 | 17.5 to 32.5 |
| SNAP | k_latch, N/m | 4500 to 5500 | 4250 to 5750 | 4000 to 6000 |
| CRANK | mu_k | 0.423 to 0.517 | 0.3995 to 0.5405 | 0.376 to 0.564 |
| BATTERY | adhesive, N | 7.2 to 8.8 | 6.8 to 9.2 | 6.4 to 9.6 |

SCREW draws independent uniform proposals and rejects the entire pair unless
mu_s >= mu_k. The accepted joint distribution is uniform on the physically
ordered part of the rectangle, not two independent marginal uniforms. The
shell also rejects pairs inside the original rectangle. No clipping, sorting
or redrawing only one component is allowed. All bounds are declared before
evaluation and are stress-test bounds relative to the existing simulation,
not newly measured physical population distributions.

For a single parameter, full-range widening places 1/3 or 1/2 of draws outside
training support in expectation. Those fractions do not apply to SCREW's
joint conditional distribution. Save every actual parameter draw, its support
classification and the number of rejection attempts; report actual fractions.

## Frozen interface and random streams

`evaluate_frozen(..., evaluation_distribution=spec)` passes an explicit spec
to the Gym adapter. The adapter calls the original reset, then overrides only
the already-existing instance fields listed above, before encoding the initial
observation. No environment source, module constant or governor is modified.
Constraint, robust-margin-policy and taxonomy hashes must equal the historical
record before and after each cell. Source hashes separately cover the plant,
governors, runtime, outcome contract and existing statistical implementation.

Parameter draws use a dedicated NumPy RandomState, seeded by the first 32 bits
of SHA-256 of `trix-ood-reset-v1|task|episode_seed` (big-endian). The condition
is omitted from that seed to couple uniform proposals across widening levels.
Rejection sampling never consumes the plant's global noise RNG. Native reset
consumption and per-step noise are unchanged. No parameter values unavailable
to the original policy/governor are added to their observations.

Each task uses 100 episode seeds, beginning at 1,200,000 plus 1,000 times the
existing fixed task offset. These are disjoint from original training,
validation and held-out test schedules. Both arms, all training seeds and all
three conditions share this declared task-specific list. Comparisons to the
historical test use training-seed pairing only, since episode seeds differ.

## Matrix, records and failures

Seven comparisons x two arms x ten training seeds x 100 episodes x three
conditions = **42,000 new episodes**, in 420 cells, using 110 distinct selected
checkpoints. Freeze checkpoint paths, selected steps, file hashes, historical
baseline counts, software versions and caps in the JSON declaration.

Keep action limits, margins, specifications, damage thresholds, observation
encoders, noise and episode caps fixed. Each new record stores the full
evaluation-distribution spec and its own hash separately from the three
unchanged definition hashes. `run_record.check_comparable` rejects different
or missing distribution specs/hashes, including old untagged records. Cross-
distribution comparisons are explicit paired contrasts, never pooled cells.

One cell is written atomically after its 100 episodes. Resuming may skip only
completed cells under the exact same freeze. Failed cells retain their traceback;
there is no silent retry, checkpoint replacement, outcome-based exclusion or
post-result change to the range. Interrupted cells without a completed record
may restart with the same seeds. A failed/missing cell makes the study incomplete.
Any implementation defect requires a documented amendment before rerunning;
all previous outputs and errors must be retained.

## Analysis and reporting

The unit is the training seed, with ten values per task/arm/condition. Report
safe completion, destructive completion, mechanical failure, unsafe intact
completion, timeouts and episodes with a constraint violation, with counts and
denominators. For each full 100-episode cell, report absolute seed rates and
changes from its historical in-range rate. Report second-arm minus first-arm
differences paired by training-seed identifier, keeping Stage 1 and Stage 2
separate.

Reuse `scripts/build_revision_seed_statistics.py:exact_bootstrap` without a
change of estimator: exact 2.5th/97.5th percentile distributions of the mean of
ten resampled seed values/differences, by convolution. With 100 episodes per
cell the percentage-point inputs are integers, as in Section 4.6. These are
descriptive intervals, not multiplicity-adjusted tests or a pass/fail gate.

For the mixed widened distributions also report within-support and outside-
support counts and descriptive rates per seed. Their varying denominators do
not enter the integer-grid bootstrap. Empty strata are marked absent, not zero.
The shell condition supplies a pure-OOD endpoint with the full 100 episodes
per cell and the same inferential procedure as the original comparisons.

Every condition and seed is reported, including collapse, improvement or no
difference. Results concern parameter-distribution shifts in the declared
simulation only. They do not establish real-world calibration or robustness
to arbitrary unseen dynamics. Paper/SI/response wording changes only after
the complete evaluation has been verified; no OOD result is claimed in advance.

## Reproduction and release

The source entrypoint is [experiments/ood_reset.py](../experiments/ood_reset.py).
`prepare --archive PATH` reads only the frozen evidence package and creates the
declaration. `verify-native --archive PATH` replays five previously recorded
in-range episodes for seed 0 of each included arm (70 episodes) and requires
exact count equality with the historical shards. This is an implementation
check with no OOD policy rollout and does not change the published estimates.
`run --archive PATH --out build/ood_reset --workers 16` refuses a
dirty Git tree and checks the declaration and checkpoint/source hashes.
`report --out build/ood_reset` verifies recorded draws/counts and regenerates
the analysis. Raw cells belong in a separate new evidence directory; the large
historical archive is never edited. Curated summaries and source maps are added
to the lean repository when results exist.
