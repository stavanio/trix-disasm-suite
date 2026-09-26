# SNAP OOD attribution checks

This is a post-hoc diagnostic of the already reported OOD study, requested by
the author after seeing its results. It is not an independent confirmatory
experiment or a revision of the prospective protocol. The diagnostic code and
this plan are hashed before the diagnostic replay starts.

## Scope

Replay every recorded SNAP cell: both Stage 2 SAC arms, ten training seeds,
the original 100 episode seeds and all three declared conditions. This is
6,000 existing episodes, with no new draws, policies, selections or outcomes.
Use the original evaluator, checkpoints, governors and environment functions.
All 39 scientific-source hashes and the OOD protocol remain frozen.

## Check 1: commands against the declared executable set

At every filter call, independently evaluate the six box inequalities for
the pre-step observation. Retain observations, nominal and governed actions,
bounds, subsequent observations and realized deflection/pull forces. Report
strict positive residuals and a separate 1e-7 normalized-coordinate numerical
tolerance; do not silently discard small violations.

The set is the frozen executable specification described by the existing
provenance audit: nominal stiffness 5,000 N/m, 0.85 deflection factor, the
observed phase flag, and the actual common-default Gaussian multiplier.
The numerical inward offset used by projection is not a new physical bound.
The later per-family margin registry is a separate diagnostic comparison.
Passing the executable-set check must not be described as passing the later
registry or as a guarantee on realized forces, dynamic state or future phase.

Also check static against its own fixed box and, separately, the TRiX box at
static's own observed states. These are distinct set-membership checks; the
two separately trained policies do not share complete trajectories.

## Check 2: static outside training support

Report the five-category outcome partition for both arms in the pure outside-
support shell and in the outside-support subsets of both mixed conditions.
Retain denominators, training-seed values and lower/upper stiffness strata.
Do not pool conditions or treat repeated episode draws across policy seeds
as independent training replicates. Existing paired intervals remain those
of the prospective analysis; subgroup counts are descriptive.

## Damage-path observations

Observe each original integration substep without altering its arguments,
state or random stream. Save first pull-damage and first break events,
including sampled phase, before/after state and realized forces. Distinguish
degraded from broken terminal states: the existing destructive-completion
category includes both. Count overdeflection and accumulated-pull first-break
routes separately, and flag phase re-engagement within a held action.

These observations identify executed branches and temporal order, not a
counterfactual estimate of the effect of removing noise, changing stiffness
or modifying a governor. Do not infer that every failure has the same cause.

## Verification and reporting

Require exact agreement with every original episode's category, step count,
violation flag and reset metadata, plus every original cell count. A mismatch
or execution error makes that attribution incomplete; preserve its record.
Observers call each original operation exactly once and draw no randomness.
Archive every command trace and diagnostic episode, with hashes. Keep compact
summaries, code and source mappings in the lean repository. Do not change the
paper, primary experiment, renderers or historical archives during this check.
