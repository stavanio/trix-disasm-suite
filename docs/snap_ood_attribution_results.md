# SNAP OOD attribution: command enforcement and static outcomes

Post-hoc diagnostic completed 26 September 2026. This supplements the already
reported prospective OOD experiment; it does not replace any outcome or change
the manuscript. The author requested two checks: whether governed commands
belonged to the declared set, and how static behaved outside training support.

## Scope and verification

All 60 SNAP cells were replayed: both Stage 2 SAC arms, ten training seeds,
100 original episode seeds, and all three OOD conditions. All 6,000 episodes
exactly reproduced their original category, step count, violation flag and
reset metadata. Every original cell count also matched.

Passive observers retained 610,548 commands and their pre/post observations.
An independent pass over the saved arrays checked every bound, command,
projection result and within-episode observation transition. All 39 frozen
scientific source hashes, checkpoints and the original OOD protocol were
preserved. No training, parameter adjustment or counterfactual intervention
was performed. [Verification](../results/snap_ood_attribution/verification.json).

## Check 1: membership in the frozen executable command set

| Arm | Replayed episodes | Governed commands | Strictly outside its own executed set | Maximum positive residual |
|---|---:|---:|---:|---:|
| TRiX | 3,000 | 293,082 | 0 | 0.0 |
| Static | 3,000 | 317,466 | 0 | 0.0 |

These are exact nonpositive-residual results, not cases hidden by a numerical
tolerance. Reconstructing each projection independently from the nominal
action and saved bounds gave zero difference from the saved executed action.

The checked TRiX set is the pre-step box in the frozen executable source:

- Deflection command: magnitude at most
  `min(1, 5000 * 0.006 * 0.85 / 25) = 1` in normalized units.
- Pull command: lower bound -1; upper bound
  `max(0, phase_limit - z * force_noise) / 40`, where the phase limit is
  5 N while engaged and 60 N while disengaged.
- Lateral command: magnitude at most `(15 - z * force_noise) / 20`.
- The executed common-default multiplier is `z = 4.3043128` approximately.
  Projection applies the recorded numerical inward factor before float32 output.

This verifies command membership at the sampled state. The check does not
identify the box with the set of commands guaranteeing safe subsequent
dynamic states under every stiffness and noise realization.

### The existing margin-registry discrepancy remains explicit

The later per-family registry records approximately `4.4266298`, rather than
the executed `4.3043128`. This discrepancy was already documented in
[the provenance audit](provenance.md); the diagnostic does not rewrite it.

Substituting that registered multiplier in the pull/lateral bounds puts
230,582 of the 293,082 TRiX commands outside that tighter alternative by more
than 1e-7 normalized units. The maximum excess is approximately 0.007552586.
Thus passing the actual executed box cannot be reported as compliance with
the later registered tightening.

The `registry_set_*` record fields name this scalar substitution diagnostic.
They retain the executed deflection bound and do not constitute a complete
validation of the registry's overstress probability or a state-invariance
claim. Neither variant supplies a dynamics-aware deflection guarantee.

Static's own box is phase-blind. In the pure shell, 98,793 of its 128,915
commands fall outside the TRiX box evaluated at static's own states. This is
a comparison of sets along the static trajectory, not a matched-action
counterfactual with the separately trained TRiX policy.

## Check 2: complete outcomes outside training support

The primary shell contains 100 reset draws shared across ten training seeds,
giving 1,000 episodes per arm. Counts below form a disjoint partition.

| Outcome in the outside-support shell | TRiX | Static |
|---|---:|---:|
| Safe completion | 599 | 35 |
| Intact completion with a violation | 0 | 30 |
| Completion with a broken latch | 369 | 419 |
| Completion with accumulated damage but no break | 0 | 467 |
| Timeout with a broken latch | 32 | 49 |
| Other outcomes | 0 | 0 |
| Total | 1,000 | 1,000 |

The original destructive-completion category groups the third and fourth
rows: 369 for TRiX and 886 for static. It does not include the timeout row.
The original outcome contract is preserved, while the diagnostic exposes
terminal damage separately from completion status.

Consequently, **actual fractures across all episodes are 401/1,000 under TRiX
and 468/1,000 under static**. The difference in fracture counts is much smaller
than the difference in the original destructive-completion category. Calling
369 versus 886 a fracture comparison would be incorrect. All 32 TRiX timeouts
and all 49 static timeouts in this shell ended with broken latches.

For the existing primary endpoint, TRiX's safe-completion advantage is 56.4
percentage points, with the original exact paired seed-bootstrap interval
[46.1, 63.7]. Its difference in destructive completion is -51.7 percentage
points, interval [-68.4, -32.6]. These intervals are retained from the original
analysis; no new inferential claim is attached to the post-hoc damage subgroups.

### Both mixed conditions: outside-support episodes only

| Condition | Arm | Outside episodes | Safe | Unsafe intact | Destructive completion | Timeout |
|---|---|---:|---:|---:|---:|---:|
| 1.5x half-width | TRiX | 320 | 205 | 0 | 104 | 11 |
| 1.5x half-width | Static | 320 | 12 | 9 | 285 | 14 |
| 2x half-width | TRiX | 480 | 294 | 0 | 172 | 14 |
| 2x half-width | Static | 480 | 18 | 14 | 424 | 24 |

These are descriptive strata of the original mixed evaluations. Conditions
are not pooled, and the repeated reset draws are not independent training
replicates. The complete summaries retain every condition and training seed.

### Direction of the shift in the primary shell

| Stiffness stratum | Episodes per arm | TRiX safe | Static safe | TRiX broken, including timeouts | Static broken, including timeouts |
|---|---:|---:|---:|---:|---:|
| Below training support: 4,000 to below 4,500 N/m | 520 | 119 | 11 | 401 | 468 |
| Above training support: above 5,500 to 6,000 N/m | 480 | 480 | 24 | 0 | 0 |

All observed TRiX shell failures occur in the softer-latch stratum. Both arms
have zero fractures in the stiffer stratum, but static still completes with
accumulated pull damage in 432 of those 480 episodes. These are observed
subgroup counts, not guarantees for either interval.

## What the substep traces establish

Every one of the 401 TRiX shell fractures first occurs in the overdeflection
branch, when the integrated absolute deflection exceeds 6 mm. None first
occurs by accumulated pull damage reaching its break threshold. Every such
break precedes the first pull-damage event, if any, in that episode.
All commands up to and including these events satisfy the executed box.

The earlier static calculation `25 N / 4000 N/m = 6.25 mm` establishes one
possible weakness of the nominal bound, but does not explain all observed
fractures. In 331 of the 401 broken TRiX episodes, no pre-break commanded
force had a static `abs(F_command / k_latch)` value exceeding 6 mm. At 99 first
break events, even the instantaneous realized-force equilibrium value was
at most 6 mm. Dynamic state therefore matters; the static calculation alone
is insufficient. The traces retain the first-event deflection, velocity,
force, phase and damage state so this observation is checkable.

TRiX subsequently records pull damage in 234 shell episodes; each has a
within-action phase re-engagement event. Because the first break already
occurred, these observations must not be presented as the initiating fracture
mechanism. Static's 468 fractures also first occur through overdeflection,
although static commonly accumulates pull damage before breaking.

**Supported attribution:** the audited TRiX projection enforces its frozen
command box; physical fractures occur downstream through overdeflection of
softer latches. Static performs worse on safe completion and total recorded
damage, while its fracture count is higher by 67 episodes per 1,000. This audit rules
out an out-of-set command as the initiating explanation for the observed TRiX
fractures. It does not quantify separate causal contributions from stiffness,
noise and transient dynamics, or isolate the governor from policy adaptation.

## Reproduction and evidence

- [Diagnostic plan](snap_ood_attribution_plan.md), explicitly post hoc.
- [Diagnostic source](../experiments/snap_ood_attribution.py).
- [Complete summaries](../results/snap_ood_attribution/summary.json).
- [All 6,000 episode attributions](../results/snap_ood_attribution/episode_attribution.json), with column definitions.
- [Verification](../results/snap_ood_attribution/verification.json) and [raw-cell/trace manifest](../results/snap_ood_attribution/manifest.json).

The separate `TRIX_snap_ood_attribution_evidence` directory preserves every
command trace, full diagnostic episode and first-event state, the diagnostic
source and plan, run log and checksum manifest. It is approximately 46 MB.
Keep it beside the unchanged primary OOD and historical evidence directories.
No upload or public-access claim accompanies this diagnostic.
