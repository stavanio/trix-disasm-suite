# Preregistration: paired held-out task

Written before any implementation exists. The predictions below follow
from the three-gate criterion, version 1.1, frozen and committed prior to
this document. If the predictions fail, the criterion is falsified and
must be reported as such.

## Why paired

A single new task that confirms the criterion is weak evidence: a
criterion that only ever predicts an advantage is unfalsifiable in
practice. Two variants of the same task with opposite predictions is a
controlled test, because everything except the deciding variable is held
fixed.

Both variants share the same mechanism, actuation, reward, observation
space, episode cap, noise model and damage model.

Both variants fail gate 2 because a componentwise box cannot exactly
represent their elliptical sets. The variants differ in whether the lost
authority is context-dependent and task-essential. The prediction
therefore turns entirely on gate 3, which is the gate that decided all
five existing tasks.

## The task: bayonet connector release

A bayonet or twist-lock coupling. A pin rides in an L-shaped slot. To
release, the tool must press axially to unload the detent, rotate through
the locking path, and then withdraw. The pin can shear if withdrawal is
attempted while it is still in the rotating leg of the slot.

Admissible commands are (F_axial, tau, F_lateral).

The coupled constraint, present in both variants: the resultant of axial
force and torque-induced tangential load at the pin must stay within the
pin's shear envelope, which is an ellipse in (F_axial, tau) rather than a
rectangle.

### Variant S, static

The shear envelope is fixed. Its size and orientation do not depend on
where the pin sits in the slot.

Successful release requires a moderate coupled action near (0.45, 0.45)
in normalised coordinates, which lies inside both the ellipse and the
inscribed box of half-width 1/sqrt(2) = 0.707. The actions uniquely
removed by the box are therefore not task-essential.

On units: normalised semi-axes of (1, 1) describe a circle in normalised
coordinates. Because the axial and rotational axes carry different
physical scales, the same set is an ellipse in physical force and torque
units.

Making the ellipse's corners essential would manufacture an advantage for
structured projection and destroy the null control. Variant S tests the
hypothesis that a geometrically imperfect baseline can still preserve
every action success requires.

### Variant P, phased

The same ellipse, but its admissible axial component is phase-gated:
while the pin is in the rotating leg, withdrawal force is limited to a
small value; once the pin reaches the release leg, withdrawal up to the
actuator limit is admissible. Nothing else differs.

## Predictions, derived from criterion 1.1

### Variant S

- Gate 1: specification valid. The envelope is preventive and sits at or
  below the shear onset; the tightening leaves positive authority.
- Gate 2: a componentwise box cannot express an ellipse. Not decided
  here.
- Gate 3: the actions a box loses are the ellipse's corners at
  simultaneous high axial force and high torque. Release requires a
  coupled action, but a moderate one that sits inside the box, so the
  removed corners are not task-essential.

Predicted: no material advantage.

### Variant P

- Gate 1: identical specification, valid.
- Gate 2: the baseline receives no phase signal and cannot express a
  phase-indexed envelope. Not decided here.
- Gate 3: a static bound must either permit release-leg withdrawal force
  while the pin is still in the rotating leg, which shears it, or
  restrict withdrawal to the rotating-leg limit throughout, which cannot
  extract the connector. The withdrawal authority available only in the
  release phase IS the extraction action, so it is task-essential.

Predicted: material advantage.

### Controls for Variant P

Comparing a phase-aware projection against a phase-blind box would
confound two things: access to the phase signal, and the geometry of the
admissible set. A reviewer could reasonably conclude the result shows
only that phase information helps.

Variant P therefore runs four arms, three of which receive identical
phase information:

| Arm | Phase signal | Set shape | Purpose |
|---|---|---|---|
| phase-blind conservative box | no | box at the rotating-leg limit | safe, expected unable to extract |
| phase-blind permissive box | no | box at the release-leg limit | able to withdraw, expected to shear during rotation |
| phase-aware box | yes | phase-conditioned box | isolates phase information from geometry |
| phase-aware projection | yes | phase-conditioned ellipse | the method under test |

Variant P has two distinct comparisons, and conflating them would let a
mechanism result masquerade as a falsification.

Primary confirmatory comparison: phase-aware projection against the
phase-blind conservative box. The criterion predicts a material advantage
of at least 25 percentage points, because the phase-blind restriction
permanently removes withdrawal authority that is essential after release.
This comparison alone confirms or falsifies the criterion.

Secondary mechanism comparison: phase-aware projection against the
phase-aware box. No directional prediction is made. If the phase-aware
box matches projection, phase conditioning is sufficient and coupled
geometry adds nothing measurable. If projection materially exceeds it,
coupled geometry contributes value beyond phase information. Either
result is informative and neither bears on confirmation.

The phase-blind permissive box is a mechanism control, expected to
preserve withdrawal capability at the cost of unsafe withdrawal during
rotation. It is not part of the primary confirmatory comparison.

## The four-cell prediction

| Variant | Frozen nominal policy | Filter-aware training |
|---|---|---|
| S, static ellipse | projection may outperform the box | projection and box within 10 points |
| P, phase-gated ellipse | phase-aware projection should outperform the phase-blind conservative box | phase-aware projection should exceed the phase-blind conservative box by at least 25 points |

Performance of the phase-aware box relative to projection is a
prespecified mechanism result and does not determine confirmation or
falsification of the context-dependence criterion.

Adaptation can overcome a static representational mismatch. Adaptation
cannot overcome a context-blind filter when the excluded authority is
task-essential.

The S row under frozen-policy evaluation is deliberately loose: the
criterion makes no commitment there, because governing a policy trained
without the filter is exactly the case where representation matters most,
and PCB already showed a 0% to 100% gap in that setting.

## What counts as confirmation

Evaluated on paired per-seed differences, using the frozen protocol: ten
training seeds, 15,000 steps, the existing checkpoint selection rule, 100
held-out test episodes, safe completion over all episodes as the primary
endpoint.

Material advantage requires all three:
- mean paired difference at least 25 percentage points
- median paired difference at least 20 percentage points
- positive difference in at least 7 of 10 seeds

No material advantage requires all three:
- absolute mean paired difference no greater than 10 percentage points
- absolute median paired difference no greater than 10 percentage points
- absolute paired difference no greater than 10 points in at least 7 of
  10 seeds

Anything else is indeterminate and reported as such.

Primary confirmatory algorithm: SAC. PPO is a secondary replication only,
because PPO failed to acquire competent policies on four of six existing
tasks under the same budget, so a PPO null result would be uninformative.

Learned-policy competence gate: if no learned arm reaches at least 50%
completion in a variant, that variant is classified as uninformative
rather than as confirming or falsifying the criterion.

## What falsifies the criterion

- Variant S shows a material advantage. Gate 3's adaptation-escape
  reasoning would be wrong.
- Variant P shows no material advantage in the PRIMARY confirmatory
  comparison, phase-aware projection against the phase-blind conservative
  box. The reasoning that also underpins SNAP and CRANK would be
  undermined. A null result in the secondary mechanism comparison does
  not falsify the criterion.
- Both variants land in the indeterminate band. The criterion would be
  too coarse to be useful.

A criterion that survives only by reinterpretation after the fact is not
a criterion. If either prediction fails, the failure is the result.

## Manipulation checks

Declared in advance, so a null result on Variant S cannot be dismissed as
"the constraint never mattered". All must pass before the learned arms
are interpreted:

1. Box and ellipse corrections differ on at least 20% of randomly drawn
   proposals, with a reported maximum difference.
2. At least 5% of evaluation steps lie within 5% of the ellipse boundary.
3. Both principal filters intervene on at least 20% of steps, so neither
   is inert.
4. A scripted controller completes Variant S safely through both the box
   and the projection.
5. A scripted controller completes Variant P safely through the
   phase-aware box and the phase-aware projection.
6. In Variant P, the phase-blind conservative box reproduces the
   predicted capability failure, and the phase-blind permissive box
   reproduces the predicted unsafe-withdrawal failure. If either control
   does not behave as described, the implementation is wrong and the
   learned arms are not interpreted.
7. The inscribed box occupies 2/pi, about 63.7%, of the normalised
   ellipse area, so roughly a third of the admissible set is unavailable
   to the baseline.

Observation parity: both variants expose observation vectors of identical
dimension and identical context encoding. Variant S includes the same
phase variable as Variant P; its admissible set is simply invariant to
that variable. Without this, the variants would differ in observation
space as well as in set structure, and any difference could be attributed
to either.

A null result on Variant S with all checks passing means the mismatch
existed and was not capability-limiting. That is the hypothesis.

## Fixed parameters

| Item | Value |
|---|---|
| Primary endpoint | paired difference in held-out safe completion, over all test episodes |
| Algorithms | SB3 SAC primary, PPO secondary |
| Training budget | 15,000 steps, identical across arms within a variant |
| Training seeds | 0 through 9 |
| Test seeds | from selection.test_seeds, disjoint from validation |
| Checkpoint selection | the existing frozen rule, unchanged |
| Ellipse | semi-axes (1, 1) normalised, scaled to task units |
| Box baseline | inscribed square, half-width 1/sqrt(2) |
| Required release combination | approximately (0.45, 0.45) normalised |
| Phase gate, Variant P | rotating leg permits 0.25 axial; release leg permits 1.0 |
| Competent policy threshold | 50% completion |
| Decision bands | material at 25 points, none within 10, indeterminate between |

No parameter above may change after comparative results are seen.

## Preregistration protocol

Two freezes, because a conceptual prediction does not pin down an
implementation and an implementation can be tuned after the fact.

Freeze 1, conceptual, before implementation:
1. Commit this document.
2. Record the commit hash and timestamp.
3. Record the criterion hash it depends on.

Freeze 2, implementation, before any learned run: commit and hash the
complete implementation, including dynamics, reward, observation space,
thresholds, physical scaling, all four filter definitions, the scripted
controller and its acceptance tests, and the environment constraint hash.
No element may change afterwards.

During implementation:
4. Implement both variants with acceptance tests.
5. Verify a scripted controller completes both variants safely through
   the projection.
6. Do not run the learned arms until the feasibility gate passes.

After running:
7. Report both variants regardless of outcome.
8. If a prediction fails, report the failure and do not revise the
   criterion to accommodate it.
9. Neither variant may be modified after comparative results are seen.

## Note on scope

This tests the criterion, not TRiX. A confirmed prediction supports the
claim that safety-governor adequacy is predictable from constraint
structure and task requirements. It does not establish that structured
projection is generally superior.

---

# Amendment 1: boundary-excitation probe

Date: 2026-08-05. Made before implementation freeze and before any
learned run. Depends on criterion 1.1 and the original preregistration,
both already committed.

## Why

Manipulation check 2 required at least 5% of evaluation steps within 5%
of the boundary. It was measured first under the scripted controller,
which sits deliberately inside at peak envelope 0.89, and then under
uniform random proposals. The random instrument recorded **zero released
steps in 2500**: random actions do not follow the press, rotate, release,
withdraw sequence, so they never leave the rotating leg and cannot test
the phase-dependent boundary at all.

The check was correct. The instrument was not. Neither a deliberately
cautious controller nor an undirected one exercises a sequential task's
constraint boundary.

Saturating every command at the action limits was considered and
rejected: it would make boundary occupancy automatic and therefore
uninformative.

## The probe

A deterministic controller that follows the scripted phase sequence so
both legs are visited, and within each phase sweeps coupled axial and
torque proposals around the admissible boundary at normalised radii of
0.8, 0.95, 1.05 and 1.2, across fixed directions. The identical nominal
sequence is supplied to every arm, and both the proposed demand and the
executed action are recorded.

## Amended checks

1. The probe visits both phases for a predeclared number of steps.
2. At least 20% of proposals in each phase lie within 5% of, or outside,
   the true boundary.
3. Box and ellipse corrections differ on at least 20% of proposals.
4. The projection arm executes actions within 5% of the boundary on at
   least 5% of probe steps.
5. The scripted controller still completes the task through the valid
   phase-aware arms.
6. Learned-policy boundary occupancy is reported descriptively and is
   **not** a pass or fail gate. A learned policy may legitimately operate
   far inside the boundary, and that must not invalidate the experiment.

Checks 1, 4 and 5 of the original list already pass: filter divergence at
20% over 5000 proposals, scripted feasibility on both variants across ten
seeds, and the inscribed square covering 2/pi of the ellipse.

Nothing about the predictions, decision bands, arms, seeds or budgets
changes.
