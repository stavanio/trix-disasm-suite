
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
