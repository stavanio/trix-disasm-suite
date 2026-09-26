# Prospective R2.4 reset-distribution evaluation results

Completed 26 September 2026. This is a new revision experiment; original results are unchanged.

## Scope and freeze

- 420 completed cells, 42,000 episodes, five eligible tasks, seven Table 4 comparisons, both arms, ten training seeds and all three declared distributions.
- 110 selected checkpoints; no training, reselection or exclusions. PRY is not applicable because its physical reset is deterministic.
- Protocol and executable interface committed before OOD rollouts as `9bc2242da260a98c5f6f1e4c10ac9bed4bf56794`.
- Protocol SHA-256: `c3e9f59d9bf1948565ba0f58a74f7dbb5b3c03a1d4b0feca8656fd64ea2ef5b5`.
- Environment/governor source, constraint, margin and taxonomy hashes remain frozen. Distribution specifications and hashes are separate record fields.
- Every cell completed without an execution error. Simulated damage and other unsuccessful outcomes are retained below.
- The three conditions are 1.5 and 2 times the native half-width about its centre, plus the 2x region conditioned outside native support. SCREW also conditions its joint friction pair on static friction being at least kinetic friction.
- These are declared simulation stress tests, not calibrated physical population ranges. All original in-range estimates are unchanged.

## Safe completion and changes

Each shifted entry is mean safe completion (%); change from historical in-range (percentage points) [exact 95% percentile bootstrap interval]. The unit is the training seed (n=10), each with 100 episodes. Historical differences are paired by training seed only; the new episode schedule means they include episode-sampling variation. Intervals describe seed variation on the fixed episode schedules and are not multiplicity-adjusted tests.

| Task / stage / policy | Arm | In-range % | 1.5x | 2x | Outside-support shell |
|---|---|---:|---|---|---|
| SCREW / stage1 / sac | `qp_matched` | 88.7 | 87.9; -0.8 [-1.3, -0.4] | 88.3; -0.4 [-1.7, +0.7] | 87.9; -0.8 [-1.8, +0.1] |
| SCREW / stage1 / sac | `trix` | 88.7 | 87.9; -0.8 [-1.3, -0.4] | 88.3; -0.4 [-1.7, +0.7] | 87.9; -0.8 [-1.8, +0.1] |
| SCREW / stage1 / ppo | `qp_matched` | 60.0 | 59.9; -0.1 [-0.3, +0.0] | 59.9; -0.1 [-0.3, +0.0] | 59.9; -0.1 [-0.3, +0.0] |
| SCREW / stage1 / ppo | `trix` | 60.0 | 59.9; -0.1 [-0.3, +0.0] | 59.9; -0.1 [-0.3, +0.0] | 59.9; -0.1 [-0.3, +0.0] |
| PCB / stage1 / sac | `box_clip` | 0.0 | 0.0; +0.0 [+0.0, +0.0] | 0.0; +0.0 [+0.0, +0.0] | 0.0; +0.0 [+0.0, +0.0] |
| PCB / stage1 / sac | `trix` | 100.0 | 99.6; -0.4 [-0.7, -0.1] | 99.6; -0.4 [-0.7, -0.1] | 99.6; -0.4 [-0.7, -0.1] |
| PCB / stage2 / sac | `box_clip` | 100.0 | 100.0; +0.0 [+0.0, +0.0] | 100.0; +0.0 [+0.0, +0.0] | 100.0; +0.0 [+0.0, +0.0] |
| PCB / stage2 / sac | `trix` | 99.8 | 99.8; +0.0 [+0.0, +0.0] | 99.8; +0.0 [+0.0, +0.0] | 99.8; +0.0 [+0.0, +0.0] |
| SNAP / stage2 / sac | `static_clip` | 5.2 | 4.7; -0.5 [-1.5, +0.0] | 4.4; -0.8 [-2.4, +0.0] | 3.5; -1.7 [-5.1, +0.0] |
| SNAP / stage2 / sac | `trix` | 96.5 | 85.1; -11.4 [-13.5, -9.5] | 78.2; -18.3 [-20.9, -16.1] | 59.9; -36.6 [-40.3, -33.3] |
| CRANK / stage2 / sac | `static_clip` | 0.0 | 0.0; +0.0 [+0.0, +0.0] | 0.0; +0.0 [+0.0, +0.0] | 0.0; +0.0 [+0.0, +0.0] |
| CRANK / stage2 / sac | `trix` | 49.4 | 47.6; -1.8 [-3.0, -0.6] | 47.6; -1.8 [-3.0, -0.6] | 47.6; -1.8 [-3.0, -0.6] |
| BATTERY / stage2 / sac | `box_clip_preventive` | 100.0 | 100.0; +0.0 [+0.0, +0.0] | 100.0; +0.0 [+0.0, +0.0] | 100.0; +0.0 [+0.0, +0.0] |
| BATTERY / stage2 / sac | `trix_preventive` | 100.0 | 100.0; +0.0 [+0.0, +0.0] | 100.0; +0.0 [+0.0, +0.0] | 100.0; +0.0 [+0.0, +0.0] |

## Between-arm paired safe-completion differences

Second minus first arm, in percentage points, paired by training-seed identifier. Stage 1 shares a policy; Stage 2 uses separately trained policies.

| Task / stage / policy | Second minus first | Condition | Mean pp | Exact 95% interval |
|---|---|---|---:|---|
| SCREW / stage1 / sac | `trix` minus `qp_matched` | wide_1p5 | +0.0 | [+0.0, +0.0] |
| SCREW / stage1 / sac | `trix` minus `qp_matched` | wide_2 | +0.0 | [+0.0, +0.0] |
| SCREW / stage1 / sac | `trix` minus `qp_matched` | shell_2 | +0.0 | [+0.0, +0.0] |
| SCREW / stage1 / ppo | `trix` minus `qp_matched` | wide_1p5 | +0.0 | [+0.0, +0.0] |
| SCREW / stage1 / ppo | `trix` minus `qp_matched` | wide_2 | +0.0 | [+0.0, +0.0] |
| SCREW / stage1 / ppo | `trix` minus `qp_matched` | shell_2 | +0.0 | [+0.0, +0.0] |
| PCB / stage1 / sac | `trix` minus `box_clip` | wide_1p5 | +99.6 | [+99.3, +99.9] |
| PCB / stage1 / sac | `trix` minus `box_clip` | wide_2 | +99.6 | [+99.3, +99.9] |
| PCB / stage1 / sac | `trix` minus `box_clip` | shell_2 | +99.6 | [+99.3, +99.9] |
| PCB / stage2 / sac | `trix` minus `box_clip` | wide_1p5 | -0.2 | [-0.5, +0.0] |
| PCB / stage2 / sac | `trix` minus `box_clip` | wide_2 | -0.2 | [-0.5, +0.0] |
| PCB / stage2 / sac | `trix` minus `box_clip` | shell_2 | -0.2 | [-0.5, +0.0] |
| SNAP / stage2 / sac | `trix` minus `static_clip` | wide_1p5 | +80.4 | [+68.8, +87.6] |
| SNAP / stage2 / sac | `trix` minus `static_clip` | wide_2 | +73.8 | [+62.7, +80.9] |
| SNAP / stage2 / sac | `trix` minus `static_clip` | shell_2 | +56.4 | [+46.1, +63.7] |
| CRANK / stage2 / sac | `trix` minus `static_clip` | wide_1p5 | +47.6 | [+18.8, +76.4] |
| CRANK / stage2 / sac | `trix` minus `static_clip` | wide_2 | +47.6 | [+18.8, +76.4] |
| CRANK / stage2 / sac | `trix` minus `static_clip` | shell_2 | +47.6 | [+18.8, +76.4] |
| BATTERY / stage2 / sac | `trix_preventive` minus `box_clip_preventive` | wide_1p5 | +0.0 | [+0.0, +0.0] |
| BATTERY / stage2 / sac | `trix_preventive` minus `box_clip_preventive` | wide_2 | +0.0 | [+0.0, +0.0] |
| BATTERY / stage2 / sac | `trix_preventive` minus `box_clip_preventive` | shell_2 | +0.0 | [+0.0, +0.0] |

## All terminal outcomes and support strata

Each row has 1,000 episodes. S = safe completion; U = unsafe intact completion; D = destructive completion; M = mechanical failure; T = timeout. S+U+D+M+T=1,000. V = episodes with outcome-affecting violations, which can overlap those terminal categories. Within/outside entries are safe/episodes; n/a denotes an empty stratum. Counts are pooled descriptively across the ten seed replicates, not treated as 1,000 independent training replicates. The same reset draws are reused across seeds and arms.

| Task / stage / policy / arm | Condition | S | U | D | M | T | V | Within support | Outside support |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| SCREW / stage1 / sac / `qp_matched` | wide_1p5 | 879 | 21 | 0 | 0 | 100 | 31 | 389/440 | 490/560 |
| SCREW / stage1 / sac / `qp_matched` | wide_2 | 883 | 17 | 0 | 0 | 100 | 28 | 268/300 | 615/700 |
| SCREW / stage1 / sac / `qp_matched` | shell_2 | 879 | 21 | 0 | 0 | 100 | 32 | n/a | 879/1000 |
| SCREW / stage1 / sac / `trix` | wide_1p5 | 879 | 21 | 0 | 0 | 100 | 31 | 389/440 | 490/560 |
| SCREW / stage1 / sac / `trix` | wide_2 | 883 | 17 | 0 | 0 | 100 | 28 | 268/300 | 615/700 |
| SCREW / stage1 / sac / `trix` | shell_2 | 879 | 21 | 0 | 0 | 100 | 32 | n/a | 879/1000 |
| SCREW / stage1 / ppo / `qp_matched` | wide_1p5 | 599 | 1 | 0 | 0 | 400 | 1 | 263/440 | 336/560 |
| SCREW / stage1 / ppo / `qp_matched` | wide_2 | 599 | 1 | 0 | 0 | 400 | 1 | 180/300 | 419/700 |
| SCREW / stage1 / ppo / `qp_matched` | shell_2 | 599 | 1 | 0 | 0 | 400 | 1 | n/a | 599/1000 |
| SCREW / stage1 / ppo / `trix` | wide_1p5 | 599 | 1 | 0 | 0 | 400 | 1 | 263/440 | 336/560 |
| SCREW / stage1 / ppo / `trix` | wide_2 | 599 | 1 | 0 | 0 | 400 | 1 | 180/300 | 419/700 |
| SCREW / stage1 / ppo / `trix` | shell_2 | 599 | 1 | 0 | 0 | 400 | 1 | n/a | 599/1000 |
| PCB / stage1 / sac / `box_clip` | wide_1p5 | 0 | 0 | 1000 | 0 | 0 | 1000 | 0/740 | 0/260 |
| PCB / stage1 / sac / `box_clip` | wide_2 | 0 | 0 | 1000 | 0 | 0 | 1000 | 0/530 | 0/470 |
| PCB / stage1 / sac / `box_clip` | shell_2 | 0 | 0 | 1000 | 0 | 0 | 1000 | n/a | 0/1000 |
| PCB / stage1 / sac / `trix` | wide_1p5 | 996 | 4 | 0 | 0 | 0 | 4 | 736/740 | 260/260 |
| PCB / stage1 / sac / `trix` | wide_2 | 996 | 4 | 0 | 0 | 0 | 4 | 527/530 | 469/470 |
| PCB / stage1 / sac / `trix` | shell_2 | 996 | 4 | 0 | 0 | 0 | 4 | n/a | 996/1000 |
| PCB / stage2 / sac / `box_clip` | wide_1p5 | 1000 | 0 | 0 | 0 | 0 | 0 | 740/740 | 260/260 |
| PCB / stage2 / sac / `box_clip` | wide_2 | 1000 | 0 | 0 | 0 | 0 | 0 | 530/530 | 470/470 |
| PCB / stage2 / sac / `box_clip` | shell_2 | 1000 | 0 | 0 | 0 | 0 | 0 | n/a | 1000/1000 |
| PCB / stage2 / sac / `trix` | wide_1p5 | 998 | 2 | 0 | 0 | 0 | 2 | 738/740 | 260/260 |
| PCB / stage2 / sac / `trix` | wide_2 | 998 | 2 | 0 | 0 | 0 | 2 | 529/530 | 469/470 |
| PCB / stage2 / sac / `trix` | shell_2 | 998 | 2 | 0 | 0 | 0 | 2 | n/a | 998/1000 |
| SNAP / stage2 / sac / `static_clip` | wide_1p5 | 47 | 40 | 891 | 0 | 22 | 953 | 35/680 | 12/320 |
| SNAP / stage2 / sac / `static_clip` | wide_2 | 44 | 37 | 891 | 0 | 28 | 956 | 26/520 | 18/480 |
| SNAP / stage2 / sac / `static_clip` | shell_2 | 35 | 30 | 886 | 0 | 49 | 965 | n/a | 35/1000 |
| SNAP / stage2 / sac / `trix` | wide_1p5 | 851 | 0 | 136 | 0 | 13 | 149 | 646/680 | 205/320 |
| SNAP / stage2 / sac / `trix` | wide_2 | 782 | 0 | 201 | 0 | 17 | 218 | 488/520 | 294/480 |
| SNAP / stage2 / sac / `trix` | shell_2 | 599 | 0 | 369 | 0 | 32 | 401 | n/a | 599/1000 |
| CRANK / stage2 / sac / `static_clip` | wide_1p5 | 0 | 0 | 0 | 0 | 1000 | 150 | 0/660 | 0/340 |
| CRANK / stage2 / sac / `static_clip` | wide_2 | 0 | 0 | 0 | 0 | 1000 | 150 | 0/530 | 0/470 |
| CRANK / stage2 / sac / `static_clip` | shell_2 | 0 | 0 | 0 | 0 | 1000 | 150 | n/a | 0/1000 |
| CRANK / stage2 / sac / `trix` | wide_1p5 | 476 | 2 | 22 | 0 | 500 | 95 | 318/660 | 158/340 |
| CRANK / stage2 / sac / `trix` | wide_2 | 476 | 2 | 22 | 0 | 500 | 95 | 253/530 | 223/470 |
| CRANK / stage2 / sac / `trix` | shell_2 | 476 | 2 | 22 | 0 | 500 | 95 | n/a | 476/1000 |
| BATTERY / stage2 / sac / `box_clip_preventive` | wide_1p5 | 1000 | 0 | 0 | 0 | 0 | 0 | 760/760 | 240/240 |
| BATTERY / stage2 / sac / `box_clip_preventive` | wide_2 | 1000 | 0 | 0 | 0 | 0 | 0 | 610/610 | 390/390 |
| BATTERY / stage2 / sac / `box_clip_preventive` | shell_2 | 1000 | 0 | 0 | 0 | 0 | 0 | n/a | 1000/1000 |
| BATTERY / stage2 / sac / `trix_preventive` | wide_1p5 | 1000 | 0 | 0 | 0 | 0 | 0 | 760/760 | 240/240 |
| BATTERY / stage2 / sac / `trix_preventive` | wide_2 | 1000 | 0 | 0 | 0 | 0 | 0 | 610/610 | 390/390 |
| BATTERY / stage2 / sac / `trix_preventive` | shell_2 | 1000 | 0 | 0 | 0 | 0 | 0 | n/a | 1000/1000 |

## Interpretation and reporting limits

SNAP is the principal adverse result: TRiX falls from 96.5% in-range safe completion to 85.1%, 78.2% and 59.9% in the three added conditions. Destructive completions rise to 136, 201 and 369 per 1,000 episodes. The shell static arm reaches 3.5% safe completion with 886 destructive completions. Relative advantage does not make TRiX damage-free. This result is stated in main Results and Discussion, Supplementary S14 and R2.4.

SCREW retains equality between its matched solution mechanisms; PCB retains its policy-regime distinction. All CRANK and BATTERY conditions, including zeros and any improvement, are retained above. The aggregate records do not establish a unique mechanism for the SNAP change. No trajectory-mechanism experiment or post-result adjustment was added.

Full widened distributions mix native-support and outside-support episodes, so their means are not pure OOD rates. The shell is the primary pure-OOD endpoint. Support strata remain descriptive because their denominators vary. The earlier 105-case constant-sensitivity study remains separate; none of its results is pooled here.

## Files and reproduction

- [Frozen declaration](ood_reset_protocol.json) and [protocol](ood_reset_protocol.md).
- [Runner and raw-record audit](../experiments/ood_reset.py).
- [Complete summary and all metric intervals](../results/ood_reset/summary.json).
- [420 seed records](../results/ood_reset/seed_records.json), [1,500 unique reset draws](../results/ood_reset/reset_draws.json) and [420 raw-cell hashes](../results/ood_reset/manifest.json).
- [Table builder](../scripts/build_ood_reset_tables.py), [generated TeX](../manuscript/data/ood_reset_results.tex); run `python3 -B scripts/build_ood_reset_tables.py --check`.

The companion private directory `TRIX_ood_reset_evidence` preserves full raw cells, freeze metadata, run log, native-replay verification and the source snapshot from the prospective commit. Its payload checksum manifest is `SHA256SUMS.json`.
Manifest SHA-256: `39e0a577e9f1d4e53c61b700b3d0309b66f15033bcfe702beec766298d42dc58`.

The original `TRIX_reproducibility_release` archive and its checkpoint files remain unchanged. Archive manifest SHA-256: `00d1f97690f4ab90b11ae7da55fabcbe1c866c9d6dbe893eb1c15f3dcf292710`. Neither archive has been uploaded or assigned a review URL or DOI in this step.

The initial detached launch exited before creating any freeze or episode; its empty log and launch metadata are preserved. The managed run produced all 420 cells without a failed-cell retry. The 70-episode native replay check reproduced historical counts before the prospective run; 47 targeted tests passed before freezing.
