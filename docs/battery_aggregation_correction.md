# BATTERY aggregation correction: 22 September 2026

The complete shard audit identified a filename-prefix collision in historical
`training/stage1.py:gather` and `training/stage2.py:gather`. Matching a prefix
ending in `test_trix_` also selected `test_trix_preventive_...`; similarly,
`test__box_clip_` also selected `test__box_clip_preventive_...` when the task,
algorithm, seed and selected step matched. These are different runtime arms.

The old 200-episode totals can be reconstructed exactly. They are not evidence
for 200 independent episodes from the arm named in the aggregate. Thirty
BATTERY records contain pooled arms; 22 repeat episode seed IDs across those
arms. Using the exact arm name gives 100 unique episode IDs per record for
all 1,020 production records.

| Historical group | Historical aggregate mean (%) | Exact-arm mean (%) |
|---|---:|---:|
| Stage 1 BATTERY/SAC/TRiX | 59.75 (printed 59.8) | 20.0 |
| Stage 1 BATTERY/PPO/TRiX | 0.0 | 0.0 |
| Stage 2 BATTERY/SAC/box_clip | 40.0 | 0.0 |
| Stage 2 BATTERY/PPO/box_clip | 81.0 | 71.0 |

Supplementary Table S8 now displays **20.0** for BATTERY/SAC/TRiX; main Results 2.3 and Supplementary S9.4 explain the correction.
The Stage 2 historical non-preventive box values in this audit table are not
numerical results claimed in the current manuscript. They are reported here
to account for the entire archive. No checkpoint was reselected or rerun;
exact-arm counts refer to the archived selected checkpoint. This audit does
not assert that redoing historical selection would select the same checkpoint.

The matched preventive arms remain **100.0% versus 100.0%**. Their distinct
full arm names were not pooled. All 13 Supplementary Table S6 contrasts and all 80 Figure 4
points remain unchanged, as do the VLM, geometry, sensitivity and hardware
results. The separate PCB seed-0 oracle correction remains in force.

The original result files and source snapshots are preserved byte for byte.
`verification/policy_record_linkage.json` records each exact-arm shard list,
historical pooled method names/counts, selected checkpoint and derived rate.
The exact lookup is `exact_arm_group_means["stage1/BATTERY/sac/trix"] = 20.0`.
The ten rates under `exact_arm_group_seed_rates` for that key are
`[0, 0, 0, 100, 0, 0, 0, 0, 0, 100]`, each from 100 unique held-out episodes.
This exact-arm group supersedes the historical pooled aggregate; the preserved
historical JSON is not silently rewritten.
`research/manuscript/data/battery_aggregation_correction.json` is the compact
reporting correction accompanying the paper.

The integrated source applies `patches/exact_arm_aggregation.patch`, which
requires the filename portion immediately after the arm prefix to be a numeric
batch index. Regression checks cover Stage 1 TRiX and Stage 2 box screening,
confirmation and test shards. No environment equation, physical constant,
policy checkpoint, archived result or frozen rendering was modified.
