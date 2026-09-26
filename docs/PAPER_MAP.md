# Paper to source to evidence

Main and Supplementary Information have separate numbering. Stable TeX labels bind
each item to its sources. Citations use the source bibliography IDs.

[Main manuscript](../manuscript/TRIX_REVISION.tex) | [Supplement](../manuscript/TRIX_SUPPLEMENT.tex) | [Editorial change record](publication_edit.md)

All 34 reviewer points are mapped in [REVIEWER_MAP.md](REVIEWER_MAP.md).
`make response` refreshes their page references; `make check` rejects stale mappings.

## Figure 1: execution architecture

`fig:submitted-1` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex); [manuscript/figures/trix_architecture_fig1.svg](../manuscript/figures/trix_architecture_fig1.svg); [manuscript/figures/trix_architecture_fig1.pdf](../manuscript/figures/trix_architecture_fig1.pdf).

Editable SVG source; paper compilation verifies the exported figure.

## Figure 2: command representation

`fig:submitted-2` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[scripts/build_revision_figure2.py](../scripts/build_revision_figure2.py); [envs/screw_env_v3.py](../envs/screw_env_v3.py); [manuscript/figures/figure2_geometry.json](../manuscript/figures/figure2_geometry.json); [manuscript/figures/figure2_representation_gap.pdf](../manuscript/figures/figure2_representation_gap.pdf); [manuscript/figures/figure2_representation_gap.svg](../manuscript/figures/figure2_representation_gap.svg).

python3 scripts/build_revision_figure2.py

## Figure 3: six workspaces

`fig:disasm-workspaces` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[scripts/render_screw_workspace_b601.py](../scripts/render_screw_workspace_b601.py); [scripts/render_pcb_workspace_b601.py](../scripts/render_pcb_workspace_b601.py); [scripts/render_snap_workspace_b601.py](../scripts/render_snap_workspace_b601.py); [scripts/render_crank_workspace_b601.py](../scripts/render_crank_workspace_b601.py); [scripts/render_battery_workspace_b601.py](../scripts/render_battery_workspace_b601.py); [scripts/render_pry_workspace_b601.py](../scripts/render_pry_workspace_b601.py); [scripts/compose_b601_workspaces.py](../scripts/compose_b601_workspaces.py); [assets/workspaces/frozen_frames.json](../assets/workspaces/frozen_frames.json); [manuscript/figures/workspaces/state_manifest.json](../manuscript/figures/workspaces/state_manifest.json); [manuscript/figures/disasm_bench_workspaces.pdf](../manuscript/figures/disasm_bench_workspaces.pdf).

Individual commands and asset/state hashes: docs/workspace_rendering.md.

Citations: `ref33`.

## Figure 4: seed distributions

`fig:seed-distributions` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[manuscript/figures/figure4_seed_distributions.pdf](../manuscript/figures/figure4_seed_distributions.pdf); [manuscript/figures/figure4_seed_data.csv](../manuscript/figures/figure4_seed_data.csv); [scripts/build_revision_seed_statistics.py](../scripts/build_revision_seed_statistics.py).

[Replacement Figure 4 builder](../scripts/build_revision_figure4.py) and
[build manifest](../manuscript/figures/figure4_rebuild.json).

`python3 scripts/build_revision_figure4.py` regenerates the vector PDF from the
retained CSV after checking all 80 points against the audited seed counts.
The historical generator remains unavailable; no original-source recovery is claimed.

## Figure 5: B601 execution

`fig:b601-hardware` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[manuscript/figures/b601_hardware_sequence.jpg](../manuscript/figures/b601_hardware_sequence.jpg); [assets/evidence/archive_manifest.json](../assets/evidence/archive_manifest.json).

Archive: hardware/results/hardware/v4_confirmation/final_submission/HARDWARE_EVIDENCE_README.md; three linked videos.

## Table 1: related methods

`tab:submitted-1` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

Bibliography ref16–ref26; editorial comparison, no generated experiment.

Citations: `ref16`, `ref17`, `ref18`, `ref19`, `ref20`, `ref21`, `ref22`, `ref23`, `ref24`, `ref25`, `ref26`.

## Table 2: task definitions

`tab:submitted-2` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py).

Environment constants, reset(), step() and _get_obs(); source hashes pinned in assets/workspaces/environment_sources.json.

## Table 3: algorithm and runtime-arm taxonomy

`tab:submitted-3` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[benchmark/registry.py](../benchmark/registry.py); [training/sb3_runner.py](../training/sb3_runner.py); [baselines/learned_cost.py](../baselines/learned_cost.py).

Registry FILTERS and SB3 ALGOS; learned-cost study identified separately.

Citations: `ref21`, `ref31`, `ref32`.

## Table 4: Policy-adaptation comparisons

`tab:policy-summary` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex); [results/stage1/stage1_records.json](../results/stage1/stage1_records.json); [results/stage2/stage2_records.json](../results/stage2/stage2_records.json); [manuscript/data/seed_statistics_summary.json](../manuscript/data/seed_statistics_summary.json); [manuscript/data/battery_aggregation_correction.json](../manuscript/data/battery_aggregation_correction.json).

Compare the eight displayed pairs with the frozen Stage 1/Stage 2 records and paired-seed statistics; details in docs/publication_edit.md.

## Table 5: VLM decisions

`tab:vlm-results` in [manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

[scripts/audit_vlm_reconciliation.py](../scripts/audit_vlm_reconciliation.py); [experiments/vlm_trial.py](../experiments/vlm_trial.py); [experiments/vlm_bank.py](../experiments/vlm_bank.py); [results/vlm_bank/manifest.json](../results/vlm_bank/manifest.json); [results/vlm/vlm_anthropic.json](../results/vlm/vlm_anthropic.json); [results/vlm/vlm_openai.json](../results/vlm/vlm_openai.json); [results/vlm/vlm_google.json](../results/vlm/vlm_google.json).

make audit-vlm: 273 decisions; visibility scaling and unavailable backend/retry IDs disclosed.

Citations: `ref36`, `ref37`, `ref38`.

## Supplementary Table S1: observations

`tab:submitted-7` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py).

Trace each row to _get_obs(); units and indices checked against current environment definitions.

## Supplementary Table S2: Physical command scales

`tab:action-scales` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py); [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

Check ACTION_SCALE in each frozen environment; no rollout.

## Supplementary Table S3: constraints and outcomes

`tab:submitted-8` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py); [benchmark/outcome.py](../benchmark/outcome.py); [benchmark/taxonomy.py](../benchmark/taxonomy.py).

Environment step()/episode_summary() and classify(); no new physics parameters.

## Supplementary Table S4: reset randomisation

`tab:submitted-9` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py).

Environment reset() and noise constants.

## Supplementary Table S5: training configuration

`tab:submitted-10` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[training/sb3_runner.py](../training/sb3_runner.py); [benchmark/protocol.py](../benchmark/protocol.py); [benchmark/selection.py](../benchmark/selection.py); [benchmark/gym_adapter.py](../benchmark/gym_adapter.py); [requirements-training.txt](../requirements-training.txt).

Recorded fingerprints in the archive; requested and actual PPO step counts distinguished.

Citations: `ref31`, `ref32`.

## Supplementary Table S6: seed contrasts

`tab:submitted-11` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[scripts/build_revision_seed_statistics.py](../scripts/build_revision_seed_statistics.py); [manuscript/data/seed_statistics_input.json](../manuscript/data/seed_statistics_input.json); [manuscript/data/seed_statistics_summary.json](../manuscript/data/seed_statistics_summary.json); [manuscript/data/seed_statistics_pairs.csv](../manuscript/data/seed_statistics_pairs.csv); [manuscript/tables/seed_statistics.tex](../manuscript/tables/seed_statistics.tex).

python3 scripts/build_revision_seed_statistics.py; 13 contrasts, 220 records, 4,400 source shards.

## Supplementary Table S7: recorded computational cost

`tab:runtime-latency` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[experiments/runtime_benchmark.py](../experiments/runtime_benchmark.py); [requirements-runtime.txt](../requirements-runtime.txt); [docs/runtime_benchmark_protocol.md](../docs/runtime_benchmark_protocol.md); [docs/runtime_benchmark_results.md](../docs/runtime_benchmark_results.md); [manuscript/tables/runtime_latency.tex](../manuscript/tables/runtime_latency.tex); [results/runtime/inputs.npz](../results/runtime/inputs.npz); [results/runtime/manifest.json](../results/runtime/manifest.json); [results/runtime/repeat_0.json](../results/runtime/repeat_0.json); [results/runtime/repeat_0.log](../results/runtime/repeat_0.log); [results/runtime/repeat_0.npz](../results/runtime/repeat_0.npz); [results/runtime/repeat_1.json](../results/runtime/repeat_1.json); [results/runtime/repeat_1.log](../results/runtime/repeat_1.log); [results/runtime/repeat_1.npz](../results/runtime/repeat_1.npz); [results/runtime/repeat_2.json](../results/runtime/repeat_2.json); [results/runtime/repeat_2.log](../results/runtime/repeat_2.log); [results/runtime/repeat_2.npz](../results/runtime/repeat_2.npz); [results/runtime/repeat_3.json](../results/runtime/repeat_3.json); [results/runtime/repeat_3.log](../results/runtime/repeat_3.log); [results/runtime/repeat_3.npz](../results/runtime/repeat_3.npz); [results/runtime/repeat_4.json](../results/runtime/repeat_4.json); [results/runtime/repeat_4.log](../results/runtime/repeat_4.log); [results/runtime/repeat_4.npz](../results/runtime/repeat_4.npz); [results/runtime/summary.json](../results/runtime/summary.json).

python3 -B experiments/runtime_benchmark.py check --out results/runtime

## Supplementary Table S8: post-hoc governance

`tab:submitted-4` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[training/stage1.py](../training/stage1.py); [results/stage1/stage1_records.json](../results/stage1/stage1_records.json); [results/audits/pcb_oracle_seed_fix/pcb_oracle_seed0_corrected.json](../results/audits/pcb_oracle_seed_fix/pcb_oracle_seed0_corrected.json); [manuscript/data/battery_aggregation_correction.json](../manuscript/data/battery_aggregation_correction.json); [scripts/audit_reproducibility_release.py](../scripts/audit_reproducibility_release.py).

Full archive audit checks exact-arm shards and all 50 displayed cells.

## Supplementary Table S9: filter-aware governance

`tab:stage2-key` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[training/stage2.py](../training/stage2.py); [results/stage2/stage2_records.json](../results/stage2/stage2_records.json).

Full archive audit; training arms stay separate from Stage 1.

## Supplementary Table S10: procedural geometry

`tab:procedural-geometry` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[experiments/geometry_bench.py](../experiments/geometry_bench.py); [benchmark/geometry_gen.py](../benchmark/geometry_gen.py); [results/geometry_bench_train.json](../results/geometry_bench_train.json); [results/geometry_bench_val.json](../results/geometry_bench_val.json); [results/geometry_bench_heldout.json](../results/geometry_bench_heldout.json).

scripts/audit_revision_claims.py against the full archive; 200 held-out geometries.

## Supplementary Table S11: hardware confirmation

`tab:b601-hardware` in [manuscript/TRIX_SUPPLEMENT.tex](../manuscript/TRIX_SUPPLEMENT.tex).

[assets/evidence/archive_manifest.json](../assets/evidence/archive_manifest.json); [scripts/audit_revision_claims.py](../scripts/audit_revision_claims.py).

Archive: hardware/results/hardware/v4_confirmation/final_submission/tables/hardware_results.csv; six logs, three video-backed trials.

## Supporting analyses

- Learned PCB: [experiment](../experiments/safelayer_decomposition.py), [record](../results/safelayer_decomposition.json), main Section 2.4 and Supplementary Section S10.2.
- PRY control: [experiment](../experiments/pry_control.py), [record](../results/control/pry_control.json), Supplementary Section S9.3; separate from selected Stage 2 means.
- Sensitivity: [experiment](../experiments/sensitivity.py), [record](../results/sensitivity.json), main Section 4.4 and Supplementary Section S7.
- BAYONET: [preregistration](bayonet_preregistration.md), [freeze](bayonet_implementation_freeze.md), [Stage 1](../results/bayonet/stage1_records.json), [Stage 2](../results/bayonet2/stage2_records.json), Supplementary Sections S10.4 and S10.5.

The separate private archive is identified by [its manifest](../assets/evidence/archive_manifest.json).
Every retained file has a declared role in the [checked inventory](repository_manifest.json).

## Prospective R2.4 addition

The [reset-distribution amendment](ood_reset_protocol.md) maps
[its declaration](ood_reset_protocol.json),
[execution and analysis](../experiments/ood_reset.py),
[distribution interface](../benchmark/evaluation_distribution.py) and
[acceptance tests](../tests/test_ood_reset.py) to R2.4 and the evaluation-use
column of Supplementary Table S4. All 42,000 episodes are now complete and
reported in main Results 2.3, Methods 4.1/4.6, Supplementary S14 and R2.4.
[Results and provenance](ood_reset_results.md) retain adverse outcomes.

## Supplementary Table S12: Added reset ranges

`tab:ood-ranges` in Supplementary S14.

[Protocol](ood_reset_protocol.json), [results](ood_reset_results.md),
[summary](../results/ood_reset/summary.json), [seed records](../results/ood_reset/seed_records.json),
[reset draws](../results/ood_reset/reset_draws.json), [raw-file manifest](../results/ood_reset/manifest.json),
[table builder](../scripts/build_ood_reset_tables.py) and
[generated TeX](../manuscript/data/ood_reset_results.tex).

`python3 -B scripts/build_ood_reset_tables.py --check`.

## Supplementary Table S13: OOD safe completion and changes

`tab:ood-safe` in Supplementary S14.

[Protocol](ood_reset_protocol.json), [results](ood_reset_results.md),
[summary](../results/ood_reset/summary.json), [seed records](../results/ood_reset/seed_records.json),
[reset draws](../results/ood_reset/reset_draws.json), [raw-file manifest](../results/ood_reset/manifest.json),
[table builder](../scripts/build_ood_reset_tables.py) and
[generated TeX](../manuscript/data/ood_reset_results.tex).

`python3 -B scripts/build_ood_reset_tables.py --check`.

## Supplementary Table S14: OOD outcomes and support strata

`tab:ood-outcomes` in Supplementary S14.

[Protocol](ood_reset_protocol.json), [results](ood_reset_results.md),
[summary](../results/ood_reset/summary.json), [seed records](../results/ood_reset/seed_records.json),
[reset draws](../results/ood_reset/reset_draws.json), [raw-file manifest](../results/ood_reset/manifest.json),
[table builder](../scripts/build_ood_reset_tables.py) and
[generated TeX](../manuscript/data/ood_reset_results.tex).

`python3 -B scripts/build_ood_reset_tables.py --check`.


## Supplementary Table S15: SNAP command and damage attribution

`tab:snap-ood-attribution` in Supplementary S14.4, cited in main Results,
Discussion and response R2.4.

[Plan](snap_ood_attribution_plan.md), [results](snap_ood_attribution_results.md),
[diagnostic source](../experiments/snap_ood_attribution.py),
[summary](../results/snap_ood_attribution/summary.json),
[6,000 episode attributions](../results/snap_ood_attribution/episode_attribution.json),
[verification](../results/snap_ood_attribution/verification.json) and
[raw-trace manifest](../results/snap_ood_attribution/manifest.json).

The checks concern the existing S14/Table S14 SNAP records. The primary
outcomes remain unchanged. Table S15 separates broken/degraded states from
completion status, including broken-latch timeouts. Section S14.4 checks
membership in the exact frozen executable command set and distinguishes it
from the later margin registry. Main Results, Discussion and R2.4 now report
safe completion and fracture totals together.

Reproduce the diagnostic with the command in [REPRODUCING.md](REPRODUCING.md),
then check the table against the retained per-episode attributions and summary.
