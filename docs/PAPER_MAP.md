# Paper → source → evidence

Figure/table labels below are stable TeX labels. Numbering refers to the current
70-page revision. Source citations refer to bibliography IDs in the manuscript.

## Figure 1 — execution architecture

`fig:submitted-1`

[manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex); [manuscript/figures/trix_architecture_fig1.svg](../manuscript/figures/trix_architecture_fig1.svg); [manuscript/figures/trix_architecture_fig1.pdf](../manuscript/figures/trix_architecture_fig1.pdf).

Editable SVG source; paper compilation verifies the exported figure.

## Figure 2 — command representation

`fig:submitted-2`

[scripts/build_revision_figure2.py](../scripts/build_revision_figure2.py); [envs/screw_env_v3.py](../envs/screw_env_v3.py); [manuscript/figures/figure2_geometry.json](../manuscript/figures/figure2_geometry.json); [manuscript/figures/figure2_representation_gap.pdf](../manuscript/figures/figure2_representation_gap.pdf); [manuscript/figures/figure2_representation_gap.svg](../manuscript/figures/figure2_representation_gap.svg).

python3 scripts/build_revision_figure2.py

## Figure 3 — six workspaces

`fig:disasm-workspaces`

[scripts/render_screw_workspace_b601.py](../scripts/render_screw_workspace_b601.py); [scripts/render_pcb_workspace_b601.py](../scripts/render_pcb_workspace_b601.py); [scripts/render_snap_workspace_b601.py](../scripts/render_snap_workspace_b601.py); [scripts/render_crank_workspace_b601.py](../scripts/render_crank_workspace_b601.py); [scripts/render_battery_workspace_b601.py](../scripts/render_battery_workspace_b601.py); [scripts/render_pry_workspace_b601.py](../scripts/render_pry_workspace_b601.py); [scripts/compose_b601_workspaces.py](../scripts/compose_b601_workspaces.py); [assets/workspaces/frozen_frames.json](../assets/workspaces/frozen_frames.json); [manuscript/figures/workspaces/state_manifest.json](../manuscript/figures/workspaces/state_manifest.json); [manuscript/figures/disasm_bench_workspaces.pdf](../manuscript/figures/disasm_bench_workspaces.pdf).

Individual commands and asset/state hashes: docs/workspace_rendering.md.

Citations: `ref33`.

## Figure 4 — seed distributions

`fig:seed-distributions`

[manuscript/figures/figure4_seed_distributions.pdf](../manuscript/figures/figure4_seed_distributions.pdf); [manuscript/figures/figure4_seed_data.csv](../manuscript/figures/figure4_seed_data.csv); [scripts/build_revision_seed_statistics.py](../scripts/build_revision_seed_statistics.py).

Full archive audit checks all 80 plotted points; original plot generator unavailable.

## Figure 5 — B601 execution

`fig:b601-hardware`

[manuscript/figures/b601_hardware_sequence.jpg](../manuscript/figures/b601_hardware_sequence.jpg); [assets/evidence/archive_manifest.json](../assets/evidence/archive_manifest.json).

Archive: hardware/results/hardware/v4_confirmation/final_submission/HARDWARE_EVIDENCE_README.md; three linked videos.

## Table 1 — related methods

`tab:submitted-1`

[manuscript/TRIX_REVISION.tex](../manuscript/TRIX_REVISION.tex).

Bibliography ref16–ref26; editorial comparison, no generated experiment.

Citations: `ref16`, `ref17`, `ref18`, `ref19`, `ref20`, `ref21`, `ref22`, `ref23`, `ref24`, `ref25`, `ref26`.

## Table 2 — task definitions

`tab:submitted-2`

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py).

Environment constants, reset(), step() and _get_obs(); source hashes pinned in assets/workspaces/environment_sources.json.

## Table 3 — algorithm and runtime-arm taxonomy

`tab:submitted-3`

[benchmark/registry.py](../benchmark/registry.py); [training/sb3_runner.py](../training/sb3_runner.py); [baselines/learned_cost.py](../baselines/learned_cost.py).

Registry FILTERS and SB3 ALGOS; learned-cost study identified separately.

Citations: `ref21`, `ref31`, `ref32`.

## Table 4 — post-hoc governance

`tab:submitted-4`

[training/stage1.py](../training/stage1.py); [results/stage1/stage1_records.json](../results/stage1/stage1_records.json); [results/audits/pcb_oracle_seed_fix/pcb_oracle_seed0_corrected.json](../results/audits/pcb_oracle_seed_fix/pcb_oracle_seed0_corrected.json); [manuscript/data/battery_aggregation_correction.json](../manuscript/data/battery_aggregation_correction.json); [scripts/audit_reproducibility_release.py](../scripts/audit_reproducibility_release.py).

Full archive audit checks exact-arm shards and all 50 displayed cells.

## Table 5 — filter-aware governance

`tab:stage2-key`

[training/stage2.py](../training/stage2.py); [results/stage2/stage2_records.json](../results/stage2/stage2_records.json).

Full archive audit; training arms stay separate from Stage 1.

## Table 6 — procedural geometry

`tab:procedural-geometry`

[experiments/geometry_bench.py](../experiments/geometry_bench.py); [benchmark/geometry_gen.py](../benchmark/geometry_gen.py); [results/geometry_bench_train.json](../results/geometry_bench_train.json); [results/geometry_bench_val.json](../results/geometry_bench_val.json); [results/geometry_bench_heldout.json](../results/geometry_bench_heldout.json).

scripts/audit_revision_claims.py against the full archive; 200 held-out geometries.

## Table 7 — VLM decisions

`tab:vlm-results`

[scripts/audit_vlm_reconciliation.py](../scripts/audit_vlm_reconciliation.py); [experiments/vlm_trial.py](../experiments/vlm_trial.py); [experiments/vlm_bank.py](../experiments/vlm_bank.py); [results/vlm_bank/manifest.json](../results/vlm_bank/manifest.json); [results/vlm/vlm_anthropic.json](../results/vlm/vlm_anthropic.json); [results/vlm/vlm_openai.json](../results/vlm/vlm_openai.json); [results/vlm/vlm_google.json](../results/vlm/vlm_google.json).

make audit-vlm: 273 decisions; visibility scaling and unavailable backend/retry IDs disclosed.

Citations: `ref36`, `ref37`, `ref38`.

## Table 8 — hardware confirmation

`tab:b601-hardware`

[assets/evidence/archive_manifest.json](../assets/evidence/archive_manifest.json); [scripts/audit_revision_claims.py](../scripts/audit_revision_claims.py).

Archive: hardware/results/hardware/v4_confirmation/final_submission/tables/hardware_results.csv; six logs, three video-backed trials.

## Table 9 — observations

`tab:submitted-7`

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py).

Trace each row to _get_obs(); units and indices checked against current environment definitions.

## Table 10 — constraints and outcomes

`tab:submitted-8`

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py); [benchmark/outcome.py](../benchmark/outcome.py); [benchmark/taxonomy.py](../benchmark/taxonomy.py).

Environment step()/episode_summary() and classify(); no new physics parameters.

## Table 11 — reset randomisation

`tab:submitted-9`

[envs/screw_env_v3.py](../envs/screw_env_v3.py); [envs/pcb_env_v2.py](../envs/pcb_env_v2.py); [envs/snap_env_v2.py](../envs/snap_env_v2.py); [envs/crank_env_v2.py](../envs/crank_env_v2.py); [envs/battery_env_v2.py](../envs/battery_env_v2.py); [envs/pry_env_v2.py](../envs/pry_env_v2.py).

Environment reset() and noise constants.

## Table 12 — training configuration

`tab:submitted-10`

[training/sb3_runner.py](../training/sb3_runner.py); [benchmark/protocol.py](../benchmark/protocol.py); [benchmark/selection.py](../benchmark/selection.py); [benchmark/gym_adapter.py](../benchmark/gym_adapter.py); [requirements-training.txt](../requirements-training.txt).

Recorded fingerprints in the archive; requested and actual PPO step counts distinguished.

Citations: `ref31`, `ref32`.

## Table 13 — seed contrasts

`tab:submitted-11`

[scripts/build_revision_seed_statistics.py](../scripts/build_revision_seed_statistics.py); [manuscript/data/seed_statistics_input.json](../manuscript/data/seed_statistics_input.json); [manuscript/data/seed_statistics_summary.json](../manuscript/data/seed_statistics_summary.json); [manuscript/data/seed_statistics_pairs.csv](../manuscript/data/seed_statistics_pairs.csv); [manuscript/tables/seed_statistics.tex](../manuscript/tables/seed_statistics.tex).

python3 scripts/build_revision_seed_statistics.py; 13 contrasts, 220 records, 4,400 source shards.

## Supporting analyses outside numbered tables

- Learned PCB: [experiment](../experiments/safelayer_decomposition.py), [record](../results/safelayer_decomposition.json), §5.3.
- PRY control: [experiment](../experiments/pry_control.py), [record](../results/control/pry_control.json), §5.2; distinct from selected Stage 2 means.
- Sensitivity: [experiment](../experiments/sensitivity.py), [record](../results/sensitivity.json), §5.5 / Appendix G.
- BAYONET: [criterion](../benchmark/criterion.py), [preregistration](bayonet_preregistration.md), [freeze](bayonet_implementation_freeze.md), [Stage 1](../results/bayonet/stage1_records.json), [Stage 2](../results/bayonet2/stage2_records.json), §5.5.

Large evidence is addressed through the [archive manifest](../assets/evidence/archive_manifest.json).
Every retained source file's import/dependency role is checked in the
[machine-readable inventory](repository_manifest.json). No pilot or retired
custom-policy entrypoint is part of the current paper interface.
