# TRiX: paper reference

Source and compact evidence for **TRiX: Task-conditioned execution governance
for robotic disassembly**, revised manuscript COMMSENG-26-0216 for
Communications Engineering. Authors and citation metadata: [CITATION.cff](CITATION.cff).

## Start here

- [Main manuscript](manuscript/TRIX_REVISION.tex) and [Supplementary Information](manuscript/TRIX_SUPPLEMENT.tex)
- [Response to the editor and reviewers](manuscript/TRIX_RESPONSE.tex), [editor correspondence](manuscript/EDITOR_EMAIL.md) and [comment-to-evidence map](docs/REVIEWER_MAP.md)
- [Every figure and table: source, data and check](docs/PAPER_MAP.md)
- [Commands and recorded environments](docs/REPRODUCING.md)
- [Historical corrections](docs/provenance.md) and [BATTERY aggregation correction](docs/battery_aggregation_correction.md)
- [Archive identity and source commits](assets/evidence/archive_manifest.json)
- [Prospective shifted-reset evaluation and complete outcomes](docs/ood_reset_results.md)
- [SNAP command-membership and damage attribution](docs/snap_ood_attribution_results.md)

TRiX governs commands at execution. The paper evaluates post-hoc governance
of frozen policies separately from filter-aware learning. PPO and SAC use
Stable-Baselines3; runtime arms are defined in [the registry](benchmark/registry.py).
The manuscript states the scope and limitations of the recorded evidence.

## Three checks

```bash
make check
python3 -B scripts/audit_vlm_reconciliation.py --archive-root . --out build/vlm.json
make paper
```

The source inventory check uses Python's standard library. The shifted-reset
table and VLM audits also need NumPy; compilation needs XeLaTeX. See the command
guide before running training. `make paper` builds the main paper and the
Supplementary Information. `make response` builds both documents, refreshes the
response letter's page references and builds the letter. All 35 requests are
quoted from the original decision letter. E.4 and R1.6 rest on a
[recorded runtime benchmark](docs/runtime_benchmark_results.md) with all raw
calls and an independently regenerable table.

## Repository and archive

Git contains maintained source, tests, TeX, final figures, small result files and
the mappings needed to trace them. Every code file has a declared entrypoint,
dependency or test role in [the checked inventory](docs/repository_manifest.json).

The **24.5 GB evidence archive** holds checkpoints, evaluation shards, raw videos,
large image banks and historical snapshots. Its manifest hash pins the exact
payload. It is deposited in Zenodo (DOI 10.5281/zenodo.23031294), private during
peer review and public on publication. Old custom agents and scratch experiments
remain in its historical snapshots. The approximately 31 MB
`TRIX_ood_reset_evidence` companion preserves all 42,000 shifted-reset episode
outcomes and the prospective source freeze. Its raw-cell hashes and curated
records are mapped in [the shifted-reset results](docs/ood_reset_results.md).

`benchmark/disasm_bench.py` remains because frozen environments import its shared
state types and constants. Its old command-line benchmark is not a paper entrypoint.
The six render scripts, their actual environment dependencies and artwork are
unchanged. Their [rendering guide](docs/workspace_rendering.md) identifies inputs.

## Release

The code for the resubmission is tagged `v1.0-resubmission`. Cite the paper and
that release. Git history and the evidence archive retain superseded work.
