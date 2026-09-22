# TRiX — paper reference

Source and compact evidence for **TRiX: Neuro-Symbolic Safety for Foundation
Model Agents in Robotic Disassembly**. Current manuscript: private revision,
not yet submitted in this form. Authors and citation metadata: [CITATION.cff](CITATION.cff).

## Start here

- [Editable paper](manuscript/TRIX_REVISION.tex)
- [Every figure and table → source, data and check](docs/PAPER_MAP.md)
- [Commands and recorded environments](docs/REPRODUCING.md)
- [Historical corrections](docs/provenance.md) and [BATTERY aggregation correction](docs/battery_aggregation_correction.md)
- [Archive identity and source commits](assets/evidence/archive_manifest.json)

TRiX governs commands at execution. The paper evaluates post-hoc governance
of frozen policies separately from filter-aware learning. PPO and SAC use
Stable-Baselines3; runtime arms are defined in [the registry](benchmark/registry.py).
The manuscript states the scope and limitations of the recorded evidence.

## Three checks

```bash
python3 -B scripts/check_paper_repository.py
python3 -B scripts/audit_vlm_reconciliation.py --archive-root . --out build/vlm.json
make paper
```

The repository check uses Python's standard library. The VLM audit also needs
NumPy; compilation needs XeLaTeX. See the command guide before running training.

## Repository and archive

Git contains maintained source, tests, TeX, final figures, small result files and
the mappings needed to trace them. Every code file has a declared entrypoint,
dependency or test role in [the checked inventory](docs/repository_manifest.json).

The **24.5 GB evidence archive** holds checkpoints, evaluation shards, raw videos,
large image banks and historical snapshots. Its manifest hash pins the exact
payload. It remains private and local; no public download URL or DOI exists yet.
Old custom agents and scratch experiments remain in those historical snapshots.

`benchmark/disasm_bench.py` remains because frozen environments import its shared
state types and constants. Its old command-line benchmark is not a paper entrypoint.
The six render scripts, their actual environment dependencies and artwork are
unchanged. Their [rendering guide](docs/workspace_rendering.md) identifies inputs.

## Git release sequence

1. Curate and check this source revision on `main`.
2. Finish the response letter and author review; commit those changes separately.
3. Tag the accepted source commit and bind it to the evidence manifest.
4. Push or publish only under the agreed access plan. Repositories remain private.

Cite the paper and exact source commit now. Add a release DOI only after one is
actually assigned. Git history and the evidence archive retain superseded work.
