# Reproduction commands

Run commands from the repository root. Python 3.12.3 is the recorded scientific
runtime. The repository checker requires Python 3.11 or later; offline numerical
audits use NumPy 1.26.4 from `requirements.txt`.

## Read-only checks

```bash
make check
make audit-vlm
python3 -B -m unittest tests.test_exact_arm_aggregation
```

These check the source map, raw VLM decisions and the aggregation fix without
training or model/API/hardware execution. `make audit-vlm` writes under `build/`.
The retained tests also include simulation and SB3 acceptance gates; running the
entire suite can train short test policies. No full-suite run is implied here.

## Full evidence audit

Obtain the private archive identified in [archive_manifest.json](../assets/evidence/archive_manifest.json).
Set `TRIX_EVIDENCE_ROOT` to its directory, then run:

```bash
python3 -B scripts/audit_reproducibility_release.py --package-root "$TRIX_EVIDENCE_ROOT" --verify-hashes
```

The archive contains its historical source tree and verification entrypoint.
The command checks that frozen payload. The current repository separately
enforces its own lean file inventory. The archive has no remote URL yet.

## Build the paper and statistics

### Overleaf upload

Run `make overleaf` to create `build/TRIX_overleaf.zip`. In Overleaf, select
**New Project > Upload Project**, upload the ZIP, and set **Main document** to
`TRIX_MAIN.tex` and **Compiler** to **XeLaTeX**. Keep all directories from the ZIP.
Select `TRIX_SUPPLEMENT.tex` to compile the separate supplement.
The package includes both documents, five figures, two table fragments, cross-references and the exact
DejaVu text/math fonts with their licence. Fonts are loaded by file path, so
the build does not require those fonts to be installed on Overleaf.
The standalone manuscript TeX file alone does not include its dependencies.

The [bundle builder](../scripts/build_overleaf_bundle.py) follows the declared
TeX/figure inputs and verifies [font hashes](../manuscript/fonts/manifest.json).
Its root `TRIX_MAIN.tex` and `TRIX_SUPPLEMENT.tex` are byte-identical to the
canonical main and supplement sources. The ZIP
contains a file-to-source/hash manifest and a XeLaTeX `latexmkrc`. It omits
build products and unrelated research records. For local verification, extract
the ZIP into a new directory and run `latexmk TRIX_MAIN.tex` and
`latexmk TRIX_SUPPLEMENT.tex` there.
See Overleaf's [project upload instructions](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project)
and [compiler settings](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler).

### Repository build

```bash
make paper
make response
python3 scripts/build_revision_seed_statistics.py
```

The statistics command regenerates the 13-contrast table from the compact saved
counts. That table is included directly by the TeX, with no second inline copy.
To rebuild counts from shards, pass `--archive-root "$TRIX_EVIDENCE_ROOT/research"`.
The full archive's original result files remain unchanged; the BATTERY correction
is recorded in [the correction note](battery_aggregation_correction.md).

`make paper` builds the main paper and Supplementary Information separately.
Their generated cross-document references are checked against both builds.
`make response` first builds both documents, then regenerates the response's section,
page and bibliography references from that build. The response map records exact
TeX source ranges and hashes; they are not journal margin line numbers.
`make check` also detects stale response mappings without requiring a TeX runtime.
Original reviewer wording remains a review gate; see [the response map](REVIEWER_MAP.md).
The new runtime benchmark supplies the measurements for E.4 and R1.6.

The response letter uses XeLaTeX with 12-point TeX Gyre Termes text, matching
TeX Gyre Termes Math, and Latin Modern Mono for literal code identifiers.
These are OpenType fonts distributed with TeX Live; install the TeX Gyre,
TeX Gyre Math and Latin Modern fonts if the build cannot locate them.
All PDF fonts should be embedded (`pdffonts manuscript/build/TRIX_RESPONSE.pdf`);
the letter's text, equations and table remain vectors, with no rasterized text.
The serif typeface and spacing are readability choices, not a mandatory Nature
response-letter template. The journal's [revised-submission guide](https://www.nature.com/commseng/submit/guide-to-authors)
requires point-by-point replies and compliance with the editor's decision letter.
[Nature's rebuttal advice](https://blogs.nature.com/blog/how-to-write-a-rebuttal-letter/)
recommends clearly distinguishing comments and replies. The draft does this
with italic dark-blue comment summaries and black responses; it still requires
the original reviewer reports before those summaries can be verified.

## Recorded runtime benchmark

Use the separate numerical environment in [requirements-runtime.txt](../requirements-runtime.txt).
This records the new timing experiment's versions; it does not reconstruct
historical solver versions. [Protocol](runtime_benchmark_protocol.md) and
[results/scope](runtime_benchmark_results.md) define the measurement boundary.

```bash
python3 -B experiments/runtime_benchmark.py check --out results/runtime
python3 -B experiments/runtime_benchmark.py table --out results/runtime
# Optional new timing replication; refuses to overwrite an existing directory:
python3 -B experiments/runtime_benchmark.py run --out build/runtime-replication
```

The first two commands only read saved measurements and regenerate the table.
The final command times the existing filter implementations; it runs no
training, environment rollout, provider call or hardware command. All 225,280
recorded calls and their source/input hashes are retained in the lean repository.

## Training and experiment entrypoints

Production: `training/stage1.py` and `training/stage2.py`; both expose `--help`.
The default task set is the six main tasks. BAYONET uses explicit task and output
arguments as recorded in `benchmark/bayonet_protocol.py` and the preregistration.
Auxiliary experiments are listed by figure/table in [PAPER_MAP.md](PAPER_MAP.md).
VLM collection requires provider credentials; the offline VLM audit does not.

`requirements-training.txt` preserves the recorded direct versions (PyTorch
2.13.0+cpu, SB3 2.9.0, Gymnasium 1.3.0). SciPy, OSQP and quadprog are declared
imports whose historical versions were not recorded. This is not a recovered
transitive lockfile or a claim of a fresh training reproduction.

Rendering uses [requirements-workspace-renders.txt](../requirements-workspace-renders.txt)
and [the unchanged workspace guide](workspace_rendering.md). Its exact meshes
and fonts are supplied in the evidence archive, with their hashes pinned here.
No renderer was run during Git curation.

## Artwork provenance

Figure 1 is editable SVG. Figure 4 is regenerated from the retained 80-point
CSV by the supplied replacement builder; the historical plotting script remains
unavailable. Figure 5 is an archived hardware image, supported by the separate
run-to-video mapping.

```bash
python3 -m pip install -r requirements-figures.txt
python3 scripts/build_revision_figure4.py
```

The builder verifies every point against `manuscript/data/seed_statistics_input.json`,
uses fixed horizontal seed offsets without perturbing the measured values, and
writes a vector PDF with embedded TrueType fonts. `figure4_rebuild.json` records
the builder, data and output hashes plus NumPy/Matplotlib versions. It explicitly
identifies this as a replacement generator. `make check` includes its 80-point
audit without importing Matplotlib. The six workspace renders remain frozen.
