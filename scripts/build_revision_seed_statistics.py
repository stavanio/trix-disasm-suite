#!/usr/bin/env python3
"""Audit archived seed counts and rebuild the manuscript's statistics table.

First run: --archive-root /path/to/full/research/checkout
Later runs reproduce the table from manuscript/data/seed_statistics_input.json.
No policy is trained, loaded or evaluated. Dependency: numpy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = []
for algo in ("sac", "ppo"):
    for arm in ("none", "qp_matched"):
        PLAN.append(("post_hoc", "SCREW", algo, "stage1", "trix", "stage1", arm))
for algo in ("sac", "ppo"):
    PLAN.append(("post_hoc", "PCB", algo, "stage1", "trix", "stage1", "box_clip"))
PLAN.append(("post_hoc", "SNAP", "sac", "stage1", "trix", "stage1", "static_clip"))
for task, a, b in [("PCB", "trix", "box_clip"), ("SNAP", "trix", "static_clip"),
                   ("CRANK", "trix", "static_clip"), ("PRY", "trix", "static_clip"),
                   ("BATTERY", "trix_preventive", "box_clip_preventive")]:
    PLAN.append(("filter_aware", task, "sac", "stage2", a, "stage2", b))
PLAN.append(("across_regimes", "PCB", "sac", "stage2", "box_clip", "stage1", "box_clip"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def key(stage, task, algo, arm, seed):
    return f"{stage}/{task}/{algo}/{arm}/{seed}"


def audit_archive(archive):
    indices, paths, sources = {}, {}, {}
    for stage in ("stage1", "stage2"):
        path = archive / f"results/{stage}/{stage}_records.json"
        sources[str(path.relative_to(archive))] = sha(path.read_bytes())
        records = json.loads(path.read_text())["records"]
        index = {}
        for r in records:
            k = key(stage, r["task"], r["algorithm"], r["evaluation_arm"], r["seed"])
            if k in index:
                raise ValueError(f"Duplicate aggregate record: {k}")
            index[k] = r
        indices[stage] = index
        paths[stage] = sorted((archive / f"results/{stage}/shards").glob("*test*.json"))
    selected = {}
    for _, task, algo, sa, aa, sb, ab in PLAN:
        for stage, arm in ((sa, aa), (sb, ab)):
            for seed in range(10):
                k = key(stage, task, algo, arm, seed)
                if k in selected:
                    continue
                r = indices[stage][k]
                assert r["summary"]["counts"] == r["counts"]
                assert r["summary"]["constraint_hash"] == r["constraint_hash"]
                assert r["training_mode"] == ("nominal" if stage == "stage1" else "filter_aware")
                tag = ("test_" if stage == "stage1" else "test__") + arm
                prefix = f'{task}_{algo}_{seed}_{r["selected_step"]}_{tag}_'
                matched = [p for p in paths[stage] if p.name.startswith(prefix)]
                ep_seeds, totals, shard_hashes = [], {}, {}
                for path in matched:
                    raw = path.read_bytes()
                    shard = json.loads(raw)
                    s = shard["summary"]
                    assert (s["task"], s["method"], s["seed"], s["constraint_hash"]) == (
                        task, f"{algo}+{arm}", seed, r["constraint_hash"])
                    assert len(shard["episode_seeds"]) == s["counts"]["episodes"]
                    ep_seeds.extend(shard["episode_seeds"])
                    for name, value in s["counts"].items():
                        totals[name] = totals.get(name, 0) + value
                    shard_hashes[str(path.relative_to(archive))] = sha(raw)
                assert totals == r["counts"], (k, totals, r["counts"])
                assert len(ep_seeds) == len(set(ep_seeds)) == 100, k
                rate = 100 * totals["safe_completions"] / totals["episodes"]
                assert abs(rate - r["summary"]["safe_completion_rate"]) < 1e-12
                required_hashes = ("constraint_hash", "robust_margin_hash", "taxonomy_hash",
                                   "protocol_freeze_hash")
                assert all(r.get(h) for h in required_hashes)
                selected[k] = {"stage": stage, "task": task, "algorithm": algo,
                               "arm": arm, "seed": seed,
                               "selected_step": r["selected_step"], "counts": totals,
                               "safe_completion_rate": rate,
                               "episode_seeds": sorted(ep_seeds),
                               "hashes": {h: r[h] for h in required_hashes},
                               "aggregate_record_sha256": sha(json.dumps(r, sort_keys=True).encode()),
                               "shard_sha256": shard_hashes}
    # Figure 4's archived points must agree with the data used for this table.
    figure4 = archive / "manuscript/figures/figure4_seed_data.csv"
    mapping = {"PCB post-hoc box": ("stage1", "PCB", "box_clip"),
               "PCB filter-aware box": ("stage2", "PCB", "box_clip"),
               "SNAP static": ("stage2", "SNAP", "static_clip"),
               "SNAP TRiX": ("stage2", "SNAP", "trix"),
               "CRANK static": ("stage2", "CRANK", "static_clip"),
               "CRANK TRiX": ("stage2", "CRANK", "trix"),
               "PRY static": ("stage2", "PRY", "static_clip"),
               "PRY TRiX": ("stage2", "PRY", "trix")}
    rows = list(csv.DictReader(figure4.open()))
    assert len(rows) == 80
    for row in rows:
        stage, task, arm = mapping[row["condition"]]
        value = selected[key(stage, task, "sac", arm, int(row["seed"]))]["safe_completion_rate"]
        assert value == float(row["safe_completion_rate"])
    sources[str(figure4.relative_to(archive))] = sha(figure4.read_bytes())
    return {"schema": 1, "endpoint": "safe completion over all held-out episodes",
            "unit": "training seed", "source_file_sha256": sources,
            "figure4_points_checked": len(rows), "records": selected}


def exact_bootstrap(differences):
    """Exact percentile law of a bootstrap mean on integral percentage points.

    Each bootstrap draw chooses one of the observed seed differences uniformly.
    Convolving its empirical PMF n times is the exact n-draw sum distribution.
    This avoids Monte Carlo error; it does not improve small-sample coverage.
    """
    d = np.asarray(differences, dtype=float)
    assert np.allclose(d, np.rint(d), atol=1e-12)
    d = np.rint(d).astype(int)
    n, lo, hi = len(d), int(d.min()), int(d.max())
    pmf = np.bincount(d - lo, minlength=hi - lo + 1).astype(float) / n
    law = np.array([1.0])
    for _ in range(n):
        law = np.convolve(law, pmf)
    assert abs(law.sum() - 1) < 1e-12
    support = (np.arange(len(law)) + n * lo) / n
    assert abs(np.dot(law, support) - np.mean(d)) < 1e-10
    variance = np.dot(law, (support - np.mean(d)) ** 2)
    assert abs(variance - np.var(d) / n) < 1e-9
    cdf = np.cumsum(law)
    return [float(support[np.searchsorted(cdf, q)]) for q in (0.025, 0.975)]


def compute(data):
    assert exact_bootstrap([0, 100]) == [0, 100]
    assert exact_bootstrap([7] * 10) == [7, 7]
    contrasts, seed_rows = [], []
    for group, task, algo, sa, aa, sb, ab in PLAN:
        va, vb, different_episode_seeds = [], [], []
        for seed in range(10):
            a, b = [data["records"][key(stage, task, algo, arm, seed)]
                    for stage, arm in ((sa, aa), (sb, ab))]
            for field in ("constraint_hash", "taxonomy_hash", "robust_margin_hash"):
                assert a["hashes"][field] == b["hashes"][field]
            if sa == sb:
                assert a["hashes"]["protocol_freeze_hash"] == b["hashes"]["protocol_freeze_hash"]
            if group == "post_hoc":
                assert a["selected_step"] == b["selected_step"]
            if a["episode_seeds"] != b["episode_seeds"]:
                different_episode_seeds.append(seed)
            va.append(a["safe_completion_rate"])
            vb.append(b["safe_completion_rate"])
            seed_rows.append({"group": group, "task": task, "algorithm": algo,
                              "stage_a": sa, "arm_a": aa, "stage_b": sb, "arm_b": ab,
                              "seed": seed, "a_percent": va[-1], "b_percent": vb[-1],
                              "difference_pp": va[-1] - vb[-1],
                              "same_episode_seeds": a["episode_seeds"] == b["episode_seeds"]})
        allowed_mismatch = group == "across_regimes" or task == "BATTERY"
        assert different_episode_seeds == (list(range(10)) if allowed_mismatch else [])
        d = np.asarray(va) - np.asarray(vb)
        contrasts.append({"group": group, "task": task, "algorithm": algo,
                          "stage_a": sa, "arm_a": aa, "stage_b": sb, "arm_b": ab,
                          "n_seed_pairs": 10, "mean_a_percent": float(np.mean(va)),
                          "mean_b_percent": float(np.mean(vb)),
                          "mean_difference_pp": float(np.mean(d)),
                          "median_difference_pp": float(np.median(d)),
                          "ci95_percentile_bootstrap_pp": exact_bootstrap(d),
                          "different_episode_seed_ids": different_episode_seeds})
    return contrasts, seed_rows


def tex_table(contrasts):
    text = r"""\begin{table}[tp]
\centering
\caption{Selected training-seed contrasts supporting the revised results.
Each row contains ten seed pairs (100 held-out episodes per policy/arm).
$A$ and $B$ are mean safe completion (\%); $\Delta=A-B$ and its 95\%
percentile bootstrap interval are in percentage points. Intervals resample
training-seed pairs, not individual episodes.}
\label{tab:submitted-11}
\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\setlength{\parskip}{0pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}llL{2.27in}rrrr@{}}
\toprule
Task & Alg. & Contrast ($A-B$) & $A$ & $B$ & $\Delta$ & 95\% interval\\
\midrule
"""
    titles = {"post_hoc": "Stage 1: same selected frozen policy",
              "filter_aware": "Stage 2: separately trained, filter-aware policies",
              "across_regimes": "Across regimes: same restriction, separately trained policies"}
    labels = {"none": "none", "qp_matched": "matched QP", "box_clip": "box",
              "static_clip": "static"}
    prior = None
    for c in contrasts:
        group = c["group"]
        if group != prior:
            if prior is not None:
                text += "\\midrule\n"
            text += f"\\multicolumn{{7}}{{l}}{{\\textit{{{titles[group]}}}}}\\\\\n\\addlinespace[3pt]\n"
            prior = group
        if group == "across_regimes":
            label = r"box: filter-aware $-$ post-hoc$^{\dagger}$"
        elif c["task"] == "BATTERY":
            label = r"preventive TRiX $-$ preventive box$^{\dagger}$"
        else:
            label = r"TRiX $-$ " + labels[c["arm_b"]]
        low, high = c["ci95_percentile_bootstrap_pp"]
        text += (f'{c["task"]} & {c["algorithm"].upper()} & {label} & '
                 f'{c["mean_a_percent"]:.1f} & {c["mean_b_percent"]:.1f} & '
                 f'{c["mean_difference_pp"]:.1f} & $[{low:.1f},\\,{high:.1f}]$\\\\\n')
    text += r"""\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{\linewidth}
\footnotesize
\textit{Interpretation.} Pairing is by training-seed identifier.
$^{\dagger}$BATTERY's preventive arms and PCB's cross-regime comparison use
different held-out episode streams; they are not episode-paired contrasts.
Stage~2 compares separately trained policies. PCB and SNAP arm comparisons
retain their documented margin/specification differences.
These descriptive intervals are not adjusted for multiple comparisons.
Collapsed intervals reflect the observed seeds, not a proof of equivalence
or population-level certainty. Historical Stage~1 BATTERY cells are excluded;
their specifications and episode counts differ. Other Stage~1 CRANK/PRY cells
and SNAP/PPO have zero safe completion in both compared arms and do not
establish governor equivalence.
\end{minipage}
\endgroup
\end{table}
"""
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "manuscript/data")
    parser.add_argument("--table-dir", type=Path, default=ROOT / "manuscript/tables")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.data_dir / "seed_statistics_input.json"
    if args.archive_root:
        data = audit_archive(args.archive_root)
        input_path.write_text(json.dumps(data, indent=2) + "\n")
    else:
        data = json.loads(input_path.read_text())
    contrasts, seed_rows = compute(data)
    report = {"method": "exact discrete percentile bootstrap of seed-paired mean differences",
              "confidence_level": 0.95, "experimental_unit": "training seed",
              "multiplicity_adjustment": None, "input_sha256": sha(input_path.read_bytes()),
              "records_audited": len(data["records"]),
              "shards_audited": sum(len(r["shard_sha256"]) for r in data["records"].values()),
              "figure4_points_checked": data["figure4_points_checked"], "contrasts": contrasts}
    (args.data_dir / "seed_statistics_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.data_dir / "seed_statistics_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    (args.table_dir / "seed_statistics.tex").write_text(tex_table(contrasts))
    print(json.dumps({k: v for k, v in report.items() if k != "contrasts"}, indent=2))
    for c in contrasts:
        print(c["group"], c["task"], c["algorithm"], c["arm_b"],
              c["mean_difference_pp"], c["ci95_percentile_bootstrap_pp"])
