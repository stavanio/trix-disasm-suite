#!/usr/bin/env python3
"""Frozen Stage-1 SCREW policies under TRiX projection with a rescaled axial metric.

The admissible polygon is unchanged; only the metric that selects the nearest
admissible command changes (S_v -> kappa * S_v). Requires the Stage-1
checkpoints and test shards from the evidence archive under --root.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
import re

import numpy as np

ALGOS = ("sac", "ppo")
KAPPAS = (0.5, 2.0)


def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def archived(root):
    data = json.loads((root / "results/stage1/stage1_records.json").read_text())
    step, safe = {}, {}
    for r in walk(data):
        if (str(r.get("task", "")).upper() != "SCREW" or r.get("training_mode") != "nominal"
                or r.get("algorithm") not in ALGOS or r.get("seed") is None):
            continue
        key = (r["algorithm"], int(r["seed"]))
        if r.get("selected_step") is not None:
            step[key] = int(r["selected_step"])
        if r.get("evaluation_arm") == "trix":
            safe[key] = int(r["counts"]["safe_completions"])
    expected = {(a, s) for a in ALGOS for s in range(10)}
    assert set(step) == expected and set(safe) == expected
    return step, safe


def episode_seeds(root, algo, seed, step):
    shards = glob.glob(str(root / f"results/stage1/shards/SCREW_{algo}_{seed}_{step}_test_trix_*.json"))
    shards.sort(key=lambda q: int(re.search(r"_test_trix_(\d+)__", q).group(1)))
    seeds = [x for q in shards for x in json.loads(Path(q).read_text())["episode_seeds"]]
    assert len(seeds) == 100 and len(set(seeds)) == 100, (algo, seed, len(seeds))
    return seeds


def weighted_filter(kappa):
    from baselines import screw_filters as SF

    poly = [np.asarray(p, dtype=np.float64) for p in SF._admissible_polygon()]
    W = np.diag([1.0 / kappa ** 2, 1.0])

    def inside(p):
        return SF._inside(np.asarray(p, dtype=np.float64), poly)

    def project(p):
        if inside(p):
            return p.copy()
        best, cost = None, np.inf
        for a, b in zip(poly, poly[1:] + poly[:1]):
            e = b - a
            den = float(e @ W @ e)
            t = 0.0 if den <= 0.0 else min(1.0, max(0.0, float((p - a) @ W @ e) / den))
            q = a + t * e
            c = float((q - p) @ W @ (q - p))
            if c < cost:
                best, cost = q, c
        return best

    def filt(obs, action):
        p = np.array([float(action[0]), float(action[1])], dtype=np.float64)
        q = project(p)
        out = np.array([q[0], q[1], np.clip(float(action[2]), -SF.RAD_LIM_N, SF.RAD_LIM_N)],
                       dtype=np.float32)
        e = SF.A_V * float(out[0]) - SF.A_W * float(out[1])
        infeasible = (abs(e) > SF.EPS + 1e-6 or abs(float(out[0])) > 1.0 + 1e-8
                      or abs(float(out[1])) > 1.0 + 1e-8 or abs(float(out[2])) > SF.RAD_LIM_N + 1e-8)
        return out, {"active": bool(not inside(p)), "infeasible": bool(infeasible), "recovery": False}

    return filt


def operator_gate(n=10000, tol=1e-6):
    from baselines import screw_solvers as SS

    rng = np.random.default_rng(20261008)
    f1 = weighted_filter(1.0)
    worst = 0.0
    for _ in range(n):
        a = rng.uniform(-1.0, 1.0, size=3)
        worst = max(worst, float(np.max(np.abs(np.asarray(SS.trix_scalar(None, a)[0], float)
                                                - np.asarray(f1(None, a)[0], float)))))
    assert worst <= tol, f"kappa=1 projector differs from production TRiX by {worst:.3e}"
    return worst


def run(job):
    root, algo, seed, step, kappa, seeds = job
    os.chdir(root)
    from benchmark import registry as REG
    from benchmark.metrics import EpisodeRecorder
    from training import sb3_runner as R

    name = f"trix_metric_{str(kappa).replace('.', 'p')}"
    REG.FILTERS["SCREW"][name] = weighted_filter(kappa)
    model = R.load(str(Path(root) / f"results/stage1/checkpoints/SCREW__{algo}__none__seed{seed}_{step}.zip"), "SCREW")
    rec = EpisodeRecorder("SCREW", f"{algo}+{name}", seed=seed)
    R.evaluate_frozen(model, "SCREW", name, seeds, rec)
    c = rec.summary()["counts"]
    return dict(algorithm=algo, training_seed=seed, selected_step=step, kappa=kappa,
                episodes=int(c["episodes"]), safe_completions=int(c["safe_completions"]),
                destructive_completions=int(c.get("destructive_completions", 0)),
                mechanical_failures=int(c.get("mechanical_failures", 0)),
                timeouts=int(c.get("timeouts", 0)),
                episodes_with_violation=int(c.get("episodes_with_violation", 0)))


def batch(root, step, kappas, workers):
    jobs = [(str(root), a, s, step[(a, s)], k, episode_seeds(root, a, s, step[(a, s)]))
            for a in ALGOS for s in range(10) for k in kappas]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return [f.result() for f in as_completed([ex.submit(run, j) for j in jobs])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=Path("results/screw_metric_sensitivity"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = a.root.resolve()
    step, safe = archived(root)
    worst = operator_gate()
    rows = batch(root, step, (1.0,), a.workers)
    mismatch = [r for r in rows if r["safe_completions"] != safe[(r["algorithm"], r["training_seed"])]]
    assert not mismatch, f"kappa=1 does not reproduce the archive: {mismatch}"
    rows += batch(root, step, KAPPAS, a.workers)
    rows.sort(key=lambda r: (r["algorithm"], r["kappa"], r["training_seed"]))
    aggregate = [dict(algorithm=al, kappa=k,
                      safe_completion_pct=float(np.mean([100.0 * r["safe_completions"] / r["episodes"]
                                                         for r in rows if r["algorithm"] == al and r["kappa"] == k])))
                 for al in ALGOS for k in (0.5, 1.0, 2.0)]
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        study="Frozen Stage-1 SCREW policies under TRiX projection with a rescaled axial metric",
        metric="S_v' = kappa * S_v; S_omega fixed; W = diag(1/kappa^2, 1)",
        operator_gate_max_difference=worst, archive_reproduced_at_kappa_1=True,
        rows=rows, aggregate=aggregate), indent=2))
    with (out / "per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in aggregate:
        print(f"{r['algorithm'].upper()} kappa={r['kappa']}: {r['safe_completion_pct']:.1f}%")


if __name__ == "__main__":
    main()
