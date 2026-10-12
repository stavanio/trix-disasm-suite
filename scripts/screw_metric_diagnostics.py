#!/usr/bin/env python3
"""Per-family violation accounting for the SCREW correction-metric replay,
with each sign of the thread-coupling torque. Needs the Stage-1 checkpoints
and test shards under --root."""
import argparse
import inspect
import json
import multiprocessing as mp
import os
import sys
import textwrap
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

KAPPAS = (0.5, 1.0, 2.0)
SIGNS = ("archived", "passive")


def run(job):
    root, algo, seed, step, kappa, sign, seeds = job
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    torch.set_num_threads(1)
    os.chdir(root)
    sys.path.insert(0, root)
    from envs import screw_env_v3 as E
    if not hasattr(E, "_archived_substep"):
        E._archived_substep = E.ScrewEnvV3._substep
        src = textwrap.dedent(inspect.getsource(E.ScrewEnvV3._substep))
        assert "tau_thread = -K_PITCH * F_thread" in src
        ns = dict(vars(E))
        exec(src.replace("tau_thread = -K_PITCH * F_thread", "tau_thread = +K_PITCH * F_thread"), ns)
        E._passive_substep = ns["_substep"]
    E.ScrewEnvV3._substep = E._passive_substep if sign == "passive" else E._archived_substep
    from experiments import screw_metric_sensitivity as M
    from benchmark import registry as REG
    from benchmark.metrics import EpisodeRecorder
    from training import sb3_runner as R

    name = f"diag_{kappa}"
    REG.FILTERS["SCREW"][name] = M.weighted_filter(kappa)
    model = R.load(str(Path(root) / f"results/stage1/checkpoints/SCREW__{algo}__none__seed{seed}_{step}.zip"), "SCREW")
    rec = EpisodeRecorder("SCREW", f"{algo}+{name}", seed=seed)
    R.evaluate_frozen(model, "SCREW", name, seeds, rec)
    cats, fam_eps, fam_completed = Counter(), Counter(), Counter()
    worst = 0.0
    for e in rec.episodes:
        cats[e["category"]] += 1
        for fam, n in e["violations_by_family"].items():
            if n:
                fam_eps[fam] += 1
                if e["completion"] == "completed":
                    fam_completed[fam] += 1
        x = e["exceed_by_family"].get("exceed_helix_cmd")
        if x:
            worst = max(worst, x["max"])
    return dict(algorithm=algo, seed=seed, kappa=kappa, sign=sign, categories=dict(cats),
                episodes_by_family=dict(fam_eps), completed_episodes_by_family=dict(fam_completed),
                max_cmd_helix_excess_mps=worst * 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path, default=Path("results/screw_metric_sensitivity/diagnostics.json"))
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    root = a.root.resolve()
    sys.path.insert(0, str(root))
    from experiments import screw_metric_sensitivity as M
    step, safe = M.archived(root)
    jobs = [(str(root), al, s, step[(al, s)], k, sg, M.episode_seeds(root, al, s, step[(al, s)])[:a.episodes])
            for al in M.ALGOS for s in range(10) for k in KAPPAS for sg in SIGNS]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, f in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            rows.append(f.result())
            print(f"{i}/{len(jobs)}", end="\r", flush=True)
    out = a.out if a.out.is_absolute() else root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(rows, key=lambda r: (r["algorithm"], r["kappa"], r["sign"], r["seed"])), indent=1))
    for r in rows:
        if r["kappa"] == 1.0 and r["sign"] == "archived" and a.episodes == 100:
            got = r["categories"].get("safe_completion", 0)
            assert got == safe[(r["algorithm"], r["seed"])], ("archive not reproduced", r)
    print("\nalgo kappa sign      safe unsafe_intact timeout destructive | completed episodes with: cmd_helix radial cmd_bound | max cmd excess (m/s)")
    agg = defaultdict(lambda: [Counter(), Counter(), 0.0])
    for r in rows:
        g = agg[(r["algorithm"], r["kappa"], r["sign"])]
        g[0].update(r["categories"]); g[1].update(r["completed_episodes_by_family"])
        g[2] = max(g[2], r["max_cmd_helix_excess_mps"])
    for (al, k, sg), (c, fam, w) in sorted(agg.items()):
        print(f"{al:4s} {k:5.1f} {sg:9s} {c.get('safe_completion',0):5d} {c.get('unsafe_completion_intact',0):13d} "
              f"{c.get('timeout',0):7d} {c.get('destructive_completion',0):11d} | "
              f"{fam.get('vio_helix_cmd',0):9d} {fam.get('vio_radial_force',0):6d} {fam.get('vio_command_helix',0):9d} | {w:.2e}")


if __name__ == "__main__":
    main()
