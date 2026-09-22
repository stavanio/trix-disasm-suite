"""Stage 1: nominal training, then one checkpoint through every arm.

The nominal policy is trained ONCE per (task, algorithm, seed). Its
checkpoint is selected by the frozen rule on validation seeds, and that
single checkpoint is then evaluated through every arm on held-out test
seeds. Training once rather than per arm matters for the comparison and
not only for compute: retraining per arm would add variance unrelated to
the filters, and the arms would no longer see identical proposals from
identical states.

Work is distributed as EVALUATION BATCHES, not whole cells. Evaluation
outweighs training by two orders of magnitude here, and episode length
varies enormously between tasks that complete and tasks that time out. A
cell-per-worker pool would let one CRANK cell with timing-out episodes
hold a worker while others idle.

Each batch writes its own shard, so an interrupted run resumes at batch
granularity rather than restarting a cell.

Usage:
    python3 training/stage1.py --seeds 0 --workers 20      # sentinel batch
    python3 training/stage1.py --workers 20                # full matrix
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BATCH = 5          # episodes per work unit


def _init_worker():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    torch.set_num_threads(1)


def shard_path(out, task, algo, seed, step, arm, i):
    key = f"{task}_{algo}_{seed}_{step}_{arm}_{i}"
    h = hashlib.sha256(key.encode()).hexdigest()[:10]
    return os.path.join(out, "shards", f"{key}__{h}.json")


def ckpt_path(out, task, algo, seed, step):
    return os.path.join(out, "checkpoints",
                        f"{task}__{algo}__none__seed{seed}_{step}.zip")


def train_cell(args):
    task, algo, seed, steps, out = args
    meta_p = os.path.join(out, "checkpoints",
                          f"{task}__{algo}__none__seed{seed}_meta.json")
    if os.path.exists(meta_p):
        return {"skipped": True}
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from training import sb3_runner as R
        os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
        R.train(task, algo, seed, steps, os.path.join(out, "checkpoints"),
                filter_name=None)
        return {"skipped": False}
    except Exception:
        return {"error": traceback.format_exc()}


def eval_batch(args):
    """One checkpoint, one arm, a few episodes. The unit of scheduling."""
    task, algo, seed, step, arm, seeds, idx, out = args
    real_arm = arm.split("_", 1)[1] if arm.startswith("test_") else (
        "trix" if arm in ("confirm", "screen") else arm)
    path = shard_path(out, task, algo, seed, step, arm, idx)
    if os.path.exists(path):
        return {"skipped": True}
    t0 = time.time()
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from benchmark.metrics import EpisodeRecorder
        from training import sb3_runner as R
        model = R.load(ckpt_path(out, task, algo, seed, step), task)
        rec = EpisodeRecorder(task, f"{algo}+{real_arm}", seed=seed)
        R.evaluate_frozen(model, task, real_arm, seeds, rec)
        s = rec.validate()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".partial"
        with open(tmp, "w") as f:
            json.dump({"summary": s, "episode_seeds": list(seeds),
                       "wall_s": time.time() - t0}, f, default=str)
        os.replace(tmp, path)
        return {"skipped": False, "wall_s": time.time() - t0}
    except Exception:
        return {"error": traceback.format_exc()}


def batches(task, algo, seed, step, arm, ep_seeds, out):
    return [(task, algo, seed, step, arm, ep_seeds[i:i + BATCH], i, out)
            for i in range(0, len(ep_seeds), BATCH)]


def _pool(summaries):
    """Combine batch summaries into one, from raw counts not rates."""
    if not summaries:
        return None
    tot = {}
    for s in summaries:
        for k, v in (s.get("counts") or {}).items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0) + v
    n = tot.get("episodes", 0)
    if not n:
        return None
    pct = lambda k: 100.0 * tot.get(k, 0) / n
    steps = [s.get("completion_steps_median") for s in summaries
             if s.get("completion_steps_median")]
    return {
        "counts": tot,
        "safe_completion_rate": pct("safe_completions"),
        "unsafe_completion_intact_rate": pct("unsafe_completions_intact"),
        "destructive_completion_rate": pct("destructive_completions"),
        "timeout_rate": pct("timeouts"),
        "completion_rate": 100.0 * (
            tot.get("safe_completions", 0)
            + tot.get("unsafe_completions_intact", 0)
            + tot.get("destructive_completions", 0)) / n,
        "episodes_with_any_violation_rate": pct("episodes_with_violation"),
        "violating_timestep_rate": (
            100.0 * tot.get("violating_steps", 0)
            / max(1, tot.get("total_steps", 1))),
        "completion_steps_median": (
            sorted(steps)[len(steps) // 2] if steps else None),
        "progress": 0.0,
        "constraint_hash": summaries[0].get("constraint_hash"),
    }


def gather(out, task, algo, seed, step, arm):
    """Pool the shards for one (checkpoint, arm) back into one summary."""
    d = os.path.join(out, "shards")
    if not os.path.isdir(d):
        return None
    prefix = f"{task}_{algo}_{seed}_{step}_{arm}_"
    found = []
    for fn in os.listdir(d):
        if (fn.startswith(prefix) and fn.endswith(".json")
                and fn[len(prefix):].split("__", 1)[0].isdigit()):
            with open(os.path.join(d, fn)) as f:
                found.append(json.load(f)["summary"])
    return _pool(found)


def run_pool(jobs, workers, label, fn):
    if not jobs:
        return
    t0, done, fails = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if "error" in r:
                fails.append(r)
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                eta = (len(jobs) - done) * el / done
                print(f"  {label}: {done}/{len(jobs)}  "
                      f"elapsed {el/60:5.1f}m eta {eta/60:5.1f}m", flush=True)
    if fails:
        print(f"  {label}: {len(fails)} FAILED")
        for f in fails[:3]:
            print("    " + f["error"].strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--algos", nargs="+", default=["sac", "ppo"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", default="results/stage1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from benchmark import protocol as PROTO, selection as SEL
    if any(t.startswith("BAYONET") for t in (args.tasks or [])):
        from benchmark import bayonet_protocol as PROTO
    from benchmark import run_record as RR
    from benchmark.registry import FILTERS, constraint_hashes
    from benchmark.selection import Checkpoint

    tasks = args.tasks or list(PROTO.TASK_STATUS)
    seeds = args.seeds or PROTO.PROTOCOL["production_seeds"]
    cells = [(t, a, s) for t in tasks for a in args.algos for s in seeds]

    print("Stage 1: nominal training plus frozen-policy evaluation")
    print(f"  protocol freeze {PROTO.freeze_hash()} ({PROTO.FROZEN_ON})")
    print(f"  {len(cells)} cells, {args.workers} workers, "
          f"{BATCH} episodes per batch\n")

    print("phase 1: training nominal policies")
    run_pool([(t, a, s, PROTO.BUDGETS[t], args.out) for t, a, s in cells],
             args.workers, "train", train_cell)

    print("\nphase 2: screening every candidate checkpoint")
    jobs = []
    for t, a, s in cells:
        vs = SEL.validation_seeds(t, SEL.SCREEN_EPISODES)
        for step in PROTO.CANDIDATE_CHECKPOINTS[t]:
            jobs += batches(t, a, s, step, "screen", vs, args.out)
    run_pool(jobs, args.workers, "screen", eval_batch)

    print("\nphase 3: confirming candidate window centres")
    chosen, jobs = {}, []
    for t, a, s in cells:
        cks = []
        for step in PROTO.CANDIDATE_CHECKPOINTS[t]:
            g = gather(args.out, t, a, s, step, "screen")
            if g:
                cks.append(Checkpoint(step=step,
                                      episodes=SEL.SCREEN_EPISODES,
                                      metrics=g))
        cands = SEL.candidate_windows(cks) if cks else []
        chosen[(t, a, s)] = [c.step for c in cands]
        vs = SEL.validation_seeds(t, SEL.CONFIRM_EPISODES,
                                  offset=SEL.SCREEN_EPISODES)
        for c in cands:
            jobs += batches(t, a, s, c.step, "confirm", vs, args.out)
    run_pool(jobs, args.workers, "confirm", eval_batch)

    print("\nphase 4: selected checkpoint through every arm")
    selected, jobs = {}, []
    for t, a, s in cells:
        cands = []
        for step in chosen.get((t, a, s), []):
            g = gather(args.out, t, a, s, step, "confirm")
            if g:
                cands.append(Checkpoint(step=step,
                                        episodes=SEL.CONFIRM_EPISODES,
                                        metrics=g))
        if not cands:
            continue
        pick = SEL.select(cands)
        selected[(t, a, s)] = pick.step
        ts = SEL.test_seeds(t, SEL.TEST_EPISODES)
        for arm in FILTERS[t]:
            jobs += batches(t, a, s, pick.step, f"test_{arm}", ts, args.out)
    run_pool(jobs, args.workers, "test", eval_batch)

    print("\nassembling records")
    H = constraint_hashes()
    records, table = [], defaultdict(dict)
    for (t, a, s), step in selected.items():
        for arm in FILTERS[t]:
            g = gather(args.out, t, a, s, step, f"test_{arm}")
            if not g:
                continue
            r = RR.make_record(t, f"{a}+{arm}", s, g, H[t],
                               training_mode="nominal", evaluation_arm=arm,
                               algorithm=a,
                               extra={"selected_step": step,
                                      "train_steps": PROTO.BUDGETS[t]})
            records.append(r)
            table[(t, a, arm)][s] = g.get("safe_completion_rate")

    path = os.path.join(args.out, "stage1_records.json")
    with open(path, "w") as f:
        json.dump({"protocol_freeze_hash": PROTO.freeze_hash(),
                   "selected_steps": {f"{k[0]}/{k[1]}/seed{k[2]}": v
                                      for k, v in selected.items()},
                   "records": records}, f, indent=1, default=str)

    print(f"\n{'task':8s} {'algo':5s} {'arm':16s} {'mean safe %':>12} {'seeds':>7}")
    for (t, a, arm), by_seed in sorted(table.items()):
        vals = [v for v in by_seed.values() if v is not None]
        mean = sum(vals) / len(vals) if vals else 0.0
        print(f"{t:8s} {a:5s} {arm:16s} {mean:12.1f} {len(vals):7d}")
    print(f"\n{len(records)} records written to {path}")


if __name__ == "__main__":
    main()
