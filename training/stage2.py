"""Stage 2: filter-aware training.

Stage 1 asked what a filter does to a policy trained without it. This
asks a different question: how does the safety mechanism alter learning
itself. The filter is active during training and at evaluation, and the
policy never sees the unfiltered action space.

MDP framing, declared in benchmark.protocol: filter-as-environment. The
policy's action space is the nominal command space and the environment is
projector plus plant, so the replay tuple (s, a_nominal, r, s') is
correct and the critic learns the value of PROPOSING a command. These
results are NOT poolable with Stage 1: the execution keys differ and
check_comparable rejects the combination.

Each filtered arm needs its own training run, because the filter is part
of the environment rather than a post-hoc wrapper.

The axis-aligned arm is the restriction control. Volume matching is
established for PRY only, where both arms admit 10.96% of the action box
at zero insertion; elsewhere the arms differ in admitted volume as well
as in form. SNAP's comparison also changes phase awareness, and CRANK's
is world-frame against rotating-frame clipping.

Usage:
    python3 training/stage2.py --seeds 0 --workers 20    # sentinel
    python3 training/stage2.py --workers 20              # full
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.stage1 import (BATCH, _init_worker, _pool, batches, run_pool,
                             shard_path)

# The filtered arms trained here. "none" is not repeated: Stage 1 already
# provides the unfiltered nominal policy, and retraining it would only add
# variance to a comparison that is otherwise paired by construction.
TRAIN_ARMS = {
    "SCREW": ["trix"],
    "PCB": ["box_clip", "trix"],
    "SNAP": ["static_clip", "trix"],
    "CRANK": ["static_clip", "trix"],
    "BATTERY": ["box_clip", "box_clip_preventive",
                "trix_preventive"],
    "PRY": ["static_clip", "trix"],
    "BAYONET_S": ["box_clip", "trix"],
    "BAYONET_P": ["box_blind_conservative", "box_blind_permissive",
                  "box_phase_aware", "trix"],
}


def s2_ckpt(out, task, algo, seed, arm, step):
    return os.path.join(out, "checkpoints",
                        f"{task}__{algo}__{arm}__seed{seed}_{step}")


def train_cell(args):
    task, algo, seed, arm, steps, out = args
    meta_p = os.path.join(out, "checkpoints",
                          f"{task}__{algo}__{arm}__seed{seed}_meta.json")
    if os.path.exists(meta_p):
        return {"skipped": True}
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from training import sb3_runner as R
        os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
        R.train(task, algo, seed, steps, os.path.join(out, "checkpoints"),
                filter_name=arm)
        return {"skipped": False}
    except Exception:
        return {"error": traceback.format_exc()}


def eval_batch(args):
    """Evaluate through the SAME filter the policy trained under."""
    task, algo, seed, step, tag, seeds, idx, out = args
    arm = tag.split("__", 1)[1]
    path = shard_path(out, task, algo, seed, step, tag, idx)
    if os.path.exists(path):
        return {"skipped": True}
    t0 = time.time()
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from benchmark.metrics import EpisodeRecorder
        from training import sb3_runner as R
        model = R.load(s2_ckpt(out, task, algo, seed, arm, step), task)
        rec = EpisodeRecorder(task, f"{algo}+{arm}", seed=seed)
        R.evaluate_frozen(model, task, arm, seeds, rec)
        s = rec.validate()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".partial"
        with open(tmp, "w") as f:
            json.dump({"summary": s, "episode_seeds": list(seeds),
                       "wall_s": time.time() - t0}, f, default=str)
        os.replace(tmp, path)
        return {"skipped": False}
    except Exception:
        return {"error": traceback.format_exc()}


def gather(out, task, algo, seed, step, tag):
    d = os.path.join(out, "shards")
    if not os.path.isdir(d):
        return None
    prefix = f"{task}_{algo}_{seed}_{step}_{tag}_"
    found = []
    for fn in os.listdir(d):
        if (fn.startswith(prefix) and fn.endswith(".json")
                and fn[len(prefix):].split("__", 1)[0].isdigit()):
            with open(os.path.join(d, fn)) as f:
                found.append(json.load(f)["summary"])
    return _pool(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--algos", nargs="+", default=["sac", "ppo"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--out", default="results/stage2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from benchmark import protocol as PROTO, selection as SEL
    if any(t.startswith("BAYONET") for t in (args.tasks or [])):
        from benchmark import bayonet_protocol as PROTO
    from benchmark import run_record as RR
    from benchmark.registry import constraint_hashes
    from benchmark.selection import Checkpoint

    tasks = args.tasks or list(PROTO.TASK_STATUS)
    seeds = args.seeds or PROTO.PROTOCOL["production_seeds"]
    cells = [(t, a, s, arm) for t in tasks for a in args.algos
             for s in seeds for arm in TRAIN_ARMS[t]]

    print("Stage 2: filter-aware training")
    print(f"  protocol freeze {PROTO.freeze_hash()} ({PROTO.FROZEN_ON})")
    framing = getattr(PROTO, "MDP_FRAMING", None)
    if framing:
        print(f"  MDP framing: "
              f"{framing['filter_active_during_training']['framing']}")
    print(f"  {len(cells)} cells, {args.workers} workers\n")

    print("phase 1: training with the filter active")
    run_pool([(t, a, s, arm, PROTO.BUDGETS[t], args.out)
              for t, a, s, arm in cells], args.workers, "train", train_cell)

    print("\nphase 2: screening every candidate checkpoint")
    jobs = []
    for t, a, s, arm in cells:
        vs = SEL.validation_seeds(t, SEL.SCREEN_EPISODES)
        for step in PROTO.CANDIDATE_CHECKPOINTS[t]:
            jobs += batches(t, a, s, step, f"screen__{arm}", vs, args.out)
    run_pool(jobs, args.workers, "screen", eval_batch)

    print("\nphase 3: confirming candidate window centres")
    chosen, jobs = {}, []
    for t, a, s, arm in cells:
        cks = []
        for step in PROTO.CANDIDATE_CHECKPOINTS[t]:
            g = gather(args.out, t, a, s, step, f"screen__{arm}")
            if g:
                cks.append(Checkpoint(step=step,
                                      episodes=SEL.SCREEN_EPISODES,
                                      metrics=g))
        cands = SEL.candidate_windows(cks) if cks else []
        chosen[(t, a, s, arm)] = [c.step for c in cands]
        vs = SEL.validation_seeds(t, SEL.CONFIRM_EPISODES,
                                  offset=SEL.SCREEN_EPISODES)
        for c in cands:
            jobs += batches(t, a, s, c.step, f"confirm__{arm}", vs, args.out)
    run_pool(jobs, args.workers, "confirm", eval_batch)

    print("\nphase 4: selected checkpoint on held-out test seeds")
    selected, jobs = {}, []
    for t, a, s, arm in cells:
        cands = []
        for step in chosen.get((t, a, s, arm), []):
            g = gather(args.out, t, a, s, step, f"confirm__{arm}")
            if g:
                cands.append(Checkpoint(step=step,
                                        episodes=SEL.CONFIRM_EPISODES,
                                        metrics=g))
        if not cands:
            continue
        pick = SEL.select(cands)
        selected[(t, a, s, arm)] = pick.step
        ts = SEL.test_seeds(t, SEL.TEST_EPISODES)
        jobs += batches(t, a, s, pick.step, f"test__{arm}", ts, args.out)
    run_pool(jobs, args.workers, "test", eval_batch)

    print("\nassembling records")
    H = constraint_hashes()
    records, table = [], defaultdict(dict)
    for (t, a, s, arm), step in selected.items():
        g = gather(args.out, t, a, s, step, f"test__{arm}")
        if not g:
            continue
        r = RR.make_record(t, f"{a}+{arm}", s, g, H[t],
                           training_mode="filter_aware",
                           evaluation_arm=arm, algorithm=a,
                           extra={"selected_step": step,
                                  "train_steps": PROTO.BUDGETS[t]})
        records.append(r)
        table[(t, a, arm)][s] = g.get("safe_completion_rate")

    path = os.path.join(args.out, "stage2_records.json")
    if os.path.exists(path):
        prior = json.load(open(path)).get("records", [])
        fresh = {(r["task"], r.get("algorithm"), r.get("evaluation_arm"),
                  r["seed"]) for r in records}
        kept = [r for r in prior
                if (r["task"], r.get("algorithm"), r.get("evaluation_arm"),
                    r["seed"]) not in fresh]
        if kept:
            print(f"  merging {len(kept)} records outside this run's scope")
            records = kept + records
    with open(path, "w") as f:
        json.dump({"protocol_freeze_hash": PROTO.freeze_hash(),
                   "training_mode": "filter_aware",
                   "records": records}, f, indent=1, default=str)

    print(f"\n{'task':8s} {'algo':5s} {'trained under':16s} "
          f"{'mean safe %':>12} {'seeds':>7}")
    for (t, a, arm), by_seed in sorted(table.items()):
        vals = [v for v in by_seed.values() if v is not None]
        mean = sum(vals) / len(vals) if vals else 0.0
        print(f"{t:8s} {a:5s} {arm:16s} {mean:12.1f} {len(vals):7d}")
    print(f"\n{len(records)} records written to {path}")


if __name__ == "__main__":
    main()
