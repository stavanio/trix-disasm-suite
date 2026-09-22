"""Volume-matched control: does task structure matter, or only restriction?

Three arms trained under identical budgets, architectures, seeds and
evaluation episodes:

    none          unrestricted action space
    static_clip   fixed bounds on the same axes
    trix          the same bounds, tracking insertion depth

At zero insertion the two restrictions admit an identical fraction of the
action box. They differ only in whether the torque bound follows depth,
so the comparison isolates task structure from the size of the
restriction.

PRIMARY ENDPOINT: safe completion rate over all held-out test episodes.
Not safety conditional on completion. Conditioning on completion selects
a different subset of episodes for each arm, since the treatment changes
how often episodes complete, and a conservative arm can appear safe among
its few completions while timing out everywhere else. Safe completion
scores capability and safety jointly and is immune to that.

Conditional safety and the count of competent seeds are reported as
secondary mechanism findings.

The competence threshold is fixed here, before any ten-seed result is
seen, at the value already used by the competence experiment.

Usage:
    python3 experiments/pry_control.py --seeds 10 --steps 15000
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from benchmark.gym_adapter import make
from benchmark.metrics import EpisodeRecorder
from benchmark import selection as SEL
from benchmark import run_record as RR
from benchmark.registry import constraint_hashes, get_filter

ARMS = (None, "static_clip", "trix")
COMPETENT_COMPLETION = 50.0     # frozen in advance, matches competence.py
NET_ARCH = [256, 256]


def admissible_fraction(task, name, n=20000):
    """Share of the action box each restriction leaves untouched."""
    if name is None:
        return 1.0
    f = get_filter(task, name)
    rng = np.random.default_rng(0)
    obs = np.zeros(10, dtype=np.float32)
    inside = 0
    for _ in range(n):
        a = rng.uniform(-1, 1, 3)
        x, _ = f(obs, a)
        inside += bool(np.allclose(np.asarray(x, dtype=np.float64), a,
                                   atol=1e-6))
    return inside / n


def run_arm(task, name, seed, steps, eval_seeds):
    env = make(task, filter_name=name)
    model = SAC("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                policy_kwargs=dict(net_arch=NET_ARCH))
    model.learn(total_timesteps=steps)
    rec = EpisodeRecorder(task, f"sac+{name}", seed=seed)
    ev = make(task, filter_name=name)
    for s in eval_seeds:
        obs, _ = ev.reset(seed=int(s))
        rec.start_episode()
        terminated = truncated = False
        for _ in range(ev.cap):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = ev.step(a)
            rec.step(info, reward=r)
            if terminated or truncated:
                break
        rec.end_episode(ev.unwrapped_task, completed=bool(terminated))
    return rec.validate()


def summarize(rows):
    """Seed-level mean and SD; the training seed is the experimental unit."""
    def ms(key):
        v = [r[key] for r in rows if r[key] is not None]
        return (float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 \
            else (float(v[0]) if v else 0.0, 0.0)

    safe = ms("safe_completion_rate")
    comp = ms("completion_rate")
    destr = ms("destructive_completion_rate")
    competent = sum(1 for r in rows
                    if r["completion_rate"] >= COMPETENT_COMPLETION)
    # Secondary and descriptive only: conditions on a post-treatment outcome.
    cond = [100.0 * r["counts"]["safe_completions"]
            / max(1, r["counts"]["safe_completions"]
                  + r["counts"]["unsafe_completions_intact"]
                  + r["counts"]["destructive_completions"])
           for r in rows if r["completion_rate"] > 0]
    return {
        "safe_completion_mean": safe[0], "safe_completion_sd": safe[1],
        "completion_mean": comp[0], "completion_sd": comp[1],
        "destructive_mean": destr[0], "destructive_sd": destr[1],
        "competent_seeds": competent, "n_seeds": len(rows),
        "conditional_safety_mean": float(np.mean(cond)) if cond else None,
        "per_seed_safe": [r["safe_completion_rate"] for r in rows],
        "per_seed_completion": [r["completion_rate"] for r in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PRY")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--episodes", type=int, default=SEL.TEST_EPISODES)
    ap.add_argument("--out", default="results/control")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_seeds = list(SEL.PRODUCTION_SEEDS)[:args.seeds]
    eval_seeds = SEL.test_seeds(args.task, args.episodes)

    print(f"{args.task} volume-matched control")
    print(f"  {len(train_seeds)} training seeds, {args.steps} steps, "
          f"{args.episodes} held-out episodes each")
    print("  admissible fraction of the action box:")
    for name in ARMS:
        print(f"    {str(name):12s} {admissible_fraction(args.task, name):.4f}")
    print(f"  competent seed threshold: >= {COMPETENT_COMPLETION:.0f}% "
          f"completion (frozen in advance)\n")

    results = {}
    for name in ARMS:
        rows = []
        for sd in train_seeds:
            s = run_arm(args.task, name, sd, args.steps, eval_seeds)
            rows.append(s)
            print(f"  {str(name):12s} seed {sd:2d}  "
                  f"safe {s['safe_completion_rate']:5.1f}  "
                  f"compl {s['completion_rate']:5.1f}  "
                  f"destr {s['destructive_completion_rate']:5.1f}", flush=True)
        results[str(name)] = {"summary": summarize(rows), "seeds": rows}

    print(f"\n  {'arm':12s} {'safe (primary)':>18} {'completion':>18} "
          f"{'destructive':>16} {'competent':>10}")
    for name in ARMS:
        s = results[str(name)]["summary"]
        print(f"  {str(name):12s} "
              f"{s['safe_completion_mean']:9.1f} +-{s['safe_completion_sd']:5.1f} "
              f"{s['completion_mean']:9.1f} +-{s['completion_sd']:5.1f} "
              f"{s['destructive_mean']:8.1f} +-{s['destructive_sd']:5.1f} "
              f"{s['competent_seeds']:6d}/{s['n_seeds']}")

    print("\n  secondary, descriptive only (conditions on completion, which "
          "the treatment changes):")
    for name in ARMS:
        c = results[str(name)]["summary"]["conditional_safety_mean"]
        print(f"    {str(name):12s} safety among completed episodes: "
              f"{'n/a' if c is None else f'{c:.1f}%'}")

    path = os.path.join(args.out, f"{args.task.lower()}_control.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args),
                   "competent_threshold": COMPETENT_COMPLETION,
                   "constraint_hash": constraint_hashes()[args.task],
                   "provenance": RR.provenance(),
                   "results": results}, f, indent=1, default=str)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
