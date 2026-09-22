"""Why safety layers fail, decomposed level by level.

Each level fails for a different, separately measurable reason, so a
single feasibility number for "SafeLayer" conceals what is actually
happening. On PCB the admissible tilt region is a disk, whose gradient
direction rotates with the action, which is exactly the structure an
action-affine cost model cannot represent.

    componentwise clipping        cannot represent a coupled set at all
    learned action-affine         gradient is constant in the action
    learned nonlinear, one step   gradient correct, step short of a curve
    learned nonlinear, iterated   residual is cost-estimation error
    per-action oracle tangent     exact cost and gradient
    exact structured projection   exact, and cheapest measured

The affine model's failure is representational rather than a matter of
training: its held-out error is flat in the size of the training set, so
more data does not help. That control is run here rather than asserted.

Usage:
    python3 experiments/safelayer_decomposition.py --samples 20000
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from benchmark.registry import make_env
from benchmark import run_record as RR
from baselines import pcb_filters as PF
from baselines.learned_cost import AffineCost, NonlinearCost, halfspace_correct

BOUNDS = [(-PF.TAU_N, PF.TAU_N), (-PF.TAU_N, PF.TAU_N),
          (-PF.FZ_N, PF.FZ_N)]

# Every arm must aim at the same interior margin. A correction that lands
# exactly on the boundary falls outside it under float32 rounding about a
# third of the time, which would measure arithmetic rather than method.
MARGIN = -PF.INTERIOR * PF.TAU_N


def phi(obs):
    o = np.asarray(obs, dtype=np.float64)
    return np.array([1.0, o[0] / 0.015, o[2] / 0.05, o[3] / 0.05])


def collect(n, seed):
    """Samples of the true constraint value, which is all a learned model
    is given. The constraint definition itself is never exposed."""
    env = make_env("PCB")
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    O, A, C = [], [], []
    for i in range(n):
        a = rng.uniform(-1, 1, 3)
        O.append(np.asarray(obs, dtype=np.float64))
        A.append(a.copy())
        C.append(PF.tilt_norm_of(a) - PF.TAU_N)
        obs, _, done, _ = env.step(a)
        if done:
            obs = env.reset(seed=seed * 7 + i)
    return np.array(O), np.array(A), np.array(C)


def feasible(x):
    return PF.tilt_norm_of(x) <= PF.TAU_N


def evaluate(name, correct, O, A):
    t0 = time.perf_counter()
    ok = sum(feasible(correct(o, a)) for o, a in zip(O, A))
    dt = (time.perf_counter() - t0) / len(A) * 1e6
    return {"arm": name, "feasible_pct": 100.0 * ok / len(A),
            "latency_us": dt}


def gradient_agreement(model, O, A):
    """Cosine and magnitude against the analytic radial gradient."""
    cos, mag = [], []
    for o, a in zip(O, A):
        _, g = model(o, a)
        _, go = PF.safelayer_oracle_cost(o, a)
        ng, no = np.linalg.norm(g), np.linalg.norm(go)
        if ng > 1e-12 and no > 1e-12:
            cos.append(float(g @ go / (ng * no)))
            mag.append(float(ng))
    return float(np.mean(cos)), float(np.mean(mag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", type=int, nargs="+",
                    default=[500, 2000, 6000, 20000])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    Otr, Atr, Ctr = collect(args.samples, args.seed)
    Ote, Ate, Cte = collect(args.test, args.seed + 99)

    affine = AffineCost(phi).fit(Otr, Atr, Ctr)
    nonlin = NonlinearCost(obs_dim=Otr.shape[1], seed=args.seed).fit(
        Otr, Atr, Ctr)

    rows = [
        evaluate("componentwise clipping",
                 lambda o, a: PF.box_clip(o, a)[0], Ote, Ate),
        evaluate("learned action-affine",
                 lambda o, a: halfspace_correct(affine, o, a, BOUNDS,
                                                margin=MARGIN, iters=1),
                 Ote, Ate),
        evaluate("learned nonlinear, one step",
                 lambda o, a: halfspace_correct(nonlin, o, a, BOUNDS,
                                                margin=MARGIN, iters=1),
                 Ote, Ate),
        evaluate("learned nonlinear, iterated",
                 lambda o, a: halfspace_correct(nonlin, o, a, BOUNDS,
                                                margin=MARGIN, iters=10),
                 Ote, Ate),
        evaluate("per-action oracle tangent",
                 lambda o, a: halfspace_correct(PF.safelayer_oracle_cost, o, a,
                                                BOUNDS, margin=MARGIN,
                                                iters=1), Ote, Ate),
        evaluate("exact structured projection",
                 lambda o, a: PF.trix_project(o, a)[0], Ote, Ate),
    ]

    print(f"{'arm':32s} {'feasible%':>10} {'us/action':>11}")
    for r in rows:
        print(f"{r['arm']:32s} {r['feasible_pct']:10.1f} "
              f"{r['latency_us']:11.1f}")

    ca, ma = gradient_agreement(affine, Ote, Ate)
    cn, mn = gradient_agreement(nonlin, Ote, Ate)
    pa = np.array([affine(o, a)[0] + affine(o, a)[1] @ a
                   for o, a in zip(Ote, Ate)])
    pn = nonlin.predict(Ote, Ate)
    print(f"\n{'model':32s} {'RMSE':>10} {'grad cos':>10} {'|g|':>8}")
    print(f"{'learned action-affine':32s} "
          f"{np.sqrt(np.mean((pa - Cte) ** 2)):10.4f} {ca:10.3f} {ma:8.3f}")
    print(f"{'learned nonlinear':32s} "
          f"{np.sqrt(np.mean((pn - Cte) ** 2)):10.4f} {cn:10.3f} {mn:8.3f}")
    print(f"{'analytic':32s} {0.0:10.4f} {1.0:10.3f} {1.0:8.3f}")

    print("\ntraining-set sweep, affine model "
          "(flat error indicates a representational limit, not a data one)")
    sweep = []
    for n in args.sweep:
        O2, A2, C2 = collect(n, args.seed + 5)
        m2 = AffineCost(phi).fit(O2, A2, C2)
        p2 = np.array([m2(o, a)[0] + m2(o, a)[1] @ a
                       for o, a in zip(Ote, Ate)])
        ok = sum(feasible(halfspace_correct(m2, o, a, BOUNDS,
                                            margin=MARGIN))
                 for o, a in zip(Ote, Ate))
        row = {"n": n, "rmse": float(np.sqrt(np.mean((p2 - Cte) ** 2))),
               "feasible_pct": 100.0 * ok / len(Ate)}
        sweep.append(row)
        print(f"  n={n:6d}  RMSE {row['rmse']:.4f}  "
              f"feasible {row['feasible_pct']:5.1f}%")

    path = os.path.join(args.out, "safelayer_decomposition.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args), "arms": rows, "sweep": sweep,
                   "gradient": {"affine_cos": ca, "affine_mag": ma,
                                "nonlinear_cos": cn, "nonlinear_mag": mn},
                   "provenance": RR.provenance()}, f, indent=1)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
