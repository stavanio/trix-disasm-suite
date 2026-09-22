"""Evaluate correction methods across procedural geometries."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import geometry_gen as G
from baselines.learned_cost import AffineCost, NonlinearCost, halfspace_correct

N_PROPOSALS = 400
FIT_SAMPLES = 3000
METHODS = ("clip", "affine", "nonlinear", "oracle_cf", "oracle_ctx",
           "exact")


def margin(g, u, ctx):
    Q, A, b = g.matrices(ctx)
    u = np.asarray(u, dtype=np.float64)
    if Q is not None:
        return float(np.sqrt(max(0.0, u @ Q @ u)) - 1.0)
    return float(np.max(A @ u - b))


def cost_samples(g, rng, n=FIT_SAMPLES):
    obs, acts, costs = [], [], []
    for _ in range(n):
        c = float(rng.uniform(0.0, 1.0))
        u = rng.uniform(-1.5, 1.5, G.DIM)
        obs.append([c])
        acts.append(u)
        costs.append(margin(g, u, c))
    return np.array(obs), np.array(acts), np.array(costs)


def oracle_cost(g, ctx):
    def f(obs, u):
        u = np.asarray(u, dtype=np.float64)
        Q, A, b = g.matrices(ctx)
        if Q is not None:
            q = float(u @ Q @ u)
            if q < 1e-12:
                return -1.0, np.zeros(G.DIM)
            grad = (Q @ u) / np.sqrt(q)
            return float(np.sqrt(q) - 1.0) - float(grad @ u), grad
        i = int(np.argmax(A @ u - b))
        return -float(b[i]), A[i]
    return f


def bounds_for(g, ctx=None):
    w = G.context_free_box(g) if ctx is None else g.inscribed_box(ctx)
    return [(-float(x), float(x)) for x in w]


def correct(method, g, u, ctx, models, bnds):
    if method == "exact":
        return g.project(u, ctx)
    if method == "clip":
        w = G.context_free_box(g)
        return np.clip(np.asarray(u, dtype=np.float64), -w, w)
    if method == "oracle_cf":
        return halfspace_correct(oracle_cost(g, ctx), [ctx], u, bnds,
                                 margin=-1e-9, iters=12)
    if method == "oracle_ctx":
        return halfspace_correct(oracle_cost(g, ctx), [ctx], u,
                                 bounds_for(g, ctx), margin=-1e-9, iters=12)
    return halfspace_correct(models[method], [ctx], u, bnds,
                             margin=-1e-9, iters=12)


def evaluate(inst, rng):
    g, ctx, req = inst["geom"], inst["ctx"], inst["required"]
    O, A_, C = cost_samples(g, rng)
    models = {
        "affine": AffineCost(lambda o: np.array([1.0, float(o[0])]),
                             act_dim=G.DIM).fit(O, A_, C),
        "nonlinear": NonlinearCost(obs_dim=1, act_dim=G.DIM,
                                   seed=0).fit(O, A_, C, steps=250),
    }
    bnds = bounds_for(g)
    props = rng.uniform(-1.5, 1.5, (N_PROPOSALS, G.DIM))
    out = {}
    for m in METHODS:
        feas = 0
        dev = []
        for u in props:
            x = correct(m, g, u, ctx, models, bnds)
            feas += bool(g.contains(x, ctx))
            dev.append(float(np.linalg.norm(np.asarray(x) - u)))
        moved = float(np.linalg.norm(
            np.asarray(correct(m, g, req, ctx, models, bnds)) - req))
        out[m] = {"feasible": 100.0 * feas / len(props),
                  "intervention": float(np.mean(dev)),
                  "required_displacement": moved,
                  "retains_required": moved < 0.02 * float(
                      np.linalg.norm(req) + 1e-9)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = G.bank(args.split)
    if args.limit:
        bank = bank[:args.limit]
    rng = np.random.default_rng(7)

    rows = []
    for i, inst in enumerate(bank):
        rows.append({"id": inst["id"], "family": inst["family"],
                     "predicted": inst["predicted_structure_needed"],
                     "methods": evaluate(inst, rng)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(bank)}", flush=True)

    print(f"\nsplit {args.split}, {len(rows)} instances, spec {G.spec_hash()}\n")
    print(f"{'':26s} " + "".join(f"{m:>12}" for m in METHODS))
    for pred in (True, False):
        sub = [r for r in rows if r["predicted"] == pred]
        if not sub:
            continue
        label = ("predicted: needs structure" if pred
                 else "predicted: box suffices")
        print(f"{label:26s} n={len(sub)}")
        keeps = {m: 100.0 * np.mean([r["methods"][m]["retains_required"]
                                     for r in sub]) for m in METHODS}
        feas = {m: float(np.mean([r["methods"][m]["feasible"]
                                  for r in sub])) for m in METHODS}
        print(f"  {'retains required %':24s}" +
              "".join(f"{keeps[m]:12.1f}" for m in METHODS))
        print(f"  {'feasible %':24s}" +
              "".join(f"{feas[m]:12.1f}" for m in METHODS))

    path = os.path.join(args.out, f"geometry_bench_{args.split}.json")
    with open(path, "w") as f:
        json.dump({"spec_hash": G.spec_hash(), "split": args.split,
                   "rows": rows}, f, indent=1, default=str)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
