"""Do the conclusions survive the constants they were measured with?

Every physical parameter in these environments is a modelling choice.
Some come from literature, some were calibrated against a healthy
controller, and some were picked to make a task well posed. A reviewer is
entitled to ask whether the qualitative results are properties of the
mechanism or artifacts of those numbers.

This perturbs each constant over a range and re-runs the comparison that
the corresponding claim rests on. Scripted policies are used throughout,
not learned ones: the question is whether the physics supports the
conclusion, and a learned policy would mix training variance into the
answer.

Each claim is reduced to an ordering that either holds or does not, for
example "the exact projection reaches safe completion where the
axis-aligned bound does not". The report is the fraction of perturbations
under which the ordering survives.

Perturbing a constant changes the constraint hash, deliberately. Rows are
therefore not poolable with production results; this is a robustness
study of its own.

Usage:
    python3 experiments/sensitivity.py --tasks PCB CRANK --episodes 8
"""

import argparse
import importlib
import json
import math
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from benchmark.registry import make_env, get_filter, FILTERS
from benchmark.metrics import EpisodeRecorder
from benchmark import run_record as RR

MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)

# Constants worth perturbing, per task, with the claim each one could
# undermine. Names are attributes of the environment module.
SWEEPS = {
    "SCREW": [
        ("envs.screw_env_v3", "I_EFF", "reflected drivetrain inertia"),
        ("envs.screw_env_v3", "K_OMEGA", "servo bandwidth, rotational"),
        ("envs.screw_env_v3", "K_V", "servo bandwidth, axial"),
        ("envs.screw_env_v3", "F_THREAD_CAPACITY", "thread load capacity"),
        ("envs.screw_env_v3", "K_WEAR", "wear coefficient"),
        ("envs.screw_env_v3", "EPS_HELIX", "coupling tolerance"),
    ],
    "PCB": [
        ("envs.pcb_env_v2", "K_BEND", "board bending stiffness"),
        ("envs.pcb_env_v2", "THETA_YIELD", "yield curvature"),
        ("envs.pcb_env_v2", "TAU_TILT_MAX", "tilt torque limit"),
    ],
    "SNAP": [
        ("envs.snap_env_v2", "K_LATCH", "latch stiffness"),
        ("envs.snap_env_v2", "F_PULL_SAFE", "premature-pull limit"),
        ("envs.snap_env_v2", "DELTA_MAX", "yield travel"),
    ],
    "CRANK": [
        ("envs.crank_env_v2", "F_AXIAL_SAFE", "phase-gated axial limit"),
        ("envs.crank_env_v2", "TORQUE_LIMIT", "torque limit"),
        ("envs.crank_env_v2", "RADIAL_LIMIT", "radial limit"),
    ],
    "BATTERY": [
        ("envs.battery_env_v2", "F_DEFORM", "deformation onset"),
        ("envs.battery_env_v2", "K_DEFORM", "deformation rate"),
        ("envs.battery_env_v2", "D_SHORT", "short initiation threshold"),
        ("envs.battery_env_v2", "ADHESIVE", "adhesive strength"),
    ],
    "PRY": [
        ("envs.pry_env_v2", "BOND_STRENGTH", "bond release force"),
        ("envs.pry_env_v2", "K_CRACK", "crack accumulation rate"),
    ],
}

# The scripted proposal each task is judged under, chosen so an unfiltered
# run is destructive: a benign policy would make every arm look identical.
def policy(task):
    if task == "SCREW":
        return lambda obs: np.array([0.8, 0.15, 0.0])
    if task == "PCB":
        return lambda obs: np.array([0.5, 0.5, 0.5])
    if task == "SNAP":
        return lambda obs: np.array([0.0, 0.5, 0.0])
    if task == "CRANK":
        from baselines.phase_filters import crank_adversarial
        return lambda obs: crank_adversarial(np.asarray(obs, dtype=np.float64))
    if task == "BATTERY":
        return lambda obs: np.array([0.6, 0.2, 0.2])
    if task == "PRY":
        return lambda obs: np.array([0.8, 0.75, 0.0])
    raise KeyError(task)


# The ordering each task's claim reduces to: (arm that should do better,
# arm it should beat, metric, description).
CLAIMS = {
    "SCREW": ("trix", "none", "destructive_completion_rate",
              "projection avoids the damage the unfiltered policy causes"),
    "PCB": ("trix", "box_clip", "destructive_completion_rate",
            "exact projection avoids damage that componentwise clipping does not"),
    "SNAP": ("trix", "none", "destructive_completion_rate",
             "phase-aware filtering prevents destructive premature pull"),
    "CRANK": ("trix", "none", "destructive_completion_rate",
              "frame projection avoids the damage the unfiltered policy causes"),
    "BATTERY": ("trix_preventive", "trix", "destructive_completion_rate",
                "the preventive bound avoids damage the flagged limit allows"),
    "PRY": ("trix", "none", "destructive_completion_rate",
            "depth-aware projection avoids damage the unfiltered policy causes"),
}


def run_arm(task, arm, act, episodes, seed0=700):
    if arm not in FILTERS[task]:
        return None
    filt = get_filter(task, arm)
    rec = EpisodeRecorder(task, f"scripted+{arm}", seed=0)
    cap = 6000
    for ep in range(episodes):
        env = make_env(task)
        obs = env.reset(seed=seed0 + ep)
        rec.start_episode()
        done = False
        for _ in range(cap):
            a = np.asarray(act(obs), dtype=np.float64)
            x, _ = filt(np.asarray(obs, dtype=np.float64), a)
            obs, r, done, info = env.step(x)
            rec.step(info, reward=r)
            if done:
                break
        rec.end_episode(env, completed=done)
    return rec.validate()


def evaluate_claim(task, episodes):
    better, worse, metric, _ = CLAIMS[task]
    act = policy(task)
    s_b = run_arm(task, better, act, episodes)
    s_w = run_arm(task, worse, act, episodes)
    if s_b is None or s_w is None:
        return None
    # lower is better for damage metrics
    holds = s_b[metric] < s_w[metric] or (
        s_b[metric] == 0.0 and s_w[metric] == 0.0
        and s_b["safe_completion_rate"] >= s_w["safe_completion_rate"])
    return {"holds": bool(holds),
            f"{better}_{metric}": s_b[metric],
            f"{worse}_{metric}": s_w[metric],
            f"{better}_safe": s_b["safe_completion_rate"],
            f"{worse}_safe": s_w["safe_completion_rate"]}


def sweep_task(task, episodes, multipliers):
    rows = []
    for mod_name, const, label in SWEEPS.get(task, []):
        mod = importlib.import_module(mod_name)
        base = getattr(mod, const)
        for m in multipliers:
            setattr(mod, const, base * m)
            # reimport filter modules that cached the constant at import
            for dep in ("baselines.pcb_filters", "baselines.phase_filters",
                        "baselines.task_filters", "baselines.screw_filters"):
                try:
                    importlib.reload(importlib.import_module(dep))
                except Exception:
                    pass
            try:
                res = evaluate_claim(task, episodes)
            except Exception as e:
                res = {"holds": None, "error": f"{type(e).__name__}: {e}"}
            setattr(mod, const, base)
            rows.append({"constant": const, "label": label,
                         "multiplier": m, "baseline": float(base),
                         "value": float(base * m), **(res or {})})
            r = rows[-1]
            mark = ("HOLDS" if r.get("holds") else
                    "FAILS" if r.get("holds") is False else "error")
            print(f"  {const:22s} x{m:<5.2f} {mark}", flush=True)
    for dep in ("baselines.pcb_filters", "baselines.phase_filters",
                "baselines.task_filters", "baselines.screw_filters"):
        try:
            importlib.reload(importlib.import_module(dep))
        except Exception:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(SWEEPS))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=list(MULTIPLIERS))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_rows = {}
    for task in args.tasks:
        better, worse, _, desc = CLAIMS[task]
        print(f"\n{task}: {desc}")
        print(f"  comparing {better} against {worse}, "
              f"{args.episodes} episodes per cell")
        all_rows[task] = sweep_task(task, args.episodes, args.multipliers)

    print(f"\n{'task':10s} {'perturbations':>14} {'claim holds':>13} "
          f"{'fraction':>10}")
    summary = {}
    for task, rows in all_rows.items():
        ok = sum(1 for r in rows if r.get("holds") is True)
        n = sum(1 for r in rows if r.get("holds") is not None)
        summary[task] = {"held": ok, "evaluated": n,
                         "fraction": ok / n if n else None}
        print(f"{task:10s} {n:14d} {ok:13d} "
              f"{(100 * ok / n if n else 0):9.0f}%")

    path = os.path.join(args.out, "sensitivity.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "rows": all_rows, "provenance": RR.provenance()},
                  f, indent=1)
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
