#!/usr/bin/env python3
"""Post-hoc command-membership and damage-path audit of frozen SNAP OOD cells.

Reuses the production evaluator and exact recorded episodes. Observers call
each original filter, reset, outer step and substep exactly once. They consume
no random numbers and write no plant, policy, governor or noise state.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import ood_reset as OOD
from benchmark import margin_policy as MP
from benchmark.margin import sigma_for
from envs import snap_env_v2 as S

TOLERANCE = 1e-7  # normalized command units; exact positive residuals also saved
CONDITIONS = ("wide_1p5", "wide_2", "shell_2")
METRICS = ("safe_completion", "unsafe_completion_intact", "destructive_completion",
           "mechanical_failure", "timeout")
COLUMNS = ([f"obs_{i}" for i in range(10)] + [f"nominal_{i}" for i in range(3)] +
           [f"command_{i}" for i in range(3)] + [f"lower_{i}" for i in range(3)] +
           [f"upper_{i}" for i in range(3)] + [f"next_obs_{i}" for i in range(10)] +
           ["realized_deflect_N", "realized_pull_N", "premature_substeps",
            "phase_reengagement_substeps", "vio_premature", "vio_overstress",
            "vio_lateral", "latch_broken", "latch_damage", "k_latch_N_per_m"])


def declared_bounds(obs, arm="trix", registry=False):
    """Independent inequality evaluation; never calls either projection function.

    This is the frozen executable set documented in the provenance audit.
    The later per-family registry is separately checked, never substituted.
    """
    if arm == "static_clip":
        upper = np.array([1.0, S.F_CATCH / S.S_PULL, S.F_LAT_MAX / S.S_LAT])
        return -upper, upper
    z = MP.POLICY["SNAP"]["premature_pull"]["sigma"] if registry else sigma_for()
    margin = z * S.BASE_FORCE_NOISE
    limit = S.F_CATCH if obs[7] > .5 else S.F_PULL_SAFE
    deflect = min(1.0, S.K_LATCH * S.DELTA_MAX * .85 / S.S_DEFLECT)
    lateral = max(0.0, S.F_LAT_MAX - margin) / S.S_LAT
    return (np.array([-deflect, -1.0, -lateral]),
            np.array([deflect, max(0.0, limit - margin) / S.S_PULL, lateral]))


def residual(a, bounds):
    lo, hi = bounds
    return float(max(0.0, np.max(lo - a), np.max(a - hi)))


def damage_state(env):
    return {"delta": float(env.delta), "delta_dot": float(env.delta_dot),
            "z": float(env.z), "released": bool(env.released),
            "latch_damage": float(env.latch_damage), "latch_broken": bool(env.latch_broken)}


class Observer:
    def __init__(self, arm):
        self.arm = arm
        self.rows = []
        self.offsets = [0]
        self.episodes = []
        self.cur = None
        self.pending = None

    def start(self, reset):
        self.cur = {"evaluation_reset": reset, "steps": 0,
                    "own_set_outside_strict": 0, "own_set_outside_tolerance": 0,
                    "trix_set_outside_strict": 0, "trix_set_outside_tolerance": 0,
                    "registry_set_outside_tolerance": 0,
                    "own_set_max_residual": 0.0, "trix_set_max_residual": 0.0,
                    "registry_set_max_residual": 0.0,
                    "premature_substeps": 0, "phase_reengagement_substeps": 0,
                    "command_equilibrium_exceeds_limit_steps_before_break": 0,
                    "first_pull_damage": None, "first_break": None,
                    "violations_by_family": {"vio_premature": 0, "vio_overstress": 0, "vio_lateral": 0}}

    def filter(self, original, obs, nominal):
        out, info = original(obs, nominal)
        action = np.asarray(out, dtype=np.float64)
        own = declared_bounds(obs, self.arm)
        trix = declared_bounds(obs)
        registry = declared_bounds(obs, registry=True)
        assert np.all(np.isfinite(action))
        for name, bounds in (("own_set", own), ("trix_set", trix), ("registry_set", registry)):
            excess = residual(action, bounds)
            self.cur[name + "_max_residual"] = max(self.cur[name + "_max_residual"], excess)
            self.cur[name + "_outside_tolerance"] += int(excess > TOLERANCE)
            if name != "registry_set":
                self.cur[name + "_outside_strict"] += int(excess > 0.0)
        self.pending = {"obs": obs.copy(), "nominal": nominal.copy(),
                        "action": action.copy(), "lo": own[0], "hi": own[1],
                        "own_set_residual": residual(action, own)}
        return out, info

    def attach(self, env):
        assert env._filter is None
        plant = env.unwrapped_task
        original_reset, original_step, original_substep = env.reset, env.step, plant._substep

        def reset(*args, **kwargs):
            obs, info = original_reset(*args, **kwargs)
            self.start(info)
            return obs, info

        def event(before, after, f_deflect, f_pull, premature):
            q = self.pending
            return {"step": self.cur["steps"] + 1, "substep": self.substep_index,
                    "before": before, "after": after,
                    "realized_deflect_N": float(f_deflect), "realized_pull_N": float(f_pull),
                    "k_latch_N_per_m": float(plant.k_latch),
                    "sampled_phase_disengaged": bool(q["obs"][7] > .5),
                    "premature": bool(premature),
                    "nominal": q["nominal"].tolist(), "command": q["action"].tolist(),
                    "declared_lower": q["lo"].tolist(), "declared_upper": q["hi"].tolist(),
                    "own_set_residual": q["own_set_residual"]}

        def substep(f_deflect, f_pull, h):
            self.substep_index += 1
            before_broken, before_damage = plant.latch_broken, plant.latch_damage
            # Detailed state is needed only until the first damage events.
            before = damage_state(plant) if (self.cur["first_break"] is None or
                                             self.cur["first_pull_damage"] is None) else None
            result = original_substep(f_deflect, f_pull, h)
            self.realized = (float(f_deflect), float(f_pull))
            self.step_premature += int(result)
            self.step_reengagement += int(result and self.pending["obs"][7] > .5 and
                                          not bool(self.pending["obs"][4]))
            if plant.latch_damage > before_damage and self.cur["first_pull_damage"] is None:
                self.cur["first_pull_damage"] = event(before, damage_state(plant), f_deflect, f_pull, result)
            if not before_broken and plant.latch_broken and self.cur["first_break"] is None:
                ev = event(before, damage_state(plant), f_deflect, f_pull, result)
                ev["deflection_exceeded"] = bool(abs(plant.delta) > S.DELTA_MAX)
                ev["pull_damage_reached_break"] = bool(plant.latch_damage >= S.DAMAGE_BREAK)
                self.cur["first_break"] = ev
            return result

        def step(action):
            assert self.pending is not None
            q = self.pending
            assert np.array_equal(np.asarray(action, dtype=np.float64), q["action"])
            self.substep_index = self.step_premature = self.step_reengagement = 0
            if not plant.latch_broken and abs(q["action"][0] * S.S_DEFLECT / plant.k_latch) > S.DELTA_MAX:
                self.cur["command_equilibrium_exceeds_limit_steps_before_break"] += 1
            result = original_step(action)
            obs, reward, terminated, truncated, info = result
            assert self.substep_index == S.N_SUBSTEPS
            self.cur["steps"] += 1
            self.cur["premature_substeps"] += self.step_premature
            self.cur["phase_reengagement_substeps"] += self.step_reengagement
            for k in self.cur["violations_by_family"]:
                self.cur["violations_by_family"][k] += int(info[k])
            row = np.concatenate((q["obs"], q["nominal"], q["action"], q["lo"], q["hi"],
                                  obs, self.realized,
                                  [self.step_premature, self.step_reengagement, info["vio_premature"],
                                   info["vio_overstress"], info["vio_lateral"], float(plant.latch_broken),
                                   plant.latch_damage, plant.k_latch]))
            assert len(row) == len(COLUMNS)
            self.rows.append(row)
            if terminated or truncated:
                self.cur.update(plant.episode_summary(bool(terminated)))
                self.cur["terminal_state"] = damage_state(plant)
                self.episodes.append(self.cur)
                self.offsets.append(len(self.rows))
            self.pending = None
            return result

        env.reset, env.step, plant._substep = reset, step, substep
        return env


def init_worker():
    import torch
    torch.set_num_threads(1)


def job(args):
    archive, raw, out, cell, condition = args
    from training import sb3_runner as R
    from benchmark.metrics import EpisodeRecorder
    started = time.monotonic()
    name = OOD.cell_key(cell).replace("/", "__") + "__" + condition
    target = out / "cells" / (name + ".json")
    try:
        p = OOD.load_protocol()
        assert OOD.software() == cell["software"]
        assert OOD.current_hashes("SNAP") == cell["hashes"]
        checkpoint = archive / "research" / cell["checkpoint"]
        assert OOD.sha(checkpoint) == cell["checkpoint_sha256"]
        original = json.loads((raw / "cells" / (name + ".json")).read_text())
        assert original["status"] == "complete"
        monitor = Observer(cell["arm"])
        original_filter, original_make = R.get_filter, R.make

        def get_filter(task, arm):
            filt = original_filter(task, arm)
            return lambda obs, nominal: monitor.filter(filt, obs, nominal)

        R.get_filter = get_filter
        R.make = lambda *a, **k: monitor.attach(original_make(*a, **k))
        try:
            model = R.load(str(checkpoint), "SNAP")
            recorder = EpisodeRecorder("SNAP", "sac+" + cell["arm"], cell["seed"])
            R.evaluate_frozen(model, "SNAP", cell["arm"], p["episode_seeds"]["SNAP"], recorder,
                              evaluation_distribution=p["conditions"]["SNAP"][condition])
        finally:
            R.get_filter, R.make = original_filter, original_make
        counts = recorder.validate()["counts"]
        assert counts == original["record"]["counts"], (counts, original["record"]["counts"])
        assert len(monitor.episodes) == len(recorder.episodes) == len(original["episodes"]) == 100
        for observer, rec, previous in zip(monitor.episodes, recorder.episodes, original["episodes"]):
            replay = {k: rec[k] for k in previous}
            assert replay == previous, (name, previous["evaluation_reset"]["episode_seed"])
            assert observer["category"] == rec["category"]
            assert observer["steps"] == rec["steps"]
            assert observer["violations_by_family"] == {
                k: rec["violations_by_family"].get(k, 0) for k in observer["violations_by_family"]}
            assert observer["damage_outcome"] == rec["damage_outcome"]
            assert observer["episode_with_violation"] == rec["episode_with_violation"]
        assert OOD.current_hashes("SNAP") == cell["hashes"]
        trace = out / "traces" / (name + ".npz")
        np.savez_compressed(trace, values=np.asarray(monitor.rows, dtype=np.float64),
                            episode_offsets=np.asarray(monitor.offsets, dtype=np.int64),
                            episode_seeds=np.asarray(p["episode_seeds"]["SNAP"], dtype=np.int64),
                            columns=np.asarray(COLUMNS))
        OOD.dump(target, {"status": "complete", "task": "SNAP", "stage": cell["stage"],
                          "algorithm": "sac", "arm": cell["arm"], "seed": cell["seed"],
                          "condition": condition, "counts": counts, "episodes": monitor.episodes,
                          "original_cell_sha256": OOD.sha(raw / "cells" / (name + ".json")),
                          "checkpoint_sha256": cell["checkpoint_sha256"], "hashes": cell["hashes"],
                          "trace": str(trace.relative_to(out)), "trace_sha256": OOD.sha(trace),
                          "exact_episode_replay_verified": True, "wall_s": time.monotonic() - started})
        return {"cell": name, "status": "complete", "steps": counts["total_steps"]}
    except Exception:
        error = {"cell": name, "status": "failed", "traceback": traceback.format_exc()}
        OOD.dump(target, error)
        return error


def run(args):
    p = OOD.load_protocol()
    assert not args.out.exists(), "Use a new output directory; preserve every earlier attempt."
    args.out.mkdir(parents=True)
    (args.out / "cells").mkdir()
    (args.out / "traces").mkdir()
    cells = [c for c in p["cells"] if c["task"] == "SNAP"]
    assert len(cells) == 20
    freeze = {"type": "post_hoc_attribution_of_existing_records", "episodes": 6000,
              "cells": 60, "scope": "all SNAP arms, all ten policy seeds, all three OOD conditions",
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "diagnostic_source_sha256": OOD.sha(__file__),
              "diagnostic_plan_sha256": OOD.sha(ROOT / "docs/snap_ood_attribution_plan.md"),
              "ood_protocol_sha256": OOD.sha(OOD.PROTOCOL_PATH), "frozen_source_sha256": p["source_sha256"],
              "numerical_tolerance": TOLERANCE, "strict_residuals_also_reported": True,
              "executed_sigma": sigma_for(), "registry_sigma": MP.POLICY["SNAP"]["premature_pull"]["sigma"],
              "columns": COLUMNS, "training": False, "reselection": False}
    OOD.dump(args.out / "freeze.json", freeze)
    jobs = [(args.archive, args.raw, args.out, c, condition) for c in cells for condition in CONDITIONS]
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for i, f in enumerate(as_completed([pool.submit(job, x) for x in jobs]), 1):
            result = f.result()
            if result["status"] != "complete":
                errors.append(result)
            print(json.dumps({"finished": i, "total": len(jobs), "execution_errors": len(errors), **result}), flush=True)
    if errors:
        raise RuntimeError(f"Attribution incomplete: {len(errors)} execution errors retained")
    report(args.out)


def summarize(eps):
    result = {"episodes": len(eps), "steps": sum(e["steps"] for e in eps),
              "categories": dict(Counter(e["category"] for e in eps)),
              "damage_states_all_episodes": dict(Counter(e["damage_outcome"] for e in eps)),
              "damage_states_destructive_completions": dict(Counter(
                  e["damage_outcome"] for e in eps if e["category"] == "destructive_completion")),
              "episodes_with_first_pull_damage": sum(e["first_pull_damage"] is not None for e in eps),
              "episodes_with_phase_reengagement": sum(e["phase_reengagement_substeps"] > 0 for e in eps),
              "first_break_routes": dict(Counter(
                  "both" if e["first_break"]["deflection_exceeded"] and e["first_break"]["pull_damage_reached_break"]
                  else "overdeflection" if e["first_break"]["deflection_exceeded"] else "accumulated_pull"
                  for e in eps if e["first_break"] is not None)),
              "violations_by_family_episodes": {k: sum(e["violations_by_family"][k] > 0 for e in eps)
                                                for k in ("vio_premature", "vio_overstress", "vio_lateral")}}
    for k in ("own_set_outside_strict", "own_set_outside_tolerance", "trix_set_outside_strict",
              "trix_set_outside_tolerance", "registry_set_outside_tolerance", "premature_substeps",
              "phase_reengagement_substeps", "command_equilibrium_exceeds_limit_steps_before_break"):
        result[k] = sum(e[k] for e in eps)
    for k in ("own_set_max_residual", "trix_set_max_residual", "registry_set_max_residual"):
        result[k] = max((e[k] for e in eps), default=0.0)
    return result


def report(out):
    files = sorted((out / "cells").glob("*.json"))
    assert len(files) == 60
    data = [json.loads(p.read_text()) for p in files]
    assert all(x["status"] == "complete" and x["exact_episode_replay_verified"] for x in data)
    groups = []
    for arm in ("static_clip", "trix"):
        for condition in CONDITIONS:
            cells = sorted([x for x in data if x["arm"] == arm and x["condition"] == condition], key=lambda x:x["seed"])
            assert [x["seed"] for x in cells] == list(range(10))
            episodes = [e for x in cells for e in x["episodes"]]
            parts = {"all": episodes,
                     "within": [e for e in episodes if not e["evaluation_reset"]["outside_training_support"]],
                     "outside": [e for e in episodes if e["evaluation_reset"]["outside_training_support"]],
                     "lower_outside": [e for e in episodes if e["evaluation_reset"]["parameters"]["k_latch"] < 4500],
                     "upper_outside": [e for e in episodes if e["evaluation_reset"]["parameters"]["k_latch"] > 5500]}
            groups.append({"arm": arm, "condition": condition,
                           "strata": {k: summarize(v) for k,v in parts.items()},
                           "by_seed": [{"seed": c["seed"], **summarize(c["episodes"])} for c in cells]})
    OOD.dump(out / "summary.json", {"status": "complete", "replayed_cells": 60,
                                    "replayed_episodes": 6000, "all_episode_records_exactly_matched": True,
                                    "groups": groups, "freeze_sha256": OOD.sha(out / "freeze.json")})
    print(json.dumps({"status": "complete", "summary": str(out / "summary.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.command == "run":
        if args.archive is None or args.raw is None:
            parser.error("run requires --archive and --raw")
        run(args)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
