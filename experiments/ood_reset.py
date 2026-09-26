#!/usr/bin/env python3
"""Prospective R2.4 reset-distribution evaluation of selected, frozen policies.

prepare records only existing evidence and hashes. run requires the amendment
and implementation to be committed. No training or checkpoint selection exists
in this entrypoint. The historical archive is read-only throughout.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import evaluation_distribution as ED
from benchmark import margin_policy as MP
from benchmark import taxonomy as TAX
from benchmark import run_record as RR
from benchmark.gym_adapter import CAPS
from benchmark.registry import ENVS
from benchmark.selection import TASK_SEED_OFFSET
from scripts.build_revision_seed_statistics import exact_bootstrap

PROTOCOL_PATH = ROOT / "docs/ood_reset_protocol.json"
ROWS = [
    ("stage1", "SCREW", "sac", "qp_matched", "trix"),
    ("stage1", "SCREW", "ppo", "qp_matched", "trix"),
    ("stage1", "PCB", "sac", "box_clip", "trix"),
    ("stage2", "PCB", "sac", "box_clip", "trix"),
    ("stage2", "SNAP", "sac", "static_clip", "trix"),
    ("stage2", "CRANK", "sac", "static_clip", "trix"),
    ("stage2", "BATTERY", "sac", "box_clip_preventive", "trix_preventive"),
]
HASH_FIELDS = ("constraint_hash", "robust_margin_hash", "taxonomy_hash")
METRICS = ("safe_completions", "destructive_completions", "mechanical_failures",
           "unsafe_completions_intact", "timeouts", "episodes_with_violation")
CATEGORY_COUNTS = {"safe_completion": "safe_completions",
                   "unsafe_completion_intact": "unsafe_completions_intact",
                   "destructive_completion": "destructive_completions",
                   "mechanical_failure": "mechanical_failures", "timeout": "timeouts"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def current_hashes(task):
    mod = importlib.import_module(ENVS[task].__module__)
    return {"constraint_hash": mod._constraint_hash(),
            "robust_margin_hash": MP.policy_hash(task),
            "taxonomy_hash": TAX.taxonomy_hash(task)}


def source_hashes():
    paths = []
    for directory in ("benchmark", "envs", "baselines", "training"):
        paths.extend(ROOT.joinpath(directory).glob("*.py"))
    paths.extend([Path(__file__), ROOT / "scripts/build_revision_seed_statistics.py"])
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def software():
    import numpy, torch, stable_baselines3, gymnasium
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "torch": torch.__version__, "sb3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__}


def episode_seeds(task):
    return [1_200_000 + 1000 * TASK_SEED_OFFSET[task] + i for i in range(100)]


def cell_key(c):
    return f'{c["stage"]}/{c["task"]}/{c["algorithm"]}/{c["arm"]}/seed{c["seed"]}'


def prepare(archive):
    """Read the historical selection/audit; do not load or evaluate policies."""
    linkage_path = archive / "verification/policy_record_linkage.json"
    audit = json.loads(linkage_path.read_text())
    seed_path = ROOT / "manuscript/data/seed_statistics_input.json"
    seeds = json.loads(seed_path.read_text())["records"]
    research = archive / "research"
    cells = []
    for stage, task, algo, first, second in ROWS:
        for arm in (first, second):
            for seed in range(10):
                matches = [r for r in audit["records"]
                           if r["aggregate"] == f"results/{stage}/{stage}_records.json"
                           and (r["task"], r["algorithm"], r["arm"], r["seed"])
                           == (task, algo, arm, seed)]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous historical selection: {stage}/{task}/{algo}/{arm}/{seed}")
                r = matches[0]
                old = seeds[f"{stage}/{task}/{algo}/{arm}/{seed}"]
                assert r["selected_step"] == old["selected_step"]
                checkpoint = research / r["checkpoint"]
                meta_path = checkpoint.with_name(checkpoint.name.rsplit("_", 1)[0] + "_meta.json")
                meta = json.loads(meta_path.read_text())
                fingerprint = meta["fingerprint"]
                expected_software = {k: fingerprint[k] for k in software()}
                assert software() == expected_software, (software(), expected_software)
                hashes = current_hashes(task)
                assert hashes == {k: old["hashes"][k] for k in HASH_FIELDS}
                assert not set(episode_seeds(task)) & set(old["episode_seeds"])
                c = {"stage": stage, "task": task, "algorithm": algo, "arm": arm,
                     "seed": seed, "selected_step": r["selected_step"],
                     "checkpoint": r["checkpoint"], "checkpoint_sha256": sha(checkpoint),
                     "metadata": str(meta_path.relative_to(research)),
                     "metadata_sha256": sha(meta_path), "hashes": hashes,
                     "software": expected_software,
                     "baseline_counts": old["counts"],
                     "baseline_episode_seeds": old["episode_seeds"],
                     "baseline_record_sha256": old["aggregate_record_sha256"],
                     "historical_protocol_hash": old["hashes"]["protocol_freeze_hash"],
                     "training_mode": "nominal" if stage == "stage1" else "filter_aware",
                     "episode_cap": CAPS[task]}
                cells.append(c)
    protocol = {
        "schema": 1, "name": "R2.4 prospective reset-distribution evaluation",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "New evaluation added during revision to address the R2.4 evaluation-use column; prior in-range results remain unchanged.",
        "protocol_document_sha256": sha(ROOT / "docs/ood_reset_protocol.md"),
        "archive_linkage_sha256": sha(linkage_path),
        "baseline_statistics_input_sha256": sha(seed_path),
        "source_sha256": source_hashes(),
        "rows": [dict(zip(("stage", "task", "algorithm", "first_arm", "second_arm"), r)) for r in ROWS],
        "excluded": {"PRY": "Not applicable: native reset is deterministic; no new parameter randomisation is introduced."},
        "conditions": {t: {name: ED.specification(t, name) for name in ED.CONDITIONS}
                       for t in ED.PARAMETERS},
        "episode_seeds": {t: episode_seeds(t) for t in ED.PARAMETERS},
        "cells": cells, "new_evaluation_cells": len(cells) * len(ED.CONDITIONS),
        "new_episodes": len(cells) * len(ED.CONDITIONS) * 100,
        "analysis": {"primary_condition": "shell_2", "unit": "training seed",
                     "primary_endpoint": "safe completion over all 100 episodes",
                     "interval": "exact percentile bootstrap by convolution, existing Section 4.6 implementation",
                     "pairing": "training seed identifier; new arms also share episode seeds and parameter draws",
                     "baseline_pairing": "training-seed pairing only; historical episode seeds are different",
                     "strata": "within-support and outside-support counts and descriptive seed rates; no stratified confidence intervals",
                     "multiplicity": "descriptive intervals, no multiplicity-adjusted claims",
                     "report_all": True, "skip_zero_baselines": False,
                     "missing_cells": "report incomplete; never replace a missing seed or silently retry a failed cell"},
        "training": False, "reselection": False,
    }
    dump(PROTOCOL_PATH, protocol)
    print(json.dumps({"protocol": str(PROTOCOL_PATH), "sha256": sha(PROTOCOL_PATH),
                      "selected_checkpoints": len({c["checkpoint"] for c in cells}),
                      "cells": protocol["new_evaluation_cells"],
                      "episodes": protocol["new_episodes"]}, indent=2))


def load_protocol():
    p = json.loads(PROTOCOL_PATH.read_text())
    assert p["source_sha256"] == source_hashes(), "Implementation changed after protocol declaration"
    assert p["protocol_document_sha256"] == sha(ROOT / "docs/ood_reset_protocol.md")
    assert p["baseline_statistics_input_sha256"] == sha(ROOT / "manuscript/data/seed_statistics_input.json")
    for task, specs in p["conditions"].items():
        for spec in specs.values():
            ED.validate(spec, task)
    return p


def init_worker():
    import torch
    torch.set_num_threads(1)


def result_path(out, cell, condition):
    return out / "cells" / (cell_key(cell).replace("/", "__") + f"__{condition}.json")


def evaluate_job(job):
    archive, out, cell, spec, seeds, protocol_hash, commit = job
    path = result_path(out, cell, spec["condition"])
    started = time.monotonic()
    try:
        from training import sb3_runner as R
        from benchmark.metrics import EpisodeRecorder
        checkpoint = archive / "research" / cell["checkpoint"]
        assert sha(checkpoint) == cell["checkpoint_sha256"]
        assert current_hashes(cell["task"]) == cell["hashes"]
        model = R.load(str(checkpoint), cell["task"])
        rec = EpisodeRecorder(cell["task"], f'{cell["algorithm"]}+{cell["arm"]}', cell["seed"])
        R.evaluate_frozen(model, cell["task"], cell["arm"], seeds, rec,
                          evaluation_distribution=spec)
        summary = rec.validate()
        assert summary["counts"]["episodes"] == len(seeds) == 100
        assert current_hashes(cell["task"]) == cell["hashes"]
        record = RR.make_record(
            cell["task"], rec.method, cell["seed"], summary, cell["hashes"]["constraint_hash"],
            training_mode=cell["training_mode"], evaluation_arm=cell["arm"],
            algorithm=cell["algorithm"], evaluation_distribution=spec,
            extra={"ood_protocol_sha256": protocol_hash, "ood_protocol_commit": commit,
                   "checkpoint_sha256": cell["checkpoint_sha256"],
                   "selected_step": cell["selected_step"], "stage": cell["stage"]})
        assert {k: record[k] for k in HASH_FIELDS} == cell["hashes"]
        episodes = [{"category": e["category"], "steps": e["steps"],
                     "episode_with_violation": bool(e["episode_with_violation"]),
                     "evaluation_reset": e["evaluation_reset"]} for e in rec.episodes]
        dump(path, {"status": "complete", "cell": cell_key(cell), "record": record,
                    "episodes": episodes, "wall_s": time.monotonic() - started})
        return {"cell": cell_key(cell), "condition": spec["condition"], "status": "complete"}
    except Exception:
        error = {"status": "failed", "cell": cell_key(cell),
                 "condition": spec["condition"], "ood_protocol_sha256": protocol_hash,
                 "traceback": traceback.format_exc(), "wall_s": time.monotonic() - started}
        dump(path, error)
        return error


def run(archive, out, workers):
    p = load_protocol()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("Commit the amendment and implementation before running OOD evaluation")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    digest = sha(PROTOCOL_PATH)
    for c in p["cells"]:
        assert software() == c["software"]
        assert current_hashes(c["task"]) == c["hashes"]
        assert sha(archive / "research" / c["checkpoint"]) == c["checkpoint_sha256"]
        assert sha(archive / "research" / c["metadata"]) == c["metadata_sha256"]
    assert sha(archive / "verification/policy_record_linkage.json") == p["archive_linkage_sha256"]
    freeze = {"protocol_sha256": digest, "protocol_commit": commit,
              "source_sha256": p["source_sha256"], "software": software(),
              "cells": p["new_evaluation_cells"], "episodes": p["new_episodes"]}
    freeze_path = out / "freeze.json"
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == freeze, "Cannot resume with a different freeze"
    else:
        dump(freeze_path, freeze)
    jobs = []
    for cell in p["cells"]:
        for condition, spec in p["conditions"][cell["task"]].items():
            path = result_path(out, cell, condition)
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["status"] != "complete":
                    raise RuntimeError(f"Recorded failure needs explicit resolution: {path}")
                assert previous["record"]["ood_protocol_sha256"] == digest
                continue
            jobs.append((archive, out, cell, spec, p["episode_seeds"][cell["task"]], digest, commit))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    print(json.dumps({"protocol_sha256": digest, "commit": commit,
                      "pending_cells": len(jobs), "workers": workers}), flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        pending = [pool.submit(evaluate_job, j) for j in jobs]
        for index, future in enumerate(as_completed(pending), 1):
            result = future.result()
            failures += result["status"] != "complete"
            if index % 10 == 0 or result["status"] != "complete" or index == len(jobs):
                print(json.dumps({"finished": index, "total": len(jobs),
                                  "failures": failures, "last": result}), flush=True)
    report(out)


def verify_native(archive, out):
    """Replay five retained in-range episodes per seed-0 headline arm."""
    import torch
    from training import sb3_runner as R
    from benchmark.metrics import EpisodeRecorder
    torch.set_num_threads(1)
    p = load_protocol()
    old = json.loads((ROOT / "manuscript/data/seed_statistics_input.json").read_text())["records"]
    checks = []
    for c in p["cells"]:
        if c["seed"] != 0:
            continue
        key = f'{c["stage"]}/{c["task"]}/{c["algorithm"]}/{c["arm"]}/0'
        saved = old[key]
        selected = None
        for path, digest in sorted(saved["shard_sha256"].items()):
            source = archive / "research" / path
            assert sha(source) == digest
            shard = json.loads(source.read_text())
            if len(shard["episode_seeds"]) == 5:
                selected = (path, shard); break
        assert selected is not None
        path, shard = selected
        model_path = archive / "research" / c["checkpoint"]
        assert sha(model_path) == c["checkpoint_sha256"]
        model = R.load(str(model_path), c["task"])
        rec = EpisodeRecorder(c["task"], f'{c["algorithm"]}+{c["arm"]}', 0)
        R.evaluate_frozen(model, c["task"], c["arm"], shard["episode_seeds"], rec,
                          evaluation_distribution=ED.specification(c["task"]))
        actual = rec.validate()["counts"]
        assert actual == shard["summary"]["counts"], (key, actual, shard["summary"]["counts"])
        # Exercise the production serializer too, without creating an OOD record.
        assert all("steps" in e and "evaluation_reset" in e for e in rec.episodes)
        checks.append({"cell": cell_key(c), "shard": path,
                       "episode_seeds": shard["episode_seeds"], "counts": actual,
                       "checkpoint_sha256": c["checkpoint_sha256"], "matches": True})
        print(json.dumps({"native_check": key, "matches": True}), flush=True)
    dump(out / "native_verification.json", {"protocol_sha256": sha(PROTOCOL_PATH),
         "checks": checks, "historical_episodes_replayed": sum(x["counts"]["episodes"] for x in checks),
         "ood_policy_rollouts": 0})


def endpoint(episodes, name):
    if name == "episodes_with_violation":
        return sum(e["episode_with_violation"] for e in episodes)
    category = next(k for k, v in CATEGORY_COUNTS.items() if v == name)
    return sum(e["category"] == category for e in episodes)


def report(out):
    p = load_protocol()
    digest = sha(PROTOCOL_PATH)
    groups = {}
    missing = []
    for c in p["cells"]:
        for condition in ED.CONDITIONS:
            path = result_path(out, c, condition)
            if not path.exists() or json.loads(path.read_text())["status"] != "complete":
                missing.append(str(path)); continue
            raw = json.loads(path.read_text())
            r, eps = raw["record"], raw["episodes"]
            assert r["ood_protocol_sha256"] == digest
            assert r["evaluation_distribution"] == p["conditions"][c["task"]][condition]
            assert (r["task"], r["algorithm"], r["evaluation_arm"], r["seed"], r["stage"]) == (
                c["task"], c["algorithm"], c["arm"], c["seed"], c["stage"])
            assert r["checkpoint_sha256"] == c["checkpoint_sha256"]
            assert {h: r[h] for h in HASH_FIELDS} == c["hashes"]
            assert [e["evaluation_reset"]["episode_seed"] for e in eps] == p["episode_seeds"][c["task"]]
            assert len(eps) == 100
            for e in eps:
                reset = e["evaluation_reset"]
                expected, attempts = ED.sample(r["evaluation_distribution"], reset["episode_seed"])
                assert reset["parameters"] == expected and reset["sampling_attempts"] == attempts
                assert reset["evaluation_distribution_hash"] == r["evaluation_distribution_hash"]
                assert reset["outside_training_support"] == ED.outside_training_support(r["evaluation_distribution"], expected)
            for metric in METRICS:
                assert endpoint(eps, metric) == r["counts"][metric]
            group = (c["stage"], c["task"], c["algorithm"], c["arm"], condition)
            groups.setdefault(group, []).append((c, raw))
    if missing:
        dump(out / "analysis.json", {"status": "incomplete", "missing_or_failed": missing,
                                     "protocol_sha256": digest})
        raise RuntimeError(f"Study incomplete: {len(missing)} missing/failed cells; no complete-study aggregate")
    arms = []
    for group, pairs in sorted(groups.items()):
        pairs.sort(key=lambda x: x[0]["seed"])
        assert [c["seed"] for c, _ in pairs] == list(range(10))
        RR.check_comparable([raw["record"] for _, raw in pairs])
        by_metric = {}
        for metric in METRICS:
            values = [raw["record"]["counts"][metric] for _, raw in pairs]
            base = [c["baseline_counts"][metric] for c, _ in pairs]
            delta = [a - b for a, b in zip(values, base)]
            by_metric[metric] = {"counts_by_seed": values, "denominator_per_seed": 100,
                "mean_percent": sum(values)/10, "ci95_percent": exact_bootstrap(values),
                "baseline_mean_percent": sum(base)/10, "delta_pp_by_seed": delta,
                "mean_delta_pp": sum(delta)/10, "ci95_delta_pp": exact_bootstrap(delta)}
        strata = {}
        for outside in (False, True):
            by_seed = []
            for c, raw in pairs:
                eps = [e for e in raw["episodes"] if e["evaluation_reset"]["outside_training_support"] == outside]
                by_seed.append({"seed": c["seed"], "episodes": len(eps),
                                "counts": {m: endpoint(eps, m) for m in METRICS},
                                "safe_completion_percent": 100*endpoint(eps, "safe_completions")/len(eps) if eps else None})
            strata["outside" if outside else "within"] = by_seed
        arms.append({"stage": group[0], "task": group[1], "algorithm": group[2],
                     "arm": group[3], "condition": group[4], "metrics": by_metric,
                     "support_strata_descriptive": strata})
    comparisons = []
    for row in p["rows"]:
        for condition in ED.CONDITIONS:
            by_arm = {a["arm"]: a for a in arms if all(a[k] == row[k] for k in ("stage", "task", "algorithm")) and a["condition"] == condition}
            metrics = {}
            for metric in METRICS:
                first = by_arm[row["first_arm"]]["metrics"][metric]["counts_by_seed"]
                second = by_arm[row["second_arm"]]["metrics"][metric]["counts_by_seed"]
                differences = [b-a for a,b in zip(first,second)]
                metrics[metric] = {"second_minus_first_pp_by_seed": differences,
                                   "mean_pp": sum(differences)/10,
                                   "ci95_pp": exact_bootstrap(differences)}
            comparisons.append({**row, "condition": condition, "metrics": metrics})
    dump(out / "analysis.json", {"status": "complete", "protocol_sha256": digest,
                                 "cells": p["new_evaluation_cells"], "episodes": p["new_episodes"],
                                 "arms": arms, "comparisons": comparisons,
                                 "analysis": p["analysis"]})
    print(json.dumps({"analysis": str(out / "analysis.json"), "status": "complete"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "verify-native", "run", "report"))
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "build/ood_reset")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    if args.command in ("prepare", "verify-native", "run") and args.archive is None:
        ap.error("--archive is required")
    if args.command == "prepare":
        prepare(args.archive.resolve())
    elif args.command == "verify-native":
        verify_native(args.archive.resolve(), args.out.resolve())
    elif args.command == "run":
        run(args.archive.resolve(), args.out.resolve(), args.workers)
    else:
        report(args.out.resolve())


if __name__ == "__main__":
    main()
