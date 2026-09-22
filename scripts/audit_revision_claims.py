#!/usr/bin/env python3
"""Read-only reconciliation of archived results. No training, replay, or API calls.

Usage: python audit_revision_claims.py --research-root /path/to/trix-disasm-suite
       --hardware-root /path/to/rebot_control --out /path/to/report.json
Requires NumPy only for the original deterministic command parser.
"""
import argparse
import csv
from datetime import date
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hardware_checksum_path(name, checksum_file, hardware_root):
    """Resolve preserved historical names inside the supplied evidence root.

    Confirmation checksum files recorded the original machine's absolute
    checkout path. Never read that checkout when auditing a relocated package.
    """
    root = hardware_root.resolve()
    path = Path(name.lstrip("*"))
    recorded_root = Path("/home/stavanio/rebot_control")
    if path.is_absolute():
        if path.is_relative_to(recorded_root):
            path = root / path.relative_to(recorded_root)
        elif not path.is_relative_to(root):
            raise ValueError(f"Unrecognized external checksum path: {path}")
    else:
        path = checksum_file.parent / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Checksum path escapes hardware root: {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-root", required=True, type=Path)
    ap.add_argument("--hardware-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root, hw = args.research_root.resolve(), args.hardware_root.resolve()
    source_hashes = {}

    def source(rel):
        p = root / rel
        source_hashes[rel] = sha(p)
        return p

    def read(rel):
        return json.loads(source(rel).read_text())

    man = read("results/vlm_bank/manifest.json")
    entries = {x["image"]: x for x in man["images"]}
    assert len(entries) == len(man["images"]) == 91
    assert len({x["state_id"] for x in entries.values()}) == 28
    spec = importlib.util.spec_from_file_location(
        "audited_vlm_primitives", source("experiments/vlm_primitives.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    reports, failures = {}, {}
    for provider in ("anthropic", "openai", "google"):
        rows = read(f"results/vlm/vlm_{provider}.json")["rows"]
        assert len(rows) == 91 and {r["image"] for r in rows} == set(entries)
        for r in rows:
            e = entries[r["image"]]
            assert r["state"] == e["state"]
            assert r["state_id"] == e["state_id"]
            assert r["bank_hash"] == man["bank_hash"]
            assert r["raw"] and "error" not in r
            action, mag, cls = parser.parse(r["raw"])
            assert (action, mag, cls) == (r["action"], r["magnitude"], r["class"])
            if cls == "grounded":
                assert parser.ground(action, mag).tolist() == r["command"]
                assert r["failure_class"] == ("executable" if r["admissible"] else "inadmissible")
                a, b = r["none"]["counts"], r["trix"]["counts"]
                assert r["projection_prevented_harm"] == bool(
                    (a["destructive_completions"] or a["episodes_with_violation"])
                    and not (b["destructive_completions"] or b["episodes_with_violation"]))
        counts = Counter(r["failure_class"] for r in rows)
        sums = {arm: dict(Counter({k: sum(r.get(arm, {}).get("counts", {}).get(k, 0)
                    for r in rows) for k in ("episodes", "safe_completions",
                    "destructive_completions", "mechanical_failures", "timeouts",
                    "episodes_with_violation")})) for arm in ("none", "trix")}
        reports[provider] = {
            "configured_aliases": sorted({r["model"] for r in rows}),
            "classes": dict(counts), "grounded": sum(r["class"] == "grounded" for r in rows),
            "outcomes": sums,
            "prevented_flags": sum(r.get("projection_prevented_harm", False) for r in rows),
            "introduced_flags": sum(r.get("projection_introduced_harm", False) for r in rows),
            "inadmissible_by_stratum": dict(Counter(r["stratum"] for r in rows
                                                   if r["failure_class"] == "inadmissible")),
            "per_call_timestamps_present": any("timestamp" in r for r in rows),
            "resolved_backend_present": any("resolved_model" in r for r in rows),
        }
        failures[provider] = {r["image"] for r in rows if r["failure_class"] == "inadmissible"}
    union = set.union(*failures.values())
    assert len(union) == 35 and not set.intersection(*failures.values())
    tilt = [{"image": e["image"], "state_id": e["state_id"],
             "display_tilt_rad": e["state"]["tilt"],
             "restored_tilt_rad": e["state"]["tilt_norm"] * 0.5 / 2.0,
             "shared_lift_m": e["state"]["lift"]} for e in entries.values()]
    assert all(abs(x["display_tilt_rad"] - x["restored_tilt_rad"]) > 1e-6 for x in tilt)
    for e in entries.values():
        source("results/vlm_bank/frames/" + e["image"])
    geometry = read("results/geometry_bench_heldout.json")
    gsummary = {}
    for pred in (True, False):
        rows = [r for r in geometry["rows"] if r["predicted"] == pred]
        gsummary["needs_structure" if pred else "box_suffices"] = {
            "n": len(rows),
            "retained": {m: sum(r["methods"][m]["retains_required"] for r in rows)
                         for m in rows[0]["methods"]}}
    assert min(v["feasible"] for r in geometry["rows"] for v in r["methods"].values()) == 100
    sensitivity = read("results/sensitivity.json")
    failed = [dict(task=t, **r) for t, rows in sensitivity["rows"].items()
              for r in rows if not r["holds"]]
    assert Counter(r["task"] for r in failed) == {"PCB": 6, "CRANK": 1, "BATTERY": 2}
    decomposed = read("results/safelayer_decomposition.json")
    stages = {}
    for rel in ("stage1/stage1_records.json", "stage2/stage2_records.json",
                "bayonet/stage1_records.json", "bayonet2/stage2_records.json"):
        rows = read("results/" + rel)["records"]
        groups = defaultdict(list)
        for r in rows:
            c = r["counts"]
            assert c == r["summary"]["counts"]
            rate = 100 * c["safe_completions"] / c["episodes"]
            assert abs(rate - r["summary"]["safe_completion_rate"]) < 1e-10
            groups[r["task"] + "/" + r["method"]].append(rate)
        stages[rel] = {"records": len(rows), "groups": {
            k: {"n": len(v), "safe_mean": sum(v)/len(v), "safe_per_seed": v}
            for k, v in groups.items()}}
    corrected = read("results/audits/pcb_oracle_seed_fix/pcb_oracle_seed0_corrected.json")
    control = read("results/control/pry_control.json")
    version_counts = Counter()
    budget_examples = []
    for stage in ("stage1", "stage2"):
        for p in sorted((root / f"results/{stage}/checkpoints").glob("*_meta.json")):
            d = read(str(p.relative_to(root)))
            f = d["fingerprint"]
            version_counts[(stage, f["sb3"], f["gymnasium"], tuple(f["net_arch"]))] += 1
            if d["task"] == "PCB" and d["seed"] == 0:
                z = p.with_name(p.name.replace("_meta.json", "_final.zip"))
                with ZipFile(z) as arc:
                    data_bytes = arc.read("data")
                    m = json.loads(data_bytes)
                budget_examples.append({
                    "file": str(z.relative_to(root)),
                    "model_metadata_sha256": hashlib.sha256(data_bytes).hexdigest(),
                    "requested_steps": d["train_steps"], "actual_steps": m["num_timesteps"],
                    "algorithm": d["algo"]})
    assert sum(version_counts.values()) == 360
    # Hash all reported hardware evidence without running controller code.
    final = hw / "results/hardware/v4_confirmation/final_submission"
    hw_rows = list(csv.DictReader((final / "tables/hardware_results.csv").open()))
    checksum_files = [final / "MEDIA_SHA256SUMS.txt"]
    for row in hw_rows:
        d = final.parent / row["run_id"]
        checksum_files.append(d / "SHA256SUMS.txt")
        log = (d / "controller_console.log").read_text()
        decisions = re.findall(r"^.*-> (ALLOW|PROJECT|REJECT)\s*$", log, re.M)
        assert Counter(decisions) == {"ALLOW": 10}, (row["run_id"], decisions)
        assert (d / "exit_code.txt").read_text().strip() == "0"
        assert "SOFT GRASP: PASS" in log
        final_angles = dict(re.findall(r"J(\d):\s*([+-][\d.]+) deg",
                            log.split("START -> FINAL:")[-1]))
        for j in range(1, 8):
            assert float(final_angles[str(j)]) == float(row[f"J{j}_final_deg"])
    hchecks = []
    for cf in checksum_files:
        for line in cf.read_text().splitlines():
            h, name = line.split(maxsplit=1)
            p = hardware_checksum_path(name, cf, hw)
            actual = sha(p)
            assert actual == h, str(p)
            hchecks.append({"path": str(p.relative_to(hw)), "sha256": actual})
    intervention = hw / "experiments/hardware/2026-08-27_trix_physical_safety_v1/adversarial/controller_console.log"
    ilog = intervention.read_text()
    assert "PROJECTED_J2:+50.00->+18.00" in ilog
    assert "J1: +0.007 deg" in ilog
    source_files = [
        "experiments/vlm_bank.py", "experiments/vlm_trial.py", "experiments/vlm_client.py",
        "envs/render.py", "envs/pcb_env_v2.py", "experiments/geometry_bench.py",
        "benchmark/geometry_gen.py", "experiments/sensitivity.py",
        "experiments/safelayer_decomposition.py", "baselines/screw_solvers.py",
        "baselines/screw_filters.py", "tests/test_battery_v2.py", "docs/provenance.md"]
    for name in source_files:
        source(name)
    result = {
        "schema": 2, "audit_date": date.today().isoformat(), "new_experiments_run": False,
        "vlm": {"bank_spec": man["spec"], "bank_hash": man["bank_hash"],
                "images": 91, "base_configurations": 28, "decisions": 273,
                "grounded_replays_per_arm": 257, "providers": reports,
                "image_union": len(union), "triple_image_overlap": 0,
                "exclusive_images": {p: len(v - set.union(*(x for q, x in failures.items() if q != p)))
                                     for p, v in failures.items()},
                "pose_mapping": tilt},
        "geometry": {"spec_hash": geometry["spec_hash"], "groups": gsummary,
                     "min_sampled_feasibility_percent": 100, "proposals_per_instance": 400,
                     "per_instance_fitting_samples": 3000, "retention_relative_tolerance": .02},
        "learned_decomposition": decomposed,
        "sensitivity": {"summary": sensitivity["summary"], "failed_rows": failed},
        "policy_aggregates": stages, "pcb_oracle_correction": corrected,
        "training_metadata": {
            "versions": [{"stage": k[0], "sb3": k[1], "gymnasium": k[2],
                          "architecture": k[3], "files": v} for k, v in version_counts.items()],
            "pcb_seed0_budget_examples": budget_examples},
        "pry_control": control,
        "hardware": {"rows": hw_rows, "checksum_verified_files": hchecks,
                     "intervention_log_sha256": sha(intervention),
                     "scope": "Logs, numeric summaries and checksums reconciled; no new hardware run."},
        "source_sha256": source_hashes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited 273 VLM decisions, 200 geometries, 105 sensitivity cases, "
          f"{sum(x['records'] for x in stages.values())} policy records and six hardware logs.")


if __name__ == "__main__":
    main()
