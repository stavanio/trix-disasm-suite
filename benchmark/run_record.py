"""Result record for one (task, method, seed) cell.

Carries everything needed to decide later whether two results may be
compared: the environment's constraint hash, the margin policy that
produced the robust set, the software and machine fingerprint, and the
raw numerators and denominators behind every rate.

A hash or configuration mismatch is a hard failure. Silent compatibility
is how results computed under different definitions get averaged
together.
"""

import os
import platform
import subprocess
import sys

import numpy as np

from benchmark import margin as M
from benchmark import margin_policy as MP
from benchmark import protocol as PROTO
from benchmark import taxonomy as TAX


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def provenance():
    try:
        import torch
        torch_v = torch.__version__
        deterministic = torch.are_deterministic_algorithms_enabled()
        threads = torch.get_num_threads()
        cuda = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        torch_v, deterministic, threads, cuda = None, None, None, None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch_v,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "torch_deterministic": deterministic,
        "torch_num_threads": threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cuda": cuda,
        "git_commit": git_commit(),
    }


def margin_policy(delta=None, horizon=None, n_constraints=1):
    d = M.DEFAULT_DELTA if delta is None else delta
    h = M.DEFAULT_HORIZON if horizon is None else horizon
    return M.describe(d, h, n_constraints=n_constraints)


class HashMismatch(RuntimeError):
    pass


def make_record(task, method, seed, summary, expected_hash,
                margin=None, extra=None, training_mode=None,
                evaluation_arm=None, algorithm=None):
    """Assemble a cell record, refusing to build one whose environment
    hash does not match what the registry declared."""
    got = summary.get("constraint_hash")
    if got != expected_hash:
        raise HashMismatch(
            f"{task}/{method}/seed{seed}: environment reported "
            f"{got!r} but the registry expects {expected_hash!r}")
    rec = {
        "task": task,
        "method": method,
        "seed": seed,
        "constraint_hash": expected_hash,
        "margin_policy": margin or margin_policy(),
        "robust_margin": MP.policy_for(task),
        "robust_margin_hash": MP.policy_hash(task),
        "taxonomy_hash": TAX.taxonomy_hash(task),
        "protocol_freeze_hash": PROTO.freeze_hash(),
        "protocol_frozen_on": PROTO.FROZEN_ON,
        "claims_admissible": {
            c: PROTO.may_support(task, c)
            for c in ("safety_enforcement", "capability_preservation",
                      "learning_enablement")},
        "provenance": provenance(),
        "summary": summary,
        "counts": summary.get("counts"),
        "training_mode": training_mode,
        "evaluation_arm": evaluation_arm,
        "algorithm": algorithm,
    }
    if extra:
        rec.update(extra)
    return rec


# Fields that must agree before records may be pooled, and must be
# present: two records both missing a hash would otherwise produce {None}
# and pass, which is exactly the silent compatibility this guards against.
REQUIRED_HASHES = ("constraint_hash", "robust_margin_hash",
                   "taxonomy_hash", "protocol_freeze_hash")

# Records can share every hash and still mean different things. A nominal
# policy filtered at evaluation and a filter-aware policy answer different
# questions, and PPO is not SAC.
EXECUTION_KEYS = ("training_mode", "evaluation_arm", "algorithm")


def check_comparable(records, require_execution_keys=True):
    """Records may only be aggregated if they agree on what they measure.

    Disagreement OR absence is a hard failure. Absence matters as much as
    mismatch: a record that cannot say which constraint definition, which
    taxonomy or which pre-registration produced it cannot be pooled with
    one that can.
    """
    if not records:
        return
    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)

    for task, rs in by_task.items():
        for field in REQUIRED_HASHES:
            vals = {r.get(field) for r in rs}
            if None in vals:
                raise HashMismatch(
                    f"{task}: a record is missing {field}; records without "
                    f"provenance cannot be pooled")
            if len(vals) > 1:
                raise HashMismatch(
                    f"{task}: records span {field} values {vals}"
                    + ("; the pre-registration changed mid-experiment"
                       if field == "protocol_freeze_hash" else ""))

        sigmas = {round(r["margin_policy"]["sigma"], 9) for r in rs
                  if r.get("margin_policy")}
        if len(sigmas) > 1:
            raise HashMismatch(
                f"{task}: records span margin policies {sigmas}")

        if require_execution_keys:
            for field in EXECUTION_KEYS:
                vals = {r.get(field) for r in rs}
                if None in vals:
                    raise HashMismatch(
                        f"{task}: a record is missing {field}; execution "
                        f"mode must be explicit before pooling")
                if len(vals) > 1:
                    raise HashMismatch(
                        f"{task}: records span {field} values {vals}; "
                        f"these measure different things")
