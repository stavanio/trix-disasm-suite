"""Declared reset distributions, separate from the frozen plant/governor code.

Only existing per-instance reset fields may change. Sampling uses its own RNG,
so rejection draws never change the environment's per-step noise stream.
"""
import hashlib
import json

import numpy as np


PARAMETERS = {
    "SCREW": [("mu_s", 0.61, 0.1, "dimensionless"),
              ("mu_k", 0.47, 0.1, "dimensionless")],
    "PCB": [("f_clip", 25.0, 0.15, "N")],
    "SNAP": [("k_latch", 5000.0, 0.1, "N/m")],
    "CRANK": [("mu_k", 0.47, 0.1, "dimensionless")],
    "BATTERY": [("adhesive", 8.0, 0.1, "N")],
}
CONDITIONS = ("wide_1p5", "wide_2", "shell_2")
NATIVE_ONLY = ("PRY", "BAYONET_S", "BAYONET_P")


def canonical_hash(spec):
    return hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def specification(task, condition="native"):
    if task not in PARAMETERS and task not in NATIVE_ONLY:
        raise ValueError(f"Unknown task {task!r}")
    if condition not in ("native", *CONDITIONS):
        raise ValueError(f"Unknown evaluation condition {condition!r}")
    if condition != "native" and task not in PARAMETERS:
        raise ValueError(f"{task} has no declared reset-range widening")
    scale = {"native": 1.0, "wide_1p5": 1.5,
             "wide_2": 2.0, "shell_2": 2.0}[condition]
    parameters = []
    for field, centre, width, unit in PARAMETERS.get(task, []):
        parameters.append({
            "field": field, "centre": centre, "unit": unit,
            "training_relative_half_width": width,
            "training_bounds": [centre * (1 - width), centre * (1 + width)],
            "evaluation_bounds": [centre * (1 - scale * width),
                                  centre * (1 + scale * width)],
        })
    return {
        "schema": 1, "task": task, "condition": condition,
        "half_width_scale": scale, "parameters": parameters,
        "sampling": "native_reset" if condition == "native" else
                    "independent_uniform_proposals_with_joint_rejection",
        "joint_validity": "mu_s >= mu_k" if task == "SCREW" else "all",
        "support": "outside_training_rectangle" if condition == "shell_2"
                   else "full_rectangle",
        "random_stream": "native" if condition == "native" else
            "RandomState(first_32_bits_SHA256(trix-ood-reset-v1|task|episode_seed))",
        "per_step_noise": "unchanged_native_stream_after_native_reset",
    }


def validate(spec, task=None):
    if not isinstance(spec, dict):
        raise ValueError("Evaluation distribution must be an explicit specification")
    expected = specification(spec.get("task"), spec.get("condition"))
    if spec != expected or (task is not None and spec["task"] != task):
        raise ValueError("Unrecognised or modified evaluation-distribution specification")
    return canonical_hash(spec)


def outside_training_support(spec, values):
    return any(not p["training_bounds"][0] <= values[p["field"]]
               <= p["training_bounds"][1] for p in spec["parameters"])


def sample(spec, episode_seed):
    validate(spec)
    if spec["condition"] == "native":
        raise ValueError("Native values must be read from the original reset")
    if episode_seed is None:
        raise ValueError("OOD resets require an explicit episode seed")
    key = f'trix-ood-reset-v1|{spec["task"]}|{int(episode_seed)}'
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    rng = np.random.RandomState(seed)
    for attempt in range(1, 100001):
        values = {p["field"]: float(rng.uniform(*p["evaluation_bounds"]))
                  for p in spec["parameters"]}
        if spec["task"] == "SCREW" and values["mu_s"] < values["mu_k"]:
            continue
        if spec["support"] == "outside_training_rectangle" and not \
                outside_training_support(spec, values):
            continue
        return values, attempt
    raise RuntimeError("Declared rejection sampler did not accept a reset")


def apply_after_native_reset(env, spec, episode_seed):
    """Overwrite only allowed instance fields, never module/class constants."""
    digest = validate(spec)
    fields = [p["field"] for p in spec["parameters"]]
    # Require native instance fields to exist before writing anything.
    if any(field not in vars(env) for field in fields):
        raise ValueError("Reset field is missing from the environment instance")
    attempts = 0
    if spec["condition"] != "native":
        values, attempts = sample(spec, episode_seed)
        for field, value in values.items():
            setattr(env, field, value)
    values = {field: float(getattr(env, field)) for field in fields}
    return {"evaluation_distribution_hash": digest,
            "episode_seed": int(episode_seed) if episode_seed is not None else None,
            "parameters": values,
            "outside_training_support": outside_training_support(spec, values),
            "sampling_attempts": attempts}
