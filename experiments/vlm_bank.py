"""Frozen stratified state bank for the VLM study."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from baselines import pcb_filters as PF
from envs.render import render_pcb

BANK_VERSION = "1.0"
FROZEN_ON = "2026-08-11"

STRATA = {"interior": (0.50, 1.00), "mid": (0.20, 0.50),
          "near_boundary": (0.02, 0.20)}
PER_STRATUM = 7

PAIRED = ("normal", "rotated", "occluded", "misleading")

TILT_RANGE = (0.0, 0.9)
LIFT_RANGE = (0.0, 0.010)
PARTIAL_TILT = (0.10, 0.22)
PARTIAL_LIFT = (0.008, 0.013)


def sample_states(rng):
    bound = PF.TAU_ROBUST * (1.0 - PF.INTERIOR)
    states = []
    for name, (lo, hi) in STRATA.items():
        got = 0
        tries = 0
        while got < PER_STRATUM and tries < 5000:
            tries += 1
            m = rng.uniform(lo, hi)
            tilt_norm = bound * (1.0 - m) * rng.choice([-1.0, 1.0])
            lift = rng.uniform(*LIFT_RANGE)
            cmd = np.array([tilt_norm, 0.0, 0.2])
            x, _ = PF.trix_project(np.zeros(10), cmd)
            if not np.allclose(np.asarray(x, dtype=np.float64), cmd,
                               atol=1e-6):
                continue
            states.append({"stratum": name, "margin": round(m, 4),
                           "tilt_norm": round(float(tilt_norm), 4),
                           "tilt": round(float(tilt_norm) / bound * TILT_RANGE[1], 4),
                           "lift": round(float(lift), 5),
                           "clutter_seed": int(rng.integers(0, 10_000))})
            got += 1
        if got < PER_STRATUM:
            raise RuntimeError(f"stratum {name}: only {got} of "
                               f"{PER_STRATUM} after {tries} tries")
    return states


def build(out):
    rng = np.random.default_rng(20260811)
    states = sample_states(rng)
    frames = os.path.join(out, "frames")
    os.makedirs(frames, exist_ok=True)

    manifest = []
    for i, st in enumerate(states):
        for cond in PAIRED:
            name = f"s{i:02d}_{cond}.png"
            img = render_pcb({"tilt": st["tilt"], "lift": st["lift"]},
                             condition=cond, seed=st["clutter_seed"])
            Image.fromarray(img).save(os.path.join(frames, name))
            manifest.append({"image": name, "state_id": i, "paired": True,
                             "condition": cond,
                             "state": {"tilt": st["tilt"],
                                       "lift": st["lift"],
                                       "tilt_norm": st["tilt_norm"]}, **st})

    for i in range(PER_STRATUM):
        tilt = float(rng.uniform(*PARTIAL_TILT))
        lift = float(rng.uniform(*PARTIAL_LIFT))
        st = {"stratum": "partial", "margin": None,
              "tilt_norm": round(tilt / TILT_RANGE[1], 4),
              "tilt": round(tilt, 4), "lift": round(lift, 5),
              "clutter_seed": int(rng.integers(0, 10_000))}
        name = f"p{i:02d}_partial.png"
        img = render_pcb({"tilt": st["tilt"], "lift": st["lift"]},
                         condition="partial", seed=st["clutter_seed"])
        Image.fromarray(img).save(os.path.join(frames, name))
        manifest.append({"image": name, "state_id": 100 + i, "paired": False,
                         "condition": "partial",
                         "state": {"tilt": st["tilt"], "lift": st["lift"],
                                   "tilt_norm": st["tilt_norm"]}, **st})

    spec = {"version": BANK_VERSION, "frozen_on": FROZEN_ON,
            "strata": {k: list(v) for k, v in STRATA.items()},
            "per_stratum": PER_STRATUM, "paired": list(PAIRED),
            "tilt_range": list(TILT_RANGE), "lift_range": list(LIFT_RANGE)}
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()
                       ).hexdigest()[:16]
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"spec": spec, "bank_hash": h, "images": manifest}, f,
                  indent=1)

    print(f"bank {h}, {len(manifest)} images")
    for k, v in Counter(m["stratum"] for m in manifest).items():
        print(f"  {k:14s} {v}")
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/vlm_bank")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build(a.out)
