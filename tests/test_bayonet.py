"""Acceptance tests and preregistered manipulation checks."""

import math
import sys

import numpy as np
sys.path.insert(0, '.')

from envs.bayonet_env import (BayonetEnv, CAP, CONSTRAINT_HASH_S,
                              CONSTRAINT_HASH_P, E_AXIAL, E_TAU, S_AXIAL,
                              S_TAU, RETENTION_FORCE, axial_bound,
                              envelope_value, nominal_action)
from baselines import bayonet_filters as F

SEEDS = range(10)
ARMS_S = [("box", F.box_static), ("proj", F.projection_static)]
ARMS_P = [("blind_cons", F.box_blind_conservative),
          ("blind_perm", F.box_blind_permissive),
          ("aware_box", F.box_phase_aware),
          ("aware_proj", F.projection_phase_aware)]


def run(variant, filt, seed):
    env = BayonetEnv(variant)
    obs = env.reset(seed=seed)
    done = False
    inter = steps = near = 0
    for t in range(CAP):
        nom = nominal_action(t, env.released)
        x, _ = filt(obs, nom) if filt else (nom, {})
        if filt and not np.allclose(np.asarray(x, dtype=np.float64), nom,
                                    atol=1e-9):
            inter += 1
        obs, _, done, info = env.step(x)
        steps += 1
        if info["envelope_value"] > 0.95:
            near += 1
        if done:
            break
    return env.episode_summary(done), inter / steps, near / steps


def test_1_variants_have_distinct_hashes():
    assert CONSTRAINT_HASH_S != CONSTRAINT_HASH_P


def test_2_observation_layout_is_identical():
    a = BayonetEnv("s").reset(seed=0)
    b = BayonetEnv("p").reset(seed=0)
    assert a.shape == b.shape == (10,)
    assert np.allclose(a, b)


def test_3_static_envelope_ignores_phase():
    assert axial_bound("s", False) == axial_bound("s", True)
    assert axial_bound("p", False) < axial_bound("p", True)


def test_4_required_action_is_inside_both_sets():
    v = envelope_value(0.45 * S_AXIAL, 0.45 * S_TAU, "s", True)
    assert v <= 1.0, f"required action outside the ellipse: {v}"
    assert 0.45 <= 1.0 / math.sqrt(2.0)


def test_5_scripted_controller_completes_variant_s_safely():
    for seed in SEEDS:
        for name, f in ARMS_S:
            s, _, _ = run("s", f, seed)
            assert s["safe_completion"], f"S/{name} seed {seed}: {s['category']}"


def test_6_phase_aware_arms_complete_variant_p_safely():
    for seed in SEEDS:
        for name in ("aware_box", "aware_proj"):
            f = dict(ARMS_P)[name]
            s, _, _ = run("p", f, seed)
            assert s["safe_completion"], f"P/{name} seed {seed}: {s['category']}"


def test_7_blind_controls_fail_as_predicted():
    for seed in SEEDS:
        s, _, _ = run("p", F.box_blind_conservative, seed)
        assert s["completion"] == "timeout", \
            f"conservative box completed on seed {seed}"
        s, _, _ = run("p", F.box_blind_permissive, seed)
        assert s["damage_outcome"] != "intact", \
            f"permissive box did not shear on seed {seed}"


def test_8_box_and_ellipse_differ_on_enough_proposals():
    rng = np.random.default_rng(0)
    obs = BayonetEnv("s").reset(seed=0)
    diff = 0
    worst = 0.0
    for _ in range(5000):
        a = rng.uniform(-1, 1, 3)
        x, _ = F.box_static(obs, a)
        y, _ = F.projection_static(obs, a)
        d = float(np.max(np.abs(np.float64(x) - np.float64(y))))
        worst = max(worst, d)
        diff += d > 1e-6
    frac = diff / 5000
    assert frac >= 0.20, f"only {frac:.1%} of proposals differ"


PROBE_RADII = (0.8, 0.95, 1.05, 1.2)
PROBE_DIRS = (0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2)


def probe_action(step, released, variant):
    """Sweep coupled proposals around the admissible boundary while
    following the phase sequence, so both legs are visited."""
    r = PROBE_RADII[(step // 7) % len(PROBE_RADII)]
    d = PROBE_DIRS[(step // 31) % len(PROBE_DIRS)]
    ax = axial_bound(variant, released)
    sign = 1.0 if released else -1.0
    f_ax = sign * r * ax * math.cos(d)
    tau = r * E_TAU * math.sin(d)
    if not released and abs(tau) < 0.2 * E_TAU:
        tau = 0.35 * E_TAU
    return np.array([f_ax / S_AXIAL, tau / S_TAU, 0.3])


def run_probe(variant, filt):
    env = BayonetEnv(variant)
    obs = env.reset(seed=0)
    rows = []
    for t in range(CAP):
        rel = env.released
        nom = probe_action(t, rel, variant)
        demand = envelope_value(float(nom[0]) * S_AXIAL,
                                float(nom[1]) * S_TAU, variant, rel)
        x, _ = filt(obs, nom)
        obs, _, done, info = env.step(x)
        rows.append((rel, demand, info["envelope_value"],
                     not np.allclose(np.asarray(x, dtype=np.float64),
                                     nom, atol=1e-9)))
        if done:
            break
    return rows


def test_9_probe_exercises_the_boundary():
    thresh = 0.95 * F.tightened_ratio()
    for variant, arms in (("s", ARMS_S), ("p", ARMS_P[2:])):
        for name, f in arms:
            rows = run_probe(variant, f)
            n = len(rows)
            phases = {False: [r for r in rows if not r[0]],
                      True: [r for r in rows if r[0]]}
            demanding = sum(1 for r in rows if r[1] >= 0.95) / n
            inter = sum(1 for r in rows if r[3]) / n
            near = sum(1 for r in rows if r[2] > thresh) / n
            print(f"    {variant}/{name:10s} rot {len(phases[False]):4d} "
                  f"rel {len(phases[True]):4d}  demand>=1 {demanding:5.1%}  "
                  f"intervene {inter:5.1%}  executed near {near:5.1%}")
            assert demanding >= 0.20, \
                f"{variant}/{name}: only {demanding:.1%} of proposals demanding"
            assert inter >= 0.20, \
                f"{variant}/{name}: intervened on {inter:.1%}"
            if "proj" in name:
                assert near >= 0.05, \
                    f"{variant}/{name}: executed near boundary {near:.1%}"


def test_10_retention_makes_the_gate_a_capability_gate():
    ax_rot, _, _ = F._bounds("p", False)
    ax_rel, _, _ = F._bounds("p", True)
    assert ax_rot < RETENTION_FORCE < ax_rel


def _order(name):
    return int(name.split("_")[1])


if __name__ == "__main__":
    for name in sorted((k for k in dict(globals()) if k.startswith("test_")),
                       key=_order):
        globals()[name]()
        print(f"PASS {name}")
    print(f"\nS {CONSTRAINT_HASH_S}\nP {CONSTRAINT_HASH_P}")
