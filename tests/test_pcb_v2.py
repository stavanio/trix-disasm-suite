"""Acceptance tests for the planarity-constrained PCBEnv v2 model."""
import sys, importlib
import numpy as np
sys.path.insert(0, '.')
from envs import pcb_env_v2 as P
from envs.pcb_env_v2 import (PCBEnvV2, S_TAU, S_FZ, TAU_TILT_MAX,
                             F_LIFT_MAX, CONSTRAINT_HASH)

def _run(tx, ty, fz, n=4000, seed=0):
    env = PCBEnvV2(); env.reset(seed=seed)
    vp = vl = 0
    for i in range(n):
        _, _, d, info = env.step(np.array([tx/S_TAU, ty/S_TAU, fz/S_FZ]))
        vp += info['vio_planarity']; vl += info['vio_lift']
        if d:
            return info, i + 1, vp, vl, env
    return info, n, vp, vl, env

def test_1_planar_lift_is_safe_completion():
    info, n, vp, vl, env = _run(0.0, 0.0, 30.0)
    assert info['completion'] == 'completed'
    assert info['damage_outcome'] == 'intact'
    assert info['safe_completion'] is True
    assert vp == 0 and vl == 0

def test_2_per_axis_legal_norm_illegal_is_flagged():
    """tau_x = tau_y = 0.08 Nm is within every per-axis bound below the
    0.1 Nm limit, but its norm 0.113 Nm violates planarity. A box-shaped
    safe set cannot represent this constraint."""
    info, n, vp, vl, env = _run(0.08, 0.08, 30.0)
    assert vp == n, f"expected planarity violation every step, got {vp}/{n}"
    assert info['tilt_norm'] > TAU_TILT_MAX

def test_3_lift_overload_damages_not_merely_flags():
    """A large vertical impulse must not yield a safe completion."""
    info, n, vp, vl, env = _run(0.0, 0.0, 55.0)
    assert vl == n, "lift overload should be flagged every step"
    assert info['damage_outcome'] == 'fractured'
    assert info['safe_completion'] is False

def test_4_gross_tilt_fractures():
    info, n, vp, vl, env = _run(0.3, 0.3, 30.0)
    assert info['damage_outcome'] == 'fractured'
    assert env.fractured

def test_5_constraint_families_reported_separately():
    env = PCBEnvV2(); env.reset(seed=0)
    _, _, _, info = env.step(np.array([1.0, 1.0, 1.0]))
    for k in ('vio_planarity', 'vio_lift', 'tilt_norm', 'curvature',
              'damage', 'fractured', 'episode_with_violation',
              'constraint_hash'):
        assert k in info, f"missing {k}"

def test_6_violation_precludes_safe_completion():
    """REQUIRED INVARIANT: no episode containing any constraint violation
    may be reported as a safe completion, whether or not damage occurred.
    Checked across a grid spanning legal, marginal and illegal commands."""
    for tx, ty, fz in [(0.0, 0.0, 30.0), (0.06, 0.06, 30.0),
                       (0.08, 0.08, 30.0), (0.12, 0.0, 30.0),
                       (0.0, 0.0, 55.0), (0.3, 0.3, 30.0),
                       (0.05, 0.05, 45.0)]:
        info, n, vp, vl, env = _run(tx, ty, fz)
        if info.get('episode_with_violation'):
            assert info['safe_completion'] is False, \
                f"violating episode reported safe: tau=({tx},{ty}) Fz={fz}"

def test_7_box_disk_gap_is_analytic():
    """Pure geometry: the fraction of the componentwise-admissible action
    box lying outside the joint torque-norm constraint, under uniform
    sampling. Independent of the simulator, its noise model and its
    dynamics -- no environment is instantiated here."""
    rng = np.random.default_rng(0)
    s = rng.uniform(-TAU_TILT_MAX, TAU_TILT_MAX, (200000, 2))
    frac = float(np.mean(np.linalg.norm(s, axis=1) > TAU_TILT_MAX))
    analytic = 1.0 - np.pi / 4.0
    assert abs(frac - analytic) < 0.005, f"{frac:.4f} vs {analytic:.4f}"

def test_8_constraint_hash_tracks_definition_changes():
    """The hash recorded with every result must change if any quantity
    defining a violation, the damage model, or the outcome rule changes."""
    base = P._constraint_hash()
    assert base == CONSTRAINT_HASH
    for attr, new in [('TAU_TILT_MAX', 0.2), ('F_LIFT_MAX', 30.0),
                      ('SIGMA_TAU_PCB', 0.01), ('DAMAGE_RATE', 60.0),
                      ('THETA_FRAC', 0.09), ('DAMAGE_LIFT_RATE', 0.2),
                      ('Z_TARGET', 0.02)]:
        old = getattr(P, attr)
        setattr(P, attr, new)
        assert P._constraint_hash() != base, f"hash insensitive to {attr}"
        setattr(P, attr, old)
    assert P._constraint_hash() == base, "hash not restored"

if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
    print(f"\nconstraint_hash: {CONSTRAINT_HASH}")
    print(f"box-disk geometry: under uniform sampling over the "
          f"componentwise-admissible box, {100*(1-np.pi/4):.1f}% of commands "
          f"violate the joint torque-norm constraint")
