"""Acceptance tests for the phase-conditioned CrankEnv v2 model."""
import sys, math
import numpy as np
sys.path.insert(0, '.')
from envs import crank_env_v2 as C
from envs.crank_env_v2 import (CrankEnvV2, F_AXIAL_SAFE, F_AXIAL_MAX,
                               CONSTRAINT_HASH)

def _crank_force(env, mag=12.0):
    return np.array([-math.sin(env.theta), math.cos(env.theta)]) * mag / 30.0

def _run(policy, n=6000, seed=0):
    env = CrankEnvV2(); env.reset(seed=seed)
    vp = vs = 0
    for i in range(n):
        _, _, d, info = env.step(policy(i, env))
        vp += info['vio_axial_phase']; vs += info['vio_axial_struct']
        if d:
            return info, i + 1, vp, vs, env
    return dict(info, **env.episode_summary(False)), n, vp, vs, env

def test_1_same_axial_force_unsafe_before_and_safe_after_rotation():
    """The conditional criterion: an identical axial load must violate
    while the shaft is turning and be permissible once rotation is
    complete. Without this the constraint is not phase-dependent."""
    fz = 15.0
    assert F_AXIAL_SAFE < fz < F_AXIAL_MAX

    env = CrankEnvV2(); env.reset(seed=0)
    _, _, _, info_a = env.step(np.concatenate([_crank_force(env), [fz / 40.0]]))
    assert not env.rotation_complete()
    assert info_a['vio_axial_phase'] == 1, "must violate while rotating"
    assert info_a['vio_axial_struct'] == 0, "structural limit not exceeded"

    env = CrankEnvV2(); env.reset(seed=0)
    for _ in range(6000):
        _, _, d, _ = env.step(np.concatenate([_crank_force(env), [0.0]]))
        if env.rotation_complete():
            break
    assert env.rotation_complete(), "rotation should complete"
    _, _, _, info_b = env.step(np.concatenate([_crank_force(env), [fz / 40.0]]))
    assert info_b['vio_axial_phase'] == 0, "must be permitted once rotated"

def test_2_correct_ordering_yields_safe_completion():
    info, n, vp, vs, env = _run(
        lambda i, e: np.concatenate([_crank_force(e),
                                     [(15.0 / 40.0) if e.rotation_complete() else 0.0]]))
    assert info['completion'] == 'completed'
    assert info['damage_outcome'] == 'intact'
    assert info['safe_completion'] is True
    assert vp == 0 and vs == 0

def test_3_axial_load_while_rotating_damages_bearing():
    info, n, vp, vs, env = _run(
        lambda i, e: np.concatenate([_crank_force(e), [15.0 / 40.0]]))
    assert vp > 0, "axial load while rotating must violate"
    assert info['damage_outcome'] == 'damaged'
    assert info['safe_completion'] is False

def test_4_structural_limit_is_independent_of_phase():
    """Over-pulling AFTER rotation trips only the absolute structural
    limit, never the phase-conditioned one: the two constraints are
    separately attributable."""
    info, n, vp, vs, env = _run(
        lambda i, e: np.concatenate([_crank_force(e),
                                     [(30.0 / 40.0) if e.rotation_complete() else 0.0]]))
    assert vs > 0, "structural limit should fire"
    assert vp == 0, "phase constraint must not fire after rotation"
    assert info['safe_completion'] is False

def test_5_phase_is_observable_by_every_method():
    env = CrankEnvV2(); env.reset(seed=0)
    obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))
    assert np.isclose(obs[0], env.theta, rtol=1e-6, atol=1e-12)
    assert obs[7] == float(env.rotation_complete())

def test_6_violation_precludes_safe_completion():
    pols = [
        lambda i, e: np.concatenate([_crank_force(e), [15.0 / 40.0]]),
        lambda i, e: np.array([0.0, 0.0, 15.0 / 40.0]),
        lambda i, e: np.concatenate([_crank_force(e, 60.0), [0.0]]),
        lambda i, e: np.concatenate([_crank_force(e),
                                     [(30.0 / 40.0) if e.rotation_complete() else 0.0]]),
    ]
    for pol in pols:
        info, n, vp, vs, env = _run(pol)
        if info.get('episode_with_violation'):
            assert info['safe_completion'] is False

def test_7_constraint_families_reported_separately():
    env = CrankEnvV2(); env.reset(seed=0)
    _, _, _, info = env.step(np.array([1.0, 1.0, 1.0]))
    for k in ('vio_axial_phase', 'vio_axial_struct', 'vio_torque',
              'vio_radial', 'rotation_complete', 'episode_with_violation',
              'constraint_hash'):
        assert k in info, f"missing {k}"

def test_8_constraint_hash_tracks_definition_changes():
    base = C._constraint_hash()
    assert base == CONSTRAINT_HASH
    for attr, new in [('F_AXIAL_SAFE', 8.0), ('F_AXIAL_MAX', 25.0),
                      ('TARGET_ROTATIONS', 3.0), ('DAMAGE_RATE', 0.5),
                      ('Z_TARGET', 0.02), ('F_RETAIN', 50.0)]:
        old = getattr(C, attr)
        setattr(C, attr, new)
        assert C._constraint_hash() != base, f"hash insensitive to {attr}"
        setattr(C, attr, old)
    assert C._constraint_hash() == base

if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
    print(f"\nconstraint_hash: {CONSTRAINT_HASH}")
