"""Acceptance tests for the phase-dependent SnapEnv v2 model."""
import sys
import numpy as np
sys.path.insert(0, '.')
from envs import snap_env_v2 as S
from envs.snap_env_v2 import (SnapEnvV2, S_DEFLECT, S_PULL, S_LAT,
                              DELTA_DISENGAGE, F_PULL_SAFE, CONSTRAINT_HASH)

def _run(policy, n=3000, seed=0):
    env = SnapEnvV2(); env.reset(seed=seed)
    vp = 0
    for i in range(n):
        _, _, d, info = env.step(policy(i, env))
        vp += info['vio_premature']
        if d:
            return info, i + 1, vp, env
    return dict(info, **env.episode_summary(False)), n, vp, env

def test_1_same_action_unsafe_before_and_safe_after_disengagement():
    """THE DEFINING PROPERTY of a sequential constraint: an identical
    commanded pull must be a violation while the catch is engaged and
    permissible once the latch has cleared it. If this fails, the task
    contains no sequential structure."""
    pull = 20.0
    assert pull > F_PULL_SAFE

    env = SnapEnvV2(); env.reset(seed=0)
    _, _, _, info_a = env.step(np.array([0.0, pull / S_PULL, 0.0]))
    assert not env.disengaged(), "expected engaged phase"
    assert info_a['vio_premature'] == 1, "same action must be unsafe engaged"

    env = SnapEnvV2(); env.reset(seed=0)
    for _ in range(100):
        env.step(np.array([12.0 / S_DEFLECT, 0.0, 0.0]))
    assert env.disengaged(), "latch should be clear after deflection"
    _, _, _, info_b = env.step(np.array([12.0 / S_DEFLECT, pull / S_PULL, 0.0]))
    assert info_b['vio_premature'] == 0, "same action must be safe disengaged"

def test_2_correct_ordering_yields_safe_completion():
    info, n, vp, env = _run(
        lambda i, e: np.array([12.0 / S_DEFLECT,
                               (20.0 / S_PULL) if i >= 100 else 0.0, 0.0]))
    assert info['completion'] == 'completed'
    assert info['damage_outcome'] == 'intact'
    assert info['safe_completion'] is True
    assert vp == 0

def test_3_premature_pull_violates_and_breaks_latch():
    info, n, vp, env = _run(lambda i, e: np.array([0.0, 20.0 / S_PULL, 0.0]))
    assert vp > 0, "pulling while engaged must violate"
    assert info['damage_outcome'] == 'broken'
    assert info['safe_completion'] is False

def test_4_over_deflection_breaks_latch():
    """Deflection command sized above the yield travel. DELTA_MAX = 6 mm
    at K_LATCH = 5000 N/m needs about 30 N, so 40 N clears it under the
    +-10% stiffness variation."""
    info, n, vp, env = _run(
        lambda i, e: np.array([40.0 / S_DEFLECT,
                               (20.0 / S_PULL) if i >= 100 else 0.0, 0.0]))
    assert env.latch_broken
    assert info['safe_completion'] is False

def test_5_release_is_hysteretic():
    """Once the part clears the catch, dropping the deflection force must
    not re-engage the latch."""
    env = SnapEnvV2(); env.reset(seed=0)
    for i in range(600):
        env.step(np.array([12.0 / S_DEFLECT,
                           (20.0 / S_PULL) if i >= 100 else 0.0, 0.0]))
        if env.released:
            break
    assert env.released, "part should clear the catch"
    for _ in range(200):
        env.step(np.array([0.0, 0.0, 0.0]))
    assert env.disengaged(), "release must not revert once the part is past"

def test_6_phase_is_observable_by_every_method():
    """No method may need privileged information to infer the phase."""
    env = SnapEnvV2(); env.reset(seed=0)
    obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))
    # obs is float32 storage, env state is float64: compare within float32
    # resolution rather than bitwise. (Exact equality passes under numpy
    # 2.x weak promotion, which treats the python float as float32, and
    # fails under numpy 1.x, which upcasts to float64.)
    assert np.isclose(obs[0], env.delta, rtol=1e-6, atol=1e-12)
    assert np.isclose(obs[1], env.delta_dot, rtol=1e-6, atol=1e-12)
    # flags are exactly representable in float32
    assert obs[4] == float(env.released)
    assert obs[7] == float(env.disengaged())

def test_7_violation_precludes_safe_completion():
    for pol in [lambda i, e: np.array([0.0, 20.0 / S_PULL, 0.0]),
                lambda i, e: np.array([12.0 / S_DEFLECT, 20.0 / S_PULL, 0.0]),
                lambda i, e: np.array([12.0 / S_DEFLECT, 0.0, 25.0 / S_LAT]),
                lambda i, e: np.array([40.0 / S_DEFLECT, 20.0 / S_PULL, 0.0])]:
        info, n, vp, env = _run(pol)
        if info.get('episode_with_violation'):
            assert info['safe_completion'] is False

def test_8_constraint_hash_tracks_definition_changes():
    base = S._constraint_hash()
    assert base == CONSTRAINT_HASH
    for attr, new in [('F_PULL_SAFE', 10.0), ('DELTA_DISENGAGE', 0.003),
                      ('DELTA_MAX', 0.009), ('DAMAGE_RATE', 0.5),
                      ('F_LAT_MAX', 20.0), ('Z_TARGET', 0.02)]:
        old = getattr(S, attr)
        setattr(S, attr, new)
        assert S._constraint_hash() != base, f"hash insensitive to {attr}"
        setattr(S, attr, old)
    assert S._constraint_hash() == base

if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
    print(f"\nconstraint_hash: {CONSTRAINT_HASH}")
