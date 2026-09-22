"""Acceptance tests for the staged BatteryEnv v2 model."""
import sys
import numpy as np
sys.path.insert(0, '.')
from envs import battery_env_v2 as B
from envs.battery_env_v2 import (BatteryEnvV2, S_PEEL, F_DEFORM, D_SHORT,
                                 T_CRIT, CONSTRAINT_HASH)


def _run(fp, n=8000, seed=900, lat=0.0):
    env = BatteryEnvV2(); env.reset(seed=seed)
    info = None
    for t in range(n):
        _, _, d, info = env.step(np.array([fp / S_PEEL, lat / 30.0, 0.0]))
        if d:
            break
    return info, t + 1, env


def test_1_peel_resists_from_first_contact():
    """A zero-resistance start let the cell accelerate freely and finish in
    a handful of steps."""
    info, n, env = _run(10.0)
    assert n > 100, f"extraction completed in {n} steps"
    assert abs(env.v_z) < 0.05


def test_2_gentle_peel_is_a_safe_completion():
    info, n, env = _run(10.0)
    assert info['completion'] == 'completed'
    assert info['damage_outcome'] == 'intact'
    assert info['safe_completion'] is True
    assert env.short == 0.0


def test_3_abusive_peel_deforms_and_shorts():
    info, n, env = _run(25.0)
    assert env.deform > D_SHORT
    assert env.short > 0.0
    assert info['damage_outcome'] == 'damaged'
    assert info['safe_completion'] is False


def test_4_heating_continues_after_the_command_stops():
    """The defining latent property: once a short initiates, removing the
    force does not stop the consequence."""
    env = BatteryEnvV2(); env.reset(seed=900)
    for _ in range(200):
        env.step(np.array([19.0 / S_PEEL, 0.0, 0.0]))
    t_stop = env.temperature
    assert env.short > 0.0, "expected a short to have initiated"
    for _ in range(2000):
        env.step(np.array([0.0, 0.0, 0.0]))
    assert env.peak_temperature > t_stop + 10.0
    assert env.peak_temperature > T_CRIT, \
        f"peak {env.peak_temperature:.1f} did not reach the hazard threshold"


def test_5_damage_is_irreversible():
    env = BatteryEnvV2(); env.reset(seed=901)
    for _ in range(300):
        env.step(np.array([25.0 / S_PEEL, 0.0, 0.0]))
    d, g = env.deform, env.short
    for _ in range(1000):
        env.step(np.array([0.0, 0.0, 0.0]))
    assert env.deform >= d - 1e-9, "deformation must not recover"
    assert env.short >= g - 1e-9, "short severity must not recover"


def test_6_constraint_families_reported_separately():
    env = BatteryEnvV2(); env.reset(seed=900)
    _, _, _, info = env.step(np.array([1.0, 1.0, 1.0]))
    for k in ('vio_puncture', 'vio_lateral', 'vio_thermal', 'deformation',
              'short_severity', 'temperature', 'episode_with_violation',
              'constraint_hash', 'exceed_puncture', 'exceed_thermal'):
        assert k in info, f"missing {k}"


def test_7_violation_precludes_safe_completion():
    for fp in (9.0, 10.0, 12.0, 16.0, 19.0, 25.0):
        info, n, env = _run(fp)
        if info.get('episode_with_violation') and 'safe_completion' in info:
            assert info['safe_completion'] is False


def test_8_constraint_hash_tracks_definition_changes():
    base = B._constraint_hash()
    assert base == CONSTRAINT_HASH
    for attr, new in [('F_DEFORM', 18.0), ('D_SHORT', 2.0), ('D_INTACT', 0.2),
                      ('P_ISC_MAX', 400.0), ('C_TH', 20.0), ('H_COOL', 8.0),
                      ('ADHESIVE', 10.0), ('Z_TARGET', 0.02)]:
        old = getattr(B, attr)
        setattr(B, attr, new)
        assert B._constraint_hash() != base, f"hash insensitive to {attr}"
        setattr(B, attr, old)
    assert B._constraint_hash() == base


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
    print(f"\nconstraint_hash: {CONSTRAINT_HASH}")
