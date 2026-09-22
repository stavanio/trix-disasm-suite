"""Acceptance tests for the retrofitted PryEnv v2 model."""
import sys
import numpy as np
sys.path.insert(0, '.')
from envs import pry_env_v2 as P
from envs.pry_env_v2 import (PryEnvV2, dynamic_torque_limit, MAX_INSERTION,
                             BOND_STRENGTH, LEVER_LENGTH, CONSTRAINT_HASH)
from benchmark.disasm_bench import TORQUE_LIMIT


def _run(tau, fins, n=6000, seed=950):
    env = PryEnvV2(); env.reset(seed=seed)
    info = None; vt = 0
    for t in range(n):
        _, _, d, info = env.step(np.array([tau / 2.5, fins / 80.0, 0.0]))
        vt += info['vio_torque']
        if d:
            break
    return info, t + 1, vt, env


def test_1_task_is_completable_within_the_torque_limit():
    """The bond released only above 3.75 Nm while the limit was 1.5, so no
    admissible torque could finish the task."""
    pry_at_limit = TORQUE_LIMIT / LEVER_LENGTH
    assert BOND_STRENGTH < pry_at_limit, \
        f"bond {BOND_STRENGTH} N unreachable at {pry_at_limit:.1f} N"
    info, n, vt, env = _run(1.2, 5.0)
    assert info['completion'] == 'completed'


def test_2_moderate_torque_shallow_is_safe():
    info, n, vt, env = _run(1.2, 5.0)
    assert info['damage_outcome'] == 'intact'
    assert info['safe_completion'] is True
    assert vt == 0


def test_3_depth_shrinks_the_admissible_torque():
    assert dynamic_torque_limit(0.0) > dynamic_torque_limit(MAX_INSERTION)
    shallow = _run(1.2, 5.0)[2]
    deep = _run(1.2, 60.0)[2]
    assert deep > shallow, "deep insertion should make the same torque illegal"


def test_4_excess_torque_cracks_the_housing():
    info, n, vt, env = _run(2.0, 5.0)
    assert env.crack > 0.0
    assert info['safe_completion'] is False


def test_5_insufficient_torque_times_out_without_damage():
    info, n, vt, env = _run(0.7, 5.0)
    assert n >= 6000
    assert env.crack == 0.0


def test_6_constraint_families_reported_separately():
    env = PryEnvV2(); env.reset(seed=950)
    _, _, _, info = env.step(np.array([1.0, 1.0, 1.0]))
    for k in ('vio_torque', 'vio_force', 'vio_lateral', 'insertion',
              'torque_limit', 'crack', 'episode_with_violation',
              'constraint_hash', 'exceed_torque'):
        assert k in info, f"missing {k}"


def test_7_violation_precludes_safe_completion():
    for tau, fins in ((1.2, 5.0), (1.2, 60.0), (2.0, 5.0), (0.7, 5.0)):
        info, n, vt, env = _run(tau, fins)
        if info.get('episode_with_violation') and 'safe_completion' in info:
            assert info['safe_completion'] is False


def test_8_constraint_hash_tracks_definition_changes():
    base = P._constraint_hash()
    assert base == CONSTRAINT_HASH
    for attr, new in [('BOND_STRENGTH', 9.0), ('K_CRACK', 0.8),
                      ('CRACK_LIMIT', 2.0), ('CRACK_INTACT', 0.2),
                      ('MAX_INSERTION', 0.08), ('Z_TARGET', 0.02)]:
        old = getattr(P, attr)
        setattr(P, attr, new)
        assert P._constraint_hash() != base, f"hash insensitive to {attr}"
        setattr(P, attr, old)
    assert P._constraint_hash() == base


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
    print(f"\nconstraint_hash: {CONSTRAINT_HASH}")
