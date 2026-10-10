"""Acceptance tests for the velocity-commanded ScrewEnv v3 model."""
import sys, math
import numpy as np
sys.path.insert(0, '.')
from envs import screw_env_v3 as S
from envs.screw_env_v3 import (ScrewEnvV3, project_helix_command,
                               nominal_action, K_PITCH, V_CMD_MAX,
                               OMEGA_CMD_MAX, EPS_HELIX, EPS_CMD_NUM,
                               CONSTRAINT_HASH)

def _run(a, n=4000, seed=0, **kw):
    env = ScrewEnvV3(**kw); env.reset(seed=seed)
    vc = vr = sat = 0
    for i in range(n):
        _, _, d, info = env.step(a(env) if callable(a) else a)
        vc += info['vio_helix_cmd']; vr += info['vio_helix_real']
        sat += info['saturated']
        if d: break
    return info, i + 1, vc, vr, sat, env

def test_1_projection_is_exact_in_command_space():
    """A projected command satisfies the helical equality to numerical
    precision: this is what a governor can guarantee."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20000):
        v = rng.uniform(-V_CMD_MAX, V_CMD_MAX)
        w = rng.uniform(-OMEGA_CMD_MAX, OMEGA_CMD_MAX)
        vs, ws = project_helix_command(v, w)
        worst = max(worst, abs(vs - K_PITCH * ws))
    assert worst < EPS_CMD_NUM, f"projection error {worst:.3e}"

def test_2_realized_deviation_is_not_forced_to_zero():
    """The environment must not algebraically enforce the coupling: a
    projected command still shows transient realized deviation."""
    info, n, vc, vr, sat, env = _run(nominal_action(25.0), eps_helix=1e9)
    env2 = ScrewEnvV3(eps_helix=1e9); env2.reset(seed=0)
    peak = 0.0
    for _ in range(200):
        _, _, d, i2 = env2.step(nominal_action(25.0))
        peak = max(peak, abs(i2['helix_error']))
        if d: break
    assert peak > 0.0, "realized deviation is identically zero"

def test_3_unfiltered_commands_can_leave_the_manifold():
    info, n, vc, vr, sat, env = _run(np.array([1.0, 0.0, 0.0]))
    assert vc > 0, "pull without rotation must violate in command space"
    assert abs(info['helix_error_cmd']) > EPS_HELIX

def test_4_matched_qp_agrees_with_closed_form():
    """For the pure helical equality the numerical minimizer and the
    closed form coincide; this is equivalence, not a differentiator."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        v = rng.uniform(-V_CMD_MAX, V_CMD_MAX)
        w = rng.uniform(-OMEGA_CMD_MAX, OMEGA_CMD_MAX)
        vs, ws = project_helix_command(v, w)
        res = minimize(lambda x: (x[0]-v)**2 + (x[1]-w)**2, x0=[v, w],
                       constraints=[{"type": "eq",
                                     "fun": lambda x: x[0] - K_PITCH*x[1]}],
                       method="SLSQP", options={"ftol": 1e-16, "maxiter": 300})
        worst = max(worst, abs(res.x[0]-vs), abs(res.x[1]-ws))
    assert worst < 1e-6, f"QP and closed form differ by {worst:.3e}"

def test_5_saturation_is_recorded_and_bounded():
    info, n, vc, vr, sat, env = _run(np.array([1.0, 0.0, 0.0]))
    assert sat > 0, "an unreachable command should saturate the actuator"
    assert np.isfinite(env.v_z) and np.isfinite(env.omega)
    assert abs(env.v_z) <= S.V_Z_MAX + 1e-9
    assert abs(env.omega) <= S.OMEGA_MAX + 1e-6

def test_6_information_symmetry():
    """Every method sees the same state through the same interface."""
    env = ScrewEnvV3(); env.reset(seed=0)
    obs, _, _, info = env.step(nominal_action(25.0))
    assert np.isclose(obs[2], env.omega, rtol=1e-6, atol=1e-12)
    assert np.isclose(obs[3], env.v_z, rtol=1e-6, atol=1e-12)
    assert np.isclose(obs[5], env.helix_error(), rtol=1e-6, atol=1e-9)
    for k in ('helix_error', 'helix_error_cmd', 'engagement', 'saturated'):
        assert k in info

def test_7_no_algebraic_coupling_in_the_dynamics():
    """v_z is integrated from its own force balance, never assigned from
    omega. With the thread stripped the two decouple entirely."""
    env = ScrewEnvV3(eps_helix=1e9); env.reset(seed=0)
    for _ in range(4000):
        _, _, d, _ = env.step(np.array([1.0, 0.0, 0.0]))
        if env.engagement == 0.0 or d: break
    assert env.engagement == 0.0, "expected the thread to strip"
    assert abs(env.helix_error()) > 1e-3, \
        "stripped thread must permit large realized deviation"

def test_8_constraint_hash_covers_interface_and_servo():
    base = S._constraint_hash()
    assert base == CONSTRAINT_HASH
    for attr, new in [('EPS_HELIX', 2e-3), ('PITCH', 0.0015),
                      ('K_OMEGA', 0.05), ('K_V', 4000.0),
                      ('V_CMD_MAX', 0.08), ('OMEGA_CMD_MAX', 90.0),
                      ('N_SUBSTEPS', 20), ('ENGAGE_INTACT', 0.95)]:
        old = getattr(S, attr)
        setattr(S, attr, new)
        assert S._constraint_hash() != base, f"hash insensitive to {attr}"
        setattr(S, attr, old)
    assert S._constraint_hash() == base

def test_9_graded_response_spans_the_outcome_categories():
    on = _run(nominal_action(25.0))[0]
    mild = _run(np.array([0.0, 25.0/OMEGA_CMD_MAX, 0.0]))[0]
    gross = _run(np.array([1.0, 0.0, 0.0]))[0]
    assert on['safe_completion'] is True
    assert mild['damage_outcome'] == 'intact' and not mild['safe_completion']
    assert gross['damage_outcome'] == 'stripped'

if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
    print(f"\nconstraint_hash: {CONSTRAINT_HASH}")
