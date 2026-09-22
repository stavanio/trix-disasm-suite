"""Tests for the phase-gated SNAP and CRANK filter arms."""
import sys, math
import numpy as np
sys.path.insert(0, '.')
from envs.snap_env_v2 import SnapEnvV2, S_DEFLECT, S_PULL
from envs.crank_env_v2 import CrankEnvV2
from baselines import phase_filters as PH

CR_XY = 30.0


def _roll(cls, filt, pol, n_ep=8, cap=8000, seed0=800):
    vio = steps = 0
    outcomes, thetas, by_bin = [], [], {}
    for ep in range(n_ep):
        env = cls(); obs = env.reset(seed=seed0 + ep); done = False
        for t in range(cap):
            a = pol(t, env, obs)
            x = a if filt is None else filt(obs, a)[0]
            th = float(obs[0]) % (2 * math.pi)
            obs, r, done, inf = env.step(x)
            vio += inf['vio']; steps += 1
            b = int(math.degrees(th) // 15) * 15
            d = by_bin.setdefault(b, [0, 0]); d[0] += inf['vio']; d[1] += 1
            if done:
                break
        thetas.append(abs(getattr(env, 'theta', 0.0)) / (2 * math.pi))
        outcomes.append(inf.get('category') if done else 'timeout')
    return {"vio_pct": 100 * vio / steps, "turns": float(np.mean(thetas)),
            "outcomes": outcomes, "by_bin": by_bin}


def test_1_snap_phase_filter_prevents_damage_without_creating_progress():
    """A reactive phase-aware filter suppresses the premature pull and
    prevents damage, but cannot supply the deflection the task needs."""
    pol = lambda t, e, o: np.array([0.0, 20.0 / S_PULL, 0.0])
    raw = _roll(SnapEnvV2, None, pol, seed0=700, cap=3000)
    filt = _roll(SnapEnvV2, PH.snap_project, pol, seed0=700, cap=3000)
    assert all(o == 'destructive_completion' for o in raw['outcomes'])
    assert filt['vio_pct'] < 1.0
    assert all(o == 'timeout' for o in filt['outcomes'])


def test_2_snap_correct_order_completes_under_every_arm():
    """The timeout above is caused by the missing preparatory action, not
    by an unwinnable environment."""
    pol = lambda t, e, o: np.array([16.0 / S_DEFLECT,
                                    (20.0 / S_PULL) if t >= 50 else 0.0, 0.0])
    for f in (None, PH.snap_static_clip, PH.snap_phase_clip, PH.snap_project):
        r = _roll(SnapEnvV2, f, pol, seed0=700, cap=3000)
        assert all(o == 'safe_completion' for o in r['outcomes']), \
            f"{f} gave {r['outcomes']}"


def test_3_snap_static_clip_is_phase_blind():
    """Named honestly: it does not condition the pull bound on phase, so
    it cannot enforce a sequential rule."""
    obs_engaged = np.zeros(10, dtype=np.float32)
    obs_clear = np.zeros(10, dtype=np.float32); obs_clear[7] = 1.0
    a = np.array([0.0, 20.0 / S_PULL, 0.0])
    assert np.allclose(PH.snap_static_clip(obs_engaged, a)[0],
                       PH.snap_static_clip(obs_clear, a)[0])
    assert not np.allclose(PH.snap_project(obs_engaged, a)[0],
                           PH.snap_project(obs_clear, a)[0])


def test_4_crank_static_bound_loses_feasibility_off_axis():
    """Frozen-state geometry: an axis-aligned bound tracks the rotating
    admissible region only near multiples of pi/2."""
    rng = np.random.default_rng(0)
    feas = {}
    for deg in (0, 30, 45, 90):
        obs = np.zeros(10, dtype=np.float32); obs[0] = math.radians(deg)
        t, r = PH.crank_frame(obs); ft, fr, fz = PH.crank_bounds(obs)
        ok_s = ok_p = 0
        for _ in range(3000):
            a = rng.uniform(-1, 1, 3)
            for x, key in ((PH.crank_static_clip(obs, a)[0], 's'),
                           (PH.crank_project(obs, a)[0], 'p')):
                f = np.array([float(x[0]), float(x[1])])
                good = (abs(f @ t) <= ft and abs(f @ r) <= fr
                        and abs(float(x[2])) <= fz)
                if key == 's':
                    ok_s += good
                else:
                    ok_p += good
        feas[deg] = (ok_s / 3000, ok_p / 3000)
    assert feas[0][0] > 0.95, "static bound should be fine when aligned"
    assert feas[45][0] < 0.75, "static bound should degrade off-axis"
    for d in feas:
        assert feas[d][1] > 0.99, f"frame projection failed at {d} deg"


def test_5_crank_static_bound_trades_safety_for_capability():
    """Closed loop under the adversarial policy: the static bound avoids
    violations only by restricting every axis, and stalls."""
    pol = lambda t, e, o: PH.crank_adversarial(o)
    raw = _roll(CrankEnvV2, None, pol)
    stat = _roll(CrankEnvV2, PH.crank_static_clip, pol)
    proj = _roll(CrankEnvV2, PH.crank_project, pol)
    assert raw['vio_pct'] > 20.0
    assert stat['vio_pct'] < 1.0 and stat['turns'] < 1.0
    assert proj['vio_pct'] < 2.0 and proj['turns'] > 1.5


def test_6_margin_is_a_tradeoff_not_a_guarantee():
    """Violations fall with the margin and progress is paid for it, so the
    choice must be reported as a sweep rather than a fixed constant."""
    pol = lambda t, e, o: np.array([0.0, 20.0 / S_PULL, 0.0])
    prev = None
    for sig in (0.0, 1.0, 3.0):
        f = lambda o, a, s=sig: PH.snap_project(o, a, sigma=s)
        r = _roll(SnapEnvV2, f, pol, seed0=700, cap=3000)
        if prev is not None:
            assert r['vio_pct'] <= prev + 1e-9, "more margin must not add violations"
        prev = r['vio_pct']
    assert prev < 1.0


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
