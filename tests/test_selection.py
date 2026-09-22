"""Tests for the frozen checkpoint-selection protocol."""
import sys
sys.path.insert(0, '.')
from benchmark import selection as S
from benchmark.selection import Checkpoint as C


def _ck(step, safe, destr=0.0, prog=0.0, vio=0.0, eps=S.CONFIRM_EPISODES):
    return C(step=step, episodes=eps, metrics={
        "safe_completion_rate": safe, "destructive_completion_rate": destr,
        "progress": prog, "episodes_with_any_violation_rate": vio})


def test_1_safe_completion_leads():
    """A do-nothing policy that never damages anything must not outrank a
    policy that completes the task safely."""
    doer = _ck(1000, safe=80.0, destr=5.0)
    idler = _ck(2000, safe=0.0, destr=0.0)
    assert S.select([doer, idler]).step == 1000


def test_2_destructive_breaks_ties_on_safe():
    a = _ck(1000, safe=50.0, destr=30.0)
    b = _ck(2000, safe=50.0, destr=10.0)
    assert S.select([a, b]).step == 2000


def test_3_progress_breaks_ties_below_destructive():
    a = _ck(1000, safe=50.0, destr=10.0, prog=0.4)
    b = _ck(2000, safe=50.0, destr=10.0, prog=0.9)
    assert S.select([a, b]).step == 2000


def test_4_violation_incidence_then_earliest_step():
    a = _ck(3000, safe=50.0, destr=10.0, prog=0.5, vio=40.0)
    b = _ck(2000, safe=50.0, destr=10.0, prog=0.5, vio=10.0)
    assert S.select([a, b]).step == 2000
    c = _ck(1000, safe=50.0, destr=10.0, prog=0.5, vio=10.0)
    assert S.select([b, c]).step == 1000, "earliest tied checkpoint wins"


def test_5_window_centre_is_selected_not_the_peak():
    """A lone spike must not win over a sustained region."""
    cks = [_ck(1000, 10.0), _ck(2000, 10.0), _ck(3000, 95.0),
           _ck(4000, 10.0), _ck(5000, 70.0), _ck(6000, 72.0),
           _ck(7000, 71.0), _ck(8000, 10.0)]
    cands = S.candidate_windows(cks)
    steps = [c.step for c in cands]
    assert 6000 in steps, f"sustained window centre missing from {steps}"
    best = max(cands, key=lambda c: c.key("safe_completion_rate"))
    assert best.step == 6000, f"spike at 3000 outranked the plateau: {steps}"


def test_6_candidates_are_distinct_and_limited():
    cks = [_ck(1000 * i, float(i)) for i in range(1, 12)]
    cands = S.candidate_windows(cks)
    assert len(cands) == S.N_CANDIDATES
    assert len({c.step for c in cands}) == len(cands)


def test_7_selection_refuses_underpowered_estimates():
    """Thirty screening episodes may not decide the reported model."""
    thin = _ck(1000, safe=90.0, eps=S.SCREEN_EPISODES)
    try:
        S.select([thin])
    except ValueError:
        return
    raise AssertionError("a screen-only checkpoint was accepted for selection")


def test_8_validation_and_test_seeds_are_disjoint():
    for task in S.PROGRESS_KEY:
        v = set(S.validation_seeds(task, S.SCREEN_EPISODES))
        v2 = set(S.validation_seeds(task, S.CONFIRM_EXTRA,
                                    offset=S.SCREEN_EPISODES))
        t = set(S.test_seeds(task, S.TEST_EPISODES))
        assert not (v & t), f"{task}: validation and test seeds overlap"
        assert not (v & v2), f"{task}: screen and confirm seeds overlap"
        assert not (v2 & t), f"{task}: confirm and test seeds overlap"


def test_9_pilot_seeds_never_appear_in_production():
    assert not (set(S.PILOT_SEEDS) & set(S.PRODUCTION_SEEDS))
    assert len(S.PRODUCTION_SEEDS) == 10


def test_10_every_task_declares_a_progress_metric():
    from benchmark.registry import TASKS
    for task in TASKS:
        assert task in S.PROGRESS_KEY, f"{task} has no frozen progress key"


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
    import json
    print("\nfrozen protocol:")
    print(json.dumps(S.describe(), indent=1))
