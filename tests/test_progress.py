"""Progress metrics are normalized, bounded, and discriminate timeouts.

Selection uses progress only below safe completion and destructive rate,
so its job is to separate a policy that nearly finished from one that
never moved. Both would otherwise tie at zero safe completions.
"""
import sys
import numpy as np
sys.path.insert(0, '.')
from benchmark.registry import TASKS, make_env, constraint_hashes
from benchmark.selection import PROGRESS_KEY

EXPECTED_HASHES = {
    "SCREW": "beb88f8d0baf5e4a", "PCB": "53fcd8e18d341d5d",
    "SNAP": "044763e531cb6bcb", "CRANK": "bc6cc80449c95466",
    "BATTERY": "09a4ae5d933db6bf", "PRY": "52be08aa340982f6",
}


def _summary(task, action, steps=6000, seed=0):
    env = make_env(task)
    env.reset(seed=seed)
    done = False
    for _ in range(steps):
        _, _, done, _ = env.step(np.asarray(action, dtype=np.float64))
        if done:
            break
    return env.episode_summary(done), env


def test_1_every_task_emits_its_declared_key():
    for task in TASKS:
        s, _ = _summary(task, np.zeros(3), steps=20)
        assert PROGRESS_KEY[task] in s, f"{task} missing {PROGRESS_KEY[task]}"


def test_2_idle_policy_scores_near_zero():
    """Sensor noise moves some states slightly, so the bar is negligible
    progress rather than exact zero."""
    for task in TASKS:
        s, _ = _summary(task, np.zeros(3), steps=50)
        v = s[PROGRESS_KEY[task]]
        assert v < 0.01, f"{task} progressed {v} while idle"


def test_3_progress_is_bounded_in_unit_interval():
    for task in TASKS:
        for act in (np.zeros(3), np.ones(3), -np.ones(3),
                    np.array([0.5, -0.5, 0.5])):
            s, _ = _summary(task, act, steps=800)
            v = s[PROGRESS_KEY[task]]
            assert 0.0 <= v <= 1.0, f"{task} progress {v} outside [0,1]"


def test_4_completion_implies_full_progress():
    for task, act in (("SCREW", [0.35, 0.35, 0.2]),
                      ("SNAP", [0.35, 0.35, 0.2]),
                      ("BATTERY", [0.35, 0.35, 0.2])):
        s, _ = _summary(task, act)
        if s["completion"] == "completed":
            assert s[PROGRESS_KEY[task]] == 1.0, \
                f"{task} completed with progress {s[PROGRESS_KEY[task]]}"


def test_5_progress_orders_two_incomplete_policies():
    """The discriminating case: neither finishes, but one got further."""
    s_idle, _ = _summary("SCREW", np.zeros(3), steps=400)
    s_move, _ = _summary("SCREW", [0.2, 0.2, 0.0], steps=400)
    assert s_move[PROGRESS_KEY["SCREW"]] > s_idle[PROGRESS_KEY["SCREW"]]


def test_6_crank_requires_both_phases():
    """Rotation alone must not score as complete, since extraction follows."""
    env = make_env("CRANK")
    env.reset(seed=0)
    for _ in range(4000):
        env.step(np.array([0.3, 0.3, 0.0]))     # turn, never pull
    s = env.episode_summary(False)
    assert s["progress_rotation"] < 1.0, "rotation alone scored as complete"


def test_7_progress_does_not_alter_constraint_hashes():
    got = constraint_hashes()
    for task, want in EXPECTED_HASHES.items():
        assert got[task] == want, \
            f"{task}: adding progress changed the constraint hash"


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
