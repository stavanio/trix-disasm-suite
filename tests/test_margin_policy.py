"""Tests for where robust margins apply and where they deliberately do not.

The governing question is whether stochastic realization separates the
filter output from the constraint check. Where it does not, a margin buys
nothing and costs admissible commands.
"""
import sys

import numpy as np
sys.path.insert(0, '.')

from envs.screw_env_v3 import ScrewEnvV3
from envs import screw_env_v3 as SE
from baselines import screw_solvers as SS
from benchmark import margin_policy as MP
from benchmark.registry import TASKS


def test_1_command_constraint_is_noise_free():
    """Realization noise must not alter the command-manifold verdict: the
    check reads the projected command itself."""
    a = np.array([0.8, 0.15, 0.4])
    x, _ = SS.trix_scalar(None, a)
    verdicts, e_cmds = set(), set()
    for seed in range(25):
        env = ScrewEnvV3(); env.reset(seed=seed)
        _, _, _, info = env.step(x)
        verdicts.add(info["vio_helix_cmd"])
        e_cmds.add(round(info["helix_error_cmd"], 15))
    assert len(verdicts) == 1, f"command verdict varied with noise: {verdicts}"
    assert len(e_cmds) == 1, f"command coupling error varied: {e_cmds}"


def test_2_realized_quantities_do_vary_with_the_same_noise():
    """The same command produces different realized outcomes, which is why
    downstream constraints need margins and the command one does not."""
    a = np.array([0.8, 0.15, 0.4])
    x, _ = SS.trix_scalar(None, a)
    realized, loads = set(), set()
    for seed in range(25):
        env = ScrewEnvV3(); env.reset(seed=seed)
        for _ in range(50):
            _, _, _, info = env.step(x)
        realized.add(round(info["helix_error"], 12))
        loads.add(round(info["peak_thread_load"], 9))
    assert len(realized) > 1, "realized deviation was identical across noise"
    assert len(loads) > 1, "thread load was identical across noise"


def test_3_screw_command_layers_declare_no_margin():
    p = MP.policy_for("SCREW")
    assert p["command_helix"]["mode"] == MP.NONE
    assert p["command_bounds"]["mode"] == MP.NONE
    assert p["realized_helix"]["mode"] == MP.DERIVED
    assert p["realized_helix"]["sigma"] > 3.0
    for k in ("command_helix", "command_bounds"):
        assert p[k].get("reason"), f"{k} must record why no margin applies"


def test_4_every_task_declares_a_mode_and_reason_per_family():
    for task in TASKS:
        p = MP.policy_for(task)
        assert p, f"{task} has no declared policy"
        for fam, spec in p.items():
            assert spec["mode"] in (MP.NONE, MP.DERIVED), f"{task}/{fam}"
            if spec["mode"] == MP.NONE:
                assert spec.get("reason"), f"{task}/{fam} needs a reason"
            else:
                assert spec.get("sigma", 0) > 0, f"{task}/{fam} needs a sigma"


def test_5_policy_hash_covers_values_and_the_no_margin_decision():
    base = MP.policy_hash("SCREW")
    saved = MP.POLICY["SCREW"]["command_helix"]["mode"]
    MP.POLICY["SCREW"]["command_helix"]["mode"] = MP.DERIVED
    try:
        assert MP.policy_hash("SCREW") != base, \
            "flipping a no-margin decision left the hash unchanged"
    finally:
        MP.POLICY["SCREW"]["command_helix"]["mode"] = saved

    saved_s = MP.POLICY["PCB"]["planarity"]["sigma"]
    b2 = MP.policy_hash("PCB")
    MP.POLICY["PCB"]["planarity"]["sigma"] = saved_s + 0.5
    try:
        assert MP.policy_hash("PCB") != b2, "changing a sigma left the hash"
    finally:
        MP.POLICY["PCB"]["planarity"]["sigma"] = saved_s
    assert MP.policy_hash("SCREW") == base and MP.policy_hash("PCB") == b2


def test_6_battery_thermal_declares_none_for_a_different_reason():
    """Not all no-margin decisions have the same justification: thermal is
    a consequence of past commands, not a bound on the current one."""
    spec = MP.policy_for("BATTERY")["thermal"]
    assert spec["mode"] == MP.NONE
    assert "past" in spec["reason"] or "consequence" in spec["reason"]


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
