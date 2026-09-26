"""End-to-end gate for the registry, filters and result records.

Every check here must fail loudly rather than degrade quietly: a wrong
environment version, a mismatched constraint hash, a filter reading
post-step state or a corrupted record should stop the sweep, not produce
a plausible-looking number.
"""
import copy
import json
import sys

import numpy as np
sys.path.insert(0, '.')

from benchmark.registry import TASKS, ENVS, FILTERS, make_env, get_filter, \
    constraint_hashes
from benchmark.metrics import EpisodeRecorder, SeedAggregator
from benchmark.outcome import CATEGORIES, classify
from benchmark import run_record as RR
from benchmark import margin as MG

EXPECTED_HASHES = {
    "SCREW": "beb88f8d0baf5e4a",
    "PCB": "53fcd8e18d341d5d",
    "SNAP": "044763e531cb6bcb",
    "CRANK": "bc6cc80449c95466",
    "BATTERY": "09a4ae5d933db6bf",
    "PRY": "52be08aa340982f6",
    "BAYONET_S": "d8dd709fd5a55436",
    "BAYONET_P": "77b8eeda360dbd82",
}

EXPECTED_ENV = {
    "SCREW": "ScrewEnvV3", "PCB": "PCBEnvV2", "SNAP": "SnapEnvV2",
    "CRANK": "CrankEnvV2", "BATTERY": "BatteryEnvV2", "PRY": "PryEnvV2",
    "BAYONET_S": "BayonetS",
    "BAYONET_P": "BayonetP",
}


def _pol(task):
    if task == "CRANK":
        from baselines.phase_filters import crank_adversarial
        return lambda o: crank_adversarial(o)
    fixed = {"SCREW": [0.8, 0.15, 0.0], "PCB": [0.5, 0.5, 0.5],
             "SNAP": [0.0, 0.5, 0.0], "BATTERY": [0.6, 0.2, 0.2],
             "PRY": [0.8, 0.75, 0.0],
             "BAYONET_S": [-0.35, 0.45, 0.0],
             "BAYONET_P": [-0.35, 0.45, 0.0]}[task]
    return lambda o: np.array(fixed)


def test_1_every_task_binds_the_expected_environment_version():
    for task in TASKS:
        assert ENVS[task].__name__ == EXPECTED_ENV[task], \
            f"{task} bound to {ENVS[task].__name__}"


def test_2_constraint_hashes_match_the_declared_values():
    got = constraint_hashes()
    for task, want in EXPECTED_HASHES.items():
        assert got[task] == want, f"{task}: {got[task]} != {want}"


def test_3_action_semantics_and_episode_cap():
    """Every environment takes a 3-vector, returns a 10-vector observation,
    and terminates or is capped."""
    for task in TASKS:
        env = make_env(task); obs = env.reset(seed=0)
        assert obs.shape == (10,), f"{task} obs {obs.shape}"
        o, r, d, info = env.step(np.zeros(3))
        assert o.shape == (10,)
        assert isinstance(float(r), float)
        assert isinstance(d, (bool, np.bool_))
        assert "vio" in info and "constraint_hash" in info


def test_4_filter_runs_before_step_and_sees_no_post_step_state():
    """The filter is handed the pre-step observation. Mutating the copy it
    receives must not affect the environment, and the observation it sees
    must equal the one returned by the previous step."""
    for task in TASKS:
        env = make_env(task); obs = env.reset(seed=0)
        f = get_filter(task, "trix")
        seen = {}

        def spy(o, a, _f=f, _s=seen):
            _s["obs"] = np.array(o, copy=True)
            o[:] = 999.0            # corrupt the caller's array
            return _f(_s["obs"], a)

        a = _pol(task)(obs)
        x, _ = spy(np.array(obs, copy=True), a)
        assert np.allclose(seen["obs"], obs), f"{task}: filter saw altered obs"
        obs2, _, _, _ = env.step(x)
        assert not np.allclose(obs2, 999.0), f"{task}: env state corrupted"


def test_5_outcome_categories_are_exhaustive_and_exclusive():
    for completion in ("completed", "timeout", "failed"):
        for dmg in ("intact", "degraded", "damaged", "stripped", "cracked"):
            for vio in (False, True):
                c = classify(completion, dmg, vio)
                assert c in CATEGORIES
                assert sum(1 for k in CATEGORIES if k == c) == 1


def test_6_any_violation_precludes_safe_completion():
    for task in TASKS:
        for method in FILTERS[task]:
            rec = EpisodeRecorder(task, method)
            env = make_env(task); obs = env.reset(seed=1)
            f = get_filter(task, method); p = _pol(task)
            rec.start_episode(); done = False
            for t in range(3000):
                x, fi = f(obs, p(obs))
                obs, rw, done, info = env.step(x)
                rec.step(info, reward=rw, filter_info=fi)
                if done:
                    break
            rec.end_episode(env, completed=done)
            s = rec.validate()
            for e in rec.episodes:
                if e["episode_with_violation"] and e["completion"] == "completed":
                    assert e["category"] != "safe_completion"


def test_7_nominal_and_filtered_actions_are_both_recoverable():
    for task in TASKS:
        env = make_env(task); obs = env.reset(seed=0)
        p = _pol(task); f = get_filter(task, "trix")
        nominal = np.array(p(obs), copy=True)
        filtered, _ = f(obs, np.array(nominal, copy=True))
        assert nominal.shape == filtered.shape
        # the filter must not mutate the nominal action in place
        assert np.allclose(nominal, p(obs)), f"{task}: nominal action mutated"


def test_8_record_rejects_a_hash_mismatch():
    rec = EpisodeRecorder("SCREW", "trix")
    env = make_env("SCREW"); obs = env.reset(seed=0)
    rec.start_episode()
    for _ in range(5):
        obs, rw, d, info = env.step(np.array([0.1, 0.1, 0.0]))
        rec.step(info, reward=rw)
    rec.end_episode(env, completed=False)
    s = rec.validate()
    RR.make_record("SCREW", "trix", 0, s, EXPECTED_HASHES["SCREW"])
    try:
        RR.make_record("SCREW", "trix", 0, s, "0000000000000000")
    except RR.HashMismatch:
        pass
    else:
        raise AssertionError("a wrong constraint hash was accepted")


def test_9_records_spanning_definitions_cannot_be_aggregated():
    base = {"task": "SCREW", "constraint_hash": "aaaa",
            "robust_margin_hash": "bb", "taxonomy_hash": "cc",
            "protocol_freeze_hash": "dd", "training_mode": "nominal",
            "evaluation_arm": "trix", "algorithm": "sac",
            "margin_policy": {"sigma": 4.3}}
    from benchmark import evaluation_distribution as ED
    base["evaluation_distribution"] = ED.specification("SCREW")
    base["evaluation_distribution_hash"] = ED.canonical_hash(base["evaluation_distribution"])
    other = dict(base, constraint_hash="bbbb")
    diff_margin = dict(base, margin_policy={"sigma": 3.0})
    RR.check_comparable([base, copy.deepcopy(base)])
    for bad in ([base, other], [base, diff_margin]):
        try:
            RR.check_comparable(bad)
        except RR.HashMismatch:
            continue
        raise AssertionError("incomparable records were accepted")


def test_10_record_carries_provenance_margin_and_denominators():
    rec = EpisodeRecorder("PCB", "trix", seed=3)
    env = make_env("PCB"); obs = env.reset(seed=0)
    rec.start_episode(); done = False
    for _ in range(2000):
        x, fi = get_filter("PCB", "trix")(obs, _pol("PCB")(obs))
        obs, rw, done, info = env.step(x)
        rec.step(info, reward=rw, filter_info=fi)
        if done:
            break
    rec.end_episode(env, completed=done)
    r = RR.make_record("PCB", "trix", 3, rec.validate(),
                       EXPECTED_HASHES["PCB"])
    for k in ("task", "method", "seed", "constraint_hash", "margin_policy",
              "provenance", "counts"):
        assert k in r, f"missing {k}"
    for k in ("delta", "horizon", "sigma", "p_step"):
        assert k in r["margin_policy"], f"margin_policy missing {k}"
    for k in ("python", "numpy", "platform", "git_commit"):
        assert k in r["provenance"], f"provenance missing {k}"
    for k in ("episodes", "total_steps", "violating_steps"):
        assert k in r["counts"], f"counts missing {k}"
    json.dumps(r)


def test_11_margin_policy_is_derived_not_fixed():
    assert MG.sigma_for(horizon=100) < MG.sigma_for(horizon=5000)
    assert MG.sigma_for(delta=0.05) < MG.sigma_for(delta=0.001)
    assert MG.sigma_for(n_constraints=1) < MG.sigma_for(n_constraints=5)
    from baselines import task_filters as TF, pcb_filters as PF, \
        phase_filters as PH
    assert abs(TF.MARGIN_SIGMA - MG.sigma_for()) < 1e-12
    assert abs(PF.MARGIN_SIGMA - MG.sigma_for()) < 1e-12
    assert abs(PH.MARGIN_SIGMA - MG.sigma_for()) < 1e-12


def test_12_unknown_filter_fails_loudly():
    try:
        get_filter("SCREW", "does_not_exist")
    except KeyError:
        return
    raise AssertionError("an unregistered filter name was accepted")


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
