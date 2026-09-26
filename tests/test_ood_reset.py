"""Guard the scientific boundary: shifted resets, frozen governors and metrics."""
import copy
import importlib
import json

import numpy as np
import pytest

from benchmark import evaluation_distribution as ED
from benchmark import run_record as RR
from benchmark.gym_adapter import make
from benchmark.registry import ENVS, get_filter
from benchmark.metrics import EpisodeRecorder
from experiments.ood_reset import current_hashes
from training import sb3_runner as R


def rng_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


@pytest.mark.parametrize("task", ED.PARAMETERS)
def test_native_path_preserves_entire_short_trajectory(task):
    """Explicit native distribution must reproduce pre-interface physics/noise."""
    def trajectory(explicit):
        env = make(task, evaluation_distribution=ED.specification(task) if explicit else None)
        obs, _ = env.reset(seed=98123)
        result = [(obs.tolist(),)]
        action = np.array([0.15, 0.2, 0.01])
        for _ in range(20):
            obs, reward, done, trunc, info = env.step(action)
            result.append((obs.tolist(), reward, done, trunc, info))
            if done or trunc:
                break
        return result, np.random.get_state()
    old, old_rng = trajectory(False)
    new, new_rng = trajectory(True)
    encode = lambda value: json.dumps(value, sort_keys=True,
                                      default=lambda a: a.tolist())
    assert encode(old) == encode(new)
    assert rng_equal(old_rng, new_rng)


@pytest.mark.parametrize("task", ED.PARAMETERS)
@pytest.mark.parametrize("condition", ED.CONDITIONS)
def test_shift_changes_only_instance_parameters_and_leaves_noise_and_governor(task, condition):
    env = make(task)
    obs, _ = env.reset(seed=1200100)
    instance = copy.deepcopy(vars(env.unwrapped_task))
    mod = importlib.import_module(ENVS[task].__module__)
    constants = {k: v for k, v in vars(mod).items() if k.isupper() and isinstance(v, (str,int,float,bool))}
    hashes = current_hashes(task)
    rng = np.random.get_state()
    filt = get_filter(task, "trix_preventive" if task == "BATTERY" else "trix")
    probe = np.array([0.8, -0.6, 0.2])
    before, _ = filt(obs, probe)
    spec = ED.specification(task, condition)
    details = ED.apply_after_native_reset(env.unwrapped_task, spec, 1200100)
    assert rng_equal(rng, np.random.get_state())
    assert current_hashes(task) == hashes
    assert constants == {k: vars(mod)[k] for k in constants}
    after, _ = filt(env.unwrapped_task._get_obs(), probe)
    np.testing.assert_array_equal(before, after)
    fields = {p["field"] for p in spec["parameters"]}
    for key, value in instance.items():
        if key not in fields:
            np.testing.assert_equal(vars(env.unwrapped_task)[key], value)
    assert details["parameters"] == ED.sample(spec, 1200100)[0]


def test_shell_has_no_in_range_samples_and_screw_joint_is_physically_ordered():
    for task in ED.PARAMETERS:
        for condition in ED.CONDITIONS:
            spec = ED.specification(task, condition)
            for seed in range(100):
                values, _ = ED.sample(spec, seed)
                assert all(p["evaluation_bounds"][0] <= values[p["field"]] <= p["evaluation_bounds"][1]
                           for p in spec["parameters"])
                if task == "SCREW":
                    assert values["mu_s"] >= values["mu_k"]
                if condition == "shell_2":
                    assert ED.outside_training_support(spec, values)


def test_pry_and_undeclared_fields_are_rejected():
    with pytest.raises(ValueError):
        ED.specification("PRY", "wide_2")
    spec = ED.specification("PCB", "wide_2")
    spec["parameters"][0]["field"] = "THETA_YIELD"
    with pytest.raises(ValueError):
        make("PCB", evaluation_distribution=spec)
    with pytest.raises(ValueError):
        make("PCB", evaluation_distribution=ED.specification("SNAP", "wide_2"))


def make_record(condition):
    spec = ED.specification("PCB", condition)
    hashes = current_hashes("PCB")
    return RR.make_record("PCB", "sac+trix", 0,
        {"constraint_hash": hashes["constraint_hash"], "counts": {}},
        hashes["constraint_hash"], training_mode="nominal",
        evaluation_arm="trix", algorithm="sac", evaluation_distribution=spec)


def test_distribution_hash_prevents_silent_pooling_or_missing_provenance():
    native = make_record("native")
    wide = make_record("wide_2")
    shell = make_record("shell_2")
    RR.check_comparable([native, copy.deepcopy(native)])
    RR.check_comparable([wide, copy.deepcopy(wide)])
    for a, b in ((native, wide), (wide, shell)):
        with pytest.raises(RR.HashMismatch):
            RR.check_comparable([a, b])
    for field in ("evaluation_distribution", "evaluation_distribution_hash"):
        broken = copy.deepcopy(native)
        del broken[field]
        with pytest.raises(RR.HashMismatch):
            RR.check_comparable([broken])
    altered = copy.deepcopy(wide)
    altered["evaluation_distribution_hash"] = native["evaluation_distribution_hash"]
    with pytest.raises(RR.HashMismatch):
        RR.check_comparable([altered])


def test_runner_records_actual_distribution_and_each_draw(monkeypatch):
    class ConstantModel:
        def predict(self, obs, deterministic):
            return np.zeros(3), None
    monkeypatch.setitem(R.CAPS, "PCB", 2)
    spec = ED.specification("PCB", "shell_2")
    recorder = EpisodeRecorder("PCB", "sac+trix", 0)
    R.evaluate_frozen(ConstantModel(), "PCB", "trix", [123, 124], recorder,
                      evaluation_distribution=spec)
    assert recorder.validate()["counts"]["episodes"] == 2
    assert [e["evaluation_reset"]["episode_seed"] for e in recorder.episodes] == [123,124]
    assert all(e["evaluation_reset"]["outside_training_support"] for e in recorder.episodes)
    assert all(e["evaluation_reset"]["evaluation_distribution_hash"] == ED.canonical_hash(spec)
               for e in recorder.episodes)
    with pytest.raises(ValueError):
        R.evaluate_frozen(ConstantModel(), "PCB", "trix", [125], recorder,
                          evaluation_distribution=spec)
