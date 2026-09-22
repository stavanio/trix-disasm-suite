"""Metrics module tests against hand-constructed episode traces.

No environment is instantiated: a fake exposing only episode_summary()
stands in, so these tests exercise the aggregation contract itself.
"""
import sys
import numpy as np
sys.path.insert(0, '.')
from benchmark.metrics import (EpisodeRecorder, SeedAggregator,
                               format_summary, format_seed_result)
from benchmark.outcome import classify, CATEGORIES


class FakeEnv:
    """Minimal stand-in emitting only the three primitives."""
    def __init__(self, damage="intact", violated=False, completion=None):
        self.damage = damage
        self.violated = violated
        self._completion = completion

    def episode_summary(self, completed):
        completion = self._completion or ("completed" if completed else "timeout")
        return {
            "completion": completion,
            "damage_outcome": self.damage,
            "episode_with_violation": self.violated,
            "safe_completion": classify(completion, self.damage,
                                        self.violated) == "safe_completion",
            "constraint_hash": "deadbeefdeadbeef",
        }


def _trace(rec, n_steps, vio_steps=(), families=("vio_alpha",), reward=1.0,
           exceed=None, filt=None):
    rec.start_episode()
    for t in range(1, n_steps + 1):
        v = int(t in vio_steps)
        info = {"vio": v, "constraint_hash": "deadbeefdeadbeef"}
        for f in families:
            info[f] = v
            if exceed is not None:
                info["exceed_" + f[len("vio_"):]] = exceed if v else 0.0
        rec.step(info, reward=reward, filter_info=filt)


def test_1_terminal_partition_is_exact():
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 10); rec.end_episode(FakeEnv(), completed=True)
    _trace(rec, 10, vio_steps={3}); rec.end_episode(
        FakeEnv(damage="degraded", violated=True), completed=True)
    _trace(rec, 10); rec.end_episode(FakeEnv(), completed=False)
    _trace(rec, 10); rec.end_episode(
        FakeEnv(completion="failed", damage="stripped"), completed=False)
    s = rec.validate()
    c = s["counts"]
    assert (c["safe_completions"], c["destructive_completions"],
            c["mechanical_failures"], c["timeouts"]) == (1, 1, 1, 1)
    assert c["safe_completions"] + c["destructive_completions"] \
        + c["mechanical_failures"] + c["timeouts"] == s["episodes"]


def test_2_violating_completed_episode_is_never_safe():
    """A violation always disqualifies a safe completion. With no damage
    the episode is unsafe-but-intact, not destructive: collapsing it into
    destructive would overstate harm."""
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 20, vio_steps={5})
    rec.end_episode(FakeEnv(damage="intact", violated=True), completed=True)
    s = rec.validate()
    assert s["safe_completion_rate"] == 0.0
    assert s["unsafe_completion_intact_rate"] == 100.0
    assert s["destructive_completion_rate"] == 0.0
    assert s["completion_rate"] == 100.0


def test_3_rate_ordering_invariants():
    rec = EpisodeRecorder("T", "M")
    for i in range(5):
        _trace(rec, 10, vio_steps={2} if i % 2 else set())
        rec.end_episode(FakeEnv(violated=bool(i % 2),
                                damage="degraded" if i % 2 else "intact"),
                        completed=True)
    s = rec.validate()
    assert s["safe_completion_rate"] <= s["completion_rate"]
    assert s["destructive_completion_rate"] <= s["completion_rate"]


def test_4_episode_and_timestep_weighting_differ():
    """One violation in a short episode and many in a long one must not
    be conflated: the two weightings are reported separately."""
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 2, vio_steps={1, 2})          # short, all violating
    rec.end_episode(FakeEnv(violated=True, damage="stripped"), completed=True)
    _trace(rec, 200, vio_steps={7})           # long, one violation
    rec.end_episode(FakeEnv(violated=True, damage="degraded"), completed=True)
    s = rec.validate()
    assert s["episodes_with_any_violation_rate"] == 100.0
    assert abs(s["violating_timestep_rate"] - 100.0 * 3 / 202) < 1e-9
    assert s["counts"]["violating_steps"] == 3
    assert s["counts"]["total_steps"] == 202


def test_5_per_constraint_attribution():
    """Two methods with the same total violation rate can fail in
    different ways; families are reported individually."""
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 10, vio_steps={1, 2}, families=("vio_alpha",))
    rec.end_episode(FakeEnv(violated=True, damage="degraded"), completed=True)
    _trace(rec, 10, vio_steps={5, 6}, families=("vio_beta",))
    rec.end_episode(FakeEnv(violated=True, damage="degraded"), completed=True)
    s = rec.validate()
    fam = s["violations_by_constraint"]
    assert set(fam) == {"vio_alpha", "vio_beta"}
    assert fam["vio_alpha"]["violating_steps"] == 2
    assert fam["vio_beta"]["episodes_affected"] == 1


def test_6_first_violation_step_and_exceedance():
    """Severity conditional on violation must not be diluted by the long
    safe stretches of an episode."""
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 50, vio_steps={10, 11}, exceed=0.25)
    rec.end_episode(FakeEnv(violated=True, damage="degraded"), completed=True)
    _trace(rec, 50, vio_steps={20}, exceed=0.75)
    rec.end_episode(FakeEnv(violated=True, damage="degraded"), completed=True)
    s = rec.validate()
    assert s["first_violation_step_mean"] == 15.0
    # 3 violating steps out of 100: all-steps mean is heavily diluted
    assert abs(s["exceedance_mean_all_steps"] - (0.25*2 + 0.75)/100) < 1e-9
    # conditional mean reflects the actual excursions
    assert abs(s["exceedance_mean_violating_steps"] - (0.25*2 + 0.75)/3) < 1e-9
    assert s["exceedance_max"] == 0.75
    fam = s["violations_by_constraint"]["vio_alpha"]
    assert abs(fam["exceedance_mean_violating_steps"] - (0.25*2+0.75)/3) < 1e-9


def test_7_filter_statistics():
    rec = EpisodeRecorder("T", "M")
    rec.start_episode()
    for t in range(100):
        rec.step({"vio": 0}, reward=0.0,
                 filter_info={"intervention_norm": 0.1 * (t % 10),
                              "latency_us": 5.0 + t,
                              "infeasible": t < 5, "recovery": t < 2})
    rec.end_episode(FakeEnv(), completed=True)
    s = rec.validate()
    assert s["projection_latency_us_p50"] is not None
    assert s["projection_latency_us_p99"] >= s["projection_latency_us_p95"] \
        >= s["projection_latency_us_p50"]
    assert abs(s["infeasibility_rate"] - 5.0) < 1e-9
    assert abs(s["recovery_event_rate"] - 2.0) < 1e-9


def test_8_missing_terminal_fields_are_rejected():
    """The evaluator must not infer terminal state when an environment
    fails to supply it."""
    class Bad:
        def episode_summary(self, completed):
            return {"completion": "completed"}
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 5)
    try:
        rec.end_episode(Bad(), completed=True)
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected ValueError for incomplete contract")


def test_9_timeout_is_not_failure():
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 30); rec.end_episode(FakeEnv(), completed=False)
    s = rec.validate()
    assert s["timeout_rate"] == 100.0
    assert s["mechanical_failure_rate"] == 0.0
    assert s["completion_rate"] == 0.0


def test_10_five_categories_partition_exactly():
    rec = EpisodeRecorder("T", "M")
    rec.start_episode(); rec.step({"vio": 0}); rec.end_episode(FakeEnv(), True)
    _trace(rec, 5, vio_steps={2}); rec.end_episode(
        FakeEnv(violated=True, damage="intact"), True)          # unsafe-intact
    _trace(rec, 5); rec.end_episode(FakeEnv(damage="degraded"), True)
    _trace(rec, 5); rec.end_episode(FakeEnv(), False)
    _trace(rec, 5); rec.end_episode(FakeEnv(completion="failed"), False)
    s = rec.validate()
    c = s["counts"]
    assert (c["safe_completions"], c["unsafe_completions_intact"],
            c["destructive_completions"], c["timeouts"],
            c["mechanical_failures"]) == (1, 1, 1, 1, 1)


def test_11_damage_without_violation_is_destructive():
    """Reachable under continuous wear: an episode may finish outside the
    intact band without ever tripping a constraint."""
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 20); rec.end_episode(
        FakeEnv(damage="degraded", violated=False), completed=True)
    s = rec.validate()
    assert s["destructive_completion_rate"] == 100.0
    assert s["safe_completion_rate"] == 0.0
    assert s["episodes_with_any_violation_rate"] == 0.0


def test_12_near_miss_is_not_destructive():
    rec = EpisodeRecorder("T", "M")
    _trace(rec, 20, vio_steps={3}); rec.end_episode(
        FakeEnv(damage="intact", violated=True), completed=True)
    s = rec.validate()
    assert s["unsafe_completion_intact_rate"] == 100.0
    assert s["destructive_completion_rate"] == 0.0
    assert s["safe_completion_rate"] == 0.0


def test_13_seed_aggregation_separates_seeds_from_episodes():
    """Three seeds of 100 episodes is n=3 experimental units, not n=300."""
    agg = SeedAggregator("T", "M")
    for seed, n_safe in enumerate([80, 90, 100]):
        rec = EpisodeRecorder("T", "M", seed=seed)
        for i in range(100):
            _trace(rec, 5, vio_steps=set() if i < n_safe else {2})
            rec.end_episode(FakeEnv(violated=i >= n_safe,
                                    damage="intact"), completed=True)
        agg.add(rec.validate())
    r = agg.validate()
    assert r["seeds"] == 3
    ss = r["seed_summary"]["safe_completion_rate"]
    assert ss["n_seeds"] == 3
    assert abs(ss["mean"] - 90.0) < 1e-9
    assert ss["sd"] is not None and ss["sd"] > 0
    # pooled counts are exact, not an average of rates
    assert r["pooled_counts"]["episodes"] == 300
    assert r["pooled_counts"]["safe_completions"] == 270
    assert abs(r["pooled_rates"]["safe_completion_rate"] - 90.0) < 1e-9


def test_14_seeds_with_different_constraint_definitions_are_rejected():
    agg = SeedAggregator("T", "M")
    rec = EpisodeRecorder("T", "M", seed=0)
    _trace(rec, 5); rec.end_episode(FakeEnv(), True)
    s = rec.summary(); agg.add(s)
    s2 = dict(s); s2["constraint_hash"] = "0000000000000000"; agg.add(s2)
    try:
        agg.result()
    except ValueError as e:
        assert "constraint definitions" in str(e)
    else:
        raise AssertionError("expected rejection of mismatched hashes")


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith('test_')}.items()):
        fn(); print(f"PASS {name}")
