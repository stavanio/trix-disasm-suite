"""The pre-registration record must be consistent and self-enforcing."""
import sys
sys.path.insert(0, '.')
from benchmark import protocol as P
from benchmark import selection as SEL
from benchmark.registry import TASKS as ALL_TASKS

# The freeze covers the original six. The prospective bayonet test
# carries its own record with different decision rules.
TASKS = tuple(t for t in ALL_TASKS if not t.startswith('BAYONET'))


def test_1_every_task_has_a_frozen_status():
    for t in TASKS:
        assert t in P.TASK_STATUS, f"{t} has no frozen status"
        for claim in ("safety_enforcement", "capability_preservation",
                      "learning_enablement"):
            assert claim in P.TASK_STATUS[t], f"{t} missing {claim}"
        assert P.TASK_STATUS[t].get("note"), f"{t} needs a stated reason"


def test_2_crank_is_excluded_from_enforcement_claims():
    """Its tightened axial set is empty at baseline, so no enforcement
    claim is admissible; the geometry result needs no margin."""
    assert not P.may_support("CRANK", "safety_enforcement")
    assert P.may_support("CRANK", "capability_preservation")


def test_3_snap_does_not_claim_capability_preservation():
    """Filtering cannot supply the missing preparatory deflection. That is
    the finding, not a failure to hide."""
    assert P.may_support("SNAP", "safety_enforcement")
    assert not P.may_support("SNAP", "capability_preservation")


def test_4_pry_enablement_requires_the_volume_matched_control():
    rule = P.CLAIM_RULES["learning_enablement"]
    assert "volume-matched" in rule
    assert "geometric attribution" in rule


def test_5_mdp_framing_is_declared_for_both_modes():
    for mode in ("filter_active_during_training", "frozen_policy_evaluation"):
        d = P.MDP_FRAMING[mode]
        for k in ("framing", "policy_action_space", "replay_stores",
                  "valid_because"):
            assert d.get(k), f"{mode} missing {k}"
    assert P.MDP_FRAMING["filter_active_during_training"]["replay_stores"] \
        == "nominal action"


def test_6_protocol_matches_the_selection_module():
    assert P.PROTOCOL["checkpoint_every"] == SEL.CHECKPOINT_EVERY
    assert P.PROTOCOL["screen_episodes"] == SEL.SCREEN_EPISODES
    assert P.PROTOCOL["confirm_episodes"] == SEL.CONFIRM_EPISODES
    assert P.PROTOCOL["test_episodes"] == SEL.TEST_EPISODES
    assert P.PROTOCOL["production_seeds"] == list(SEL.PRODUCTION_SEEDS)


def test_7_budgets_are_identical_across_methods_within_a_task():
    assert P.PROTOCOL["budget_identical_across_methods_within_task"]
    for t in TASKS:
        assert P.BUDGETS[t] > 0


def test_8_primary_endpoint_is_not_conditioned_on_completion():
    ep = P.PROTOCOL["primary_endpoint"]
    assert "never conditioned on completion" in ep


def test_9_sensitivity_claims_necessity_not_sufficiency():
    s = P.SENSITIVITY_INTERPRETATION
    assert "necessary preconditions, not" in s["not_supported"]
    assert s["stratification_decided"].startswith("after")


def test_10_candidate_checkpoints_are_explicit():
    """The set is part of the search space: screening fewer checkpoints
    changes which policy is selected, so it cannot be left implicit."""
    for task, ck in P.CANDIDATE_CHECKPOINTS.items():
        assert ck, f"{task} has no candidate checkpoints"
        assert ck[0] == P.PROTOCOL["checkpoint_every"]
        assert ck[-1] == P.BUDGETS[task]
        assert len(ck) == P.BUDGETS[task] // P.PROTOCOL["checkpoint_every"]
    assert P.PROTOCOL["screen_all_candidates"] is True


def test_11_freeze_hash_changes_when_a_decision_changes():
    base = P.freeze_hash()
    saved = P.TASK_STATUS["CRANK"]["safety_enforcement"]
    P.TASK_STATUS["CRANK"]["safety_enforcement"] = True
    try:
        assert P.freeze_hash() != base, "flipping a status left the hash"
    finally:
        P.TASK_STATUS["CRANK"]["safety_enforcement"] = saved
    assert P.freeze_hash() == base


def _order(name):
    return int(name.split('_')[1])


if __name__ == '__main__':
    for name in sorted((k for k in globals() if k.startswith('test_')),
                       key=_order):
        globals()[name](); print(f"PASS {name}")
    print(f"\nfreeze hash: {P.freeze_hash()}  frozen on {P.FROZEN_ON}")
