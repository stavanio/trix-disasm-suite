"""Decisions frozen before the production matrix runs.

Everything here was settled from pilot and control data, on seeds that
never appear in reported results. Changing any of it after seeing
production numbers would reopen exactly the post-hoc criticism the
revision exists to answer, so the record is hashed and dated: the hash
appears in every production result, and a later edit makes that visible
rather than silent.

Four things are frozen.

  1. Which task configurations may support which claims.
  2. The claim-eligibility rules themselves.
  3. Where each filter sits relative to the MDP.
  4. Budgets, selection, seeds and failure handling.

The provenance audit of CRANK's 5 N axial limit is deliberately NOT a
blocker. Its possible outcomes are bounded and none of them touches the
other five tasks: if the limit and the noise estimate both stand, the
finding is chance-constraint infeasibility; if either is unsupported,
that is a new environment version and only CRANK cells rerun.
"""

import hashlib
import json

FROZEN_ON = "2026-08-02"

# --- 1. what each task configuration is allowed to support ---------------
# A configuration excluded from one claim can still carry another. CRANK's
# tightened axial set is empty at baseline, so it cannot speak to
# enforcement, but its geometry result needs no margin at all.
TASK_STATUS = {
    "SCREW": {
        "safety_enforcement": True,
        "capability_preservation": True,
        "learning_enablement": True,
        "note": "scripted feasibility gate passed at three speeds",
    },
    "PCB": {
        "safety_enforcement": True,
        "capability_preservation": True,
        "learning_enablement": True,
        "note": "coupled disk; the clean demonstration",
    },
    "SNAP": {
        "safety_enforcement": True,
        "capability_preservation": False,
        "learning_enablement": False,
        "note": "sequential limitation: filtering cannot supply the "
                "missing preparatory deflection, so capability is not "
                "preserved and that is the finding, not a failure",
    },
    "CRANK": {
        "safety_enforcement": False,
        "capability_preservation": True,
        "learning_enablement": False,
        "note": "the chance-tightened axial set is EMPTY at baseline "
                "(margin 5.54 N against a 5.00 N limit, max feasible "
                "z = 4.05 against a required 4.488), so no enforcement "
                "claim is admissible; the static-box-stalls result needs "
                "no margin and stands. No learned arm completed the task",
    },
    "BATTERY": {
        "safety_enforcement": True,
        "capability_preservation": True,
        "learning_enablement": False,
        "note": "BATTERY supports only that a damage threshold and a "
                "preventive threshold are not interchangeable. It does "
                "NOT support the representation-form claim until the "
                "matched-specification arm reports. "
                "the preventive arm is the primary configuration; the "
                "flagged 20 N arm is a specification-layer ablation, not "
                "a competing version of the algorithm",
    },
    "PRY": {
        "safety_enforcement": True,
        "capability_preservation": False,
        "learning_enablement": True,
        "note": "no learner completes it unfiltered, so a frozen policy "
                "cannot support capability preservation; enablement is "
                "supported but attributed to restriction, not geometry",
    },
}

# --- 2. what evidence each claim requires --------------------------------
CLAIM_RULES = {
    "environment_feasibility":
        "a scripted controller completes the task safely through the "
        "identical projector, margins and episode cap",
    "safety_enforcement":
        "the tightened set is nonempty, and the filtered arm shows lower "
        "destructive completion than its comparator on held-out episodes",
    "capability_preservation":
        "the same frozen nominal policy is competent BEFORE filtering, so "
        "any loss of completion is attributable to the filter",
    "learning_enablement":
        "filtered training beats unfiltered reproducibly across seeds, "
        "AND a volume-matched restriction with the wrong geometry does "
        "not achieve the same effect, before any geometric attribution",
    "task_learnability":
        "permitted when a scripted controller succeeds but no evaluated "
        "learner does; reported as a benchmark result, never as evidence "
        "about the filter",
}

# --- 3. where each filter sits relative to the MDP -----------------------
# SB3 stores the action the policy emitted, whatever the environment then
# does with it. Under filter-as-environment that tuple is correct: the
# policy's action space IS the nominal command space and the critic learns
# the value of PROPOSING a command. Verified on PRY, where 91% of stored
# actions would still be altered by the filter.
MDP_FRAMING = {
    "filter_active_during_training": {
        "framing": "filter-as-environment",
        "policy_action_space": "nominal command",
        "environment": "projector plus plant",
        "replay_stores": "nominal action",
        "valid_because": "the policy is solving the projected MDP, so the "
                         "stored tuple is the transition it actually took",
        "invalid_for": "claims about a safety layer bolted onto a policy "
                       "trained elsewhere",
    },
    "frozen_policy_evaluation": {
        "framing": "filter as an evaluation-time governor",
        "policy_action_space": "nominal command",
        "environment": "plant only during training",
        "replay_stores": "not applicable, no training",
        "valid_because": "every arm receives identical nominal actions "
                         "from identical states, so the comparison is "
                         "exactly paired",
    },
}

# --- 4. budgets, selection, seeds, failure handling ----------------------
BUDGETS = {t: 15000 for t in TASK_STATUS}

# The candidate checkpoint set is part of the search space, not an
# implementation detail: screening fewer checkpoints changes the chance of
# finding a narrow peak and therefore which policy is selected. Given the
# peak-and-collapse behaviour measured on SNAP and SCREW, every checkpoint
# is screened. Naming the set here closes an underspecification that would
# otherwise let a runtime shortcut silently alter results.
CANDIDATE_CHECKPOINTS = {
    t: list(range(1000, BUDGETS[t] + 1, 1000)) for t in BUDGETS
}

PROTOCOL = {
    "training_budget_steps": BUDGETS,
    "budget_identical_across_methods_within_task": True,
    "candidate_checkpoints": CANDIDATE_CHECKPOINTS,
    "screen_all_candidates": True,
    "checkpoint_every": 1000,
    "screen_episodes": 30,
    "confirm_episodes": 100,
    "test_episodes": 100,
    "production_seeds": list(range(10)),
    "pilot_seeds": [900, 901, 902],
    "selection_rule": "centre of the best contiguous 3-checkpoint window "
                      "by mean safe completion, then lexicographic: max "
                      "safe completion, min destructive, max progress, "
                      "min episode violation incidence, earliest step",
    "primary_endpoint": "safe completion rate over ALL held-out test "
                        "episodes, never conditioned on completion",
    "experimental_unit": "training seed",
    "failure_handling": "a cell that raises is recorded as failed with "
                        "its traceback and excluded from aggregates; it "
                        "is never silently retried or replaced",
    "competence_threshold_completion_pct": 50.0,
}

# --- what the sensitivity sweep does and does not establish --------------
SENSITIVITY_INTERPRETATION = {
    "supported":
        "every observed failure occurred where the enforced limit "
        "exceeded modelled damage onset, or the uncertainty tightening "
        "emptied the admissible set",
    "not_supported":
        "that a projection filter is SAFE whenever those two conditions "
        "hold; the sweep identifies necessary preconditions, not "
        "sufficiency against omitted dynamics, an incorrect damage "
        "model, sequencing failures or unmodelled constraints",
    "paper_wording":
        "exact projection cannot compensate for an incorrectly specified "
        "safety boundary, and probabilistic tightening may itself make a "
        "nominally valid command set infeasible",
    "stratification_decided": "after the sweep, from its results",
}


# Amendments made after the original freeze. Each records WHY, so the
# distinction between a correction and a convenient change is auditable.
AMENDMENTS = [
    {
        "date": "2026-09-04",
        "change": "clarification only; no declaration altered",
        "note": "The CRANK entry in TASK_STATUS states that no learned "
                "arm acquired a completing policy. That statement was "
                "written from Stage 1, where every CRANK arm read 0.0, "
                "and predates Stage 2. Filter-aware training later gave "
                "CRANK/sac/trix 49.4% safe completion over ten seeds. "
                "The original declaration is preserved as written; this "
                "entry records that it is superseded by Stage 2 for the "
                "filter-aware execution mode only.",
        "scope": "documentation; no protocol parameter changed",
    },
    {
        "date": "2026-08-02",
        "change": "added the BATTERY box_clip_preventive arm",
        "reason": "the completed analysis revealed that the original "
                  "BATTERY arms enforced DIFFERENT specifications: "
                  "box_clip used the 20 N puncture limit while "
                  "trix_preventive used the 15 N deformation onset. That "
                  "comparison varied both the filter form and the "
                  "threshold, so it could not attribute the difference to "
                  "either. The new arm holds the specification fixed at "
                  "15 N and varies only the form.",
        "not_because": "the numerical result was unfavourable; the "
                       "original comparison favoured the projection",
        "scope": "BATTERY only; no other task is affected or rerun",
        "original_records_retained_as":
            "specification ablation, 20 N box against 15 N projection",
    },
]


def freeze_hash():
    spec = json.dumps({"frozen_on": FROZEN_ON, "task_status": TASK_STATUS,
                       "claim_rules": CLAIM_RULES, "mdp": MDP_FRAMING,
                       "protocol": PROTOCOL, "amendments": AMENDMENTS},
                      sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


def may_support(task, claim):
    return bool(TASK_STATUS.get(task, {}).get(claim, False))


def describe():
    return {"frozen_on": FROZEN_ON, "freeze_hash": freeze_hash(),
            "task_status": TASK_STATUS, "claim_rules": CLAIM_RULES,
            "mdp_framing": MDP_FRAMING, "protocol": PROTOCOL,
            "sensitivity_interpretation": SENSITIVITY_INTERPRETATION}
