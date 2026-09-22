"""A predictive criterion for safety-governor adequacy.

Three gates, applied in order. The criterion predicts whether a
task-structured projection will materially outperform a generic
axis-aligned restriction, from the structure of the constraint and the
task alone, without reference to measured outcomes.

    Gate 1, specification validity.
        Is the enforced boundary preventive rather than diagnostic, at or
        below the modelled damage onset, and does the uncertainty
        tightening leave a set with positive authority? If not, no filter
        geometry can rescue the outcome and the comparison is void.

    Gate 2, representational adequacy.
        Let the true admissible set be A(c) for a governing context c
        (phase, frame, contact state). Does the baseline receive c, and
        can its functional form express A(c)?

    Gate 3, adaptation escape.
        Even where the baseline is imperfect, does a successful policy
        exist entirely within the actions the baseline safely preserves?
        If so, filter-aware training makes the mismatch irrelevant.

The prediction:

    Task-structured projection materially improves outcomes when the
    specification is valid, the baseline cannot represent the
    context-indexed admissible set, AND the resulting representational
    loss removes task-essential actions that policy adaptation cannot
    avoid.

This is a criterion, not a law. It is stated here so it can be frozen
before the held-out task is implemented, and so a failed prediction is
visible rather than absorbable.
"""

import hashlib
import json

CRITERION_VERSION = "1.1"
FROZEN_ON = "2026-08-05"

# Revision 1.1, made after the criterion retrodicted 4/5 and BEFORE any
# held-out task was implemented. Gate 1 was applied at task level, which
# voided CRANK entirely because its phase-gated axial family has an empty
# tightened set. But CRANK's observed advantage rests on the torque and
# radial families in the rotating frame, whose specifications are valid.
# Gate 1 is therefore per constraint family: an invalid family voids
# comparisons that depend on THAT family, not the whole task.
REVISIONS = [
    {"version": "1.1", "date": "2026-08-05",
     "change": "gate 1 evaluated per constraint family rather than per task",
     "reason": "CRANK's axial_phase family has an empty tightened set, but "
               "the measured advantage derives from the rotating-frame "
               "torque and radial families, which are validly specified",
     "prompted_by": "retrodiction failure on CRANK, 4/5",
     "before_held_out_task": True},
]

# --- Gate 1: is the specification valid? ---------------------------------
SPECIFICATION = {
    "SCREW": {"preventive": True, "at_or_below_damage_onset": True,
              "tightening_has_positive_authority": True,
              "note": "command manifold is exact; realized layer is "
                      "diagnostic and declared as such"},
    "PCB": {"preventive": True, "at_or_below_damage_onset": True,
            "tightening_has_positive_authority": True,
            "note": "limit coincides with damage onset at baseline: "
                    "TAU_TILT_MAX = K_BEND * THETA_YIELD"},
    "SNAP": {"preventive": True, "at_or_below_damage_onset": True,
             "tightening_has_positive_authority": True},
    # Per family. The comparison in question rests on the families listed
    # in DECIDING_FAMILIES below; other families may be invalid without
    # voiding it.
    "CRANK": {"preventive": True, "at_or_below_damage_onset": True,
              "tightening_has_positive_authority": True,
              "invalid_families": ["axial_phase"],
              "note": "the phase-gated axial family has an EMPTY tightened "
                      "set (sigma 1.235 N against a 5.00 N limit gives max "
                      "feasible z = 4.05 against a required 4.488) and is "
                      "excluded from enforcement claims. The torque and "
                      "radial families, on which the rotating-frame "
                      "comparison rests, are validly specified"},
    "BATTERY_flagged": {"preventive": False,
                        "at_or_below_damage_onset": False,
                        "tightening_has_positive_authority": True,
                        "note": "20 N puncture limit sits above the 15 N "
                                "deformation onset"},
    "BATTERY": {"preventive": True, "at_or_below_damage_onset": True,
                "tightening_has_positive_authority": True,
                "note": "15 N deformation onset"},
    "PRY": {"preventive": True, "at_or_below_damage_onset": True,
            "tightening_has_positive_authority": True},
}

# --- Gate 2: can the baseline represent the context-indexed set? ---------
# Which constraint families the filter-form comparison depends on.
DECIDING_FAMILIES = {
    "SCREW": ["helix_cmd"],
    "PCB": ["planarity", "lift"],
    "SNAP": ["premature_pull"],
    "CRANK": ["torque", "radial"],
    "BATTERY": ["puncture", "lateral"],
    "PRY": ["torque"],
}

REPRESENTATION = {
    "SCREW": {"context": None, "set_shape": "slab in command space",
              "baseline": "none registered", "expressible": None},
    "PCB": {"context": None, "set_shape": "disk x interval",
            "baseline": "componentwise box", "expressible": False,
            "note": "a box cannot exactly represent a disk"},
    "SNAP": {"context": "latch phase", "set_shape": "phase-conditioned box",
             "baseline": "static box", "expressible": False,
             "note": "the baseline does not receive the phase"},
    "CRANK": {"context": "crank angle", "set_shape": "rotating rectangle",
              "baseline": "world-frame box", "expressible": False,
              "note": "a fixed frame cannot track a rotating set"},
    "BATTERY": {"context": None, "set_shape": "interval x lateral disk",
                "baseline": "componentwise box", "expressible": False,
                "note": "a box cannot exactly represent the lateral norm"},
    "PRY": {"context": "insertion depth", "set_shape": "depth-scaled box",
            "baseline": "static box", "expressible": False,
            "note": "the baseline does not receive the depth"},
}

# --- Gate 3: can adaptation escape the mismatch? -------------------------
ADAPTATION = {
    "SCREW": {"lost_authority_essential": None,
              "note": "no axis-aligned baseline registered"},
    "PCB": {"lost_authority_essential": False,
            "note": "the box's inscribed region still contains tilt and "
                    "lift commands sufficient to complete the task; only "
                    "the disk's corners are lost"},
    "SNAP": {"lost_authority_essential": True,
             "note": "the disengaged phase's larger pull bound IS the "
                     "extraction action; a static bound either forbids it "
                     "or permits it while engaged"},
    "CRANK": {"lost_authority_essential": True,
              "note": "tangential authority along the rotating direction "
                      "is what turns the crank; a fixed box costs a factor "
                      "sqrt(2) in every direction"},
    "BATTERY": {"lost_authority_essential": False,
                "note": "peel proceeds on the axial axis; the lateral "
                        "corners are not required"},
    "PRY": {"lost_authority_essential": False,
            "note": "the depth-dependent torque bound tightens by half "
                    "over full insertion, but a successful policy operates "
                    "at shallow depth where the static bound is adequate"},
}


def predict(task):
    """Apply the three gates. Returns the prediction and the gate that
    decided it, so a wrong prediction is attributable."""
    spec = SPECIFICATION.get(task, {})
    if not spec:
        return {"task": task, "prediction": "unknown", "decided_by": None}

    invalid = set(spec.get("invalid_families", []))
    deciding = set(DECIDING_FAMILIES.get(task, []))
    voids = invalid & deciding if deciding else invalid
    if voids or not (spec.get("preventive")
                     and spec.get("at_or_below_damage_onset")
                     and spec.get("tightening_has_positive_authority")):
        return {"task": task, "prediction": "comparison void",
                "decided_by": "gate 1, specification invalid"
                              + (f" for {sorted(voids)}" if voids else ""),
                "detail": spec.get("note")}

    rep = REPRESENTATION.get(task, {})
    if rep.get("expressible") is None:
        return {"task": task, "prediction": "not applicable",
                "decided_by": "gate 2, no baseline registered"}
    if rep.get("expressible"):
        return {"task": task, "prediction": "no material advantage",
                "decided_by": "gate 2, baseline can express the set"}

    ad = ADAPTATION.get(task, {})
    if ad.get("lost_authority_essential"):
        return {"task": task, "prediction": "material advantage",
                "decided_by": "gate 3, lost authority is task-essential",
                "detail": ad.get("note")}
    return {"task": task, "prediction": "no material advantage",
            "decided_by": "gate 3, adaptation escapes the mismatch",
            "detail": ad.get("note")}


def criterion_hash():
    spec = json.dumps({"version": CRITERION_VERSION, "frozen_on": FROZEN_ON,
                       "specification": SPECIFICATION,
                       "representation": REPRESENTATION,
                       "adaptation": ADAPTATION}, sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


# Measured Stage 2 outcomes, for retrodiction only. Not an input to the
# criterion: predict() never reads this.
OBSERVED = {
    "PCB": "no material advantage",       # box 100.0, trix 99.8
    "SNAP": "material advantage",         # static 5.2, trix 96.5
    "CRANK": "material advantage",        # static 0.0, trix 49.4
    "BATTERY": "no material advantage",   # box 100.0, trix 100.0
    "PRY": "no material advantage",       # static 88.5, trix 89.1
}


if __name__ == "__main__":
    print(f"predictive criterion {CRITERION_VERSION}, "
          f"hash {criterion_hash()}\n")
    print(f"{'task':10s} {'predicted':22s} {'observed':22s} {'decided by'}")
    hits = 0
    for task, obs in OBSERVED.items():
        p = predict(task)
        ok = p["prediction"] == obs
        hits += ok
        print(f"{task:10s} {p['prediction']:22s} {obs:22s} "
              f"{p['decided_by']}")
    print(f"\nretrodicts {hits}/{len(OBSERVED)} tasks")
    if hits < len(OBSERVED):
        print("the criterion does not account for the observed pattern; "
              "it must be revised BEFORE the held-out task, and the "
              "revision recorded")
    else:
        print("the criterion accounts for every observed outcome. It is "
              "retrodictive only until the held-out task reports.")
    print(f"\nBATTERY_flagged (specification ablation): "
          f"{predict('BATTERY_flagged')['prediction']}")
