"""Protocol for the prospective bayonet test.

Separate from the six-task freeze: the predictions, decision bands and
primary algorithm differ, and the original record stands as evidence
that the six-task results were preregistered before they ran.

    preregistration          9d91891
    amendment 1              49ea367
    implementation freeze    b9d72a2
    criterion                1.1
"""

import hashlib
import json

FROZEN_ON = "2026-08-05"

# Amendment: production_seeds aliases training_seeds, and TASK_STATUS was
# added, both for runner compatibility. The hash moved from
# e56ffbe4343192dc to 686273f0caad3a10. No prediction, decision band, arm,
# seed list or budget changed, and no result had been read.
AMENDED_ON = "2026-08-05"

BUDGETS = {"BAYONET_S": 15000, "BAYONET_P": 15000}
CANDIDATE_CHECKPOINTS = {t: list(range(1000, BUDGETS[t] + 1, 1000))
                         for t in BUDGETS}

CONSTRAINT_HASHES = {"BAYONET_S": "d8dd709fd5a55436",
                     "BAYONET_P": "77b8eeda360dbd82"}

ARMS = {
    "BAYONET_S": ["none", "box_clip", "trix"],
    "BAYONET_P": ["none", "box_blind_conservative", "box_blind_permissive",
                  "box_phase_aware", "trix"],
}

TASK_STATUS = {t: {"note": "prospective test"} for t in BUDGETS}

PRIMARY_ALGO = "sac"
SECONDARY_ALGO = "ppo"

# phase-aware projection against the phase-blind conservative box
CONFIRMATORY = {"BAYONET_S": ("trix", "box_clip"),
                "BAYONET_P": ("trix", "box_blind_conservative")}

# phase-aware projection against the phase-aware box; no directional
# prediction, and a null here does not falsify the criterion
MECHANISM = {"BAYONET_P": ("trix", "box_phase_aware")}

PREDICTION = {
    ("BAYONET_S", "nominal"): "unspecified",
    ("BAYONET_S", "filter_aware"): "no material advantage",
    ("BAYONET_P", "nominal"): "material advantage",
    ("BAYONET_P", "filter_aware"): "material advantage",
}

MATERIAL = {"mean_pp": 25.0, "median_pp": 20.0, "seeds_positive": 7}
NO_ADVANTAGE = {"mean_pp": 10.0, "median_pp": 10.0, "seeds_within": 7}
COMPETENCE_COMPLETION_PCT = 50.0

PROTOCOL = {
    "training_seeds": list(range(10)),
    "production_seeds": list(range(10)),
    "checkpoint_every": 1000,
    "screen_episodes": 30,
    "confirm_episodes": 100,
    "test_episodes": 100,
    "primary_endpoint": "safe completion over all held-out test episodes",
    "experimental_unit": "training seed",
}


def classify(mean_pp, median_pp, per_seed):
    """Apply the frozen three-part rule to paired per-seed differences."""
    n = len(per_seed)
    pos = sum(1 for d in per_seed if d > 0)
    within = sum(1 for d in per_seed if abs(d) <= NO_ADVANTAGE["median_pp"])
    if (mean_pp >= MATERIAL["mean_pp"]
            and median_pp >= MATERIAL["median_pp"]
            and pos >= MATERIAL["seeds_positive"]):
        return "material advantage"
    if (abs(mean_pp) <= NO_ADVANTAGE["mean_pp"]
            and abs(median_pp) <= NO_ADVANTAGE["median_pp"]
            and within >= NO_ADVANTAGE["seeds_within"]):
        return "no material advantage"
    return "indeterminate"


def freeze_hash():
    spec = json.dumps({"frozen_on": FROZEN_ON, "budgets": BUDGETS,
                       "arms": ARMS, "confirmatory": CONFIRMATORY,
                       "mechanism": MECHANISM, "prediction":
                       {f"{k[0]}/{k[1]}": v for k, v in PREDICTION.items()},
                       "material": MATERIAL, "no_advantage": NO_ADVANTAGE,
                       "protocol": PROTOCOL,
                       "hashes": CONSTRAINT_HASHES}, sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:16]
