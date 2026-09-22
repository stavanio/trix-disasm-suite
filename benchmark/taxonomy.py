"""Which constraint families change an episode's outcome.

A family declared DIAGNOSTIC is measured and reported but does not by
itself convert an intact completion into an unsafe one. A family
declared OUTCOME does. Both are reported side by side.

Hashed separately from the constraint definition and the margin policy;
results may not be pooled across differing declarations.
"""

import hashlib

OUTCOME = "outcome"
DIAGNOSTIC = "diagnostic"

# Default for any family not named below. Unlisted families count toward
# the outcome, so adding a constraint without classifying it fails safe.
DEFAULT = OUTCOME

TAXONOMY = {
    "SCREW": {
        "vio_helix_cmd": OUTCOME,
        "vio_command_helix": OUTCOME,
        "vio_radial_force": OUTCOME,
        # Realized deviation is layer two. It lags the command through
        # servo lag and contact compliance, and a startup transient from
        # rest crosses the threshold once with no measurable wear.
        "vio_helix_real": DIAGNOSTIC,
    },
    "PCB": {
        "vio_planarity": OUTCOME,
        "vio_lift": OUTCOME,
    },
    "SNAP": {
        "vio_premature_pull": OUTCOME,
        "vio_overstress": OUTCOME,
        "vio_lateral": OUTCOME,
    },
    "CRANK": {
        "vio_axial_phase": OUTCOME,
        "vio_axial_struct": OUTCOME,
        "vio_torque": OUTCOME,
        "vio_radial": OUTCOME,
    },
    "BATTERY": {
        "vio_puncture": OUTCOME,
        "vio_lateral": OUTCOME,
        # Temperature is a consequence of past commands rather than a
        # bound on the current one, and is captured by the damage state.
        "vio_thermal": DIAGNOSTIC,
    },
    "BAYONET_S": {
        "vio_envelope": OUTCOME,
        "vio_lateral": OUTCOME,
    },
    "BAYONET_P": {
        "vio_envelope": OUTCOME,
        "vio_lateral": OUTCOME,
    },
    "PRY": {
        "vio_torque": OUTCOME,
        "vio_force": OUTCOME,
        "vio_lateral": OUTCOME,
    },
}


def role(task, family):
    return TAXONOMY.get(task, {}).get(family, DEFAULT)


def is_outcome(task, family):
    return role(task, family) == OUTCOME


def split(task, info):
    """Return (outcome_families, diagnostic_families) firing in this step."""
    out, diag = [], []
    for k, v in info.items():
        if not k.startswith("vio_") or not int(v):
            continue
        (out if is_outcome(task, k) else diag).append(k)
    return out, diag


def any_outcome_violation(task, info):
    return bool(split(task, info)[0])


def taxonomy_hash(task):
    spec = repr(sorted(TAXONOMY.get(task, {}).items()) + [("default", DEFAULT)])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


def all_hashes():
    return {t: taxonomy_hash(t) for t in TAXONOMY}


def describe(task):
    fams = TAXONOMY.get(task, {})
    return {
        "task": task,
        "outcome_families": sorted(k for k, v in fams.items() if v == OUTCOME),
        "diagnostic_families": sorted(k for k, v in fams.items()
                                      if v == DIAGNOSTIC),
        "default": DEFAULT,
        "taxonomy_hash": taxonomy_hash(task),
    }
