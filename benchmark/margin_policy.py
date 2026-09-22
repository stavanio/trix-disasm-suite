"""Where robust margins apply, and where they deliberately do not.

A margin is warranted only when stochastic realization separates the
filter output from the constraint check. Adding one elsewhere shrinks the
admissible set without reducing any modelled risk.

SCREW's command-manifold condition is evaluated on the projected command
itself, with no noise injected between projection and check, so it takes
numerical tolerance only. Its realized and contact constraints sit
downstream of servo dynamics, saturation and contact, and carry derived
margins. The other tasks check noisy realized actions directly.

The policy is hashed separately from the environment constraint hash:
the environment defines what counts as a violation, the policy defines
how conservatively a filter aims inside it. Both must match before
results are pooled.
"""

import hashlib

from benchmark.margin import sigma_for

NONE = "none"
DERIVED = "episode_risk_derived"


def _derived(n_constraints=1, horizon=None, delta=None):
    kw = {"n_constraints": n_constraints}
    if horizon is not None:
        kw["horizon"] = horizon
    if delta is not None:
        kw["delta"] = delta
    return {"mode": DERIVED, "sigma": sigma_for(**kw),
            "n_constraints": n_constraints}


POLICY = {
    "SCREW": {
        "command_helix": {
            "mode": NONE,
            "reason": "evaluated directly on the noiseless projected "
                      "command; no realization step separates filter "
                      "output from check",
        },
        "command_bounds": {
            "mode": NONE,
            "reason": "commanded velocities are applied without added "
                      "noise; the radial axis is clipped inside its limit",
        },
        "realized_helix": {
            "mode": DERIVED, **_derived(n_constraints=2, horizon=900),
            "reason": "downstream of servo lag, saturation and contact",
        },
    },
    "PCB": {
        "planarity": {"mode": DERIVED, **_derived(n_constraints=2, horizon=300),
                      "reason": "checked on the noisy realized torque"},
        "lift": {"mode": DERIVED, **_derived(n_constraints=2, horizon=300),
                 "reason": "checked on the noisy realized force"},
    },
    "SNAP": {
        "premature_pull": {"mode": DERIVED, **_derived(n_constraints=3,
                                                       horizon=700),
                           "reason": "checked on the noisy realized force"},
        "overstress": {"mode": DERIVED, **_derived(n_constraints=3,
                                                   horizon=700),
                       "reason": "deflection driven by a noisy force"},
        "lateral": {"mode": DERIVED, **_derived(n_constraints=3, horizon=700),
                    "reason": "checked on the noisy realized force"},
    },
    "CRANK": {
        "axial_phase": {"mode": DERIVED, **_derived(n_constraints=4,
                                                    horizon=700),
                        "reason": "checked on the noisy realized force"},
        "axial_struct": {"mode": DERIVED, **_derived(n_constraints=4,
                                                     horizon=700),
                         "reason": "checked on the noisy realized force"},
        "torque": {"mode": DERIVED, **_derived(n_constraints=4, horizon=700),
                   "reason": "torque derived from noisy planar forces"},
        "radial": {"mode": DERIVED, **_derived(n_constraints=4, horizon=700),
                   "reason": "checked on the noisy realized force"},
    },
    "BATTERY": {
        "puncture": {"mode": DERIVED, **_derived(n_constraints=3,
                                                 horizon=1000),
                     "reason": "checked on the noisy realized peel force"},
        "lateral": {"mode": DERIVED, **_derived(n_constraints=3,
                                                horizon=1000),
                    "reason": "checked on the noisy realized force norm"},
        "thermal": {
            "mode": NONE,
            "reason": "a state consequence of past commands rather than a "
                      "bound on the current one; no instantaneous margin "
                      "reduces it",
        },
    },
    "PRY": {
        "torque": {"mode": DERIVED, **_derived(n_constraints=3, horizon=1200),
                   "reason": "checked on the noisy realized torque against a "
                             "depth-dependent bound"},
        "insertion_force": {"mode": DERIVED, **_derived(n_constraints=3,
                                                        horizon=1200),
                            "reason": "checked on the noisy realized force"},
        "lateral": {"mode": DERIVED, **_derived(n_constraints=3,
                                                horizon=1200),
                    "reason": "checked on the noisy realized force"},
    },
    "BAYONET_S": {
        "envelope": {"mode": DERIVED, **_derived(n_constraints=2,
                                                 horizon=2500),
                     "reason": "checked on realized axial force and torque, "
                               "both carrying noise added after the filter"},
        "lateral": {"mode": DERIVED, **_derived(n_constraints=2,
                                                horizon=2500),
                    "reason": "checked on the noisy realized force"},
    },
    "BAYONET_P": {
        "envelope": {"mode": DERIVED, **_derived(n_constraints=2,
                                                 horizon=2500),
                     "reason": "checked on realized axial force and torque "
                               "against a phase-gated bound"},
        "lateral": {"mode": DERIVED, **_derived(n_constraints=2,
                                                horizon=2500),
                    "reason": "checked on the noisy realized force"},
    },
}


def policy_for(task):
    return POLICY[task]


def policy_hash(task):
    """Covers both the margin values and the decision not to apply one."""
    p = POLICY[task]
    spec = repr(sorted(
        (k, v["mode"], round(v.get("sigma", 0.0), 9), v.get("n_constraints", 0))
        for k, v in p.items()))
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


def all_hashes():
    return {t: policy_hash(t) for t in POLICY}
