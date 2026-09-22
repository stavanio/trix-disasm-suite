"""Robust margin sizing.

Three distinct sets:

    nominal feasible    what a deterministic projection satisfies
    robust interior     the nominal set tightened by a margin
    realised criterion  what the environment evaluates, on the noisy
                        command

For independent per-step noise and horizon N,

    P(any violation) = 1 - (1 - p_step)^N

so a 3 sigma per-step tail compounds to about 0.79 over 1156 steps. The
margin is sized from a target episode risk delta and the task horizon,
both recorded with the results. Independence is an assumption; the
measured sweep is what the paper reports.
"""

import math

from scipy.stats import norm

DEFAULT_DELTA = 0.01     # target probability that an episode contains any
                         # noise-induced violation
DEFAULT_HORIZON = 1200   # steps; set per task from measured episode length


def per_step_tail(delta=DEFAULT_DELTA, horizon=DEFAULT_HORIZON):
    """Per-step tail probability consistent with an episode-level target."""
    if horizon <= 0:
        return delta
    return 1.0 - (1.0 - delta) ** (1.0 / horizon)


def sigma_for(delta=DEFAULT_DELTA, horizon=DEFAULT_HORIZON, two_sided=False,
              n_constraints=1):
    """Gaussian quantile giving that tail, with a Bonferroni allowance for
    several constraint families and for two-sided bounds."""
    p = per_step_tail(delta, horizon) / max(1, n_constraints)
    if two_sided:
        p *= 0.5
    p = min(max(p, 1e-15), 0.5 - 1e-15)
    return float(norm.ppf(1.0 - p))


def describe(delta=DEFAULT_DELTA, horizon=DEFAULT_HORIZON, two_sided=False,
             n_constraints=1):
    return {
        "delta": delta,
        "horizon": horizon,
        "two_sided": two_sided,
        "n_constraints": n_constraints,
        "p_step": per_step_tail(delta, horizon) / max(1, n_constraints),
        "sigma": sigma_for(delta, horizon, two_sided, n_constraints),
    }


def chance_tightened_limit(nominal_limit, margin, tol=1e-12):
    """Tighten a limit by an uncertainty allowance, with exact semantics.

    Three distinct states, which a single boolean would conflate:

        remaining > 0    a usable interval
        remaining == 0   the singleton {0}: mathematically nonempty, but
                         no nonzero command is admissible
        remaining < 0    genuinely empty; the requested tightening cannot
                         be represented

    Clamping a negative result to zero would silently turn the third case
    into the second, which looks like conservative enforcement and is not.

    The tightening is CHANCE-CONSTRAINED, not robust. The allowance comes
    from a Gaussian tail, and Gaussian noise is unbounded, so no finite
    margin gives an absolute guarantee. It targets a stated episode-level
    risk under an assumed distribution, and the name says so.
    """
    remaining = float(nominal_limit) - float(margin)
    return {
        "nominal_limit": float(nominal_limit),
        "margin": float(margin),
        "tightened_limit": float(max(0.0, remaining)),
        "remaining_authority": float(remaining),
        "set_nonempty": bool(remaining >= -tol),
        "positive_authority": bool(remaining > tol),
        "set_infeasible": bool(remaining < -tol),
    }


def max_feasible_sigma(nominal_limit, sigma_noise):
    """Largest z for which the tightened set on this constraint is nonempty."""
    if sigma_noise <= 0:
        return float("inf")
    return float(nominal_limit) / float(sigma_noise)


def margin_spec(task, family):
    """The single authoritative record for one family's allowance.

    Declaration and implementation previously computed z separately and
    disagreed (4.304 against 4.488 on CRANK). Both now read this.
    """
    from benchmark import margin_policy as MP
    d = MP.policy_for(task)[family]
    if d["mode"] == MP.NONE:
        return {"task": task, "family": family, "mode": MP.NONE,
                "z_value": 0.0, "reason": d.get("reason")}
    return {
        "task": task, "family": family, "mode": d["mode"],
        "target_episode_risk": DEFAULT_DELTA,
        "tail": "one-sided",
        "allocation": "bonferroni",
        "constraint_families": d.get("n_constraints"),
        "z_value": d["sigma"],
    }
