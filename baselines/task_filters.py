"""Safety filter arms for the BATTERY and PRY interfaces.

BATTERY  the flagged limit is puncture force at 20 N, but deformation
         begins at 15 N, so a filter that enforces only the stated
         criterion intervenes too late to guarantee an intact cell. Both
         bounds are provided so the difference is measurable rather than
         assumed. The lateral axes carry a norm constraint, not a
         per-axis one.

PRY      the torque bound depends on insertion depth, so the admissible
         set is axis-aligned but state-dependent and must be recomputed
         each step.
"""

import math

import numpy as np

from envs import battery_env_v2 as B
from envs import pry_env_v2 as P
from benchmark.margin import sigma_for
from benchmark.disasm_bench import (BASE_FORCE_NOISE, BASE_TORQUE_NOISE,
                                    FORCE_LIMIT, RADIAL_LIMIT)

INTERIOR = 1e-6

# Derived from a target episode-level risk and the task horizon rather
# than fixed, since a per-step tail compounds over an episode. The target,
# horizon and derived value are recorded with each result.
MARGIN_SIGMA = sigma_for()


def _margin(sigma=None, base=BASE_FORCE_NOISE):
    return (MARGIN_SIGMA if sigma is None else sigma) * base


# ============================================================ BATTERY
def battery_lateral_norm(a):
    return float(np.hypot(float(a[1]), float(a[2])))


def battery_box_clip(obs, a, sigma=None):
    """Per-axis bounds only. Cannot represent the lateral norm constraint."""
    m = _margin(sigma)
    pk = max(0.0, B.F_PUNCTURE - m) / B.S_PEEL
    lt = max(0.0, B.F_LAT_MAX - m) / B.S_LAT
    return np.array([np.clip(float(a[0]), -pk, pk),
                     np.clip(float(a[1]), -lt, lt),
                     np.clip(float(a[2]), -lt, lt)], dtype=np.float32), {}


def _battery_project(obs, a, peel_lim, sigma):
    m = _margin(sigma)
    pk = max(0.0, peel_lim - m) / B.S_PEEL * (1 - INTERIOR)
    lt = max(0.0, B.F_LAT_MAX - m) / B.S_LAT * (1 - INTERIOR)
    x = float(np.clip(float(a[0]), -pk, pk))
    y, z = float(a[1]), float(a[2])
    n = math.hypot(y, z)
    if n > lt:
        s = lt / n
        y, z = y * s, z * s
    return np.array([x, y, z], dtype=np.float32)


def battery_box_clip_preventive(obs, a, sigma=None):
    """Componentwise clipping at the SAME preventive threshold TRiX uses.

    Without this the only axis-aligned BATTERY arm enforced the 20 N
    puncture limit while the projection arm enforced the 15 N deformation
    onset, so that comparison varied both the filter form and the
    specification and could attribute the difference to neither. This
    holds the specification fixed and varies only the form.

    The comparison is a real geometry test rather than a formality: the
    projection arm scales the lateral PAIR radially, enforcing a norm,
    while this clips each lateral axis independently. That is the same
    disk-against-box structure as PCB.
    """
    from benchmark.margin import chance_tightened_limit
    m = _margin(sigma)
    pk = max(0.0, B.F_DEFORM - m) / B.S_PEEL
    lt = max(0.0, B.F_LAT_MAX - m) / B.S_LAT
    st = {"peel": chance_tightened_limit(B.F_DEFORM, m),
          "lateral": chance_tightened_limit(B.F_LAT_MAX, m)}
    return np.array([np.clip(float(a[0]), -pk, pk),
                     np.clip(float(a[1]), -lt, lt),
                     np.clip(float(a[2]), -lt, lt)], dtype=np.float32), {
        "chance_set_infeasible": bool(any(v["set_infeasible"]
                                          for v in st.values())),
        "tightening_status": st}


def battery_project_flagged(obs, a, sigma=None):
    """Enforces the criterion the manuscript states: puncture force."""
    return _battery_project(obs, a, B.F_PUNCTURE, sigma), {}


def battery_project_preventive(obs, a, sigma=None):
    """Enforces the deformation onset instead, which is where the damage
    the puncture limit is meant to prevent actually begins."""
    from benchmark.margin import chance_tightened_limit
    m = _margin(sigma)
    st = {"peel": chance_tightened_limit(B.F_DEFORM, m),
          "lateral": chance_tightened_limit(B.F_LAT_MAX, m)}
    return _battery_project(obs, a, B.F_DEFORM, sigma), {
        "chance_set_infeasible": bool(any(v["set_infeasible"]
                                          for v in st.values())),
        "tightening_status": st}


# ================================================================ PRY
def pry_static_clip(obs, a, sigma=None):
    """Fixed torque bound, ignoring insertion depth."""
    mt = _margin(sigma, BASE_TORQUE_NOISE)
    mf = _margin(sigma)
    tq = max(0.0, P.dynamic_torque_limit(0.0) - mt) / 2.5
    fi = max(0.0, FORCE_LIMIT - mf) / 80.0
    fl = max(0.0, RADIAL_LIMIT - mf) / 40.0
    return np.array([np.clip(float(a[0]), -tq, tq),
                     np.clip(float(a[1]), -fi, fi),
                     np.clip(float(a[2]), -fl, fl)], dtype=np.float32), {}


def pry_project(obs, a, sigma=None):
    """Depth-aware bound.

    Information timing: the filter uses the observed insertion and the
    insertion force it is about to command, both available at decision
    time. It does not read post-step state. The prediction is needed
    because the environment advances insertion before applying the
    depth-dependent torque limit, so the bound in force at check time is
    tighter than the one implied by the entry observation.
    """
    from benchmark.disasm_bench import DT
    mt = _margin(sigma, BASE_TORQUE_NOISE)
    mf = _margin(sigma)
    fi = max(0.0, FORCE_LIMIT - mf) / 80.0 * (1 - INTERIOR)
    f_ins = min(abs(float(a[1])), fi) * 80.0 + mf
    ins_next = min(P.MAX_INSERTION, float(obs[4]) + f_ins * DT * 0.0001)
    lim = P.dynamic_torque_limit(ins_next)
    tq = max(0.0, lim - mt) / 2.5 * (1 - INTERIOR)
    fl = max(0.0, RADIAL_LIMIT - mf) / 40.0 * (1 - INTERIOR)
    return np.array([np.clip(float(a[0]), -tq, tq),
                     np.clip(float(a[1]), -fi, fi),
                     np.clip(float(a[2]), -fl, fl)], dtype=np.float32), \
        {"torque_limit": lim}


# ---------------------------- oracle tangent cost models ----------------
# Exact linearisation of the most violated constraint at the current
# action. Where a task's admissible set is axis-aligned the tangent
# coincides with exact projection on that axis, so the two arms agree by
# construction; the comparison only separates them where a constraint
# couples axes, which here means BATTERY's lateral force norm.

def battery_oracle_cost(obs, a, sigma=None):
    """Peel force is axis-aligned; the lateral pair carries a norm."""
    m = _margin(sigma)
    pk = max(0.0, B.F_PUNCTURE - m) / B.S_PEEL
    lt = max(0.0, B.F_LAT_MAX - m) / B.S_LAT
    x = np.asarray(a, dtype=np.float64)
    n = math.hypot(float(x[1]), float(x[2]))
    c_peel = abs(float(x[0])) - pk
    c_lat = n - lt
    if c_lat >= c_peel:
        if n < 1e-12:
            return -lt, np.zeros(3)
        g = np.array([0.0, float(x[1]) / n, float(x[2]) / n])
        return -lt, g
    s = 1.0 if x[0] >= 0 else -1.0
    return -pk, np.array([s, 0.0, 0.0])


def pry_oracle_cost(obs, a, sigma=None):
    """Depth-dependent torque bound plus two axis-aligned force bounds."""
    from benchmark.disasm_bench import DT
    mt = _margin(sigma, BASE_TORQUE_NOISE)
    mf = _margin(sigma)
    fi = max(0.0, FORCE_LIMIT - mf) / 80.0
    x = np.asarray(a, dtype=np.float64)
    f_ins = min(abs(float(x[1])), fi) * 80.0 + mf
    ins_next = min(P.MAX_INSERTION, float(obs[4]) + f_ins * DT * 0.0001)
    tq = max(0.0, P.dynamic_torque_limit(ins_next) - mt) / 2.5
    fl = max(0.0, RADIAL_LIMIT - mf) / 40.0
    lims = (tq, fi, fl)
    over = [abs(float(x[i])) - lims[i] for i in range(3)]
    i = int(np.argmax(over))
    g = np.zeros(3)
    g[i] = 1.0 if x[i] >= 0 else -1.0
    return -lims[i], g


def snap_oracle_cost(obs, a, sigma=None):
    """Phase-dependent bounds, axis-aligned once the phase is known."""
    from baselines.phase_filters import snap_bounds
    lo, hi = snap_bounds(obs, sigma=sigma)
    x = np.asarray(a, dtype=np.float64)
    over = [max(float(x[i]) - hi[i], lo[i] - float(x[i])) for i in range(3)]
    i = int(np.argmax(over))
    g = np.zeros(3)
    g[i] = 1.0 if float(x[i]) >= 0 else -1.0
    lim = hi[i] if float(x[i]) >= 0 else -lo[i]
    return -lim, g
