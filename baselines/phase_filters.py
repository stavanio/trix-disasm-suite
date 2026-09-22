"""Safety filter arms for the phase-gated SNAP and CRANK interfaces.

Both admissible sets depend on observed state, so the filter recomputes
them each step rather than projecting onto a fixed region.

SNAP   the pull bound flips with latch phase: F_pull <= F_PULL_SAFE while
       the catch is engaged, actuator limit once it is clear. The
       deflection bound comes from the yield travel and the stiffness,
       which varies between episodes and so must be estimated.

CRANK  the torque and radial constraints act along the tangent and radial
       directions, which rotate with crank angle. The admissible region
       in (F_x, F_y) is therefore a rectangle rotated by theta, and a
       componentwise bound can only represent it when theta is a multiple
       of pi/2.
"""

import math

import numpy as np

from envs import snap_env_v2 as S
from envs import crank_env_v2 as C
from benchmark.disasm_bench import BASE_FORCE_NOISE

from benchmark.margin import sigma_for

# Two feasible sets are distinct. The nominal set is what a deterministic
# projection can satisfy exactly. The environment then adds realisation
# noise, so a command on the nominal boundary violates about half the
# time. The robust interior set tightens each bound by MARGIN_SIGMA
# standard deviations, derived from a target episode-level risk and the
# task horizon since a per-step tail compounds over an episode.
MARGIN_SIGMA = sigma_for()


def noise_margin(sigma=None):
    return (MARGIN_SIGMA if sigma is None else sigma) * BASE_FORCE_NOISE


# ============================================================== SNAP
SN_DEFLECT = S.S_DEFLECT
SN_PULL = S.S_PULL
SN_LAT = S.S_LAT
INTERIOR = 1e-6


def snap_phase(obs):
    """Disengaged when the latch is clear or the part has passed."""
    return bool(obs[7] > 0.5)


def snap_bounds(obs, k_est=S.K_LATCH, sigma=None):
    """Admissible command box for the observed phase."""
    disengaged = snap_phase(obs)
    pull_lim = S.F_CATCH if disengaged else S.F_PULL_SAFE
    pull_hi = max(0.0, pull_lim - noise_margin(sigma)) / SN_PULL
    # deflection that stays inside the yield travel for the estimated
    # stiffness; the +-10% episode variation is covered by a margin
    defl_hi = min(1.0, (k_est * S.DELTA_MAX * 0.85) / SN_DEFLECT)
    lat = max(0.0, S.F_LAT_MAX - noise_margin(sigma)) / SN_LAT
    return (np.array([-defl_hi, -1.0, -lat]),
            np.array([defl_hi, pull_hi, lat]))


def snap_static_clip(obs, a):
    """Static componentwise clip: fixed bounds, phase-blind. Named so the
    comparison is not misleading, since a phase-aware componentwise bound
    is a different and stronger method."""
    lat = S.F_LAT_MAX / SN_LAT
    hi = S.F_CATCH / SN_PULL
    return np.array([np.clip(float(a[0]), -1.0, 1.0),
                     np.clip(float(a[1]), -hi, hi),
                     np.clip(float(a[2]), -lat, lat)],
                    dtype=np.float32), {}


def snap_phase_clip(obs, a, sigma=None):
    """Phase-aware componentwise bound. For SNAP the constraint is
    axis-aligned once the phase is known, so this coincides with the
    projection below; both are reported to keep the hierarchy explicit."""
    return snap_project(obs, a, sigma)


def snap_project(obs, a, sigma=None):
    """Projection onto the phase-dependent robust interior set."""
    lo, hi = snap_bounds(obs, sigma=sigma)
    out = np.clip(np.asarray(a, dtype=np.float64),
                  lo * (1 - INTERIOR), hi * (1 - INTERIOR))
    return out.astype(np.float32), {"disengaged": snap_phase(obs)}


# ============================================================= CRANK
CR_XY = 30.0
CR_Z = 40.0


def crank_frame(obs):
    """Tangent and radial unit directions at the observed crank angle."""
    th = float(obs[0])
    return (np.array([-math.sin(th), math.cos(th)]),
            np.array([math.cos(th), math.sin(th)]))


def crank_tightening_status(obs, sigma=None):
    """Whether each family retains admissible commands at this state.

    Reported rather than clamped, and named for what it is: a Gaussian
    tail gives a chance constraint under an assumed distribution, not an
    absolute guarantee.
    """
    from benchmark.margin import chance_tightened_limit, margin_spec
    rotating = obs[7] < 0.5
    fam = "axial_phase" if rotating else "axial_struct"
    z = sigma if sigma is not None else margin_spec("CRANK", fam)["z_value"]
    m = z * BASE_FORCE_NOISE
    return {
        "axial": chance_tightened_limit(
            C.F_AXIAL_SAFE if rotating else C.F_AXIAL_MAX, m),
        "torque": chance_tightened_limit(C.TORQUE_LIMIT / C.RADIUS, m),
        "radial": chance_tightened_limit(C.RADIAL_LIMIT, m),
        "z_value": float(z),
    }


def crank_bounds(obs, sigma=None):
    """Axial bound depends on whether rotation is complete.

    Where the margin exceeds the nominal limit the bound collapses to
    zero. That is a fallback, not enforcement, and crank_tightening_status
    reports it so results can exclude those configurations from claims
    about robust guarantees.
    """
    rotating = obs[7] < 0.5
    fz_lim = C.F_AXIAL_SAFE if rotating else C.F_AXIAL_MAX
    m = noise_margin(sigma)
    fz = max(0.0, fz_lim - m) / CR_Z
    ft = max(0.0, C.TORQUE_LIMIT / C.RADIUS - m) / CR_XY
    fr = max(0.0, C.RADIAL_LIMIT - m) / CR_XY
    return ft, fr, fz


def crank_static_clip(obs, a, sigma=None):
    """Static componentwise clip in world x and y. Safe only if every axis
    is held to the tightest orientation-dependent bound, which is why it
    loses tangential drive."""
    ft, fr, fz = crank_bounds(obs, sigma)
    lim = min(ft, fr) * (1 - INTERIOR)
    fz = fz * (1 - INTERIOR)
    return np.array([np.clip(float(a[0]), -lim, lim),
                     np.clip(float(a[1]), -lim, lim),
                     np.clip(float(a[2]), -fz, fz)], dtype=np.float32), {}


def crank_project(obs, a, sigma=None):
    """Projection onto the rotated rectangle crossed with the axial
    interval. The constraint is separable in the tangent-radial frame, so
    the projection is a clip in that frame."""
    t, r = crank_frame(obs)
    ft, fr, fz = crank_bounds(obs, sigma)
    f = np.array([float(a[0]), float(a[1])])
    ct, cr = float(f @ t), float(f @ r)
    ct = np.clip(ct, -ft * (1 - INTERIOR), ft * (1 - INTERIOR))
    cr = np.clip(cr, -fr * (1 - INTERIOR), fr * (1 - INTERIOR))
    g = ct * t + cr * r
    st = crank_tightening_status(obs, sigma)
    return np.array([g[0], g[1],
                     np.clip(float(a[2]), -fz * (1 - INTERIOR),
                             fz * (1 - INTERIOR))],
                    dtype=np.float32), {
        "rotating": bool(obs[7] < 0.5),
        "chance_set_infeasible": bool(any(
            v["set_infeasible"] for k, v in st.items() if k != "z_value")),
        "positive_authority_lost": bool(any(
            not v["positive_authority"] for k, v in st.items()
            if k != "z_value")),
        "tightening_status": st}


def crank_oracle_cost(obs, a):
    """Exact linearisation of the most violated constraint at the current
    action. Only one inequality is corrected per call, which is the form
    the cited safety layer uses."""
    t, r = crank_frame(obs)
    ft, fr, fz = crank_bounds(obs)
    f = np.array([float(a[0]), float(a[1])])
    ct, cr = float(f @ t), float(f @ r)
    cands = [(abs(ct) - ft, np.array([t[0], t[1], 0.0]) * np.sign(ct)),
             (abs(cr) - fr, np.array([r[0], r[1], 0.0]) * np.sign(cr)),
             (abs(float(a[2])) - fz,
              np.array([0.0, 0.0, np.sign(float(a[2]))]))]
    c, g = max(cands, key=lambda p: p[0])
    return c - float(g @ np.asarray(a, dtype=np.float64)), g


def crank_adversarial(obs, f_t=13.0, f_r=26.0, f_z=3.0):
    """Nominal policy that rotates the crank while proposing a radial
    component past the limit. Expressed in the rotating frame, so the
    world-frame command sweeps as theta advances and the admissible
    region rotates out from under a static bound."""
    th = float(obs[0])
    tx, ty = -math.sin(th), math.cos(th)
    rx, ry = math.cos(th), math.sin(th)
    return np.array([(tx * f_t + rx * f_r) / CR_XY,
                     (ty * f_t + ry * f_r) / CR_XY,
                     f_z / 40.0])
