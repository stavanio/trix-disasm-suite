"""Filter arms for the bayonet task.

Variant S: box and projection, both static.
Variant P: four arms, three of which receive the phase signal, so the
phase-aware box isolates context information from set geometry.
"""

import math

import numpy as np

from benchmark.margin import sigma_for
from benchmark.disasm_bench import BASE_FORCE_NOISE, BASE_TORQUE_NOISE
from envs import bayonet_env as B

MARGIN_SIGMA = sigma_for(n_constraints=2, horizon=B.CAP)
INTERIOR = 1e-3

M_FORCE = MARGIN_SIGMA * BASE_FORCE_NOISE
M_TORQUE = MARGIN_SIGMA * BASE_TORQUE_NOISE


def _bounds(variant, released):
    a = max(0.0, B.axial_bound(variant, released) - M_FORCE)
    t = max(0.0, B.E_TAU - M_TORQUE)
    l = max(0.0, B.LAT_LIMIT - M_FORCE)
    return a, t, l


def _lateral(a2, l_lim):
    lim = (l_lim / B.S_LAT) * (1.0 - INTERIOR)
    return float(np.clip(a2, -lim, lim))


def box_static(obs, a, released=None):
    """Inscribed box: half-width limit/sqrt(2) on each envelope axis."""
    ax, tq, lt = _bounds("s", True)
    k = (1.0 - INTERIOR) / math.sqrt(2.0)
    return np.array([
        np.clip(float(a[0]), -ax * k / B.S_AXIAL, ax * k / B.S_AXIAL),
        np.clip(float(a[1]), -tq * k / B.S_TAU, tq * k / B.S_TAU),
        _lateral(float(a[2]), lt)], dtype=np.float32), {}


def _project(a, ax, tq, lt):
    """Radial scaling onto the ellipse, then the lateral interval."""
    fa, tv = float(a[0]) * B.S_AXIAL, float(a[1]) * B.S_TAU
    r = math.hypot(fa / ax, tv / tq) if ax > 0 and tq > 0 else float("inf")
    s = (1.0 - INTERIOR) / r if r > (1.0 - INTERIOR) else 1.0
    return np.array([fa * s / B.S_AXIAL, tv * s / B.S_TAU,
                     _lateral(float(a[2]), lt)], dtype=np.float32)


def projection_static(obs, a, released=None):
    ax, tq, lt = _bounds("s", True)
    return _project(a, ax, tq, lt), {}


def _released(obs, released):
    return bool(obs[6] > 0.5) if released is None else bool(released)


def box_blind_conservative(obs, a, released=None):
    """Rotating-leg axial limit in every phase."""
    ax, tq, lt = _bounds("p", False)
    k = (1.0 - INTERIOR) / math.sqrt(2.0)
    return np.array([
        np.clip(float(a[0]), -ax * k / B.S_AXIAL, ax * k / B.S_AXIAL),
        np.clip(float(a[1]), -tq * k / B.S_TAU, tq * k / B.S_TAU),
        _lateral(float(a[2]), lt)], dtype=np.float32), {}


def box_blind_permissive(obs, a, released=None):
    """Release-leg axial limit in every phase."""
    ax, tq, lt = _bounds("p", True)
    k = (1.0 - INTERIOR) / math.sqrt(2.0)
    return np.array([
        np.clip(float(a[0]), -ax * k / B.S_AXIAL, ax * k / B.S_AXIAL),
        np.clip(float(a[1]), -tq * k / B.S_TAU, tq * k / B.S_TAU),
        _lateral(float(a[2]), lt)], dtype=np.float32), {}


def box_phase_aware(obs, a, released=None):
    """Gated axial limit, componentwise. Same phase signal as projection."""
    rel = _released(obs, released)
    ax, tq, lt = _bounds("p", rel)
    k = (1.0 - INTERIOR) / math.sqrt(2.0)
    return np.array([
        np.clip(float(a[0]), -ax * k / B.S_AXIAL, ax * k / B.S_AXIAL),
        np.clip(float(a[1]), -tq * k / B.S_TAU, tq * k / B.S_TAU),
        _lateral(float(a[2]), lt)], dtype=np.float32), {"released": rel}


def projection_phase_aware(obs, a, released=None):
    """Gated axial limit, elliptical."""
    rel = _released(obs, released)
    ax, tq, lt = _bounds("p", rel)
    return _project(a, ax, tq, lt), {"released": rel}


def tightened_ratio():
    """Envelope value of a command on the tightened boundary, measured
    against the nominal envelope."""
    ax, tq, _ = _bounds("s", True)
    return min(ax / B.E_AXIAL, tq / B.E_TAU)
