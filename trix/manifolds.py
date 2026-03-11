"""
Differentiable manifold definitions and projections.

Each projection is a closed-form mathematical operation derived from
first-principles physics (Appendix B).
"""

import numpy as np
import math


def helical_projection(v_z: float, omega_z: float, pitch: float) -> tuple:
    """
    Project (v_z, omega_z) onto the helical manifold (Eq. 6).

    M_helix = {(v_z, omega_z) : v_z = k * omega_z}
    where k = pitch / (2*pi)

    Orthogonal projection onto line through origin with direction (k, 1):
      Psi(u) = (k*v_z + omega_z) / (k^2 + 1) * [k, 1]^T

    Computation: 3 multiplications, 2 additions, 1 division = O(1)
    """
    k = pitch / (2 * np.pi)
    scale = (k * v_z + omega_z) / (k**2 + 1)
    v_safe = k * scale
    omega_safe = scale
    return v_safe, omega_safe


def thermal_projection(F: float, v: float, P_max: float) -> tuple:
    """
    Project (F, v) onto thermal-viscous manifold (Section B.2).

    C_batt = {(F, v) : F * v <= P_max(T)}

    If F*v <= P_max: action is safe, return unchanged.
    If F*v > P_max: scale velocity to satisfy constraint.

    Computation: 1 multiplication, 1 comparison, 1 division = O(1)
    """
    power = F * v
    if power <= P_max:
        return F, v
    # Scale velocity (preserves peel direction)
    v_safe = P_max / max(abs(F), 1e-8) * np.sign(v)
    return F, v_safe


def planar_projection(wrench: np.ndarray, F_frac: float = 30.0) -> np.ndarray:
    """
    Project onto planar invariant manifold (Eq. 8, Section B.3).

    M_pcb = {u in R^6 : tau_x = 0, tau_y = 0, Fz <= F_frac}

    Zeros tilt torques and clips normal force.
    Computation: 1 comparison, 2 assignments = O(1)
    """
    result = wrench.copy()
    result[3] = 0.0  # tau_x = 0
    result[4] = 0.0  # tau_y = 0
    result[2] = min(result[2], F_frac)  # clip normal force
    return result


def tangential_projection(Fx: float, Fy: float, angle: float,
                          F_max: float = 12.0) -> tuple:
    """
    Project force onto tangent direction at given angle.

    Removes radial component that would cause binding in rotational
    extraction tasks.
    """
    tx, ty = -math.sin(angle), math.cos(angle)
    F_tangent = Fx * tx + Fy * ty
    F_tangent = np.clip(F_tangent, -F_max, F_max)
    return F_tangent * tx, F_tangent * ty


def path_projection(Fx: float, Fy: float) -> tuple:
    """Zero lateral forces for path-constrained extraction."""
    return 0.0, 0.0


def sequential_gate(F_pull: float, latch_deflected: bool) -> float:
    """
    Sequential constraint gate (Eq. 16).

    Returns zero pull force if latch not deflected.
    """
    if not latch_deflected:
        return max(F_pull, 0.0)  # prevent pulling
    return F_pull
