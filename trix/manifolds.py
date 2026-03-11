"""
Differentiable Safe Operating Manifolds (Section 3)

Closed-form projections for each DISASM-Bench task primitive.
Each projection is O(1) computation derived from first-principles physics.

    u_safe = Psi_M(u_pi) = argmin_{u in M} ||u - u_pi||   (Eq. 3)
"""

import numpy as np
import math


def helical_projection(v_z: float, omega_z: float, pitch: float) -> tuple:
    """
    Orthogonal projection onto the helical manifold (Eq. 6, Appendix B.1).

    M_helix = { (v_z, omega_z) : v_z = k * omega_z },  k = pitch / (2*pi)

    Returns the closest point on the manifold to (v_z, omega_z).
    Cost: 3 multiplications, 2 additions, 1 division.
    """
    k = pitch / (2 * np.pi)
    s = (k * v_z + omega_z) / (k ** 2 + 1)
    return k * s, s


def thermal_projection(F: float, v: float, P_max: float) -> tuple:
    """
    Thermal-viscous manifold projection (Appendix B.2).

    C_batt = { (F, v) : F * v <= P_max(T) }

    Scales velocity to satisfy power constraint; preserves force direction.
    """
    if F * v <= P_max:
        return F, v
    return F, P_max / max(abs(F), 1e-8) * np.sign(v)


def planar_projection(wrench: np.ndarray, F_frac: float = 30.0) -> np.ndarray:
    """
    Planar invariant manifold projection (Eq. 8, Appendix B.3).

    M_pcb = { u in R^6 : tau_x = 0, tau_y = 0, Fz <= F_frac }

    Zeros tilt torques, clips normal force. O(1).
    """
    out = wrench.copy()
    out[3] = 0.0
    out[4] = 0.0
    out[2] = min(out[2], F_frac)
    return out


def tangential_projection(Fx: float, Fy: float, theta: float,
                          F_max: float = 12.0) -> tuple:
    """
    Tangential force projection for curved-manifold tasks (CRANK).

    Decomposes Cartesian force into tangential/radial components at angle theta,
    discards radial, clips tangential.
    """
    tx, ty = -math.sin(theta), math.cos(theta)
    F_t = np.clip(Fx * tx + Fy * ty, -F_max, F_max)
    return F_t * tx, F_t * ty
