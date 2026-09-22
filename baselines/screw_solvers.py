"""Solver arms for the SCREW admissible command set.

All arms solve min ||a - a_nom||^2 over the same set and differ only in
how the solution is obtained.

  trix_scalar    closed form, scalar arithmetic
  qp_osqp        operator splitting, setup reused across calls
  qp_quadprog    Goldfarb-Idnani dual active set
  slsqp_generic  general-purpose SQP; a reference point, not a
                 representative QP method
"""

import numpy as np
import osqp
import quadprog
from scipy import sparse

from baselines.screw_filters import A_V, A_W, EPS, RAD_LIM_N, INTERIOR

_N = np.array([A_V, -A_W, 0.0])
_EPS_I = EPS - INTERIOR
_LB = np.array([-1.0, -1.0, -RAD_LIM_N])
_UB = np.array([1.0, 1.0, RAD_LIM_N])


def _poly():
    from baselines.screw_filters import _admissible_polygon
    return [(float(p[0]), float(p[1])) for p in _admissible_polygon()]


_P = _poly()
_EDGES = [(_P[i], _P[(i + 1) % len(_P)]) for i in range(len(_P))]


def trix_scalar(obs, a):
    """Same geometry as the array version; scalar arithmetic isolates the
    projection cost from numpy allocation."""
    v, w, r = float(a[0]), float(a[1]), float(a[2])
    r = RAD_LIM_N if r > RAD_LIM_N else (-RAD_LIM_N if r < -RAD_LIM_N else r)

    inside = True
    for (px, py), (qx, qy) in _EDGES:
        if (qx - px) * (w - py) - (qy - py) * (v - px) < -1e-12:
            inside = False
            break
    if inside:
        return np.array([v, w, r], dtype=np.float32), {"active": False}

    bx = by = 0.0
    bd = 1e300
    for (px, py), (qx, qy) in _EDGES:
        ex, ey = qx - px, qy - py
        d2 = ex * ex + ey * ey
        t = ((v - px) * ex + (w - py) * ey) / d2 if d2 > 0.0 else 0.0
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        cx, cy = px + t * ex, py + t * ey
        dx, dy = cx - v, cy - w
        d = dx * dx + dy * dy
        if d < bd:
            bd, bx, by = d, cx, cy
    return np.array([bx, by, r], dtype=np.float32), {"active": True}


# --------------------------------------------------------------- OSQP
class OSQPFilter:
    """Operator-splitting QP, set up once and updated per call."""

    def __init__(self):
        P = sparse.csc_matrix(2.0 * np.eye(3))
        A = sparse.csc_matrix(np.vstack([_N, np.eye(3)]))
        self._l = np.concatenate([[-_EPS_I], _LB])
        self._u = np.concatenate([[_EPS_I], _UB])
        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=np.zeros(3), A=A, l=self._l, u=self._u,
                        verbose=False, eps_abs=1e-9, eps_rel=1e-9,
                        max_iter=20000, polish=True,
                        polish_refine_iter=0)

    def __call__(self, obs, a):
        a0 = np.clip(np.asarray(a, dtype=np.float64), _LB, _UB)
        self.prob.update(q=-2.0 * a0)
        res = self.prob.solve()
        x = res.x
        ok = (x is not None and np.all(np.isfinite(x))
              and abs(float(_N @ x)) <= EPS + 1e-9)
        if not ok:
            return np.zeros(3, dtype=np.float32), {"infeasible": True,
                                                   "recovery": True}
        return x.astype(np.float32), {"infeasible": False}


# ----------------------------------------------------------- quadprog
def qp_quadprog(obs, a):
    """Goldfarb-Idnani dual active set; exact for a problem this small."""
    a0 = np.clip(np.asarray(a, dtype=np.float64), _LB, _UB)
    G = 2.0 * np.eye(3)
    a_lin = 2.0 * a0
    # C^T x >= b
    C = np.column_stack([-_N, _N, np.eye(3), -np.eye(3)])
    b = np.concatenate([[-_EPS_I], [-_EPS_I], _LB, -_UB])
    try:
        x = quadprog.solve_qp(G, a_lin, C, b)[0]
    except Exception:
        return np.zeros(3, dtype=np.float32), {"infeasible": True,
                                               "recovery": True}
    if abs(float(_N @ x)) > EPS + 1e-9:
        return np.zeros(3, dtype=np.float32), {"infeasible": True,
                                               "recovery": True}
    return x.astype(np.float32), {"infeasible": False}
