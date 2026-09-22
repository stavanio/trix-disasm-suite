"""Safety filter arms for the SCREW v3 command interface.

Admissible set, in normalised action units:
    |A_V a_v - A_W a_w| <= EPS      helical coupling
    |a_v|, |a_w| <= 1,  |a_r| <= RAD_LIM_N

safelayer_oracle linearises the true constraint exactly at the current
action, so a failure there is a property of the correction structure
rather than of the learned model.
"""

import numpy as np
from scipy.optimize import minimize

from envs import screw_env_v3 as E

# normalised-space coefficients: e_cmd = A_V * a_v - A_W * a_w
A_V = E.V_CMD_MAX
A_W = E.K_PITCH * E.OMEGA_CMD_MAX
EPS = E.EPS_HELIX
from benchmark.margin import sigma_for
from benchmark.disasm_bench import BASE_FORCE_NOISE

# F_rad = a[2] * S_FRAD + noise, so realization separates the filter output
# from the check on this axis and a robust interior bound is required. The
# helical command carries no such step and takes numerical tolerance only.
RAD_MARGIN_SIGMA = sigma_for(n_constraints=2, horizon=900)
RAD_LIM_N = max(0.0, E.RADIAL_LIMIT
                - RAD_MARGIN_SIGMA * BASE_FORCE_NOISE) / E.S_FRAD


def e_cmd_of(a):
    return A_V * float(a[0]) - A_W * float(a[1])


def _clip_box(a):
    return np.array([np.clip(a[0], -1.0, 1.0), np.clip(a[1], -1.0, 1.0),
                     np.clip(a[2], -RAD_LIM_N, RAD_LIM_N)])


# ------------------------------------------- admissible command polygon
def _clip_halfplane(poly, a, b):
    """Sutherland-Hodgman clip of a convex polygon by a.x <= b."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = float(a @ p) - b, float(a @ q) - b
        if dp <= 0:
            out.append(p)
        if (dp > 0) != (dq > 0):
            t = dp / (dp - dq)
            out.append(p + t * (q - p))
    return out


INTERIOR = 1e-6 * EPS   # dominates float32 output quantisation near the
                        # slab boundary (~4e-9 in e units) at negligible cost


def _admissible_polygon():
    """{|A_V v - A_W w| <= EPS} intersected with [-1,1]^2, as a polygon."""
    poly = [np.array([-1.0, -1.0]), np.array([1.0, -1.0]),
            np.array([1.0, 1.0]), np.array([-1.0, 1.0])]
    n2 = np.array([A_V, -A_W])
    poly = _clip_halfplane(poly, n2, EPS - INTERIOR)
    poly = _clip_halfplane(poly, -n2, EPS - INTERIOR)
    return [np.asarray(p, dtype=np.float64) for p in poly]


_POLY = _admissible_polygon()


def _proj_segment(p, a, b):
    ab = b - a
    t = float(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-300))
    return a + min(1.0, max(0.0, t)) * ab


def _inside(p, poly):
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < -1e-12:
            return False
    return True


def _proj_polygon(p, poly):
    if _inside(p, poly):
        return p.copy()
    best, bd = None, np.inf
    n = len(poly)
    for i in range(n):
        q = _proj_segment(p, poly[i], poly[(i + 1) % n])
        d = float(np.sum((q - p) ** 2))
        if d < bd:
            best, bd = q, d
    return best


# --------------------------------------------------------------- TRiX
def trix_project(obs, a):
    """Exact projection onto the slab-box intersection. Projecting onto
    the slab alone can leave the box, and clipping afterwards leaves the
    slab, so the two are enforced jointly as one convex polygon."""
    p = np.array([float(a[0]), float(a[1])], dtype=np.float64)
    q = _proj_polygon(p, _POLY)
    out = np.array([q[0], q[1], np.clip(float(a[2]), -RAD_LIM_N, RAD_LIM_N)])
    e = A_V * out[0] - A_W * out[1]
    return out.astype(np.float32), {"active": bool(not _inside(p, _POLY)),
                                    "infeasible": abs(e) > EPS + 1e-6}


# ------------------------------------------------- generic SQP baseline
def cbf_qp_solve(obs, a):
    """Generic SQP over the same set. The constraint row is normalised:
    unscaled, |n| ~ 5e-2 against an order-1 objective leaves the solver
    unable to resolve the slab."""
    a0 = _clip_box(a)
    n_raw = np.array([A_V, -A_W, 0.0])
    scale = float(np.linalg.norm(n_raw))
    n = n_raw / scale
    eps_s = EPS / scale
    eps_s -= INTERIOR / scale
    cons = [{"type": "ineq", "fun": lambda x: eps_s - (n @ x)},
            {"type": "ineq", "fun": lambda x: eps_s + (n @ x)}]
    bnds = [(-1.0, 1.0), (-1.0, 1.0), (-RAD_LIM_N, RAD_LIM_N)]
    # warm start from a known feasible point: starting at the infeasible
    # nominal leaves SLSQP reporting failure on ~30% of calls
    res = minimize(lambda x: float(np.sum((x - a0) ** 2)), x0=np.zeros(3),
                   jac=lambda x: 2.0 * (x - a0), bounds=bnds,
                   constraints=cons, method="SLSQP",
                   options={"ftol": 1e-16, "maxiter": 300})
    x = res.x
    ok = res.success and abs(A_V * x[0] - A_W * x[1]) <= EPS
    if not ok:
        return np.zeros(3, dtype=np.float32), {"infeasible": True,
                                               "recovery": True}
    return x.astype(np.float32), {"infeasible": False}


# --------------------------------------------------- SafeLayer (oracle)
def safelayer_oracle_cost(obs, a):
    """c(a) = |e_cmd| - EPS is piecewise linear, so its linearisation is
    exact on the active side."""
    e = e_cmd_of(a)
    s = 1.0 if e >= 0 else -1.0
    g = s * np.array([A_V, -A_W, 0.0])
    c0 = -EPS
    return c0, g


# -------------------------------------------------- SafeLayer (learned)
class LearnedCost:
    """Least-squares fit of c(s,a) ~ c0(s) + g(s)^T a. A fixed feature map
    rather than a network, so the fit is deterministic and inspectable."""

    def __init__(self, n_feat=4):
        self.n_feat = n_feat
        self.W = None

    @staticmethod
    def _phi(obs):
        o = np.asarray(obs, dtype=np.float64)
        return np.array([1.0, o[2] / 60.0, o[3] / 0.05, o[4]])

    def fit(self, obs_list, act_list, cost_list, ridge=1e-6):
        rows, y = [], []
        for o, a, c in zip(obs_list, act_list, cost_list):
            p = self._phi(o)
            rows.append(np.concatenate([p, np.kron(p, a)]))
            y.append(c)
        X = np.array(rows); y = np.array(y)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ y)
        return self

    def __call__(self, obs, a):
        p = self._phi(obs)
        k = self.n_feat
        c0 = float(self.W[:k] @ p)
        G = self.W[k:].reshape(k, 3)
        g = p @ G
        return c0, g
