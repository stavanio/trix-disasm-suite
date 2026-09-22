"""Safety filter arms for the PCB v2 action interface.

Admissible set, in normalised action units:
    sqrt(a_x^2 + a_y^2) * S_TAU <= TAU_TILT_MAX     tilt disk
    |a_z| * S_FZ <= F_LIFT_MAX                      lift interval

The tilt axes carry a norm constraint, so the admissible region is a disk
crossed with an interval. Componentwise clipping can only produce a box.
"""

import numpy as np
from scipy.optimize import minimize

from envs import pcb_env_v2 as P
from benchmark.margin import sigma_for
from benchmark.disasm_bench import BASE_FORCE_NOISE

TAU_N = P.TAU_TILT_MAX / P.S_TAU      # disk radius in action units
FZ_N = P.F_LIFT_MAX / P.S_FZ          # lift bound in action units
INTERIOR = 1e-6
MARGIN_SIGMA = sigma_for()

# The damage onset coincides with the flagged limit here, since
# THETA_YIELD = TAU_TILT_MAX / K_BEND, so a command projected exactly onto
# the limit bends the board to the yield point and accumulates damage.
# Sensing noise adds to that. The robust interior set backs both off.
TAU_ROBUST = max(0.0, P.TAU_TILT_MAX - MARGIN_SIGMA * P.SIGMA_TAU_PCB) / P.S_TAU
FZ_ROBUST = max(0.0, P.F_LIFT_MAX - MARGIN_SIGMA * BASE_FORCE_NOISE) / P.S_FZ


def tilt_norm_of(a):
    return float(np.hypot(float(a[0]), float(a[1])))


def _clip_z(z, robust=True):
    lim = FZ_ROBUST if robust else FZ_N
    return max(-lim, min(lim, float(z)))


# ------------------------------------------------- componentwise clipping
def box_clip(obs, a):
    """Per-axis bounds only, the form the submitted benchmark used."""
    return np.array([np.clip(float(a[0]), -TAU_N, TAU_N),
                     np.clip(float(a[1]), -TAU_N, TAU_N),
                     _clip_z(a[2])], dtype=np.float32), {}


# ----------------------------------------------------- exact projection
def trix_project(obs, a):
    """Radial scaling onto the robust interior disk; lift is separable."""
    x, y = float(a[0]), float(a[1])
    n = np.hypot(x, y)
    r = TAU_ROBUST * (1.0 - INTERIOR)
    if n > r:
        s = r / n
        x, y = x * s, y * s
    return np.array([x, y, _clip_z(a[2])], dtype=np.float32), {}


# ------------------------------------------------- SafeLayer (oracle)
def safelayer_oracle_cost(obs, a):
    """c(a) = |tau| - TAU_N linearised at the current action. The gradient
    is the unit radial direction, so the correction moves radially."""
    x, y = float(a[0]), float(a[1])
    n = np.hypot(x, y)
    if n < 1e-12:
        return -TAU_N, np.zeros(3)
    g = np.array([x / n, y / n, 0.0])
    c0 = -TAU_N
    return c0, g


# ------------------------------------------------ SafeLayer (learned)
class LearnedCost:
    """Least-squares fit of c(s,a) ~ c0(s) + g(s)^T a."""

    def __init__(self, n_feat=4):
        self.n_feat = n_feat
        self.W = None

    @staticmethod
    def _phi(obs):
        o = np.asarray(obs, dtype=np.float64)
        return np.array([1.0, o[0] / 0.015, o[2] / 0.05, o[3] / 0.05])

    def fit(self, obs_list, act_list, cost_list, ridge=1e-6):
        rows, y = [], []
        for o, a, c in zip(obs_list, act_list, cost_list):
            p = self._phi(o)
            rows.append(np.concatenate([p, np.kron(p, a)]))
            y.append(c)
        X = np.array(rows); y = np.array(y)
        self.W = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
        return self

    def __call__(self, obs, a):
        p = self._phi(obs)
        k = self.n_feat
        c0 = float(self.W[:k] @ p)
        g = p @ self.W[k:].reshape(k, 3)
        return c0, g


# ------------------------------------- matched numerical projection (SOC)
def socp_project(obs, a):
    """The tilt constraint is a second-order cone, not a linear inequality,
    so a QP cannot represent it exactly. Solved numerically with a
    nonlinear constraint for comparison against the closed form."""
    a0 = np.array([float(a[0]), float(a[1]), _clip_z(a[2])])
    r = TAU_N * (1.0 - INTERIOR)
    cons = [{"type": "ineq", "fun": lambda x: r - np.hypot(x[0], x[1])}]
    bnds = [(-1.0, 1.0), (-1.0, 1.0), (-FZ_N, FZ_N)]
    res = minimize(lambda x: float(np.sum((x - a0) ** 2)),
                   x0=np.zeros(3), jac=lambda x: 2.0 * (x - a0),
                   bounds=bnds, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 400})
    x = res.x
    if not res.success or np.hypot(x[0], x[1]) > TAU_N:
        return np.zeros(3, dtype=np.float32), {"infeasible": True,
                                               "recovery": True}
    return x.astype(np.float32), {"infeasible": False}
