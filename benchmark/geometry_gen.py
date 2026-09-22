"""Procedural admissible-set geometries.

Six families spanning separable, coupled, state-dependent and sequential
structure. Each instance carries a required action drawn from its own
interior, standing in for the task-essential command a real task needs.

The criterion predicts, per instance and before any method runs, whether
richer structure is needed: it is when an inscribed axis-aligned box
excludes the required action.

Spec frozen 2026-08-06 and hashed. Held-out instances are generated from
a separate seed stream and evaluated once.
"""

import hashlib
import json

import numpy as np

SPEC_VERSION = "1.3"
FROZEN_ON = "2026-08-06"

DIM = 3
INTERIOR = 1e-6
_SIGNS = np.array(
    [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]],
    dtype=float)
FAMILIES = ("box", "ellipsoid", "polytope", "state_scaled",
            "rotating", "phase_gated")

N_TRAIN, N_VAL, N_HELDOUT = 120, 40, 200
SEED_TRAIN, SEED_VAL, SEED_HELDOUT = 10_000, 20_000, 30_000

RANGES = {
    "half_width": (0.30, 0.95),
    "aspect": (1.0, 6.0),
    "rotation": (0.0, np.pi),
    "n_faces": (4, 10),
    "scale_lo": (0.15, 0.60),
    "gate_ratio": (0.15, 0.60),
    "required_radius": (0.55, 0.92),
}


def _u(rng, key):
    lo, hi = RANGES[key]
    return float(rng.uniform(lo, hi))


class Geometry:

    def __init__(self, family, params, ctx=0.0):
        self.family = family
        self.p = params
        self.ctx = ctx

    def matrices(self, ctx=None):
        """Return (Q, A, b) for the active set at this context."""
        c = self.ctx if ctx is None else ctx
        p = self.p
        if self.family == "box":
            w = np.array(p["w"])
            A = np.vstack([np.eye(DIM), -np.eye(DIM)])
            return None, A, np.concatenate([w, w])
        if self.family == "ellipsoid":
            return np.array(p["Q"]), None, None
        if self.family == "polytope":
            return None, np.array(p["A"]), np.array(p["b"])
        if self.family == "state_scaled":
            s = p["lo"] + (1.0 - p["lo"]) * c
            w = np.array(p["w"]) * s
            A = np.vstack([np.eye(DIM), -np.eye(DIM)])
            return None, A, np.concatenate([w, w])
        if self.family == "rotating":
            th = p["rate"] * c
            R = _rot_z(th)
            Q0 = np.array(p["Q"])
            return R @ Q0 @ R.T, None, None
        if self.family == "phase_gated":
            w = np.array(p["w"]).copy()
            if c < 0.5:
                w[0] *= p["gate"]
            A = np.vstack([np.eye(DIM), -np.eye(DIM)])
            return None, A, np.concatenate([w, w])
        raise KeyError(self.family)

    def contains(self, u, ctx=None, tol=1e-9):
        Q, A, b = self.matrices(ctx)
        u = np.asarray(u, dtype=np.float64)
        if Q is not None:
            return float(u @ Q @ u) <= 1.0 + tol
        return bool(np.all(A @ u <= b + tol))

    def project(self, u, ctx=None):
        """Exact projection onto the active set."""
        Q, A, b = self.matrices(ctx)
        u = np.asarray(u, dtype=np.float64)
        if Q is not None:
            r = float(u @ Q @ u)
            return u if r <= 1.0 else u / np.sqrt(r)
        if self.family in ("box", "state_scaled", "phase_gated"):
            w = b[:DIM]
            return np.clip(u, -w, w)
        return _project_polytope(u, A, b)

    def inscribed_box(self, ctx=None):
        """Largest axis-aligned box inside the active set."""
        Q, A, b = self.matrices(ctx)
        if Q is not None:
            d = np.sqrt(np.diag(np.linalg.inv(Q)))
            worst = max(float((sg * d) @ Q @ (sg * d))
                        for sg in _SIGNS)
            return d / np.sqrt(worst) * (1.0 - INTERIOR)
        if self.family in ("box", "state_scaled", "phase_gated"):
            return b[:DIM]
        w = np.array([b[i] / max(1e-9, np.sum(np.abs(A[i]))) for i in
                      range(len(b))]).min()
        return np.full(DIM, w)


def _rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _project_polytope(u, A, b, iters=200):
    x = np.asarray(u, dtype=np.float64).copy()
    for _ in range(iters):
        v = A @ x - b
        i = int(np.argmax(v))
        if v[i] <= 1e-12:
            break
        a = A[i]
        x = x - (v[i] / float(a @ a)) * a
    return x


def sample(rng, family):
    if family == "box":
        w = np.array([_u(rng, "half_width") for _ in range(DIM)])
        return {"w": w.tolist()}
    if family == "ellipsoid":
        d = np.array([_u(rng, "half_width") for _ in range(DIM)])
        d[rng.integers(DIM)] /= _u(rng, "aspect")
        R = _rot_z(_u(rng, "rotation"))
        Q = R @ np.diag(1.0 / d ** 2) @ R.T
        return {"Q": Q.tolist()}
    if family == "polytope":
        n = int(rng.integers(*RANGES["n_faces"]))
        A = rng.normal(size=(n, DIM))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        b = np.array([_u(rng, "half_width") for _ in range(n)])
        return {"A": A.tolist(), "b": b.tolist()}
    if family == "state_scaled":
        w = np.array([_u(rng, "half_width") for _ in range(DIM)])
        return {"w": w.tolist(), "lo": _u(rng, "scale_lo")}
    if family == "rotating":
        d = np.array([_u(rng, "half_width") for _ in range(DIM)])
        d[0] /= _u(rng, "aspect")
        Q = np.diag(1.0 / d ** 2)
        return {"Q": Q.tolist(), "rate": float(rng.uniform(0.5, 3.0))}
    if family == "phase_gated":
        w = np.array([_u(rng, "half_width") for _ in range(DIM)])
        return {"w": w.tolist(), "gate": _u(rng, "gate_ratio")}
    raise KeyError(family)


def required_action(g, rng, ctx=0.0):
    """A command the instance must retain, drawn from its own interior."""
    for _ in range(400):
        d = rng.normal(size=DIM)
        d /= np.linalg.norm(d)
        r = _u(rng, "required_radius")
        u = g.project(d * 10.0, ctx) * r
        if g.contains(u, ctx):
            return u
    return np.zeros(DIM)


CTX_GRID = np.linspace(0.0, 1.0, 21)
CONTEXT_DEPENDENT = ("state_scaled", "rotating", "phase_gated")


def context_free_box(g):
    if g.family not in CONTEXT_DEPENDENT:
        return g.inscribed_box(g.ctx)
    return np.min([g.inscribed_box(c) for c in CTX_GRID], axis=0)


def predicts_structure_needed(g, req, ctx=0.0):
    return bool(np.any(np.abs(req) > context_free_box(g) + 1e-9))


def bank(split):
    seed = {"train": SEED_TRAIN, "val": SEED_VAL,
            "heldout": SEED_HELDOUT}[split]
    n = {"train": N_TRAIN, "val": N_VAL, "heldout": N_HELDOUT}[split]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fam = FAMILIES[i % len(FAMILIES)]
        g = Geometry(fam, sample(rng, fam))
        ctx = float(rng.uniform(0.0, 1.0))
        g.ctx = ctx
        req = required_action(g, rng, ctx)
        out.append({"id": f"{split}_{i:04d}", "family": fam, "geom": g,
                    "ctx": ctx, "required": req,
                    "predicted_structure_needed":
                        predicts_structure_needed(g, req, ctx)})
    return out


def spec_hash():
    spec = json.dumps({"version": SPEC_VERSION, "frozen_on": FROZEN_ON,
                       "dim": DIM, "families": list(FAMILIES),
                       "counts": [N_TRAIN, N_VAL, N_HELDOUT],
                       "seeds": [SEED_TRAIN, SEED_VAL, SEED_HELDOUT],
                       "ranges": {k: list(v) for k, v in RANGES.items()}},
                      sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()[:16]
