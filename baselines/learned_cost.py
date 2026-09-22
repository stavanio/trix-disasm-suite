"""Learned constraint models for safety-layer baselines.

    AffineCost      c(s,a) = c0(s) + g(s)^T a, the form used by the
                    cited safety layer
    NonlinearCost   c_phi(s,a) from a small network, action gradient
                    taken by automatic differentiation at the proposed
                    action

Both correct an action by the single-halfspace step the cited method
uses. Correction can be iterated.
"""

import numpy as np
import torch
import torch.nn as nn


class AffineCost:
    """Least squares fit of c(s,a) = c0(s) + g(s)^T a.

    A fixed feature map rather than a network, so the fit is a linear
    solve and reproduces exactly. The functional form is what matters,
    not the capacity of the state encoder.
    """

    def __init__(self, phi, act_dim=3, ridge=1e-6):
        self.phi = phi
        self.act_dim = act_dim
        self.ridge = ridge
        self.W = None
        self.n_feat = None

    def fit(self, obs, acts, costs):
        rows = []
        for o, a in zip(obs, acts):
            p = np.asarray(self.phi(o), dtype=np.float64)
            rows.append(np.concatenate([p, np.kron(p, np.asarray(a))]))
        X = np.array(rows)
        y = np.asarray(costs, dtype=np.float64)
        self.n_feat = len(self.phi(obs[0]))
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ y)
        return self

    def __call__(self, obs, a):
        p = np.asarray(self.phi(obs), dtype=np.float64)
        k = self.n_feat
        c0 = float(self.W[:k] @ p)
        g = p @ self.W[k:].reshape(k, self.act_dim)
        return c0, np.asarray(g, dtype=np.float64)


class NonlinearCost:
    """c_phi(s,a) from a network; the action gradient comes from autodiff.

    The gradient is evaluated at the proposed action rather than being a
    function of state alone, which is the capability the affine form
    lacks.
    """

    def __init__(self, obs_dim, act_dim=3, hidden=128, seed=0, lr=1e-3):
        torch.manual_seed(seed)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, obs, acts, costs, steps=400, batch=512):
        X = torch.tensor(np.hstack([np.asarray(obs, dtype=np.float32),
                                    np.asarray(acts, dtype=np.float32)]))
        y = torch.tensor(np.asarray(costs, dtype=np.float32)).unsqueeze(1)
        for _ in range(steps):
            idx = torch.randint(0, len(X), (min(batch, len(X)),))
            loss = ((self.net(X[idx]) - y[idx]) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self

    def __call__(self, obs, a):
        x = torch.tensor(np.concatenate([np.asarray(obs, dtype=np.float32),
                                         np.asarray(a, dtype=np.float32)]),
                         requires_grad=True)
        val = self.net(x)
        val.backward()
        g = x.grad.numpy()[self.obs_dim:].astype(np.float64)
        c = float(val)
        # Return the intercept in the same convention as AffineCost, so a
        # single correction routine serves both.
        return c - float(g @ np.asarray(a, dtype=np.float64)), g

    def predict(self, obs, acts):
        X = torch.tensor(np.hstack([np.asarray(obs, dtype=np.float32),
                                    np.asarray(acts, dtype=np.float32)]))
        with torch.no_grad():
            return self.net(X).squeeze(1).numpy()


def halfspace_correct(cost_model, obs, a, bounds, margin=0.0, iters=1):
    """The single-halfspace correction the cited method uses.

    Iterating is a separate knob from the model class: one linear step
    lands short of a curved boundary even with an exact gradient.
    """
    def clip(v):
        if bounds is None:
            return v
        for i, (lo, hi) in enumerate(bounds):
            v[i] = min(hi, max(lo, v[i]))
        return v

    # Bounds are part of the admissible set, not a repair applied only
    # after a correction. Clipping inside the loop alone would leave a
    # separable axis untouched whenever the coupled constraint happened
    # to be satisfied, so the arm would enforce the box only sometimes.
    x = clip(np.asarray(a, dtype=np.float64).copy())
    for _ in range(iters):
        c0, g = cost_model(obs, x)
        c = c0 + float(g @ x)
        if c <= margin or float(g @ g) < 1e-12:
            break
        x = clip(x - ((c - margin) / float(g @ g)) * g)
    return x
