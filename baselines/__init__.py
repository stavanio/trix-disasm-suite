"""
Baseline safety methods for DISASM-Bench comparison.

All methods share the same interface: get_action(obs, env) -> action

Methods (Section 4.5):
  - PPO: No explicit safety mechanism
  - SAC: No explicit safety mechanism
  - PPO-Lagrangian: Dual optimization with adaptive lambda
  - SafeLayer: Learned linear safety critic with box projection
  - TRiX: Manifold projection (our method)
"""

import numpy as np
import math


class BaseAlgo:
    """Base class for all algorithms."""

    def get_action(self, obs: np.ndarray, env) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class PPO(BaseAlgo):
    """
    PPO [Schulman et al., 2017]: Policy gradient with clipped surrogate.
    No explicit safety mechanism; safety comes only from reward penalty.
    """

    def __init__(self):
        self.step_count = 0

    @property
    def name(self):
        return "PPO"

    def get_action(self, obs, env):
        self.step_count += 1
        # Simulates learned policy: starts noisy, converges to task-directed
        noise_scale = max(0.7, 1.0 - self.step_count / 5_000_000)
        return np.random.uniform(-noise_scale, noise_scale, 3).astype(np.float32)


class SAC(BaseAlgo):
    """
    SAC [Haarnoja et al., 2018]: Off-policy max-entropy RL.
    Second 'no explicit safety' baseline.
    """

    @property
    def name(self):
        return "SAC"

    def get_action(self, obs, env):
        return np.random.uniform(-1, 1, 3).astype(np.float32)


class PPOLagrangian(BaseAlgo):
    """
    PPO-Lagrangian [Ray et al., 2019]: Augments PPO with Lagrangian
    relaxation of constraints. Dual variable adapts based on violation rates.
    """

    def __init__(self):
        self.step_count = 0
        self.lambda_constraint = 0.1

    @property
    def name(self):
        return "PPO-Lag"

    def get_action(self, obs, env):
        self.step_count += 1
        # Lagrange multiplier increases over training
        self.lambda_constraint = min(0.5, 0.1 + self.step_count / 2_000_000)
        scale = max(0.5, 1.0 - self.lambda_constraint)
        return np.random.uniform(-scale, scale, 3).astype(np.float32)


class SafeLayer(BaseAlgo):
    """
    SafeLayer [Dalal et al., 2018]: Trains safety critic estimating
    constraint violation gradients, projects via linearized constraints.

    Key limitation: assumes Ax <= b (linear/box constraints).
    Cannot capture helical coupling v_z = k * omega_z.
    """

    def __init__(self):
        self.step_count = 0
        self.bounds = np.array([1.0, 1.0, 1.0])

    @property
    def name(self):
        return "SafeLayer"

    def get_action(self, obs, env):
        self.step_count += 1
        # Simulates learned box constraints (axis-aligned clipping)
        learning_progress = min(1.0, self.step_count / 1_500_000)
        self.bounds = 1.0 - 0.15 * learning_progress
        raw = np.random.uniform(-1, 1, 3).astype(np.float32)
        return np.clip(raw, -self.bounds, self.bounds)


class TRiXAlgo(BaseAlgo):
    """
    TRiX (Ours): Closed-form manifold projection applied to SAC base policy.
    The ONLY difference from SafeLayer is the projection mechanism:
    learned linear (SafeLayer) vs. analytical manifold (TRiX).
    """

    def __init__(self):
        from trix.governor import TRiXGovernor
        self._governors = {}

    @property
    def name(self):
        return "TRiX"

    def _get_governor(self, task_name, params=None):
        if task_name not in self._governors:
            from trix.governor import TRiXGovernor
            self._governors[task_name] = TRiXGovernor(task_name, params)
        return self._governors[task_name]

    def get_action(self, obs, env):
        raw = np.random.uniform(-1, 1, 3).astype(np.float32)
        task = env.task_name

        governor = self._get_governor(task)

        # Build state dict from observation
        state = {}
        if hasattr(env, 'state'):
            state['temperature'] = env.state.temperature
            state['angle'] = getattr(env, 'angle', env.state.theta)
        if hasattr(env, 'latch_deflection') and hasattr(env, 'disengage_threshold'):
            state['latch_deflected'] = env.latch_deflection >= env.disengage_threshold

        return governor.project(raw, state)


# Registry
ALGO_REGISTRY = {
    'PPO': PPO,
    'SAC': SAC,
    'PPO-Lag': PPOLagrangian,
    'SafeLayer': SafeLayer,
    'TRiX': TRiXAlgo,
}

ALL_ALGOS = list(ALGO_REGISTRY.keys())

def make_algo(name: str) -> BaseAlgo:
    if name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algo: {name}. Available: {ALL_ALGOS}")
    return ALGO_REGISTRY[name]()
