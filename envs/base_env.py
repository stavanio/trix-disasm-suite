"""
Base environment for DISASM-Bench tasks.

All tasks share a common interface for fair comparison.
Observation space: R^19 (pose, velocities, wrench, temperature)
Action space: R^6 (commanded wrench)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict
from enum import Enum


# ============================================================================
# Physical Constants (Table 8, Appendix D)
# ============================================================================
# Material: Steel/Aluminum contact (engineering handbooks)
MU_STATIC = 0.61
MU_KINETIC = 0.47

# Sensor noise model: ATI Mini45 F/T sensor (~0.5% full scale)
# + actuator noise (~2% force control error)
# Combined (RSS): sqrt(0.725^2 + 1.0^2) ~ 1.235 N
FORCE_NOISE_STD = 1.235     # N
TORQUE_NOISE_STD = 0.039    # Nm
POSITION_NOISE_STD = 0.001  # m
VELOCITY_NOISE_STD = 0.01   # m/s
ANGULAR_VEL_NOISE_STD = 0.05  # rad/s

# Safety limits (collaborative robot standards)
TORQUE_LIMIT = 1.5    # Nm
FORCE_LIMIT = 50.0    # N
RADIAL_LIMIT = 20.0   # N

# Simulation
DT = 1.0 / 240        # 240 Hz integration
CONTROL_DT = 1.0 / 100  # 100 Hz control (Sec 4.3)
STEPS_PER_CONTROL = int(CONTROL_DT / DT)


class FrictionState(Enum):
    STUCK = 0
    SLIPPING = 1


@dataclass
class PhysicsState:
    """Full physics state for disassembly tasks."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    external_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))
    temperature: float = 25.0
    theta: float = 0.0
    omega: float = 0.0
    z: float = 0.0
    friction: FrictionState = FrictionState.STUCK

    def reset(self, T0: float = 25.0):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.external_wrench = np.zeros(6)
        self.temperature = T0
        self.theta = 0.0
        self.omega = 0.0
        self.z = 0.0
        self.friction = FrictionState.STUCK


class BaseEnv:
    """
    Base class for DISASM-Bench environments.

    Observation: R^19 = [position(3), lin_vel(3), ang_vel(3), wrench(6),
                         temperature(1), task_vars(3)]
    Action: R^6 = [Fx, Fy, Fz, tau_x, tau_y, tau_z]
             Forces: [-50, 50] N, Torques: [-1.5, 1.5] Nm

    Reward: r_t = alpha * v_progress - beta * I_violation
                 - lambda * ||u||^2 * dt + r_success  (Eq. 13)
    """

    def __init__(self, max_steps: int = 1000):
        self.state = PhysicsState()
        self.step_count = 0
        self.max_steps = max_steps

        # Reward coefficients (Section 4.4)
        self.alpha = 1.0
        self.beta = 10.0
        self.lam = 1e-3
        self.r_success = 100.0

    @property
    def observation_dim(self) -> int:
        return 19

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def task_name(self) -> str:
        raise NotImplementedError

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(int(seed))
        self.state.reset()
        self.step_count = 0
        self._randomize_parameters()
        return self._get_obs()

    def _randomize_parameters(self):
        """Domain randomization (Table 10, Appendix D)."""
        pass

    def _get_obs(self) -> np.ndarray:
        """Return noisy observation (Eq. 41, Appendix D)."""
        obs = np.concatenate([
            self.state.position + np.random.normal(0, POSITION_NOISE_STD, 3),
            self.state.velocity + np.random.normal(0, VELOCITY_NOISE_STD, 3),
            self.state.angular_velocity + np.random.normal(0, ANGULAR_VEL_NOISE_STD, 3),
            self.state.external_wrench[:3] + np.random.normal(0, FORCE_NOISE_STD, 3),
            self.state.external_wrench[3:] + np.random.normal(0, TORQUE_NOISE_STD, 3),
            np.array([self.state.temperature]),
            self._get_task_vars(),
        ])
        return obs.astype(np.float32)

    def _get_task_vars(self) -> np.ndarray:
        """Task-specific state variables (3-dim)."""
        return np.array([
            self.state.theta / (2 * np.pi),
            self.state.z,
            float(self.state.friction.value),
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one control step (100 Hz)."""
        self.step_count += 1

        # Handle both 3-dim and 6-dim actions
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        if len(action) >= 6:
            F = action[:3] * FORCE_LIMIT
            tau = action[3:6] * TORQUE_LIMIT
        else:
            # 3-dim action: interpret as [tau_z, F_z, F_radial] for screw-like tasks
            F = np.zeros(3)
            tau = np.zeros(3)
            F[2] = action[1] * FORCE_LIMIT if len(action) > 1 else 0.0
            tau[2] = action[0] * TORQUE_LIMIT
            if len(action) > 2:
                F[0] = action[2] * FORCE_LIMIT

        # Add actuator noise
        F += np.random.normal(0, FORCE_NOISE_STD, 3)
        tau += np.random.normal(0, TORQUE_NOISE_STD, 3)

        # Physics step
        violation = self._physics_step(F, tau)

        # Reward (Eq. 13)
        progress = self._compute_progress(F, tau)
        energy = np.sum(action ** 2) * CONTROL_DT
        reward = (self.alpha * progress
                  - self.beta * float(violation)
                  - self.lam * energy)

        done = self._check_done()
        if done and self._check_success():
            reward += self.r_success

        info = {
            'violation': int(violation),
            'success': self._check_success(),
            'progress': progress,
        }

        return self._get_obs(), reward, done, info

    def _physics_step(self, F: np.ndarray, tau: np.ndarray) -> bool:
        """Execute physics and return whether violation occurred."""
        raise NotImplementedError

    def _compute_progress(self, F: np.ndarray, tau: np.ndarray) -> float:
        raise NotImplementedError

    def _check_done(self) -> bool:
        return self.step_count >= self.max_steps

    def _check_success(self) -> bool:
        return False
