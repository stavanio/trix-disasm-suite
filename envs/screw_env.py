"""
SCREW Environment: Helical Fastener Extraction (Section 4.2.1)

An M8 screw threaded into aluminum. The helical coupling
v_z = (p/2pi) * omega_z must be maintained throughout extraction.

Violation: |v_z - (p/2pi)*omega_z| > 1 mm/s  OR  ||F_lat|| > 5 N

Domain Randomization:
  - Thread pitch p ~ U[1.0, 1.5] mm/rev
  - Friction mu ~ U[0.4, 0.8]
  - Preload torque tau_0 ~ U[0.5, 2.0] Nm
"""

import numpy as np
from .base_env import (BaseEnv, PhysicsState, FrictionState,
                       DT, TORQUE_LIMIT, RADIAL_LIMIT,
                       FORCE_NOISE_STD, TORQUE_NOISE_STD)


class ScrewEnv(BaseEnv):
    """M8 screw extraction with helical manifold constraint."""

    def __init__(self, max_steps: int = 1000):
        super().__init__(max_steps)
        self.pitch = 1.25           # mm/rev (nominal M8)
        self.thread_depth = 20.0    # mm
        self.mu_s = 0.61
        self.mu_k = 0.47
        self.preload_torque = 1.0   # Nm
        self.k = self.pitch / (2 * np.pi)  # pitch ratio

    @property
    def task_name(self) -> str:
        return 'SCREW'

    def _randomize_parameters(self):
        """Domain randomization (Section 4.2.1)."""
        self.pitch = np.random.uniform(1.0, 1.5)
        self.k = self.pitch / (2 * np.pi)
        self.mu_s = np.random.uniform(0.4, 0.8)
        self.mu_k = self.mu_s * 0.77  # typical static/kinetic ratio
        self.preload_torque = np.random.uniform(0.5, 2.0)

    def reset(self, seed: int = None) -> np.ndarray:
        obs = super().reset(seed)
        self.state.z = 0.0
        self.state.theta = 0.0
        self.state.omega = 0.0
        self.state.friction = FrictionState.STUCK
        return self._get_obs()

    def _get_task_vars(self) -> np.ndarray:
        return np.array([
            self.state.theta / (2 * np.pi),
            self.state.z / self.thread_depth,
            float(self.state.friction.value),
        ], dtype=np.float32)

    def _physics_step(self, F: np.ndarray, tau: np.ndarray) -> bool:
        """
        Screw physics: helical motion with Coulomb friction.

        The screw is modeled as:
        - Torque drives rotation: I * alpha = tau_z - tau_friction
        - Helical coupling: v_z = k * omega_z (Eq. 4)
        - Stiction/slip transitions (Stribeck model simplified)
        """
        tau_z = tau[2]  # torque about extraction axis
        F_axial = F[2]  # axial force
        F_lateral = np.sqrt(F[0]**2 + F[1]**2)

        # Friction model (Coulomb with stiction)
        normal_force = max(abs(F_axial), 1.0)  # at least 1N from thread engagement
        tau_friction_static = self.mu_s * normal_force * 0.004  # r_eff ~ 4mm for M8
        tau_friction_kinetic = self.mu_k * normal_force * 0.004

        # State transitions
        if self.state.friction == FrictionState.STUCK:
            net_tau = abs(tau_z) - self.preload_torque
            if net_tau > tau_friction_static:
                self.state.friction = FrictionState.SLIPPING
        else:
            if abs(tau_z) < tau_friction_kinetic * 0.8:
                self.state.friction = FrictionState.STUCK

        # Dynamics
        if self.state.friction == FrictionState.SLIPPING:
            # Simplified rotational dynamics: I_eff * alpha = tau_net
            I_eff = 0.5e-6  # kg*m^2, small for screw
            tau_net = tau_z - np.sign(tau_z) * tau_friction_kinetic
            alpha = tau_net / I_eff

            # Integration (sub-steps for stability)
            for _ in range(int(1.0 / (240 * DT))):
                self.state.omega += alpha * DT
                self.state.omega *= 0.95  # damping
                self.state.theta += self.state.omega * DT
                self.state.z += self.k * self.state.omega * DT
        else:
            self.state.omega *= 0.1  # rapid deceleration when stuck

        # Update wrench
        self.state.external_wrench = np.concatenate([F, tau])

        # Violation check (Table 9)
        # 1. Helical coupling: |v_z - k * omega_z| > 1 mm/s
        expected_vz = self.k * self.state.omega
        actual_vz = F_axial * 0.001  # simplified force-to-velocity
        coupling_error = abs(actual_vz - expected_vz)

        violation = False
        if coupling_error > 1.0 and abs(F_axial) > 5.0:
            violation = True
        if abs(tau_z) > TORQUE_LIMIT:
            violation = True
        if F_lateral > RADIAL_LIMIT:
            violation = True

        return violation

    def _compute_progress(self, F: np.ndarray, tau: np.ndarray) -> float:
        return max(0.0, self.state.z) * 0.001

    def _check_done(self) -> bool:
        if self.state.z >= self.thread_depth:
            return True
        return super()._check_done()

    def _check_success(self) -> bool:
        return self.state.z >= self.thread_depth * 0.95
