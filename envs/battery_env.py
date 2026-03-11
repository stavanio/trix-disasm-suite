"""
BATTERY Environment: Thermally-Constrained Adhesive Separation (Section 4.2.2)

A pouch-cell Li-ion battery adhered with viscoelastic adhesive.
Temperature evolves: dT/dt = (eta/C) * F*v - kappa*(T - T_amb)  (Eq. 12)

Violation: T > 60 deg C

Domain Randomization:
  - Adhesive viscosity nu ~ U[5, 15] kPa*s
  - Initial temperature T0 ~ U[20, 35] deg C
"""

import numpy as np
from .base_env import BaseEnv, DT, FORCE_NOISE_STD


BATTERY_FORCE_LIMIT = 20.0   # N (puncture threshold)
BATTERY_TEMP_LIMIT = 60.0    # deg C (thermal runaway onset)
T_AMBIENT = 25.0             # deg C


class BatteryEnv(BaseEnv):
    """Li-ion battery removal with thermal constraint."""

    def __init__(self, max_steps: int = 1000):
        super().__init__(max_steps)
        self.eta = 0.7       # viscous dissipation coefficient
        self.C = 50.0        # J/K, thermal capacity
        self.kappa = 0.1     # W/K, thermal conductance (called gamma in some parts)
        self.T_crit = BATTERY_TEMP_LIMIT
        self.adhesive_length = 50.0  # mm, total peel length
        self.peel_position = 0.0

    @property
    def task_name(self) -> str:
        return 'BATTERY'

    def _randomize_parameters(self):
        self.viscosity = np.random.uniform(5.0, 15.0)  # kPa*s
        self.state.temperature = np.random.uniform(20.0, 35.0)
        self.C = np.random.uniform(40.0, 60.0)

    def reset(self, seed: int = None) -> np.ndarray:
        obs = super().reset(seed)
        self.peel_position = 0.0
        return self._get_obs()

    def _get_task_vars(self) -> np.ndarray:
        return np.array([
            self.state.temperature / 100.0,
            self.peel_position / self.adhesive_length,
            (self.T_crit - self.state.temperature) / self.T_crit,
        ], dtype=np.float32)

    def _physics_step(self, F: np.ndarray, tau: np.ndarray) -> bool:
        """
        Battery removal physics:
        - Peel force drives separation
        - Viscous dissipation heats adhesive: P = F * v
        - Temperature evolves per Eq. 12
        """
        F_peel = max(0, F[2])  # upward force for peeling
        v_peel = F_peel * 0.001  # simplified force-to-velocity

        # Power dissipation
        P = F_peel * v_peel

        # Temperature dynamics (Eq. 12)
        dT = (self.eta / self.C) * P - (self.kappa / self.C) * (
            self.state.temperature - T_AMBIENT)
        self.state.temperature += dT * (1.0 / 100)  # at control rate

        # Peel progress
        self.peel_position += v_peel * (1.0 / 100)
        self.peel_position = min(self.peel_position, self.adhesive_length)

        # Update state
        self.state.position[2] = self.peel_position * 0.001
        self.state.velocity[2] = v_peel
        self.state.external_wrench = np.concatenate([F, tau])

        # Violation: T > 60 deg C (Table 9)
        violation = self.state.temperature > self.T_crit

        # Also check puncture force
        if abs(F[0]) > BATTERY_FORCE_LIMIT or abs(F[1]) > BATTERY_FORCE_LIMIT:
            violation = True

        return violation

    def _compute_progress(self, F: np.ndarray, tau: np.ndarray) -> float:
        return self.peel_position / self.adhesive_length

    def _check_success(self) -> bool:
        return self.peel_position >= self.adhesive_length * 0.95

    def _check_done(self) -> bool:
        if self._check_success():
            return True
        if self.state.temperature > self.T_crit + 20:
            return True  # catastrophic thermal event
        return super()._check_done()
