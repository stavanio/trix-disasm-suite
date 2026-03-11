"""
PCB Environment: Planar-Constrained Extraction (Section 4.2.3)

Rectangular PCB secured by snap-fit clips. Brittle FR-4 composite
fractures under bending stress.

Violation: sqrt(tau_x^2 + tau_y^2) > 0.1 Nm  (Table 9)

Domain Randomization:
  - Clip retention force F_clip ~ U[2, 8] N
  - PCB thickness t ~ U[1.2, 2.0] mm
"""

import numpy as np
from .base_env import BaseEnv, DT, FORCE_NOISE_STD, TORQUE_NOISE_STD


PCB_TORQUE_LIMIT = 0.1  # Nm, bending moment limit
PCB_FRACTURE_FORCE = 30.0  # N


class PCBEnv(BaseEnv):

    def __init__(self, max_steps: int = 1000):
        super().__init__(max_steps)
        self.clip_force = 5.0  # N
        self.thickness = 1.6   # mm
        self.extraction_distance = 10.0  # mm
        self.position_z = 0.0

    @property
    def task_name(self) -> str:
        return 'PCB'

    def _randomize_parameters(self):
        self.clip_force = np.random.uniform(2.0, 8.0)
        self.thickness = np.random.uniform(1.2, 2.0)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self.position_z = 0.0
        return self._get_obs()

    def _get_task_vars(self):
        return np.array([
            self.position_z / self.extraction_distance,
            self.thickness / 2.0,
            self.clip_force / 10.0,
        ], dtype=np.float32)

    def _physics_step(self, F, tau):
        Fz = F[2]
        tau_tilt = np.sqrt(tau[0]**2 + tau[1]**2)

        if Fz > self.clip_force:
            self.position_z += (Fz - self.clip_force) * 0.001
            self.position_z = min(self.position_z, self.extraction_distance)

        self.state.position[2] = self.position_z * 0.001
        self.state.external_wrench = np.concatenate([F, tau])

        violation = tau_tilt > PCB_TORQUE_LIMIT
        if Fz > PCB_FRACTURE_FORCE:
            violation = True
        return violation

    def _compute_progress(self, F, tau):
        return self.position_z / self.extraction_distance

    def _check_success(self):
        return self.position_z >= self.extraction_distance * 0.95


class SnapEnv(BaseEnv):
    """
    SNAP Environment: Sequential Constraint (Section 4.2, Table 2)

    Pull before latch is deflected = violation.
    Safe(x,u) = Fz <= Fmax if delta_latch >= delta_disengage, else Fz <= 0  (Eq. 16)
    """

    def __init__(self, max_steps=1000):
        super().__init__(max_steps)
        self.latch_deflection = 0.0
        self.disengage_threshold = 5.0  # mm
        self.pull_distance = 0.0
        self.extraction_target = 15.0  # mm

    @property
    def task_name(self):
        return 'SNAP'

    def _randomize_parameters(self):
        self.disengage_threshold = np.random.uniform(3.0, 7.0)
        self.spring_k = np.random.uniform(0.5, 2.0)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self.latch_deflection = 0.0
        self.pull_distance = 0.0
        return self._get_obs()

    def _get_task_vars(self):
        return np.array([
            self.latch_deflection / self.disengage_threshold,
            self.pull_distance / self.extraction_target,
            float(self.latch_deflection >= self.disengage_threshold),
        ], dtype=np.float32)

    def _physics_step(self, F, tau):
        F_lateral = F[0]  # lateral force deflects latch
        F_pull = -F[2]    # pull force (negative z)

        # Latch dynamics
        if F_lateral > 1.0:
            self.latch_deflection += F_lateral * 0.01
            self.latch_deflection = min(self.latch_deflection, self.disengage_threshold * 1.5)

        # Pull dynamics
        latch_engaged = self.latch_deflection < self.disengage_threshold
        if F_pull > 0:
            if not latch_engaged:
                self.pull_distance += F_pull * 0.001
            # else pulling while latch engaged does nothing useful

        self.state.external_wrench = np.concatenate([F, tau])

        # Violation (Eq. 16): pulling with force while latch still engaged
        violation = (F_pull > 5.0 and latch_engaged)
        return violation

    def _compute_progress(self, F, tau):
        return self.pull_distance / self.extraction_target

    def _check_success(self):
        return self.pull_distance >= self.extraction_target * 0.95


class PryEnv(BaseEnv):
    """
    PRY Environment: Path Constraint (Table 2)

    Prying tool must stay within 2mm of vertical path.
    Violation: ||d_xy|| > 2 mm
    """

    def __init__(self, max_steps=1000):
        super().__init__(max_steps)
        self.lateral_limit = 2.0  # mm
        self.insertion_depth = 0.0
        self.target_depth = 10.0  # mm

    @property
    def task_name(self):
        return 'PRY'

    def _randomize_parameters(self):
        self.bond_strength = np.random.uniform(5.0, 15.0)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self.insertion_depth = 0.0
        self.state.position[:2] = 0.0
        return self._get_obs()

    def _get_task_vars(self):
        lateral = np.sqrt(self.state.position[0]**2 + self.state.position[1]**2)
        return np.array([
            self.insertion_depth / self.target_depth,
            lateral / (self.lateral_limit * 0.001),
            0.0,
        ], dtype=np.float32)

    def _physics_step(self, F, tau):
        # Lateral drift from force
        self.state.position[0] += F[0] * 1e-5
        self.state.position[1] += F[1] * 1e-5

        # Vertical progress
        if F[2] > self.bond_strength:
            self.insertion_depth += (F[2] - self.bond_strength) * 0.001

        self.state.external_wrench = np.concatenate([F, tau])

        # Violation: lateral deviation > 2mm
        lateral_mm = np.sqrt(self.state.position[0]**2 + self.state.position[1]**2) * 1000
        violation = lateral_mm > self.lateral_limit
        return violation

    def _compute_progress(self, F, tau):
        return self.insertion_depth / self.target_depth

    def _check_success(self):
        return self.insertion_depth >= self.target_depth * 0.95


class CrankEnv(BaseEnv):
    """
    CRANK Environment: Hybrid Kinematic Constraint (Table 2)

    Rotational extraction with phase-dependent constraint.
    Violation: Fz > 5N before rotation complete
    """

    def __init__(self, max_steps=1000):
        super().__init__(max_steps)
        self.angle = 0.0
        self.target_angle = np.pi / 2  # 90 degrees
        self.radius = 0.10  # m

    @property
    def task_name(self):
        return 'CRANK'

    def _randomize_parameters(self):
        self.radius = np.random.uniform(0.08, 0.12)
        self.friction_torque = np.random.uniform(0.1, 0.5)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self.angle = 0.0
        return self._get_obs()

    def _get_task_vars(self):
        return np.array([
            self.angle / self.target_angle,
            np.cos(self.angle),
            np.sin(self.angle),
        ], dtype=np.float32)

    def _physics_step(self, F, tau):
        # Tangential force drives rotation
        tx, ty = -np.sin(self.angle), np.cos(self.angle)
        F_tangent = F[0] * tx + F[1] * ty
        F_radial = abs(F[0] * np.cos(self.angle) + F[1] * np.sin(self.angle))

        # Rotation dynamics
        torque = F_tangent * self.radius
        if abs(torque) > self.friction_torque:
            self.angle += np.sign(torque) * (abs(torque) - self.friction_torque) * 0.01
            self.angle = np.clip(self.angle, 0, self.target_angle * 1.1)

        self.state.theta = self.angle
        self.state.external_wrench = np.concatenate([F, tau])

        # Violation: pulling (Fz > 5N) before rotation complete
        rotation_complete = abs(self.angle - self.target_angle) < np.radians(5)
        violation = (F[2] > 5.0 and not rotation_complete)

        # Also violation if radial force too high
        if F_radial > 10.0:
            violation = True

        return violation

    def _compute_progress(self, F, tau):
        return self.angle / self.target_angle

    def _check_success(self):
        return self.angle >= self.target_angle * 0.95
