"""Crank backed out by rotation, then drawn off the shaft.

Axial load is constrained while the shaft turns (bearing brinelling) and
permitted once rotation completes. A separate absolute structural limit
applies at any phase; the two are reported independently.
"""

import math

import numpy as np

from benchmark.outcome import classify, is_safe_completion
from benchmark.disasm_bench import (
    MU_KINETIC, BASE_FORCE_NOISE, TORQUE_LIMIT, RADIAL_LIMIT, DT,
)

# --- geometry / inertia ---------------------------------------------------
RADIUS = 0.10        # m, crank arm
MASS = 0.5           # kg
INERTIA = MASS * RADIUS ** 2
M_AXIAL = 0.3        # kg, crank + puller along the shaft
LOAD_TORQUE = 0.3    # Nm, resisting load
TARGET_ROTATIONS = 2.0

# --- constraint thresholds ------------------------------------------------
F_AXIAL_SAFE = 5.0     # N, axial load permitted while the shaft turns
F_AXIAL_MAX = 20.0     # N, absolute structural limit at any phase

# --- damage ---------------------------------------------------------------
DAMAGE_RATE = 0.20     # per s per N above F_AXIAL_SAFE while rotating
DAMAGE_LIMIT = 1.0

# --- extraction -----------------------------------------------------------
Z_TARGET = 0.010       # m, travel to draw the crank off the shaft
V_Z_MAX = 0.05         # m/s
F_RETAIN = 40.0        # N, shaft retention until rotation completes
N_SUBSTEPS = 10


def _constraint_hash() -> str:
    """Fingerprint of every quantity defining a violation or damage."""
    import hashlib
    spec = repr([
        ("f_axial_safe", F_AXIAL_SAFE), ("f_axial_max", F_AXIAL_MAX),
        ("torque_limit", TORQUE_LIMIT), ("radial_limit", RADIAL_LIMIT),
        ("target_rotations", TARGET_ROTATIONS), ("damage_rate", DAMAGE_RATE),
        ("damage_limit", DAMAGE_LIMIT), ("z_target", Z_TARGET),
        ("f_retain", F_RETAIN), ("sigma_f", BASE_FORCE_NOISE),
        ("gate", "rotation_complete"),
        ("exceed_norm", "e = max(0, g/s), s = documented threshold"),
        ("outcome_rule", "benchmark.outcome.classify v1"),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


CONSTRAINT_HASH = _constraint_hash()


class CrankEnvV2:
    """Rotate the crank fully, then extract it axially."""

    def __init__(self, noise_mult: float = 1.0):
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.theta = 0.0
        self.omega = 0.0
        self.z = 0.0
        self.v_z = 0.0
        self.damage = 0.0
        self.damaged = False
        self.any_violation = False
        self.step_count = 0
        self.mu_k = MU_KINETIC * np.random.uniform(0.9, 1.1)
        return self._get_obs()

    def rotation_complete(self) -> bool:
        return abs(self.theta) >= 2 * math.pi * TARGET_ROTATIONS

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.theta
        obs[1] = self.omega
        obs[2] = math.cos(self.theta)
        obs[3] = math.sin(self.theta)
        obs[4] = self.z
        obs[5] = self.v_z
        obs[6] = self.damage
        # explicit phase indicator, observable by every method
        obs[7] = float(self.rotation_complete())
        return obs

    def _substep(self, tau, F_radial, F_z, h):
        rotating = not self.rotation_complete()

        # phase-conditioned damage: axial load while the shaft turns
        if rotating and abs(F_z) > F_AXIAL_SAFE:
            self.damage += DAMAGE_RATE * (abs(F_z) - F_AXIAL_SAFE) * h
            if self.damage >= DAMAGE_LIMIT:
                self.damaged = True

        if rotating:
            bearing = self.mu_k * abs(F_radial) * 0.01
            load = LOAD_TORQUE * np.sign(self.omega + 1e-9)
            net = tau - load - bearing - 0.1 * self.omega
            self.omega = float(np.clip(self.omega + (net / INERTIA) * h, -30, 30))
            self.theta += self.omega * h
        else:
            self.omega *= 0.9

        # axial motion, retained until rotation completes
        retain = F_RETAIN if rotating else 0.0
        reaction = min(retain, max(0.0, F_z))
        dv = (F_z - reaction - 6.0 * self.v_z) / M_AXIAL
        self.v_z = float(np.clip(self.v_z + dv * h, -V_Z_MAX, V_Z_MAX))
        self.z = float(np.clip(self.z + self.v_z * h, -0.002, 0.02))

    def step(self, action: np.ndarray):
        self.step_count += 1
        raw = np.array([float(action[0]) * 30.0, float(action[1]) * 30.0,
                        float(action[2]) * 40.0])
        F_x, F_y, F_z = raw + np.random.normal(0, self.force_noise, 3)

        t_x, t_y = -math.sin(self.theta), math.cos(self.theta)
        r_x, r_y = math.cos(self.theta), math.sin(self.theta)
        F_tangent = F_x * t_x + F_y * t_y
        F_radial = F_x * r_x + F_y * r_y
        tau = F_tangent * RADIUS

        rotating_at_entry = not self.rotation_complete()

        h = DT / N_SUBSTEPS
        for _ in range(N_SUBSTEPS):
            self._substep(tau, F_radial, F_z, h)

        # constraint families, reported separately
        vio_axial_phase = bool(rotating_at_entry and abs(F_z) > F_AXIAL_SAFE)
        vio_axial_struct = bool(abs(F_z) > F_AXIAL_MAX)
        vio_torque = bool(abs(tau) > TORQUE_LIMIT)
        vio_radial = bool(abs(F_radial) > RADIAL_LIMIT)
        violation = int(vio_axial_phase or vio_axial_struct
                        or vio_torque or vio_radial)
        if violation:
            self.any_violation = True

        reward = (abs(self.theta) / (2 * math.pi * TARGET_ROTATIONS)
                  + max(0.0, self.v_z * 100.0) - 10.0 * violation)
        done = self.z >= Z_TARGET
        info = {
            "vio": violation,
            "vio_axial_phase": int(vio_axial_phase),
            "vio_axial_struct": int(vio_axial_struct),
            "vio_torque": int(vio_torque),
            "vio_radial": int(vio_radial),
            "theta": self.theta,
            "rotation_complete": self.rotation_complete(),
            "damage": self.damage,
            "damaged": self.damaged,
            "episode_with_violation": self.any_violation,
            "constraint_hash": CONSTRAINT_HASH,
            # e = max(0, (x - s)/s)
            "exceed_axial_phase": (max(0.0, (abs(F_z) - F_AXIAL_SAFE)
                                       / F_AXIAL_SAFE)
                                   if vio_axial_phase else 0.0),
            "exceed_axial_struct": max(0.0, (abs(F_z) - F_AXIAL_MAX)
                                       / F_AXIAL_MAX),
            "exceed_torque": max(0.0, (abs(tau) - TORQUE_LIMIT) / TORQUE_LIMIT),
            "exceed_radial": max(0.0, (abs(F_radial) - RADIAL_LIMIT)
                                 / RADIAL_LIMIT),
        }
        if done:
            info.update(self.episode_summary(True))
        return self._get_obs(), reward, done, info

    def damage_outcome(self) -> str:
        if self.damaged:
            return "damaged"
        return "degraded" if self.damage > 0.0 else "intact"

    def episode_summary(self, completed: bool) -> dict:
        completion = "completed" if completed else "timeout"
        dmg = self.damage_outcome()
        return {
            "completion": completion,
            "damage_outcome": dmg,
            "episode_with_violation": self.any_violation,
            "category": classify(completion, dmg, self.any_violation),
            "safe_completion": is_safe_completion(
                completion, dmg, self.any_violation),
            "progress_rotation": 0.5 * min(1.0, abs(self.theta)
                / (2 * math.pi * TARGET_ROTATIONS))
                + 0.5 * min(1.0, max(0.0, self.z / Z_TARGET)),
            "constraint_hash": CONSTRAINT_HASH,
        }
