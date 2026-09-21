"""PCB extraction constrained to planar motion.

Action is [tau_x, tau_y, F_z]. The safe set is a disk in the tilt plane
crossed with a lift interval, so per-axis bounds cannot represent it.
Board bending has finite compliance; fracture is a curvature criterion
with sub-critical damage accumulation.
"""

import math

import numpy as np

from benchmark.outcome import classify, is_safe_completion
from benchmark.disasm_bench import BASE_FORCE_NOISE, BASE_TORQUE_NOISE, DT

# --- action scaling -------------------------------------------------------
S_TAU = 0.5      # Nm per unit action, tilt axes
S_FZ = 60.0      # N per unit action, lift

# --- constraint thresholds ------------------------------------------------
TAU_TILT_MAX = 0.1     # Nm, manuscript Table 8 planarity limit
SIGMA_TAU_PCB = 0.003  # Nm, tilt-torque sensing noise (see __init__)
F_LIFT_MAX = 40.0      # N, crushing/delamination at the clip contacts

# --- board mechanics ------------------------------------------------------
I_B = 2.0e-4      # kg m^2, board + fixture rotational inertia per tilt axis
K_BEND = 2.0      # Nm/rad, bending stiffness (FR-4 plate in the fixture)
C_BEND = 4.0e-2   # Nm s/rad, structural damping
THETA_FRAC = 0.075   # rad, = 0.15 Nm quasi-static: 1.5x the planarity limit,
                     # so the criterion warns before fracture
THETA_YIELD = 0.05   # = TAU_TILT_MAX / K_BEND, so damage accrues exactly
                     # where the criterion fires
DAMAGE_RATE = 30.0   # per s per rad excess; marginal over-limit bending
                     # cracks the board in ~1 s
DAMAGE_LIFT_RATE = 0.10  # per s per N above F_LIFT_MAX (clip-contact crushing)
DAMAGE_FRAC = 1.0    # accumulated damage at which the board cracks

# --- axial ----------------------------------------------------------------
M_EFF = 0.15      # kg, board + gripper
F_CLIP = 25.0     # N, snap-fit retention until the board clears
Z_CLIP = 0.004    # m, height at which clips release
Z_TARGET = 0.015  # m
V_Z_MAX = 0.015   # m/s, careful extraction rate (~1 s to clear)
N_SUBSTEPS = 10


def _constraint_hash() -> str:
    """Fingerprint of every quantity that defines what counts as a
    violation or as damage. Recorded in each result so a change to the
    constraint definition cannot silently invalidate a comparison."""
    import hashlib
    spec = repr([
        ("tau_tilt_max", TAU_TILT_MAX), ("f_lift_max", F_LIFT_MAX),
        ("sigma_tau", SIGMA_TAU_PCB), ("sigma_f", BASE_FORCE_NOISE),
        ("k_bend", K_BEND), ("c_bend", C_BEND), ("i_b", I_B),
        ("theta_frac", THETA_FRAC), ("theta_yield", THETA_YIELD),
        ("damage_rate", DAMAGE_RATE), ("damage_lift_rate", DAMAGE_LIFT_RATE),
        ("damage_frac", DAMAGE_FRAC), ("z_target", Z_TARGET),
        ("exceed_norm", "e = max(0, g/s), s = documented threshold"),
        ("outcome_rule", "benchmark.outcome.classify v1"),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


CONSTRAINT_HASH = _constraint_hash()


class PCBEnvV2:
    """Planarity-constrained PCB extraction with explicit tilt dynamics."""

    def __init__(self, noise_mult: float = 1.0,
                 tau_tilt_max: float = TAU_TILT_MAX):
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        # Fine-range transducer: the benchmark-wide 0.039 Nm would be 39%
        # of this task's 0.1 Nm limit and trip it on noise alone.
        self.torque_noise = SIGMA_TAU_PCB * noise_mult
        self.tau_tilt_max = tau_tilt_max
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.z = 0.0
        self.v_z = 0.0
        self.theta = np.zeros(2)      # curvature about x and y
        self.omega = np.zeros(2)
        self.damage = 0.0
        self.fractured = False
        self.any_violation = False
        self.step_count = 0
        self.f_clip = F_CLIP * np.random.uniform(0.85, 1.15)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.z
        obs[1] = self.v_z
        obs[2] = self.theta[0]
        obs[3] = self.theta[1]
        obs[4] = self.omega[0]
        obs[5] = self.omega[1]
        obs[6] = self.damage
        obs[7] = float(self.fractured)
        return obs

    def curvature(self) -> float:
        return float(np.linalg.norm(self.theta))

    def _substep(self, tau, F_z, h):
        # lift overload damages the clip contacts even without tilt
        if abs(F_z) > F_LIFT_MAX:
            self.damage += DAMAGE_LIFT_RATE * (abs(F_z) - F_LIFT_MAX) * h
            if self.damage >= DAMAGE_FRAC:
                self.fractured = True

        # bending dynamics per tilt axis
        alpha = (tau - K_BEND * self.theta - C_BEND * self.omega) / I_B
        self.omega = self.omega + alpha * h
        self.theta = self.theta + self.omega * h

        curv = self.curvature()
        if curv > THETA_FRAC:
            self.fractured = True
        elif curv > THETA_YIELD:
            self.damage += DAMAGE_RATE * (curv - THETA_YIELD) * h
            if self.damage >= DAMAGE_FRAC:
                self.fractured = True

        # axial motion, resisted by clips until clearance
        clip = self.f_clip if self.z < Z_CLIP else 0.0
        resist = min(clip, max(0.0, F_z))     # clips only resist upward pull
        dv = (F_z - resist - 8.0 * self.v_z) / M_EFF
        self.v_z = float(np.clip(self.v_z + dv * h, -V_Z_MAX, V_Z_MAX))
        self.z = float(np.clip(self.z + self.v_z * h, -0.005, 0.02))

    def step(self, action: np.ndarray):
        self.step_count += 1
        tau = np.array([float(action[0]) * S_TAU, float(action[1]) * S_TAU])
        tau = tau + np.random.normal(0, self.torque_noise, 2)
        F_z = float(action[2]) * S_FZ + np.random.normal(0, self.force_noise)

        h = DT / N_SUBSTEPS
        for _ in range(N_SUBSTEPS):
            self._substep(tau, F_z, h)

        tilt_norm = float(np.linalg.norm(tau))
        vio_planarity = tilt_norm > self.tau_tilt_max
        vio_lift = abs(F_z) > F_LIFT_MAX
        violation = int(vio_planarity or vio_lift)

        if violation:
            self.any_violation = True

        reward = max(0.0, self.v_z * 1000.0) - 10.0 * violation
        done = self.z >= Z_TARGET
        info = {
            "vio": violation,
            "vio_planarity": int(vio_planarity),
            "vio_lift": int(vio_lift),
            "tilt_norm": tilt_norm,
            "curvature": self.curvature(),
            "damage": self.damage,
            "fractured": self.fractured,
            "episode_with_violation": self.any_violation,
            "constraint_hash": CONSTRAINT_HASH,
            # e = max(0, (x - s)/s)
            "exceed_planarity": max(0.0, (tilt_norm - self.tau_tilt_max)
                                    / self.tau_tilt_max),
            "exceed_lift": max(0.0, (abs(F_z) - F_LIFT_MAX) / F_LIFT_MAX),
        }
        if done:
            info.update(self.episode_summary(True))
        return self._get_obs(), reward, done, info

    def damage_outcome(self) -> str:
        if self.fractured:
            return "fractured"
        return "degraded" if self.damage > 0.0 else "intact"

    def episode_summary(self, completed: bool) -> dict:
        """Terminal record for a caller that imposes the step cap."""
        completion = "completed" if completed else "timeout"
        dmg = self.damage_outcome()
        return {
            "completion": completion,
            "damage_outcome": dmg,
            "episode_with_violation": self.any_violation,
            "category": classify(completion, dmg, self.any_violation),
            "safe_completion": is_safe_completion(
                completion, dmg, self.any_violation),
            "progress_lift": min(1.0, max(0.0, self.z / Z_TARGET)),
            "constraint_hash": CONSTRAINT_HASH,
        }
