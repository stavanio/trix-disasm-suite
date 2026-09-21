"""Screw extraction under commanded axial and angular velocity.

The policy commands (v_z, omega_z, F_r). A shared finite-bandwidth servo
converts the velocity commands into torque and axial force.

Two coupling quantities are distinct and both reported:
    e_cmd  = v_z_cmd - (p/2pi) omega_cmd   command space, a filter can
                                           zero this exactly
    e_real = v_z     - (p/2pi) omega       realized; the thread absorbs a
                                           commanded mismatch as load, so
                                           this lags and grows once the
                                           thread fails

The servo is part of the actuator interface and is identical for every
method. No filter receives a different controller.
"""

import math

import numpy as np

from benchmark.outcome import classify, is_safe_completion
from benchmark.disasm_bench import (
    MU_STATIC, MU_KINETIC, BASE_FORCE_NOISE, BASE_TORQUE_NOISE,
    TORQUE_LIMIT, FORCE_LIMIT, RADIAL_LIMIT, DT,
)

PITCH = 0.00125
RADIUS = 0.004
K_PITCH = PITCH / (2 * math.pi)

I_EFF = 5e-4
M_EFF = 0.5

# --- command interface ----------------------------------------------------
V_CMD_MAX = 0.05     # m/s, commanded axial rate at action = 1
OMEGA_CMD_MAX = 60.0  # rad/s, commanded angular rate at action = 1
S_FRAD = 40.0        # N per unit action, radial

# --- inner servo (shared by every method) ---------------------------------
K_OMEGA = 0.025      # Nm per rad/s; I_EFF / K_OMEGA = 20 ms rotational
K_V = 2000.0         # N per m/s. Sized so a full-scale velocity command
                     # error saturates the axial actuator: a weaker axial
                     # loop cannot load the thread at all against a
                     # 1e6 N/m coupling, and no command could then depart
                     # from the manifold
D_OMEGA = 0.0
D_V = 0.0

OMEGA_MAX = 60.0
V_Z_MAX = 0.05

K_C = 1.0e6
C_C = 4.0e2
F_THREAD_CAPACITY = 35.0
N_SUBSTEPS = 40     # M_EFF / K_V = 0.25 ms, so h must be well below it

ENGAGE_MIN = 0.05
ENGAGE_OK = 0.90
ENGAGE_INTACT = 0.99
DEGRADE_OVERLOAD = 4.0
K_WEAR = 2.0

Z_TARGET = 0.015
PRELOAD = 20.0
EPS_HELIX = 1.0e-3   # m/s, coupling tolerance, applied to both the
                     # commanded and the realized deviation
EPS_CMD_NUM = 1.0e-9  # m/s, numerical exactness bound for a projector


def _constraint_hash() -> str:
    import hashlib
    spec = repr([
        ("eps_helix", EPS_HELIX), ("eps_cmd_num", EPS_CMD_NUM),
        ("pitch", PITCH), ("radius", RADIUS),
        ("v_cmd_max", V_CMD_MAX), ("omega_cmd_max", OMEGA_CMD_MAX),
        ("k_omega", K_OMEGA), ("k_v", K_V), ("d_omega", D_OMEGA), ("d_v", D_V),
        ("torque_limit", TORQUE_LIMIT), ("force_limit", FORCE_LIMIT),
        ("radial_limit", RADIAL_LIMIT),
        ("omega_max", OMEGA_MAX), ("v_z_max", V_Z_MAX),
        ("k_c", K_C), ("c_c", C_C), ("i_eff", I_EFF), ("m_eff", M_EFF),
        ("f_thread_capacity", F_THREAD_CAPACITY), ("k_wear", K_WEAR),
        ("engage_ok", ENGAGE_OK), ("engage_min", ENGAGE_MIN),
        ("engage_intact", ENGAGE_INTACT), ("z_target", Z_TARGET),
        ("sigma_f", BASE_FORCE_NOISE), ("sigma_tau", BASE_TORQUE_NOISE),
        ("n_substeps", N_SUBSTEPS),
        ("violation_space", "commanded_and_realized"),
        ("families", "helix_cmd|helix_real|command_helix|radial_force"),
        ("exceed_norm", "e = max(0, g/s), s = documented threshold"),
        ("outcome_rule", "benchmark.outcome.classify v1"),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


CONSTRAINT_HASH = _constraint_hash()


def project_helix_command(v_cmd, omega_cmd):
    """Orthogonal projection onto v_z = (p/2pi) omega_z in physical units."""
    k = K_PITCH
    s = (k * v_cmd + omega_cmd) / (k * k + 1.0)
    return k * s, s


class ScrewEnvV3:
    """Velocity-commanded screw with a shared inner servo."""

    def __init__(self, noise_mult: float = 1.0, eps_helix: float = EPS_HELIX,
                 k_omega: float = K_OMEGA, k_v: float = K_V):
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.torque_noise = BASE_TORQUE_NOISE * noise_mult
        self.eps_helix = eps_helix
        self.k_omega = k_omega
        self.k_v = k_v
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.theta = 0.0
        self.omega = 0.0
        self.z = 0.0
        self.v_z = 0.0
        self.engagement = 1.0
        self.slip_integral = 0.0
        self.any_violation = False
        self.peak_thread_load = 0.0
        self.saturation_steps = 0
        self.step_count = 0
        self.mu_s = MU_STATIC * np.random.uniform(0.9, 1.1)
        self.mu_k = MU_KINETIC * np.random.uniform(0.9, 1.1)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.theta
        obs[1] = self.z
        obs[2] = self.omega
        obs[3] = self.v_z
        obs[4] = self.engagement
        obs[5] = self.helix_error()
        return obs

    def helix_error(self) -> float:
        return self.v_z - K_PITCH * self.omega

    def _substep(self, v_cmd, omega_cmd, h):
        tau_raw = self.k_omega * (omega_cmd - self.omega) - D_OMEGA * self.omega
        F_raw = self.k_v * (v_cmd - self.v_z) - D_V * self.v_z
        tau_sat = float(np.clip(tau_raw, -TORQUE_LIMIT, TORQUE_LIMIT))
        F_sat = float(np.clip(F_raw, -FORCE_LIMIT, FORCE_LIMIT))
        saturated = (tau_sat != tau_raw) or (F_sat != F_raw)

        if tau_sat * self.omega > 0:
            tau_motor = tau_sat * max(0.0, 1.0 - abs(self.omega) / OMEGA_MAX)
        else:
            tau_motor = tau_sat

        e_pos = self.z - K_PITCH * self.theta
        e_vel = self.v_z - K_PITCH * self.omega
        F_thread = self.engagement * (K_C * e_pos + C_C * e_vel)
        self.peak_thread_load = max(self.peak_thread_load, abs(F_thread))
        tau_thread = -K_PITCH * F_thread

        normal = abs(F_sat) + PRELOAD
        mu = self.mu_k if abs(self.omega) > 0.01 else self.mu_s
        tau_fric = -np.sign(self.omega if abs(self.omega) > 1e-9 else tau_motor) \
            * mu * normal * RADIUS * 0.1

        domega = (tau_motor + tau_thread + tau_fric) / I_EFF
        dv_z = (F_sat - F_thread - 5.0 * self.v_z) / M_EFF

        self.omega = float(np.clip(self.omega + domega * h, -OMEGA_MAX, OMEGA_MAX))
        self.v_z = float(np.clip(self.v_z + dv_z * h, -V_Z_MAX, V_Z_MAX))
        self.theta += self.omega * h
        self.z += self.v_z * h

        wear = K_WEAR * (abs(F_thread) / F_THREAD_CAPACITY) * abs(e_vel)
        self.engagement = max(0.0, self.engagement - wear * h)
        self.slip_integral += abs(e_vel) * h
        if abs(F_thread) > F_THREAD_CAPACITY:
            over = abs(F_thread) / F_THREAD_CAPACITY - 1.0
            self.engagement = max(0.0, self.engagement - DEGRADE_OVERLOAD * over * h)
        if self.engagement < ENGAGE_MIN:
            self.engagement = 0.0
        return saturated, tau_sat, F_sat

    def step(self, action: np.ndarray):
        self.step_count += 1
        v_cmd = float(action[0]) * V_CMD_MAX
        omega_cmd = float(action[1]) * OMEGA_CMD_MAX
        F_rad = float(action[2]) * S_FRAD + np.random.normal(0, self.force_noise)

        e_cmd = v_cmd - K_PITCH * omega_cmd

        h = DT / N_SUBSTEPS
        sat_any = False
        for _ in range(N_SUBSTEPS):
            sat, tau_sat, F_sat = self._substep(v_cmd, omega_cmd, h)
            sat_any |= sat
        if sat_any:
            self.saturation_steps += 1

        e_real = self.helix_error()

        # Commanded coupling is the actionable constraint: a governor acts
        # on the command. Realized deviation lags, since the thread absorbs
        # the mismatch as load until it fails.
        vio_helix_cmd = abs(e_cmd) > self.eps_helix
        vio_helix_real = abs(e_real) > self.eps_helix
        # Two different layers were sharing one name. The velocity bounds
        # are checked on the commanded values, with no realization step;
        # the radial bound is checked on F_rad, which carries sensor noise
        # added after the filter acts.
        vio_command_helix = (abs(v_cmd) > V_CMD_MAX + 1e-12
                             or abs(omega_cmd) > OMEGA_CMD_MAX + 1e-12)
        vio_radial_force = abs(F_rad) > RADIAL_LIMIT
        violation = int(vio_helix_cmd or vio_helix_real
                        or vio_command_helix or vio_radial_force)
        if violation:
            self.any_violation = True

        reward = max(0.0, self.v_z * 1000.0) - 10.0 * violation
        done = self.z >= Z_TARGET
        info = {
            "vio": violation,
            "vio_helix_cmd": int(vio_helix_cmd),
            "vio_helix_real": int(vio_helix_real),
            "vio_command_helix": int(vio_command_helix),
            "vio_radial_force": int(vio_radial_force),
            "helix_error": e_real,
            "helix_error_cmd": e_cmd,
            "engagement": self.engagement,
            "slip_integral": self.slip_integral,
            "peak_thread_load": self.peak_thread_load,
            "saturated": bool(sat_any),
            "episode_with_violation": self.any_violation,
            "constraint_hash": CONSTRAINT_HASH,
            "exceed_helix_cmd": max(0.0, (abs(e_cmd) - self.eps_helix)
                                    / self.eps_helix),
            "exceed_helix_real": max(0.0, (abs(e_real) - self.eps_helix)
                                     / self.eps_helix),
            "exceed_command_helix": max(
                0.0,
                max(abs(v_cmd) / V_CMD_MAX,
                    abs(omega_cmd) / OMEGA_CMD_MAX) - 1.0),
            "exceed_radial_force": max(
                0.0, abs(F_rad) / RADIAL_LIMIT - 1.0),
        }
        if done:
            info.update(self.episode_summary(True))
        return self._get_obs(), reward, done, info

    def damage_outcome(self) -> str:
        if self.engagement < ENGAGE_OK:
            return "stripped"
        return "intact" if self.engagement >= ENGAGE_INTACT else "degraded"

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
            "progress_axial": min(1.0, max(0.0, self.z / Z_TARGET)),
            "constraint_hash": CONSTRAINT_HASH,
        }


def nominal_action(omega_cmd=25.0):
    """On-manifold command: axial rate derived from the rotation rate."""
    v_cmd = K_PITCH * omega_cmd
    return np.array([v_cmd / V_CMD_MAX, omega_cmd / OMEGA_CMD_MAX, 0.0])
