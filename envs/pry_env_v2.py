"""Pry-bar separation with an insertion-dependent torque limit.

The admissible torque shrinks as the bar goes deeper, so the constraint
is state-dependent even though each bound is axis-aligned once the depth
is known. Dynamics and violation criteria are those of the submitted
environment; what is added here is a damage state and the shared outcome
contract, since the original reported only a violation flag.

The manuscript described this task by a lateral-displacement criterion
that the environment never implemented. The implemented criteria are
insertion-dependent torque, insertion force and lateral force, and those
are what the revised text describes.
"""

import numpy as np

from benchmark.outcome import classify, is_safe_completion
from benchmark.disasm_bench import (
    State, FrictionState, BASE_FORCE_NOISE, BASE_TORQUE_NOISE,
    TORQUE_LIMIT, FORCE_LIMIT, RADIAL_LIMIT, DT,
)

LEVER_LENGTH = 0.15
MASS = 0.2
INERTIA = (1.0 / 3.0) * MASS * LEVER_LENGTH ** 2
BOND_STRENGTH = 7.0  # N. At 150 the bond could not be released by any
                     # torque inside the limit, so the task could not be
                     # completed without violating and a filter that
                     # clipped torque scored zero violations and zero
                     # completions. Set between the pry force available
                     # shallow (10 N at 1.5 Nm) and deep (5 N at 0.75 Nm),
                     # so depth decides whether release is still possible
BOND_STIFFNESS = 5000.0
MAX_INSERTION = 0.05
Z_TARGET = 0.015

# Over-torquing past the depth-dependent limit stresses the housing. The
# original environment flagged this without consequence, so a policy could
# violate freely and still be scored as completing the task.
K_CRACK = 0.4        # per second per Nm above the limit
CRACK_LIMIT = 1.0
CRACK_INTACT = 0.05  # noise excursions are ordinary handling


def dynamic_torque_limit(insertion):
    depth = min(1.0, abs(insertion) / MAX_INSERTION)
    return TORQUE_LIMIT * (1.0 - 0.5 * depth)


def _constraint_hash() -> str:
    import hashlib
    spec = repr([
        ("torque_limit", TORQUE_LIMIT), ("force_limit", FORCE_LIMIT),
        ("radial_limit", RADIAL_LIMIT), ("max_insertion", MAX_INSERTION),
        ("depth_derate", 0.5), ("k_crack", K_CRACK),
        ("crack_limit", CRACK_LIMIT), ("crack_intact", CRACK_INTACT),
        ("bond_strength", BOND_STRENGTH), ("bond_stiffness", BOND_STIFFNESS),
        ("z_target", Z_TARGET), ("sigma_f", BASE_FORCE_NOISE),
        ("sigma_tau", BASE_TORQUE_NOISE),
        ("exceed_norm", "e = max(0, (x - s)/s)"),
        ("outcome_rule", "benchmark.outcome.classify v1"),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


CONSTRAINT_HASH = _constraint_hash()


class PryEnvV2:
    """Pry separation with a depth-dependent torque bound and crack state."""

    def __init__(self, noise_mult: float = 1.0):
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.torque_noise = BASE_TORQUE_NOISE * noise_mult
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state = State()
        self.state.reset()
        self.crack = 0.0
        self.cracked = False
        self.any_violation = False
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.state.theta
        obs[1] = self.state.omega
        obs[2] = self.state.position[2]
        obs[3] = float(self.state.friction.value)
        obs[4] = self.state.position[0]
        obs[5] = dynamic_torque_limit(self.state.position[0])
        obs[6] = self.crack
        return obs

    def step(self, action: np.ndarray):
        self.step_count += 1
        tau = float(action[0]) * 2.5 + np.random.normal(0, self.torque_noise)
        F_ins = float(action[1]) * 80.0 + np.random.normal(0, self.force_noise)
        F_lat = float(action[2]) * 40.0 + np.random.normal(0, self.force_noise)

        self.state.position[0] = float(np.clip(
            self.state.position[0] + F_ins * DT * 0.0001, 0.0, MAX_INSERTION))
        tau_lim = dynamic_torque_limit(self.state.position[0])

        vio_torque = abs(tau) > tau_lim
        vio_force = abs(F_ins) > FORCE_LIMIT
        vio_lateral = abs(F_lat) > RADIAL_LIMIT
        violation = int(vio_torque or vio_force or vio_lateral)
        if violation:
            self.any_violation = True

        if abs(tau) > tau_lim:
            self.crack += K_CRACK * (abs(tau) - tau_lim) * DT
            if self.crack >= CRACK_LIMIT:
                self.cracked = True

        pry_force = tau / LEVER_LENGTH
        gap = float(self.state.position[2])
        if self.state.friction == FrictionState.STUCK:
            bond = BOND_STIFFNESS * gap
            if pry_force > BOND_STRENGTH or gap > 0.005:
                self.state.friction = FrictionState.SLIPPING
                bond = 0.0
        else:
            bond = 0.0

        acc = (pry_force - bond) / MASS
        self.state.velocity[2] += acc * DT * 0.001
        self.state.velocity[2] *= 0.95
        self.state.position[2] = float(np.clip(
            self.state.position[2] + self.state.velocity[2] * DT, 0.0, 0.02))

        self.state.omega += (tau / INERTIA) * DT
        self.state.omega *= 0.95
        self.state.theta += self.state.omega * DT

        reward = float(self.state.position[2]) * 100.0 - 10.0 * violation
        done = float(self.state.position[2]) >= Z_TARGET
        info = {
            "vio": violation,
            "vio_torque": int(vio_torque),
            "vio_force": int(vio_force),
            "vio_lateral": int(vio_lateral),
            "insertion": float(self.state.position[0]),
            "torque_limit": tau_lim,
            "crack": self.crack,
            "episode_with_violation": self.any_violation,
            "constraint_hash": CONSTRAINT_HASH,
            "exceed_torque": max(0.0, (abs(tau) - tau_lim) / max(tau_lim, 1e-9)),
            "exceed_force": max(0.0, (abs(F_ins) - FORCE_LIMIT) / FORCE_LIMIT),
            "exceed_lateral": max(0.0, (abs(F_lat) - RADIAL_LIMIT) / RADIAL_LIMIT),
        }
        if done:
            info.update(self.episode_summary(True))
        return self._get_obs(), reward, done, info

    def damage_outcome(self) -> str:
        if self.cracked:
            return "cracked"
        return "degraded" if self.crack > CRACK_INTACT else "intact"

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
            "progress_gap": min(1.0, max(0.0, float(self.state.position[2]) / Z_TARGET)),
            "constraint_hash": CONSTRAINT_HASH,
        }
