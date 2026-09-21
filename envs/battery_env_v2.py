"""Adhesive battery extraction with a delayed thermal consequence.

Safety here depends on state history rather than on the current command.
Mechanical abuse accumulates deformation damage; past a threshold that
damage initiates an internal short whose heating continues after the
initiating force is removed. An instantaneous projection can stop the
abuse but cannot reverse it.

This is a reduced-order model of the documented failure path
(deformation, separator damage, internal short, heating). It is sized so
the hazard threshold is reachable within an episode and is not intended
to predict pouch-cell behaviour quantitatively; the constants carry a
sensitivity sweep rather than a fidelity claim. Nothing beyond the
hazard threshold is modelled.
"""

import math

import numpy as np

from benchmark.outcome import classify, is_safe_completion
from benchmark.disasm_bench import (
    BASE_FORCE_NOISE, BATTERY_FORCE_LIMIT, BATTERY_TEMP_LIMIT, DT,
)

S_PEEL = 40.0
S_LAT = 30.0

F_PUNCTURE = BATTERY_FORCE_LIMIT          # 20 N
F_LAT_MAX = BATTERY_FORCE_LIMIT * 0.75    # 15 N
T_CRIT = BATTERY_TEMP_LIMIT               # 60 C
T_AMBIENT = 25.0

# --- peel mechanics -------------------------------------------------------
ADHESIVE = 8.0       # N, steady peel resistance, present from first contact.
                     # Set below the deformation onset so a feasible peel
                     # window exists: at 12 N the force needed to peel
                     # overlapped the force that deforms the cell
M_EFF = 0.2          # kg
C_DAMP = 400.0       # N s/m, viscous adhesive drag; bounds peel rate near
                     # 1e-2 m/s under the admissible force range
V_MAX = 0.05
Z_TARGET = 0.015
N_SUBSTEPS = 10

# --- deformation ----------------------------------------------------------
F_DEFORM = 15.0      # N, onset of plastic deformation, below the puncture
                     # limit so damage can precede a flagged violation
K_DEFORM = 0.6       # per second per N above onset
D_SHORT = 1.0        # deformation at which an internal short initiates
D_INTACT = 0.05      # below this the cell counts as undamaged; noise
                     # excursions past the onset are ordinary handling and
                     # an exact-zero test would mark every episode degraded

# --- internal short and thermal ------------------------------------------
P_ISC_MAX = 250.0    # W at full short severity, local dissipation
C_TH = 10.0          # J/K, locally heated region
H_COOL = 5.0         # W/K; C_TH / H_COOL = 2 s thermal time constant
G_RATE = 1.5         # per second, growth of short severity once initiated


def _constraint_hash() -> str:
    import hashlib
    spec = repr([
        ("f_puncture", F_PUNCTURE), ("f_lat_max", F_LAT_MAX),
        ("t_crit", T_CRIT), ("f_deform", F_DEFORM), ("k_deform", K_DEFORM),
        ("d_short", D_SHORT), ("d_intact", D_INTACT),
        ("p_isc_max", P_ISC_MAX), ("c_th", C_TH),
        ("h_cool", H_COOL), ("g_rate", G_RATE), ("adhesive", ADHESIVE),
        ("c_damp", C_DAMP), ("m_eff", M_EFF), ("z_target", Z_TARGET),
        ("sigma_f", BASE_FORCE_NOISE), ("n_substeps", N_SUBSTEPS),
        ("exceed_norm", "e = max(0, (x - s)/s)"),
        ("outcome_rule", "benchmark.outcome.classify v1"),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


CONSTRAINT_HASH = _constraint_hash()


class BatteryEnvV2:
    """Peel extraction with staged mechanical, electrical and thermal state."""

    def __init__(self, noise_mult: float = 1.0):
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.z = 0.0
        self.v_z = 0.0
        self.deform = 0.0
        self.short = 0.0
        self.temperature = T_AMBIENT
        self.peak_temperature = T_AMBIENT
        self.any_violation = False
        self.step_count = 0
        self.adhesive = ADHESIVE * np.random.uniform(0.9, 1.1)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.z
        obs[1] = self.v_z
        obs[2] = self.deform
        obs[3] = self.short
        obs[4] = self.temperature / 100.0
        obs[5] = float(self.short > 0.0)
        return obs

    def _substep(self, F_peel, F_lat, h):
        # deformation from abuse, on either axis
        over = max(abs(F_peel) - F_DEFORM, abs(F_lat) - F_DEFORM, 0.0)
        if over > 0.0:
            self.deform += K_DEFORM * over * h

        # an internal short initiates once deformation crosses the threshold
        # and then grows on its own: removing the force does not undo it
        if self.deform >= D_SHORT:
            self.short = min(1.0, self.short + G_RATE * h)

        p_isc = P_ISC_MAX * self.short
        self.temperature += (p_isc - H_COOL * (self.temperature - T_AMBIENT)) / C_TH * h
        self.peak_temperature = max(self.peak_temperature, self.temperature)

        resist = self.adhesive if self.z < Z_TARGET else 0.0
        drive = F_peel - min(resist, max(0.0, F_peel))
        acc = (drive - C_DAMP * self.v_z) / M_EFF
        self.v_z = float(np.clip(self.v_z + acc * h, -V_MAX, V_MAX))
        self.z = float(np.clip(self.z + self.v_z * h, 0.0, 0.02))

    def step(self, action: np.ndarray):
        self.step_count += 1
        raw = np.array([float(action[0]) * S_PEEL, float(action[1]) * S_LAT,
                        float(action[2]) * S_LAT])
        F_peel, F_l1, F_l2 = raw + np.random.normal(0, self.force_noise, 3)
        F_lateral = math.hypot(F_l1, F_l2)

        h = DT / N_SUBSTEPS
        for _ in range(N_SUBSTEPS):
            self._substep(F_peel, F_lateral, h)

        vio_puncture = abs(F_peel) > F_PUNCTURE
        vio_lateral = F_lateral > F_LAT_MAX
        vio_thermal = self.temperature > T_CRIT
        violation = int(vio_puncture or vio_lateral or vio_thermal)
        if violation:
            self.any_violation = True

        reward = max(0.0, self.v_z * 1000.0) - 10.0 * violation
        done = self.z >= Z_TARGET
        info = {
            "vio": violation,
            "vio_puncture": int(vio_puncture),
            "vio_lateral": int(vio_lateral),
            "vio_thermal": int(vio_thermal),
            "deformation": self.deform,
            "short_severity": self.short,
            "temperature": self.temperature,
            "peak_temperature": self.peak_temperature,
            "episode_with_violation": self.any_violation,
            "constraint_hash": CONSTRAINT_HASH,
            "exceed_puncture": max(0.0, (abs(F_peel) - F_PUNCTURE) / F_PUNCTURE),
            "exceed_lateral": max(0.0, (F_lateral - F_LAT_MAX) / F_LAT_MAX),
            "exceed_thermal": max(0.0, (self.temperature - T_CRIT) / T_CRIT),
        }
        if done:
            info.update(self.episode_summary(True))
        return self._get_obs(), reward, done, info

    def damage_outcome(self) -> str:
        if self.short > 0.0 or self.temperature > T_CRIT:
            return "damaged"
        return "degraded" if self.deform > D_INTACT else "intact"

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
            "progress_peel": min(1.0, max(0.0, self.z / Z_TARGET)),
            "constraint_hash": CONSTRAINT_HASH,
        }
