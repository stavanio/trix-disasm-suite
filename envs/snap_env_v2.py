"""Snap-fit release with an explicit latch deflection state.

Pulling is a violation while the catch is engaged and permissible once
the latch clears it, so the constraint is phase-dependent. Release is
hysteretic. All phase state is observable, so no method needs privileged
information to infer it.
"""

import numpy as np

from benchmark.outcome import classify, is_safe_completion
from benchmark.disasm_bench import BASE_FORCE_NOISE, DT

# --- action scaling -------------------------------------------------------
S_DEFLECT = 25.0    # N per unit action, latch arm
S_PULL = 40.0       # N per unit action, extraction axis
S_LAT = 20.0        # N per unit action, off-axis

# --- latch mechanics ------------------------------------------------------
K_LATCH = 5000.0        # N/m, cantilever stiffness
M_LATCH = 0.004         # kg, effective arm mass
C_LATCH = 5.0           # N s/m; zeta ~ 0.56, a moulded polymer arm. Below
                        # ~0.2 a step command overshoots into DELTA_MAX
DELTA_DISENGAGE = 0.002  # m, deflection that clears the catch
DELTA_MAX = 0.006        # m. sigma_delta ~ 0.25 mm and K_LATCH varies +-10%,
                         # so a 2-4 mm window admits no robust command;
                         # 3x the disengagement travel leaves a feasible band

# --- constraint thresholds ------------------------------------------------
F_PULL_SAFE = 5.0    # N, pull permitted while the catch is still engaged
F_LAT_MAX = 15.0     # N, off-axis contact limit

# --- damage ---------------------------------------------------------------
DAMAGE_RATE = 0.25   # per second per N of pull above F_PULL_SAFE while engaged
DAMAGE_BREAK = 1.0

# --- extraction -----------------------------------------------------------
M_PART = 0.08        # kg
Z_CLEAR = 0.001      # m, travel at which the part passes the catch
Z_TARGET = 0.012     # m
V_Z_MAX = 0.05       # m/s
F_CATCH = 60.0       # N, catch reaction while engaged (part cannot pass)
N_SUBSTEPS = 10


def _constraint_hash() -> str:
    """Fingerprint of every quantity defining a violation or damage."""
    import hashlib
    spec = repr([
        ("f_pull_safe", F_PULL_SAFE), ("f_lat_max", F_LAT_MAX),
        ("delta_disengage", DELTA_DISENGAGE), ("delta_max", DELTA_MAX),
        ("k_latch", K_LATCH), ("m_latch", M_LATCH), ("c_latch", C_LATCH),
        ("damage_rate", DAMAGE_RATE), ("damage_break", DAMAGE_BREAK),
        ("f_catch", F_CATCH), ("z_clear", Z_CLEAR), ("z_target", Z_TARGET),
        ("sigma_f", BASE_FORCE_NOISE),
        # evaluation timing is part of the definition, not just thresholds
        ("premature_eval", "substep_any"),
        ("n_substeps", N_SUBSTEPS),
        ("exceed_norm", "e = max(0, g/s), s = documented threshold"),
        ("outcome_rule", "benchmark.outcome.classify v1"),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


CONSTRAINT_HASH = _constraint_hash()


class SnapEnvV2:
    """Snap-fit release with an explicit, hysteretic latch phase."""

    def __init__(self, noise_mult: float = 1.0):
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.delta = 0.0
        self.delta_dot = 0.0
        self.z = 0.0
        self.v_z = 0.0
        self.released = False
        self.latch_damage = 0.0
        self.latch_broken = False
        self.any_violation = False
        self.step_count = 0
        self.k_latch = K_LATCH * np.random.uniform(0.9, 1.1)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.delta
        obs[1] = self.delta_dot
        obs[2] = self.z
        obs[3] = self.v_z
        obs[4] = float(self.released)
        obs[5] = self.latch_damage
        obs[6] = float(self.latch_broken)
        # explicit phase indicator, observable by every method
        obs[7] = float(self.delta >= DELTA_DISENGAGE or self.released)
        return obs

    def disengaged(self) -> bool:
        """True when pulling is permissible: either the latch is currently
        deflected past the catch, or the part has already cleared it."""
        return bool(self.released or self.delta >= DELTA_DISENGAGE)

    def _substep(self, F_deflect, F_pull, h):
        """True if a premature pull held during this substep."""
        # latch arm: driven cantilever; a broken arm no longer resists
        k = 0.0 if self.latch_broken else self.k_latch
        acc = (F_deflect - k * self.delta - C_LATCH * self.delta_dot) / M_LATCH
        self.delta_dot += acc * h
        self.delta = float(np.clip(self.delta + self.delta_dot * h, -0.01, 0.01))
        if abs(self.delta) > DELTA_MAX:
            self.latch_broken = True

        engaged = not self.disengaged()
        premature = bool(engaged and F_pull > F_PULL_SAFE)
        if premature:
            self.latch_damage += DAMAGE_RATE * (F_pull - F_PULL_SAFE) * h
            if self.latch_damage >= DAMAGE_BREAK:
                self.latch_broken = True

        # the catch blocks travel while engaged and intact
        blocking = engaged and not self.latch_broken and self.z < Z_CLEAR
        reaction = min(F_CATCH, max(0.0, F_pull)) if blocking else 0.0
        dv = (F_pull - reaction - 6.0 * self.v_z) / M_PART
        self.v_z = float(np.clip(self.v_z + dv * h, -V_Z_MAX, V_Z_MAX))
        self.z = float(np.clip(self.z + self.v_z * h, -0.002, 0.02))

        # hysteresis: once the part is past the catch it stays released
        if self.z >= Z_CLEAR and (self.delta >= DELTA_DISENGAGE
                                  or self.latch_broken):
            self.released = True
        return premature

    def step(self, action: np.ndarray):
        self.step_count += 1
        F_deflect = float(action[0]) * S_DEFLECT + np.random.normal(0, self.force_noise)
        F_pull = float(action[1]) * S_PULL + np.random.normal(0, self.force_noise)
        F_lat = float(action[2]) * S_LAT + np.random.normal(0, self.force_noise)

        # Per-substep: the latch oscillates about the threshold, so a
        # policy pulling at the boundary can re-engage mid-step.
        h = DT / N_SUBSTEPS
        premature_any = False
        for _ in range(N_SUBSTEPS):
            premature_any |= self._substep(F_deflect, F_pull, h)

        vio_premature = bool(premature_any)
        vio_overstress = bool(abs(self.delta) > DELTA_MAX)
        vio_lateral = bool(abs(F_lat) > F_LAT_MAX)
        violation = int(vio_premature or vio_overstress or vio_lateral)

        if violation:
            self.any_violation = True

        reward = max(0.0, self.v_z * 1000.0) - 10.0 * violation
        done = self.z >= Z_TARGET
        info = {
            "vio": violation,
            "vio_premature": int(vio_premature),
            "vio_overstress": int(vio_overstress),
            "vio_lateral": int(vio_lateral),
            "delta": self.delta,
            "released": self.released,
            "disengaged": self.disengaged(),
            "latch_damage": self.latch_damage,
            "latch_broken": self.latch_broken,
            "episode_with_violation": self.any_violation,
            "constraint_hash": CONSTRAINT_HASH,
            # e = max(0, (x - s)/s)
            "exceed_premature": (max(0.0, (F_pull - F_PULL_SAFE) / F_PULL_SAFE)
                                 if vio_premature else 0.0),
            "exceed_overstress": max(0.0, (abs(self.delta) - DELTA_MAX)
                                     / DELTA_MAX),
            "exceed_lateral": max(0.0, (abs(F_lat) - F_LAT_MAX) / F_LAT_MAX),
        }
        if done:
            info.update(self.episode_summary(True))
        return self._get_obs(), reward, done, info

    def damage_outcome(self) -> str:
        if self.latch_broken:
            return "broken"
        return "degraded" if self.latch_damage > 0.0 else "intact"

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
            "progress_extract": min(1.0, max(0.0, self.z / Z_TARGET)),
            "constraint_hash": CONSTRAINT_HASH,
        }
