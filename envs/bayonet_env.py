"""Bayonet connector release.

Pin in an L-slot: press to unload the detent, rotate through the locking
leg, withdraw. Shears if (F_axial, tau) leaves an elliptical envelope.

    's'   envelope fixed
    'p'   axial component gated by slot leg

Observation layout is identical in both variants. Action is
[F_axial, tau, F_lateral] normalised to [-1, 1].
"""

import hashlib
import math

import numpy as np

from benchmark.disasm_bench import DT, BASE_FORCE_NOISE, BASE_TORQUE_NOISE
from benchmark.outcome import classify, is_safe_completion

S_AXIAL = 60.0
S_TAU = 2.0
S_LAT = 40.0

E_AXIAL = 45.0
E_TAU = 1.5
LAT_LIMIT = 18.0

ROT_LEG_AXIAL_FRAC = 0.25

THETA_RELEASE = math.pi / 3.0
SEAT_DEPTH = 0.004
Z_TARGET = 0.012
DETENT_FORCE = 5.0
RETENTION_FORCE = 16.0

K_ROT = 9.0
K_AXIAL = 2.2e-4
SHEAR_RATE = 6.0e-3
SHEAR_FAIL = 1.0

CAP = 2500


def _hash(variant):
    spec = repr([
        ("variant", variant),
        ("envelope", E_AXIAL, E_TAU),
        ("lateral_limit", LAT_LIMIT),
        ("rot_leg_axial_frac", ROT_LEG_AXIAL_FRAC if variant == "p" else 1.0),
        ("theta_release", THETA_RELEASE),
        ("z_target", Z_TARGET),
        ("scales", S_AXIAL, S_TAU, S_LAT),
        ("shear", SHEAR_RATE, SHEAR_FAIL),
        ("retention", RETENTION_FORCE),
        ("dt", DT),
    ])
    return hashlib.sha256(spec.encode()).hexdigest()[:16]


CONSTRAINT_HASH_S = _hash("s")
CONSTRAINT_HASH_P = _hash("p")


def axial_bound(variant, released):
    """Admissible axial force."""
    if variant == "p" and not released:
        return E_AXIAL * ROT_LEG_AXIAL_FRAC
    return E_AXIAL


def envelope_value(f_axial, tau, variant="s", released=True):
    """Normalised envelope radius; <= 1 is admissible."""
    a = axial_bound(variant, released)
    return math.hypot(float(f_axial) / a, float(tau) / E_TAU)


class BayonetEnv:

    def __init__(self, variant="s"):
        if variant not in ("s", "p"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.constraint_hash = _hash(variant)
        self.reset(seed=0)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.theta = 0.0
        self.z = 0.0
        self.seated = 0.0
        self.shear = 0.0
        self.steps = 0
        self.any_violation = False
        self.viol_steps = 0
        self.peak_envelope = 0.0
        return self._get_obs()

    @property
    def released(self):
        return abs(self.theta) >= THETA_RELEASE

    def _get_obs(self):
        o = np.zeros(10, dtype=np.float32)
        o[0] = self.theta
        o[1] = self.z
        o[2] = self.seated
        o[3] = math.cos(self.theta)
        o[4] = math.sin(self.theta)
        o[5] = self.shear
        o[6] = float(self.released)
        o[7] = float(self.released)
        o[8] = self.z / Z_TARGET
        o[9] = abs(self.theta) / THETA_RELEASE
        return o

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        f_ax = a[0] * S_AXIAL + self.rng.normal(0.0, BASE_FORCE_NOISE)
        tau = a[1] * S_TAU + self.rng.normal(0.0, BASE_TORQUE_NOISE)
        f_lat = a[2] * S_LAT + self.rng.normal(0.0, BASE_FORCE_NOISE)

        env_val = envelope_value(f_ax, tau, self.variant, self.released)
        vio_envelope = env_val > 1.0
        vio_lateral = abs(f_lat) > LAT_LIMIT
        violation = bool(vio_envelope or vio_lateral)
        self.peak_envelope = max(self.peak_envelope, env_val)

        if violation:
            self.viol_steps += 1
            self.any_violation = True
            self.shear += SHEAR_RATE * max(0.0, env_val - 1.0)

        if f_ax < -DETENT_FORCE * 0.5:
            self.seated = min(SEAT_DEPTH, self.seated - f_ax * K_AXIAL * DT)
        if self.seated >= SEAT_DEPTH * 0.6 and not self.released:
            self.theta += K_ROT * (tau / S_TAU) * DT
        if self.released and f_ax > RETENTION_FORCE:
            self.z += K_AXIAL * (f_ax - RETENTION_FORCE) * DT * 8.0

        self.steps += 1
        done = self.z >= Z_TARGET or self.shear >= SHEAR_FAIL
        obs = self._get_obs()
        reward = 12.0 * (self.z / Z_TARGET) + 4.0 * min(
            1.0, abs(self.theta) / THETA_RELEASE) - 10.0 * float(violation)

        info = {
            "vio": int(violation),
            "vio_envelope": int(vio_envelope),
            "vio_lateral": int(vio_lateral),
            "exceed_envelope": max(0.0, env_val - 1.0),
            "exceed_lateral": max(0.0, abs(f_lat) / LAT_LIMIT - 1.0),
            "envelope_value": env_val,
            "released": bool(self.released),
            "constraint_hash": self.constraint_hash,
        }
        return obs, reward, done, info

    def episode_summary(self, completed):
        sheared = self.shear >= SHEAR_FAIL
        completion = ("completed" if (completed and not sheared)
                      else "failed" if sheared else "timeout")
        damage = "sheared" if sheared else "intact"
        return {
            "completion": completion,
            "damage_outcome": damage,
            "episode_with_violation": self.any_violation,
            "category": classify(completion, damage, self.any_violation),
            "safe_completion": is_safe_completion(
                completion, damage, self.any_violation),
            "progress_release": min(1.0, max(0.0, self.z / Z_TARGET)),
            "violating_steps": self.viol_steps,
            "total_steps": self.steps,
            "peak_envelope": self.peak_envelope,
            "constraint_hash": self.constraint_hash,
        }


def make_static():
    return BayonetEnv("s")


def make_phased():
    return BayonetEnv("p")


def nominal_action(t, released, press=-0.35, turn=0.45, pull=0.62):
    """Press, turn, withdraw."""
    if not released:
        return np.array([press, turn, 0.0])
    return np.array([pull, 0.0, 0.0])


class BayonetS(BayonetEnv):
    def __init__(self):
        super().__init__("s")


class BayonetP(BayonetEnv):
    def __init__(self):
        super().__init__("p")
