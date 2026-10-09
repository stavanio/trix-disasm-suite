#!/usr/bin/env python3

import os
import runpy
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pinocchio as pin

import rebot
from rebot import ReBotRSMITController as GoldController


ALPHA_TARGET = float(os.environ.get("TRIX_GRAVITY_ALPHA", "0.0"))
RAMP_SECONDS = 2.0

if not 0.0 <= ALPHA_TARGET <= 0.50:
    raise ValueError(
        f"TRIX_GRAVITY_ALPHA must be between 0.0 and 0.50, got {ALPHA_TARGET}"
    )

URDF = (
    Path.home()
    / "reBotArm_control_py"
    / "urdf"
    / "00-arm-rs_asm-v3"
    / "urdf"
    / "00-arm-rs_asm-v3.urdf"
)


class J3GravityController(GoldController):
    """
    Experimental J3-only gravity feed-forward.

    Everything else remains inherited from the frozen gold controller:
      - same Kp/Kd
      - same 200 Hz rate
      - same trajectory/rate limiting
      - same stop behavior

    Gravity:
      - Pinocchio model has 8 q coordinates.
      - q[0:6] = physical arm J1-J6.
      - model finger joints remain zero.
      - physical gripper motor J7 receives zero gravity feed-forward.
      - ONLY physical J3 receives gravity FF in this experiment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._gravity_model = pin.buildModelFromUrdf(str(URDF))
        self._gravity_data = self._gravity_model.createData()

        if self._gravity_model.nq != 8:
            raise RuntimeError(
                f"Expected RS model nq=8, got {self._gravity_model.nq}"
            )

        self._gravity_start_time = None
        self._gravity_counter = 0

        print(
            f"[GRAVITY EXPERIMENT] J3 only | "
            f"alpha_target={ALPHA_TARGET:.3f} | "
            f"ramp={RAMP_SECONDS:.1f}s"
        )
        print(f"[GRAVITY MODEL] {URDF}")

    def _gravity_for_commands(self, positions_rad: Sequence[float]):
        q = np.zeros(self._gravity_model.nq, dtype=float)

        # Physical arm J1-J6 map directly to URDF joint1-joint6.
        q[:6] = np.asarray(positions_rad[:6], dtype=float)

        pin.computeGeneralizedGravity(
            self._gravity_model,
            self._gravity_data,
            q,
        )

        return np.asarray(self._gravity_data.g, dtype=float).copy()

    def _effective_alpha(self):
        if self._gravity_start_time is None:
            self._gravity_start_time = time.monotonic()
            return 0.0

        elapsed = time.monotonic() - self._gravity_start_time
        ramp = min(1.0, max(0.0, elapsed / RAMP_SECONDS))

        return ALPHA_TARGET * ramp

    def _send_mit_positions(
        self,
        positions_rad: Sequence[float],
        *,
        lock_timeout: float | None = None,
    ) -> None:

        gravity = self._gravity_for_commands(positions_rad)
        alpha = self._effective_alpha()

        with self._io_lock_guard(lock_timeout):
            for index, motor in enumerate(self.motors):
                motor_config = self.config.motors[index]

                # FIRST PHYSICAL EXPERIMENT:
                # J3 ONLY. All other joints stay tau_ff = 0.
                tau_ff = 0.0

                if index == 2:  # physical J3
                    tau_ff = alpha * float(gravity[2])

                motor.send_mit(
                    float(positions_rad[index]),
                    0.0,
                    float(motor_config.kp),
                    float(motor_config.kd),
                    float(tau_ff),
                )

        self._gravity_counter += 1

        # Approx once per second at the 200 Hz gold control rate.
        if self._gravity_counter % 200 == 0:
            print(
                f"[GRAV] "
                f"alpha={alpha:.3f} "
                f"gJ3={gravity[2]:+.3f}Nm "
                f"tauFF_J3={alpha * gravity[2]:+.3f}Nm"
            )


# j3_closed_loop_proof.py imports this name from `rebot`.
# Replace it only inside this Python process.
rebot.ReBotRSMITController = J3GravityController

print("=" * 72)
print("RUNNING FROZEN J3 PROOF THROUGH EXPERIMENTAL GRAVITY WRAPPER")
print(f"TRIX_GRAVITY_ALPHA = {ALPHA_TARGET}")
print("GOLD controller source itself is UNMODIFIED")
print("=" * 72)

runpy.run_path(
    str(Path(__file__).resolve().parent / "trix_physical_manipulation_v4_soft_grasp.py"),
    run_name="__main__",
)
