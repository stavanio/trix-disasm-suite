#!/usr/bin/env python3
from pathlib import Path

import time

from rebot import ReBotRSMITController

from trix_runtime import (
    Action,
    DecisionType,
    GripperState,
    Proposal,
    RobotState,
    TrixExecutionBridge,
    TrixPhysicalTargetAdapter,
    TrixProjector,
)


SPEEDS_DEG_S = [
    4.50,   # J1 HOLD
    3.60,   # J2
    1.50,   # J3
    3.60,   # J4
    4.50,   # J5
    4.50,   # J6
    6.00,   # J7 gripper
]

TIMEOUT_S = 45.0
TOL_DEG = 0.08
HOLD_S = 0.8


EXPECTED = {
    Action.OPEN_GRIPPER:
        [0, 0, 0, 0, 0, 0, 40],

    Action.APPROACH:
        [0, 18, 0, 0, 0, 0, 40],

    Action.PRESHAPE:
        [0, 18, 0, 14, 0, 0, 40],

    Action.LOW_HOVER:
        [0, 30, 0, 14, 0, 0, 40],

    Action.DEEP_HOVER:
        [0, 36, -3, 14, 0, 0, 40],

    Action.ENGAGE:
        [0, 36, -3, 14, 0, 0, 0],

    Action.RETURN_LOW_HOVER:
        [0, 30, 0, 14, 0, 0, 0],

    Action.RETRACT:
        [0, 18, 0, 14, 0, 0, 0],

    Action.UNPRESHAPE:
        [0, 18, 0, 0, 0, 0, 0],

    Action.RETURN_SAFE:
        [0, 0, 0, 0, 0, 0, 0],

    Action.RELEASE:
        [0, 0, 0, 0, 0, 0, 40],

    Action.CLOSE_GRIPPER:
        [0, 0, 0, 0, 0, 0, 0],
}


SEQUENCE = (
    Action.OPEN_GRIPPER,
    Action.APPROACH,
    Action.PRESHAPE,
    Action.LOW_HOVER,
    Action.DEEP_HOVER,
    Action.ENGAGE,
    Action.RETURN_LOW_HOVER,
    Action.RETRACT,
    Action.UNPRESHAPE,
    Action.RETURN_SAFE,
    Action.RELEASE,
    Action.CLOSE_GRIPPER,
)



# Hardware-specific gripper calibration for the 2-inch cardboard object.
#
# TRiX continues to reason symbolically about OPEN/CLOSED.
# These values map those states to physical J7 actuator positions.
HW_J7_OPEN_DELTA_DEG = 180.0
HW_J7_GRASP_DELTA_DEG = 165.0
HW_J7_STOW_DELTA_DEG = 0.0


def hardware_j7_delta(action):
    if action in (
        Action.OPEN_GRIPPER,
        Action.APPROACH,
        Action.PRESHAPE,
        Action.LOW_HOVER,
        Action.DEEP_HOVER,
        Action.RELEASE,
    ):
        return HW_J7_OPEN_DELTA_DEG

    if action in (
        Action.ENGAGE,
        Action.RETURN_LOW_HOVER,
        Action.RETRACT,
        Action.UNPRESHAPE,
        Action.RETURN_SAFE,
    ):
        return HW_J7_GRASP_DELTA_DEG

    if action == Action.CLOSE_GRIPPER:
        return HW_J7_STOW_DELTA_DEG

    raise RuntimeError(
        f"NO_HARDWARE_J7_MAPPING:{action.value}"
    )

def fmt(values):
    return "  ".join(
        f"J{i+1}={float(v):+7.2f}"
        for i, v in enumerate(values)
    )


def wait_command(arm, target):

    deadline = time.monotonic() + TIMEOUT_S

    while True:

        command = arm.get_command_angles()

        error = max(
            abs(float(a) - float(b))
            for a, b in zip(command, target)
        )

        if error <= TOL_DEG:
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"COMMAND_TIMEOUT error={error:.3f}"
            )

        time.sleep(0.05)


def build(
    trix,
    bridge,
    adapter,
    state,
    action,
):

    proposal = Proposal(
        action,
        source="trix_physical_manipulation_v4_soft_grasp",
    )

    decision = trix.evaluate(
        state,
        proposal,
    )

    if decision.decision == DecisionType.REJECT:
        raise RuntimeError(
            f"TRIX_REJECT:{action.value}:"
            + ";".join(decision.reasons)
        )

    command = bridge.build(
        state,
        decision,
    )

    physical = adapter.build(
        command,
    )

    if physical is None:
        raise RuntimeError(
            f"NO_PHYSICAL_TARGET:{action.value}"
        )

    adapter.assert_j1_invariant(
        physical.angles_deg
    )

    next_state = trix.apply(
        state,
        decision,
    )

    return (
        decision,
        physical,
        next_state,
    )


def verify_target(
    captured,
    action,
    target,
):

    delta = [
        float(t) - float(s)
        for s, t in zip(
            captured,
            target,
        )
    ]

    expected = EXPECTED[action]

    for index, (actual, exp) in enumerate(
        zip(delta, expected),
        1,
    ):

        if abs(actual - exp) > 1e-6:
            raise RuntimeError(
                f"{action.value}:"
                f"BAD_J{index}_DELTA:"
                f"{actual:+.6f},"
                f"expected={exp:+.6f}"
            )

    return delta


def execute(
    arm,
    trix,
    bridge,
    adapter,
    captured,
    state,
    action,
):

    decision, physical, next_state = build(
        trix,
        bridge,
        adapter,
        state,
        action,
    )

    symbolic_target = physical.angles_deg

    # Verify the frozen TRiX symbolic command before hardware mapping.
    verify_target(
        captured,
        action,
        symbolic_target,
    )

    target = list(symbolic_target)

    # Hardware-specific J7 mapping only.
    # J1-J6 remain exactly the TRiX-approved target.
    target[6] = (
        float(captured[6])
        + hardware_j7_delta(action)
    )

    adapter.assert_j1_invariant(target)

    delta = [
        float(t) - float(c)
        for c, t in zip(captured, target)
    ]

    print()
    print("=" * 80)

    print(
        f"{state.mode.value}/"
        f"{state.gripper_state.value}"
        f" -> {action.value}"
        f" -> {decision.decision.value}"
    )

    print("=" * 80)

    print("TARGET:")
    print(fmt(target))

    print()
    print("DELTA:")

    for index, value in enumerate(
        delta,
        1,
    ):
        print(
            f"  J{index}: "
            f"{value:+.3f} deg"
        )

    arm.set_joint_angles(
        target
    )

    wait_command(
        arm,
        target,
    )

    time.sleep(0.40)

    actual = arm.read_joint_angles()

    print()
    print("ACTUAL:")
    print(fmt(actual))

    print()
    print("ACTUAL - TARGET:")

    for index, (a, t) in enumerate(
        zip(actual, target),
        1,
    ):

        print(
            f"  J{index}: "
            f"{float(a)-float(t):+.3f} deg"
        )

    print()
    print(
        f"STATE AFTER: "
        f"{next_state.mode.value}/"
        f"{next_state.gripper_state.value}"
        f" J7_delta="
        f"{next_state.gripper_delta_deg:+.1f}"
    )

    time.sleep(HOLD_S)

    return next_state


def main():

    arm = ReBotRSMITController()
    started = False

    try:

        print("=" * 80)
        print("TRIX MANIPULATION V4 — SOFT OBJECT GRASP")
        print("FULL REACH + GRIPPER STATE CHAIN")
        print("J1 HARD-FROZEN")
        print("=" * 80)

        arm.start(
            enable_esc=False,
            install_signal_handlers=False,
        )

        started = True

        arm.set_max_speeds(
            SPEEDS_DEG_S
        )

        captured = arm.read_joint_angles()

        print()
        print("CAPTURED START:")
        print(fmt(captured))

        trix = TrixProjector()
        bridge = TrixExecutionBridge()

        adapter = TrixPhysicalTargetAdapter(
            captured
        )

        # --------------------------------------------------------
        # PRECOMPUTE / VERIFY FULL TRAJECTORY BEFORE MOVING
        # --------------------------------------------------------

        state = RobotState()

        print()
        print("=" * 80)
        print("FULL PRE-MOTION TRIX PLAN")
        print("=" * 80)

        preview_state = state

        for action in SEQUENCE:

            decision, physical, next_state = build(
                trix,
                bridge,
                adapter,
                preview_state,
                action,
            )

            delta = verify_target(
                captured,
                action,
                physical.angles_deg,
            )

            print(
                f"{preview_state.mode.value}/"
                f"{preview_state.gripper_state.value}"
                f" -> {action.value}"
                f" -> {decision.decision.value}"
                f" | "
                + " ".join(
                    f"J{i+1}={x:+.0f}"
                    for i, x in enumerate(delta)
                )
            )

            preview_state = next_state

        if (
            preview_state.mode.value
            != "START_SAFE"
        ):
            raise RuntimeError(
                "PREVIEW_NOT_START_SAFE"
            )

        if (
            preview_state.gripper_state
            != GripperState.CLOSED
        ):
            raise RuntimeError(
                "PREVIEW_GRIPPER_NOT_CLOSED"
            )

        print()
        print(
            "PRE-MOTION PLAN VALIDATED."
        )

        time.sleep(2.0)

        # --------------------------------------------------------
        # PHYSICAL EXECUTION
        # --------------------------------------------------------

        state = RobotState()

        # Known-good TRiX path to DEEP_HOVER, gripper OPEN.
        for action in (
            Action.OPEN_GRIPPER,
            Action.APPROACH,
            Action.PRESHAPE,
            Action.LOW_HOVER,
            Action.DEEP_HOVER,
        ):
            state = execute(
                arm,
                trix,
                bridge,
                adapter,
                captured,
                state,
                action,
            )

        # Hardware-calibrated GRASP_READY:
        # additional +5 deg J2 from validated DEEP_HOVER.
        deep_target = [
            float(captured[i]) + float(EXPECTED[Action.DEEP_HOVER][i])
            for i in range(7)
        ]

        # Keep the gripper physically wide open during descent.
        deep_target[6] = (
            float(captured[6])
            + HW_J7_OPEN_DELTA_DEG
        )

        grasp_target = list(deep_target)
        grasp_target[1] += 5.0

        # J1 remains absolutely frozen.
        if abs(grasp_target[0] - captured[0]) > 1e-9:
            raise RuntimeError("J1_GRASP_READY_INVARIANT_BROKEN")

        print()
        print("=" * 80)
        print("LOWERING TO GRASP_READY: J2 +5 DEG FROM DEEP_HOVER")
        print("=" * 80)

        arm.set_joint_angles(grasp_target)
        wait_command(arm, grasp_target)
        time.sleep(0.50)

        actual = arm.read_joint_angles()

        print("GRASP_READY TARGET:")
        print(fmt(grasp_target))
        print()
        print("GRASP_READY ACTUAL:")
        print(fmt(actual))

        go_file = Path("/tmp/trix_grasp_go")
        go_file.unlink(missing_ok=True)

        print()
        print("=" * 80)
        print("GRASP_READY REACHED AND HOLDING — GRIPPER OPEN")
        print("Position box between the jaws.")
        print()
        print("When clear, Terminal 2:")
        print("    touch /tmp/trix_grasp_go")
        print("=" * 80)

        while not go_file.exists():
            time.sleep(0.10)

        go_file.unlink(missing_ok=True)

        # Evaluate ENGAGE through TRiX.
        decision, physical, next_state = build(
            trix,
            bridge,
            adapter,
            state,
            Action.ENGAGE,
        )

        if decision.decision == DecisionType.REJECT:
            raise RuntimeError(
                "TRIX_REJECT:ENGAGE:"
                + ";".join(decision.reasons)
            )

        # Keep the calibrated GRASP_READY arm pose.
        # Change only J7 to the TRiX-approved CLOSED target.
        engage_target = list(grasp_target)
        engage_target[6] = (
            float(captured[6])
            + HW_J7_GRASP_DELTA_DEG
        )

        if abs(engage_target[0] - captured[0]) > 1e-9:
            raise RuntimeError("J1_ENGAGE_INVARIANT_BROKEN")

        print()
        print("=" * 80)
        print("ENGAGE -> ALLOW")
        print("CLOSING J7 ONLY AT GRASP_READY")
        print("=" * 80)
        print("TARGET:")
        print(fmt(engage_target))

        arm.set_joint_angles(engage_target)
        wait_command(arm, engage_target)
        time.sleep(0.50)

        actual = arm.read_joint_angles()

        print()
        print("ACTUAL:")
        print(fmt(actual))

        state = next_state

        print()
        print("=" * 80)
        print("GRIPPER CLOSED AT GRASP_READY.")
        print("AUTOMATIC RETURN STARTING.")
        print("=" * 80)

        # First undo only the +5 deg J2 GRASP_READY offset,
        # keeping the gripper CLOSED.
        deep_closed_target = list(deep_target)
        deep_closed_target[6] = (
            float(captured[6])
            + HW_J7_GRASP_DELTA_DEG
        )

        arm.set_joint_angles(deep_closed_target)
        wait_command(arm, deep_closed_target)
        time.sleep(0.40)

        # Resume the known-good TRiX return path.
        for action in (
            Action.RETURN_LOW_HOVER,
            Action.RETRACT,
            Action.UNPRESHAPE,
            Action.RETURN_SAFE,
        ):
            state = execute(
                arm,
                trix,
                bridge,
                adapter,
                captured,
                state,
                action,
            )

        final = arm.read_joint_angles()

        print()
        print("=" * 80)
        print("TRIX MANIPULATION V4 SOFT GRASP RESULT")
        print("=" * 80)

        print(
            f"final arm state: "
            f"{state.mode.value}"
        )

        print(
            f"final gripper state: "
            f"{state.gripper_state.value}"
        )

        print()
        print("START -> FINAL:")

        for index, (s, f) in enumerate(
            zip(captured, final),
            1,
        ):
            print(
                f"  J{index}: "
                f"{float(f)-float(s):+.3f} deg"
            )

        if (
            state.mode.value
            != "START_SAFE"
        ):
            raise RuntimeError(
                "FINAL_ARM_NOT_START_SAFE"
            )

        if (
            state.gripper_state
            != GripperState.CLOSED
        ):
            raise RuntimeError(
                "FINAL_GRIPPER_NOT_CLOSED"
            )

        print()
        print(
            "TRIX MANIPULATION V4 SOFT GRASP: PASS — OBJECT RETAINED"
        )

        time.sleep(1.0)

    finally:

        if started:
            arm.stop(
                return_to_zero=False,
                wait=True,
            )

        print(
            "MOTORS DISABLED / CAN CLOSED"
        )


if __name__ == "__main__":
    main()
