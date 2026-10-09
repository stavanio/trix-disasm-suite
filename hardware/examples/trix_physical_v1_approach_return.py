#!/usr/bin/env python3

import math
import time

from rebot import ReBotRSMITController

from trix_runtime import (
    Action,
    DecisionType,
    Proposal,
    RobotState,
    TrixExecutionBridge,
    TrixPhysicalTargetAdapter,
    TrixProjector,
)


# Proven FAST-3X velocity baseline.
SPEEDS_DEG_S = [
    4.50,   # J1 -- target NEVER changes
    3.60,   # J2
    1.50,   # J3
    3.60,   # J4
    4.50,   # J5
    4.50,   # J6
    2.25,   # J7
]

COMMAND_TOL_DEG = 0.08
MOTION_TIMEOUT_S = 12.0


def fmt_pose(values):
    return "  ".join(
        f"J{i+1}={float(v):+7.2f}"
        for i, v in enumerate(values)
    )


def wait_command_target(
    arm,
    target,
    timeout_s=MOTION_TIMEOUT_S,
):
    deadline = time.monotonic() + timeout_s

    while True:
        cmd = arm.get_command_angles()

        errors = [
            abs(float(a) - float(b))
            for a, b in zip(cmd, target)
        ]

        if max(errors) <= COMMAND_TOL_DEG:
            return cmd

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "COMMAND_TARGET_TIMEOUT "
                f"max_error={max(errors):.3f}deg "
                f"cmd={cmd} "
                f"target={list(target)}"
            )

        time.sleep(0.05)


def build_physical(
    trix,
    bridge,
    adapter,
    state,
    action,
):
    proposal = Proposal(
        action=action,
        source="physical_trix_v1",
    )

    decision = trix.evaluate(
        state,
        proposal,
    )

    command = bridge.build(
        state,
        decision,
    )

    physical = adapter.build(
        command,
    )

    next_state = trix.apply(
        state,
        decision,
    )

    return (
        proposal,
        decision,
        command,
        physical,
        next_state,
    )


def run_motion(
    arm,
    adapter,
    label,
    physical,
):
    if physical is None:
        raise RuntimeError(
            f"{label}: no physical target produced"
        )

    target = physical.angles_deg

    # HARD PHYSICAL J1 CHECK immediately before actuator API.
    adapter.assert_j1_invariant(target)

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)
    print("TARGET:")
    print(fmt_pose(target))

    print(
        f"J1 invariant: "
        f"{target[0]:+.6f} deg "
        f"== captured "
        f"{adapter.frozen_j1_deg:+.6f} deg"
    )

    arm.set_joint_angles(target)

    cmd = wait_command_target(
        arm,
        target,
    )

    time.sleep(0.40)

    actual = arm.read_joint_angles()

    adapter.assert_j1_invariant(target)

    print()
    print("COMMAND ARRIVED:")
    print(fmt_pose(cmd))

    print()
    print("ACTUAL:")
    print(fmt_pose(actual))

    print()
    print(
        "Tracking error actual-target:"
    )

    for i, (a, t) in enumerate(
        zip(actual, target),
        1,
    ):
        print(
            f"  J{i}: "
            f"{float(a)-float(t):+.3f} deg"
        )

    return actual


def main():

    arm = ReBotRSMITController()

    started = False
    returned_safe = False

    try:

        print("=" * 72)
        print("TRIX PHYSICAL EXPERIMENT V1")
        print("ONLY J2 DISPLACEMENT IS PERMITTED")
        print("J1 TARGET WILL NEVER CHANGE")
        print("=" * 72)

        # Disable controller-installed Esc/signal behavior so this
        # experiment controls its own shutdown policy.
        arm.start(
            enable_esc=False,
            install_signal_handlers=False,
        )

        started = True

        arm.set_max_speeds(
            SPEEDS_DEG_S
        )

        # One synchronous position capture.
        captured = arm.read_joint_angles()

        if len(captured) != 7:
            raise RuntimeError(
                f"Expected 7 joints, got {len(captured)}"
            )

        print()
        print("CAPTURED RELOCATED START:")
        print(fmt_pose(captured))

        trix = TrixProjector()
        bridge = TrixExecutionBridge()

        adapter = (
            TrixPhysicalTargetAdapter(
                captured
            )
        )

        state = RobotState()

        # ------------------------------------------------------
        # BUILD APPROACH WITHOUT MOVING ROBOT
        # ------------------------------------------------------

        (
            proposal,
            decision,
            command,
            physical,
            approach_state,
        ) = build_physical(
            trix,
            bridge,
            adapter,
            state,
            Action.APPROACH,
        )

        print()
        print("=" * 72)
        print("TRIX PRE-MOTION DECISION")
        print("=" * 72)

        print(
            f"proposal : "
            f"{proposal.action.value}"
        )

        print(
            f"decision : "
            f"{decision.decision.value}"
        )

        print(
            "reason   : "
            + "; ".join(decision.reasons)
        )

        if decision.decision == DecisionType.REJECT:
            raise RuntimeError(
                "TRiX rejected nominal APPROACH"
            )

        if physical is None:
            raise RuntimeError(
                "APPROACH produced no physical target"
            )

        adapter.assert_j1_invariant(
            physical.angles_deg
        )

        print()
        print("CAPTURED START:")
        print(fmt_pose(captured))

        print()
        print("APPROACH TARGET:")
        print(fmt_pose(
            physical.angles_deg
        ))

        deltas = [
            float(t) - float(s)
            for t, s in zip(
                physical.angles_deg,
                captured,
            )
        ]

        print()
        print("REQUESTED PHYSICAL DELTAS:")

        for i, delta in enumerate(
            deltas,
            1,
        ):
            print(
                f"  J{i}: {delta:+.3f} deg"
            )

        # ------------------------------------------------------
        # EXTREMELY IMPORTANT FIRST-RUN GUARD
        # ------------------------------------------------------

        expected = [
            0.0,
            18.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        for i, (actual_delta, expected_delta) in enumerate(
            zip(deltas, expected),
            1,
        ):
            if abs(
                actual_delta - expected_delta
            ) > 1e-6:
                raise RuntimeError(
                    f"UNEXPECTED_J{i}_DELTA:"
                    f"{actual_delta:+.6f}"
                )

        print()
        print(
            "PRECHECK PASSED:"
        )
        print(
            "  ONLY J2 WILL CHANGE (+18 deg)"
        )
        print(
            "  J1 target remains EXACTLY captured"
        )
        print(
            "  J3/J4/J5/J6/J7 remain captured"
        )

        print()
        confirmation = input(
            "Type RUN to execute this bounded physical episode: "
        ).strip()

        if confirmation != "RUN":
            print(
                "Operator did not type RUN. "
                "NO MOTION EXECUTED."
            )
            return

        # ------------------------------------------------------
        # PHYSICAL APPROACH
        # ------------------------------------------------------

        before_actual = arm.read_joint_angles()

        print()
        print("ACTUAL BEFORE APPROACH:")
        print(fmt_pose(before_actual))

        approach_actual = run_motion(
            arm,
            adapter,
            "TRIX ACTION 1: APPROACH",
            physical,
        )

        state = approach_state

        # ------------------------------------------------------
        # TRIX DECIDES SAFE RETURN
        # ------------------------------------------------------

        (
            proposal,
            decision,
            command,
            physical,
            return_state,
        ) = build_physical(
            trix,
            bridge,
            adapter,
            state,
            Action.RETURN_SAFE,
        )

        print()
        print("=" * 72)
        print("TRIX RETURN DECISION")
        print("=" * 72)

        print(
            f"proposal : "
            f"{proposal.action.value}"
        )

        print(
            f"decision : "
            f"{decision.decision.value}"
        )

        print(
            "reason   : "
            + "; ".join(decision.reasons)
        )

        if decision.decision == DecisionType.REJECT:
            raise RuntimeError(
                "TRiX rejected RETURN_SAFE"
            )

        if physical is None:
            raise RuntimeError(
                "RETURN_SAFE produced no target"
            )

        adapter.assert_j1_invariant(
            physical.angles_deg
        )

        # RETURN_SAFE must equal exact captured pose.
        for i, (target, start) in enumerate(
            zip(
                physical.angles_deg,
                captured,
            ),
            1,
        ):
            if abs(
                float(target)
                - float(start)
            ) > 1e-9:
                raise RuntimeError(
                    f"RETURN_TARGET_J{i}_NOT_START"
                )

        return_actual = run_motion(
            arm,
            adapter,
            "TRIX ACTION 2: RETURN_SAFE",
            physical,
        )

        state = return_state
        returned_safe = True

        print()
        print("=" * 72)
        print("PHYSICAL TRIX V1 RESULT")
        print("=" * 72)

        print(
            f"final symbolic state: "
            f"{state.mode.value}"
        )

        print()
        print("START -> FINAL ACTUAL ERROR:")

        for i, (start, final) in enumerate(
            zip(captured, return_actual),
            1,
        ):
            print(
                f"  J{i}: "
                f"{float(final)-float(start):+.3f} deg"
            )

        j1_drift = (
            float(return_actual[0])
            - float(captured[0])
        )

        print()
        print(
            f"J1 measured start->final drift: "
            f"{j1_drift:+.3f} deg"
        )

        print()
        print(
            "TRIX PHYSICAL V1 COMPLETE:"
        )

        print(
            "  APPROACH allowed and executed"
        )

        print(
            "  RETURN_SAFE allowed and executed"
        )

        print(
            "  J1 target invariant maintained"
        )

        print()
        input(
            "Robot is holding captured pose. "
            "Press ENTER to disable motors and close CAN: "
        )

    finally:

        if started:

            print()
            print(
                "Disabling motors WITHOUT "
                "automatic return-to-zero..."
            )

            try:
                arm.stop(
                    return_to_zero=False,
                    wait=True,
                )
            except Exception as exc:
                print(
                    f"Shutdown exception: {exc}"
                )

        print(
            "MOTORS DISABLED / CAN CLOSED"
        )


if __name__ == "__main__":
    main()
