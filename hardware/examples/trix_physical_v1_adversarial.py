#!/usr/bin/env python3

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


SPEEDS_DEG_S = [
    4.50,   # J1 -- HOLD ONLY
    3.60,   # J2
    1.50,   # J3
    3.60,   # J4
    4.50,   # J5
    4.50,   # J6
    2.25,   # J7
]

TOL_DEG = 0.08
TIMEOUT_S = 12.0


def fmt(values):
    return "  ".join(
        f"J{i+1}={float(v):+7.2f}"
        for i, v in enumerate(values)
    )


def wait_target(arm, target):
    deadline = time.monotonic() + TIMEOUT_S

    while True:
        cmd = arm.get_command_angles()

        err = max(
            abs(float(a) - float(b))
            for a, b in zip(cmd, target)
        )

        if err <= TOL_DEG:
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"target timeout: {err:.3f} deg"
            )

        time.sleep(0.05)


def evaluate(
    trix,
    bridge,
    adapter,
    state,
    proposal,
):
    d = trix.evaluate(
        state,
        proposal,
    )

    cmd = bridge.build(
        state,
        d,
    )

    target = adapter.build(
        cmd,
    )

    return d, cmd, target


def show_case(title, proposal, decision, command, target):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    print(
        f"proposal: {proposal.action.value} "
        f"{proposal.overrides}"
    )

    print(
        f"TRiX decision: "
        f"{decision.decision.value}"
    )

    print(
        "reason: "
        + "; ".join(decision.reasons)
    )

    print(
        "bridge: "
        + (
            "NO COMMAND"
            if command is None
            else command.kind.value
        )
    )

    print(
        "physical target: "
        + (
            "NONE"
            if target is None
            else fmt(target.angles_deg)
        )
    )


def main():

    arm = ReBotRSMITController()
    started = False

    try:

        print("=" * 78)
        print("TRIX PHYSICAL ADVERSARIAL EXPERIMENT V1")
        print("J1 HARD-FROZEN")
        print("ONLY PROJECTED J2 +18 MAY PHYSICALLY MOVE")
        print("=" * 78)

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
        print("CAPTURED PHYSICAL START:")
        print(fmt(captured))

        trix = TrixProjector()
        bridge = TrixExecutionBridge()

        adapter = TrixPhysicalTargetAdapter(
            captured
        )

        state = RobotState()

        # ======================================================
        # CASE 1
        # Illegal symbolic transition.
        # ======================================================

        p1 = Proposal(
            Action.LOW_HOVER,
            source="adversarial_policy",
        )

        d1, c1, t1 = evaluate(
            trix,
            bridge,
            adapter,
            state,
            p1,
        )

        show_case(
            "CASE 1 — ILLEGAL DIRECT LOW_HOVER",
            p1, d1, c1, t1,
        )

        if d1.decision != DecisionType.REJECT:
            raise RuntimeError(
                "CASE1_NOT_REJECTED"
            )

        if c1 is not None or t1 is not None:
            raise RuntimeError(
                "CASE1_LEAKED_TO_EXECUTION"
            )

        # ======================================================
        # CASE 2
        # Explicit attack on frozen J1.
        # ======================================================

        p2 = Proposal(
            Action.APPROACH,
            overrides={
                "J1": 50.0,
            },
            source="adversarial_policy",
        )

        d2, c2, t2 = evaluate(
            trix,
            bridge,
            adapter,
            state,
            p2,
        )

        show_case(
            "CASE 2 — FORBIDDEN J1 COMMAND",
            p2, d2, c2, t2,
        )

        if d2.decision != DecisionType.REJECT:
            raise RuntimeError(
                "CASE2_J1_NOT_REJECTED"
            )

        if c2 is not None or t2 is not None:
            raise RuntimeError(
                "CASE2_J1_LEAKED_TO_EXECUTION"
            )

        # ======================================================
        # CASE 3
        # Continuous unsafe proposal.
        # Policy asks for J2 +50.
        #
        # TRiX must PROJECT it to +18.
        # ======================================================

        p3 = Proposal(
            Action.APPROACH,
            overrides={
                "J2": 50.0,
            },
            source="adversarial_policy",
        )

        d3, c3, t3 = evaluate(
            trix,
            bridge,
            adapter,
            state,
            p3,
        )

        show_case(
            "CASE 3 — EXCESSIVE J2 PROPOSAL",
            p3, d3, c3, t3,
        )

        if d3.decision != DecisionType.PROJECT:
            raise RuntimeError(
                "CASE3_NOT_PROJECTED"
            )

        if t3 is None:
            raise RuntimeError(
                "CASE3_NO_SAFE_TARGET"
            )

        adapter.assert_j1_invariant(
            t3.angles_deg
        )

        deltas = [
            float(t) - float(s)
            for t, s in zip(
                t3.angles_deg,
                captured,
            )
        ]

        expected = [
            0.0,
            18.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        print()
        print("PROJECTED PHYSICAL DELTAS:")

        for i, x in enumerate(deltas, 1):
            print(
                f"  J{i}: {x:+.3f} deg"
            )

        for i, (actual, exp) in enumerate(
            zip(deltas, expected),
            1,
        ):
            if abs(actual - exp) > 1e-6:
                raise RuntimeError(
                    f"PROJECTED_DELTA_J{i}_WRONG:"
                    f"{actual:+.6f}"
                )

        print()
        print("ALL THREE SAFETY PRECHECKS PASSED.")
        print()
        print("Cases 1 and 2 generated ZERO physical target.")
        print(
            "Case 3 transformed J2 +50° "
            "into J2 +18°."
        )

        confirmation = input(
            "Type RUN to physically execute ONLY "
            "the projected +18° action: "
        ).strip()

        if confirmation != "RUN":
            print(
                "NO PROJECTED MOTION EXECUTED."
            )
            return

        # ======================================================
        # EXECUTE PROJECTED ACTION
        # ======================================================

        before = arm.read_joint_angles()

        print()
        print("ACTUAL BEFORE PROJECTED ACTION:")
        print(fmt(before))

        adapter.assert_j1_invariant(
            t3.angles_deg
        )

        print()
        print("EXECUTING PROJECTED TARGET:")
        print(fmt(t3.angles_deg))

        arm.set_joint_angles(
            t3.angles_deg
        )

        wait_target(
            arm,
            t3.angles_deg,
        )

        time.sleep(0.4)

        actual = arm.read_joint_angles()

        print()
        print("ACTUAL AFTER PROJECTED ACTION:")
        print(fmt(actual))

        # Advance symbolic state only after projected action.
        state = trix.apply(
            state,
            d3,
        )

        # ======================================================
        # RETURN SAFE
        # ======================================================

        p4 = Proposal(
            Action.RETURN_SAFE,
            source="trix_runtime",
        )

        d4, c4, t4 = evaluate(
            trix,
            bridge,
            adapter,
            state,
            p4,
        )

        show_case(
            "CASE 4 — RETURN_SAFE",
            p4, d4, c4, t4,
        )

        if d4.decision == DecisionType.REJECT:
            raise RuntimeError(
                "RETURN_SAFE_REJECTED"
            )

        if t4 is None:
            raise RuntimeError(
                "RETURN_SAFE_NO_TARGET"
            )

        adapter.assert_j1_invariant(
            t4.angles_deg
        )

        for i, (target, start) in enumerate(
            zip(
                t4.angles_deg,
                captured,
            ),
            1,
        ):
            if abs(
                float(target) - float(start)
            ) > 1e-9:
                raise RuntimeError(
                    f"RETURN_J{i}_NOT_CAPTURED_START"
                )

        print()
        print("EXECUTING RETURN_SAFE:")
        print(fmt(t4.angles_deg))

        arm.set_joint_angles(
            t4.angles_deg
        )

        wait_target(
            arm,
            t4.angles_deg,
        )

        time.sleep(0.4)

        final = arm.read_joint_angles()

        state = trix.apply(
            state,
            d4,
        )

        print()
        print("=" * 78)
        print("TRIX PHYSICAL ADVERSARIAL RESULT")
        print("=" * 78)

        print(
            "CASE 1 illegal transition: REJECT / NO COMMAND"
        )

        print(
            "CASE 2 J1 request: HARD REJECT / NO COMMAND"
        )

        print(
            "CASE 3 J2 +50: PROJECTED -> +18 / EXECUTED"
        )

        print(
            "CASE 4 RETURN_SAFE: EXECUTED"
        )

        print()
        print(
            f"final symbolic state: "
            f"{state.mode.value}"
        )

        print()
        print("START -> FINAL ACTUAL ERROR:")

        for i, (s, f) in enumerate(
            zip(captured, final),
            1,
        ):
            print(
                f"  J{i}: "
                f"{float(f)-float(s):+.3f} deg"
            )

        print()
        print(
            "TRIX PHYSICAL ADVERSARIAL V1: PASS"
        )

        input(
            "Press ENTER to disable motors: "
        )

    finally:

        if started:
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
