import unittest

from trix_runtime import (
    Action,
    BridgeCommand,
    BridgeKind,
    PhysicalAdapterSafetyError,
    Proposal,
    RobotState,
    TrixExecutionBridge,
    TrixPhysicalTargetAdapter,
    TrixProjector,
)


class TestTrixPhysicalAdapterV1(unittest.TestCase):

    def setUp(self):

        # Deliberately arbitrary J1 value.
        #
        # This simulates the robot having been physically relocated
        # and J1 now being in a completely different pose.
        self.start = [
            +37.42,   # J1 -- FROZEN
            +0.11,    # J2
            +0.27,    # J3
            -1.94,    # J4
            -1.68,    # J5
            -3.44,    # J6
            +1.55,    # J7 -- HOLD
        ]

        self.trix = TrixProjector()
        self.bridge = TrixExecutionBridge()

        self.adapter = (
            TrixPhysicalTargetAdapter(
                self.start
            )
        )

    def make(
        self,
        state,
        action,
    ):

        proposal = Proposal(action)

        decision = self.trix.evaluate(
            state,
            proposal,
        )

        command = self.bridge.build(
            state,
            decision,
        )

        physical = self.adapter.build(
            command
        )

        new_state = self.trix.apply(
            state,
            decision,
        )

        return (
            new_state,
            physical,
        )

    def test_j1_stays_exactly_captured(self):

        state = RobotState()

        for action in (
            Action.APPROACH,
            Action.PRESHAPE,
            Action.LOW_HOVER,
            Action.RETRACT,
            Action.UNPRESHAPE,
            Action.RETURN_SAFE,
        ):

            state, physical = self.make(
                state,
                action,
            )

            self.assertIsNotNone(
                physical
            )

            self.assertEqual(
                physical.angles_deg[0],
                37.42,
            )

    def test_nominal_absolute_low_hover(self):

        state = RobotState()

        for action in (
            Action.APPROACH,
            Action.PRESHAPE,
            Action.LOW_HOVER,
        ):

            state, physical = self.make(
                state,
                action,
            )

        expected = (
            37.42,      # J1 unchanged
            30.11,      # J2 start +30
            0.27,       # J3 start +0
            12.06,      # J4 start +14
            -1.68,      # J5 start
            -3.44,      # J6 start
            1.55,       # J7 unchanged
        )

        for actual, exp in zip(
            physical.angles_deg,
            expected,
        ):
            self.assertAlmostEqual(
                actual,
                exp,
                places=9,
            )

    def test_return_safe_is_exact_start(self):

        state = RobotState()

        actions = (
            Action.APPROACH,
            Action.PRESHAPE,
            Action.LOW_HOVER,
            Action.RETRACT,
            Action.UNPRESHAPE,
            Action.RETURN_SAFE,
        )

        for action in actions:
            state, physical = self.make(
                state,
                action,
            )

        for actual, exp in zip(
            physical.angles_deg,
            self.start,
        ):
            self.assertAlmostEqual(
                actual,
                exp,
                places=9,
            )

    def test_malicious_j1_at_physical_boundary_fails(self):

        malicious = BridgeCommand(
            kind=BridgeKind.MOVE,
            action=Action.APPROACH,
            target_delta_deg={
                "J1": 50.0,
                "J2": 18.0,
            },
            reason="MALICIOUS_TEST",
        )

        with self.assertRaises(
            PhysicalAdapterSafetyError
        ):
            self.adapter.build(
                malicious
            )

    def test_rejected_proposal_creates_no_target(self):

        state = RobotState()

        decision = self.trix.evaluate(
            state,
            Proposal(
                Action.LOW_HOVER
            ),
        )

        command = self.bridge.build(
            state,
            decision,
        )

        physical = self.adapter.build(
            command
        )

        self.assertIsNone(
            physical
        )


if __name__ == "__main__":
    unittest.main()
