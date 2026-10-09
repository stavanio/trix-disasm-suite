import unittest

from trix_runtime import (
    Action,
    BridgeKind,
    BridgeSafetyError,
    Decision,
    DecisionType,
    Mode,
    Proposal,
    RobotState,
    TrixExecutionBridge,
    TrixProjector,
)


class TestTrixBridgeV1(unittest.TestCase):

    def setUp(self):
        self.trix = TrixProjector()
        self.bridge = TrixExecutionBridge()

    def evaluate_and_build(
        self,
        state,
        proposal,
    ):
        decision = self.trix.evaluate(
            state,
            proposal,
        )

        command = self.bridge.build(
            state,
            decision,
        )

        next_state = self.trix.apply(
            state,
            decision,
        )

        return (
            decision,
            command,
            next_state,
        )

    def test_approach_pose(self):

        state = RobotState()

        d, cmd, state = (
            self.evaluate_and_build(
                state,
                Proposal(Action.APPROACH),
            )
        )

        self.assertEqual(
            d.decision,
            DecisionType.ALLOW,
        )

        self.assertEqual(
            cmd.kind,
            BridgeKind.MOVE,
        )

        self.assertEqual(
            cmd.target_delta_deg,
            {
                "J2": 18.0,
                "J3": 0.0,
                "J4": 0.0,
                "J5": 0.0,
                "J6": 0.0,
            },
        )

    def test_nominal_tabletop_trace(self):

        state = RobotState()

        expected = [
            (
                Action.APPROACH,
                {
                    "J2": 18.0,
                    "J3": 0.0,
                    "J4": 0.0,
                    "J5": 0.0,
                    "J6": 0.0,
                },
            ),
            (
                Action.PRESHAPE,
                {
                    "J2": 18.0,
                    "J3": 0.0,
                    "J4": 14.0,
                    "J5": 0.0,
                    "J6": 0.0,
                },
            ),
            (
                Action.LOW_HOVER,
                {
                    "J2": 30.0,
                    "J3": 0.0,
                    "J4": 14.0,
                    "J5": 0.0,
                    "J6": 0.0,
                },
            ),
            (
                Action.RETRACT,
                {
                    "J2": 18.0,
                    "J3": 0.0,
                    "J4": 14.0,
                    "J5": 0.0,
                    "J6": 0.0,
                },
            ),
            (
                Action.UNPRESHAPE,
                {
                    "J2": 18.0,
                    "J3": 0.0,
                    "J4": 0.0,
                    "J5": 0.0,
                    "J6": 0.0,
                },
            ),
            (
                Action.RETURN_SAFE,
                {
                    "J2": 0.0,
                    "J3": 0.0,
                    "J4": 0.0,
                    "J5": 0.0,
                    "J6": 0.0,
                },
            ),
        ]

        for action, expected_target in expected:

            d, cmd, state = (
                self.evaluate_and_build(
                    state,
                    Proposal(action),
                )
            )

            self.assertNotEqual(
                d.decision,
                DecisionType.REJECT,
            )

            self.assertEqual(
                cmd.kind,
                BridgeKind.MOVE,
            )

            self.assertEqual(
                cmd.target_delta_deg,
                expected_target,
            )

        self.assertEqual(
            state.mode,
            Mode.START_SAFE,
        )

    def test_rejected_proposal_emits_nothing(self):

        state = RobotState()

        decision = self.trix.evaluate(
            state,
            Proposal(Action.LOW_HOVER),
        )

        command = self.bridge.build(
            state,
            decision,
        )

        self.assertEqual(
            decision.decision,
            DecisionType.REJECT,
        )

        self.assertIsNone(command)

    def test_projected_j2_crosses_bridge_as_safe_value(self):

        state = RobotState()

        decision = self.trix.evaluate(
            state,
            Proposal(
                Action.APPROACH,
                overrides={
                    "J2": 50.0,
                },
            ),
        )

        command = self.bridge.build(
            state,
            decision,
        )

        self.assertEqual(
            decision.decision,
            DecisionType.PROJECT,
        )

        self.assertEqual(
            command.target_delta_deg["J2"],
            18.0,
        )

    def test_defensive_boundary_rejects_unsupported_joint(self):

        state = RobotState()

        malicious = Decision(
            decision=DecisionType.ALLOW,
            proposal=Proposal(
                Action.APPROACH
            ),
            projected_targets={
                "J1": 10.0,
            },
            reasons=[
                "MALICIOUS_TEST"
            ],
            next_mode=Mode.APPROACHED,
        )

        with self.assertRaises(
            BridgeSafetyError
        ):
            self.bridge.build(
                state,
                malicious,
            )


if __name__ == "__main__":
    unittest.main()
