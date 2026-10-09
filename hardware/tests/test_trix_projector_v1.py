import unittest

from trix_runtime import (
    Action,
    DecisionType,
    Mode,
    Proposal,
    RobotState,
    TrixProjector,
)


class TestTrixProjectorV1(unittest.TestCase):

    def setUp(self):
        self.trix = TrixProjector()

    def test_nominal_approach(self):

        state = RobotState()

        d = self.trix.evaluate(
            state,
            Proposal(Action.APPROACH),
        )

        self.assertEqual(
            d.decision,
            DecisionType.ALLOW,
        )

        self.assertEqual(
            d.projected_targets["J2"],
            18.0,
        )

    def test_j1_is_hard_rejected(self):

        state = RobotState()

        d = self.trix.evaluate(
            state,
            Proposal(
                Action.APPROACH,
                overrides={"J1": 10.0},
            ),
        )

        self.assertEqual(
            d.decision,
            DecisionType.REJECT,
        )

        self.assertIn(
            "HARD_INVARIANT_J1_FROZEN",
            d.reasons,
        )

    def test_illegal_transition(self):

        state = RobotState()

        d = self.trix.evaluate(
            state,
            Proposal(Action.LOW_HOVER),
        )

        self.assertEqual(
            d.decision,
            DecisionType.REJECT,
        )

    def test_excessive_j2_is_projected(self):

        state = RobotState()

        d = self.trix.evaluate(
            state,
            Proposal(
                Action.APPROACH,
                overrides={"J2": 50.0},
            ),
        )

        self.assertEqual(
            d.decision,
            DecisionType.PROJECT,
        )

        self.assertEqual(
            d.projected_targets["J2"],
            18.0,
        )

    def test_low_hover_requires_j3_hold(self):

        state = RobotState(
            mode=Mode.PRESHAPED
        )

        d = self.trix.evaluate(
            state,
            Proposal(
                Action.LOW_HOVER,
                overrides={
                    "J3": 9.0,
                },
            ),
        )

        self.assertEqual(
            d.decision,
            DecisionType.PROJECT,
        )

        self.assertEqual(
            d.projected_targets["J3"],
            0.0,
        )

    def test_fault_blocks_motion(self):

        state = RobotState(
            fault="CAN_FAULT"
        )

        d = self.trix.evaluate(
            state,
            Proposal(Action.APPROACH),
        )

        self.assertEqual(
            d.decision,
            DecisionType.REJECT,
        )


if __name__ == "__main__":
    unittest.main()
