import unittest

from trix_runtime import (
    Action,
    BridgeKind,
    BridgeSafetyError,
    Decision,
    DecisionType,
    GripperState,
    Mode,
    PhysicalAdapterSafetyError,
    Proposal,
    RobotState,
    TrixExecutionBridge,
    TrixPhysicalTargetAdapter,
    TrixProjector,
)


class TestTrixGripperV1(unittest.TestCase):

    def setUp(self):

        self.trix = TrixProjector()
        self.bridge = TrixExecutionBridge()

        self.start = [
            +37.42,  # J1 -- frozen
            +0.11,
            +0.27,
            -1.94,
            -1.68,
            -3.44,
            +1.55,   # captured J7
        ]

        self.adapter = (
            TrixPhysicalTargetAdapter(
                self.start
            )
        )

    def step(
        self,
        state,
        action,
        overrides=None,
    ):

        proposal = Proposal(
            action,
            overrides=overrides or {},
        )

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

        next_state = self.trix.apply(
            state,
            decision,
        )

        return (
            decision,
            command,
            physical,
            next_state,
        )

    def test_open_gripper_from_start(self):

        state = RobotState()

        d, cmd, physical, state = self.step(
            state,
            Action.OPEN_GRIPPER,
        )

        self.assertEqual(
            d.decision,
            DecisionType.ALLOW,
        )

        self.assertEqual(
            cmd.kind,
            BridgeKind.MOVE,
        )

        # Legacy arm dictionary remains J2-J6 only.
        self.assertEqual(
            cmd.target_delta_deg,
            {
                "J2": 0.0,
                "J3": 0.0,
                "J4": 0.0,
                "J5": 0.0,
                "J6": 0.0,
            },
        )

        self.assertEqual(
            cmd.gripper_delta_deg,
            40.0,
        )

        self.assertAlmostEqual(
            physical.angles_deg[0],
            self.start[0],
            places=9,
        )

        self.assertAlmostEqual(
            physical.angles_deg[6],
            self.start[6] + 40.0,
            places=9,
        )

        self.assertEqual(
            state.gripper_state,
            GripperState.OPEN,
        )

        self.assertEqual(
            state.gripper_delta_deg,
            40.0,
        )

    def test_arm_motion_preserves_open_gripper(self):

        state = RobotState()

        _, _, _, state = self.step(
            state,
            Action.OPEN_GRIPPER,
        )

        d, cmd, physical, state = self.step(
            state,
            Action.APPROACH,
        )

        self.assertNotEqual(
            d.decision,
            DecisionType.REJECT,
        )

        self.assertEqual(
            cmd.gripper_delta_deg,
            40.0,
        )

        self.assertAlmostEqual(
            physical.angles_deg[6],
            self.start[6] + 40.0,
            places=9,
        )

        self.assertAlmostEqual(
            physical.angles_deg[1],
            self.start[1] + 18.0,
            places=9,
        )

    def test_close_returns_to_captured_j7(self):

        state = RobotState()

        _, _, _, state = self.step(
            state,
            Action.OPEN_GRIPPER,
        )

        d, cmd, physical, state = self.step(
            state,
            Action.CLOSE_GRIPPER,
        )

        self.assertEqual(
            d.decision,
            DecisionType.ALLOW,
        )

        self.assertEqual(
            cmd.gripper_delta_deg,
            0.0,
        )

        self.assertAlmostEqual(
            physical.angles_deg[6],
            self.start[6],
            places=9,
        )

        self.assertEqual(
            state.gripper_state,
            GripperState.CLOSED,
        )

    def test_engage_rejected_from_start_even_if_open(self):

        state = RobotState()

        _, _, _, state = self.step(
            state,
            Action.OPEN_GRIPPER,
        )

        decision = self.trix.evaluate(
            state,
            Proposal(Action.ENGAGE),
        )

        self.assertEqual(
            decision.decision,
            DecisionType.REJECT,
        )

    def test_engage_allowed_at_deep_hover_when_open(self):

        state = RobotState(
            mode=Mode.DEEP_HOVER,
            gripper_state=GripperState.OPEN,
            gripper_delta_deg=40.0,
            joint_delta_deg={
                "J2": 36.0,
                "J3": -3.0,
                "J4": 14.0,
                "J5": 0.0,
                "J6": 0.0,
            },
        )

        d, cmd, physical, state = self.step(
            state,
            Action.ENGAGE,
        )

        self.assertEqual(
            d.decision,
            DecisionType.ALLOW,
        )

        self.assertEqual(
            cmd.gripper_delta_deg,
            0.0,
        )

        self.assertAlmostEqual(
            physical.angles_deg[6],
            self.start[6],
            places=9,
        )

        self.assertEqual(
            state.mode,
            Mode.DEEP_HOVER,
        )

        self.assertEqual(
            state.gripper_state,
            GripperState.CLOSED,
        )

    def test_malicious_j7_on_arm_action_blocked_by_bridge(self):

        state = RobotState()

        malicious = Decision(
            decision=DecisionType.ALLOW,
            proposal=Proposal(
                Action.APPROACH
            ),
            projected_targets={
                "J7": 40.0,
            },
            reasons=[
                "MALICIOUS_TEST",
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

    def test_physical_boundary_rejects_j7_above_40(self):

        from trix_runtime import (
            BridgeCommand,
        )

        malicious = BridgeCommand(
            kind=BridgeKind.MOVE,
            action=Action.OPEN_GRIPPER,
            target_delta_deg={
                "J2": 0.0,
                "J3": 0.0,
                "J4": 0.0,
                "J5": 0.0,
                "J6": 0.0,
            },
            reason="MALICIOUS_TEST",
            gripper_delta_deg=100.0,
        )

        with self.assertRaises(
            PhysicalAdapterSafetyError
        ):
            self.adapter.build(
                malicious
            )

    def test_j1_still_hard_rejected(self):

        state = RobotState()

        decision = self.trix.evaluate(
            state,
            Proposal(
                Action.OPEN_GRIPPER,
                overrides={
                    "J1": 1.0,
                },
            ),
        )

        self.assertEqual(
            decision.decision,
            DecisionType.REJECT,
        )


if __name__ == "__main__":
    unittest.main()
