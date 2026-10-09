from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .types import (
    Action,
    Decision,
    DecisionType,
    RobotState,
)


# The physical execution interface intentionally contains ONLY
# the joints that the current TRiX experiment is allowed to use.
SUPPORTED_JOINTS = (
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
)

# J7 is carried separately from the arm target dictionary.
# This preserves the legacy J2-J6 command contract while allowing
# explicit, state-gated gripper actions.
GRIPPER_JOINT = "J7"

GRIPPER_ACTIONS = {
    Action.OPEN_GRIPPER,
    Action.CLOSE_GRIPPER,
    Action.ENGAGE,
    Action.RELEASE,
}


class BridgeKind(str, Enum):
    MOVE = "MOVE"
    NO_MOTION = "NO_MOTION"
    STOP_REQUEST = "STOP_REQUEST"


class BridgeSafetyError(RuntimeError):
    pass


@dataclass
class BridgeCommand:
    kind: BridgeKind
    action: Action
    target_delta_deg: Dict[str, float]
    reason: str

    # Relative J7 target from captured start.
    #
    # Kept separate so the legacy arm command dictionary remains
    # exactly J2-J6.
    gripper_delta_deg: float = 0.0


class TrixExecutionBridge:
    """
    Convert an approved TRiX decision into an executor-facing command.

    This module:
      - does NOT import rebot
      - does NOT import socketcan
      - does NOT open CAN
      - does NOT know about motor IDs
      - exposes J2-J6 arm targets plus separately gated J7
    """

    def build(
        self,
        state_before: RobotState,
        decision: Decision,
    ) -> Optional[BridgeCommand]:

        # A rejected proposal must produce NO actuator command.
        if decision.decision == DecisionType.REJECT:
            return None

        # Defensive boundary:
        # even if an upstream software bug somehow creates an
        # invalid Decision object, unsupported joints may not
        # cross this bridge.
        projected_joints = set(
            decision.projected_targets
        )

        unsupported = (
            projected_joints
            - set(SUPPORTED_JOINTS)
            - {GRIPPER_JOINT}
        )

        if unsupported:
            raise BridgeSafetyError(
                "UNSUPPORTED_JOINT_AT_EXECUTION_BOUNDARY:"
                + ",".join(sorted(unsupported))
            )

        # J7 may cross the execution boundary ONLY when the
        # approved action itself is an explicit gripper action.
        if (
            GRIPPER_JOINT in projected_joints
            and decision.proposal.action not in GRIPPER_ACTIONS
        ):
            raise BridgeSafetyError(
                "J7_NOT_ALLOWED_FOR_ACTION:"
                f"{decision.proposal.action.value}"
            )

        if decision.proposal.action == Action.EMERGENCY_STOP:
            return BridgeCommand(
                kind=BridgeKind.STOP_REQUEST,
                action=decision.proposal.action,
                target_delta_deg={},
                reason="TRIX_EMERGENCY_STOP_REQUEST",
            )

        # Symbolic actions that intentionally cause no joint motion.
        if decision.proposal.action == Action.HOLD:
            return BridgeCommand(
                kind=BridgeKind.NO_MOTION,
                action=decision.proposal.action,
                target_delta_deg={},
                reason="SYMBOLIC_ACTION_NO_JOINT_MOTION",
            )

        # Build a FULL J2-J6 target pose by starting with the
        # currently accepted symbolic state and applying only the
        # approved/projected changes from TRiX.
        target = {
            joint: float(
                state_before.joint_delta_deg[joint]
            )
            for joint in SUPPORTED_JOINTS
        }

        # Preserve the currently accepted gripper state during
        # ordinary arm motion.
        gripper_delta = float(
            state_before.gripper_delta_deg
        )

        for joint, value in (
            decision.projected_targets.items()
        ):
            if joint == GRIPPER_JOINT:
                gripper_delta = float(value)
            else:
                target[joint] = float(value)

        return BridgeCommand(
            kind=BridgeKind.MOVE,
            action=decision.proposal.action,
            target_delta_deg=target,
            reason="ADMISSIBLE_TRIX_TARGET",
            gripper_delta_deg=gripper_delta,
        )
