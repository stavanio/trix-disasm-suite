from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class Action(str, Enum):
    HOLD = "HOLD"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    APPROACH = "APPROACH"
    PRESHAPE = "PRESHAPE"
    LOW_HOVER = "LOW_HOVER"
    DEEP_HOVER = "DEEP_HOVER"
    RETURN_LOW_HOVER = "RETURN_LOW_HOVER"
    ENGAGE = "ENGAGE"
    RELEASE = "RELEASE"
    RETRACT = "RETRACT"
    UNPRESHAPE = "UNPRESHAPE"
    RETURN_SAFE = "RETURN_SAFE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class Mode(str, Enum):
    START_SAFE = "START_SAFE"
    APPROACHED = "APPROACHED"
    PRESHAPED = "PRESHAPED"
    LOW_HOVER = "LOW_HOVER"
    DEEP_HOVER = "DEEP_HOVER"
    ENGAGED = "ENGAGED"
    FAULT = "FAULT"


class GripperState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"


class DecisionType(str, Enum):
    ALLOW = "ALLOW"
    PROJECT = "PROJECT"
    REJECT = "REJECT"


@dataclass
class RobotState:
    mode: Mode = Mode.START_SAFE

    # Gripper is independently tracked from arm pose.
    gripper_state: GripperState = GripperState.CLOSED

    # Relative J7 motor angle from captured physical start.
    gripper_delta_deg: float = 0.0

    # Relative to captured physical start.
    #
    # J1 deliberately does not exist here.
    joint_delta_deg: Dict[str, float] = field(
        default_factory=lambda: {
            "J2": 0.0,
            "J3": 0.0,
            "J4": 0.0,
            "J5": 0.0,
            "J6": 0.0,
        }
    )

    fault: str = "NONE"


@dataclass
class Proposal:
    action: Action

    # Optional policy/VLM-proposed modifications.
    overrides: Dict[str, float] = field(default_factory=dict)

    source: str = "synthetic_policy"


@dataclass
class Decision:
    decision: DecisionType
    proposal: Proposal

    projected_targets: Dict[str, float] = field(default_factory=dict)

    reasons: List[str] = field(default_factory=list)

    next_mode: Mode | None = None
