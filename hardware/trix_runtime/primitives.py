from dataclasses import dataclass
from typing import Dict, Tuple

from .types import Action, Mode


@dataclass(frozen=True)
class Primitive:
    action: Action

    allowed_from: Tuple[Mode, ...]

    next_mode: Mode

    # Relative joint targets from captured start.
    targets: Dict[str, float]

    # Action-specific admissible interval.
    #
    # This is intentionally tighter than generic motor limits.
    limits: Dict[str, tuple[float, float]]


PRIMITIVES = {

    Action.OPEN_GRIPPER: Primitive(
        action=Action.OPEN_GRIPPER,
        allowed_from=(Mode.START_SAFE,),
        next_mode=Mode.START_SAFE,
        targets={
            "J7": 40.0,
        },
        limits={
            "J7": (40.0, 40.0),
        },
    ),

    Action.CLOSE_GRIPPER: Primitive(
        action=Action.CLOSE_GRIPPER,
        allowed_from=(Mode.START_SAFE,),
        next_mode=Mode.START_SAFE,
        targets={
            "J7": 0.0,
        },
        limits={
            "J7": (0.0, 0.0),
        },
    ),

    Action.APPROACH: Primitive(
        action=Action.APPROACH,
        allowed_from=(Mode.START_SAFE,),
        next_mode=Mode.APPROACHED,

        targets={
            "J2": 18.0,
        },

        limits={
            "J2": (0.0, 18.0),
        },
    ),

    Action.PRESHAPE: Primitive(
        action=Action.PRESHAPE,
        allowed_from=(Mode.APPROACHED,),
        next_mode=Mode.PRESHAPED,

        targets={
            "J4": 14.0,
        },

        limits={
            "J4": (0.0, 14.0),
        },
    ),

    Action.LOW_HOVER: Primitive(
        action=Action.LOW_HOVER,
        allowed_from=(Mode.PRESHAPED,),
        next_mode=Mode.LOW_HOVER,

        targets={
            "J2": 30.0,

            # Critical tabletop rule:
            # do NOT recreate the old J3 +9 rise.
            "J3": 0.0,
        },

        limits={
            "J2": (18.0, 30.0),

            # Current LOW_HOVER primitive requires J3 to hold
            # captured-start position.
            "J3": (0.0, 0.0),
        },
    ),


    Action.DEEP_HOVER: Primitive(
        action=Action.DEEP_HOVER,
        allowed_from=(Mode.LOW_HOVER,),
        next_mode=Mode.DEEP_HOVER,

        # Extend farther/down from the proven LOW_HOVER.
        targets={
            "J2": 36.0,
            "J3": -3.0,
        },

        limits={
            "J2": (30.0, 36.0),
            "J3": (-3.0, 0.0),
        },
    ),

    Action.RETURN_LOW_HOVER: Primitive(
        action=Action.RETURN_LOW_HOVER,
        allowed_from=(Mode.DEEP_HOVER,),
        next_mode=Mode.LOW_HOVER,

        # First retreat from DEEP_HOVER to the already-proven
        # LOW_HOVER geometry before any larger reverse motion.
        targets={
            "J2": 30.0,
            "J3": 0.0,
        },

        limits={
            "J2": (30.0, 30.0),
            "J3": (0.0, 0.0),
        },
    ),

    Action.ENGAGE: Primitive(
        action=Action.ENGAGE,
        allowed_from=(Mode.DEEP_HOVER,),
        next_mode=Mode.DEEP_HOVER,

        # Close gripper around an object.
        targets={
            "J7": 0.0,
        },
        limits={
            "J7": (0.0, 0.0),
        },
    ),

    Action.RELEASE: Primitive(
        action=Action.RELEASE,
        allowed_from=(Mode.START_SAFE,),
        next_mode=Mode.START_SAFE,

        # Re-open gripper after returning to the safe arm pose.
        targets={
            "J7": 40.0,
        },
        limits={
            "J7": (40.0, 40.0),
        },
    ),

    Action.RETRACT: Primitive(
        action=Action.RETRACT,
        allowed_from=(
            Mode.LOW_HOVER,
            Mode.ENGAGED,
        ),

        # This returns to the same physical geometry as
        # PRESHAPED: J2 +18, J4 still +14.
        next_mode=Mode.PRESHAPED,

        targets={
            "J2": 18.0,
        },

        limits={
            "J2": (18.0, 18.0),
        },
    ),

    Action.UNPRESHAPE: Primitive(
        action=Action.UNPRESHAPE,
        allowed_from=(Mode.PRESHAPED,),
        next_mode=Mode.APPROACHED,

        targets={
            "J4": 0.0,
        },

        limits={
            "J4": (0.0, 0.0),
        },
    ),

    Action.RETURN_SAFE: Primitive(
        action=Action.RETURN_SAFE,
        allowed_from=(Mode.APPROACHED,),
        next_mode=Mode.START_SAFE,

        targets={
            "J2": 0.0,
            "J3": 0.0,
            "J4": 0.0,
            "J5": 0.0,
            "J6": 0.0,
        },

        limits={
            "J2": (0.0, 0.0),
            "J3": (0.0, 0.0),
            "J4": (0.0, 0.0),
            "J5": (0.0, 0.0),
            "J6": (0.0, 0.0),
        },
    ),
}
