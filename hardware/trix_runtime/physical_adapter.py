from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .execution_bridge import (
    BridgeCommand,
    BridgeKind,
    SUPPORTED_JOINTS,
)


class PhysicalAdapterSafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhysicalTarget:
    """
    Full actuator target consumed by the existing arm executor.

    J1 remains a captured hard invariant.
    J2-J6 are generated from approved TRiX relative deltas.
    J7 is generated only from the separately gated gripper delta.
    """
    angles_deg: tuple[float, ...]
    action: str


class TrixPhysicalTargetAdapter:

    # Robot full-angle ordering from the proven controller:
    #
    # index:
    # 0 J1
    # 1 J2
    # 2 J3
    # 3 J4
    # 4 J5
    # 5 J6
    # 6 J7

    INDEX = {
        "J2": 1,
        "J3": 2,
        "J4": 3,
        "J5": 4,
        "J6": 5,
    }

    def __init__(
        self,
        captured_start_deg: Sequence[float],
    ):

        if len(captured_start_deg) != 7:
            raise PhysicalAdapterSafetyError(
                "EXPECTED_EXACTLY_7_CAPTURED_JOINTS"
            )

        self._start = tuple(
            float(x)
            for x in captured_start_deg
        )

        self._frozen_j1 = self._start[0]
        self._captured_j7 = self._start[6]

    @property
    def captured_start_deg(self):
        return self._start

    @property
    def frozen_j1_deg(self):
        return self._frozen_j1

    def build(
        self,
        command: Optional[BridgeCommand],
    ) -> Optional[PhysicalTarget]:

        # Rejected TRiX proposals produce command=None.
        if command is None:
            return None

        # Symbolic no-motion actions do not create a target.
        if command.kind == BridgeKind.NO_MOTION:
            return None

        # STOP_REQUEST is handled by the physical runtime,
        # never transformed into a position command.
        if command.kind == BridgeKind.STOP_REQUEST:
            return None

        if command.kind != BridgeKind.MOVE:
            raise PhysicalAdapterSafetyError(
                f"UNKNOWN_BRIDGE_KIND:{command.kind}"
            )

        unsupported = (
            set(command.target_delta_deg)
            - set(SUPPORTED_JOINTS)
        )

        if unsupported:
            raise PhysicalAdapterSafetyError(
                "UNSUPPORTED_JOINT_AT_PHYSICAL_BOUNDARY:"
                + ",".join(sorted(unsupported))
            )

        # Begin from the exact physical pose captured at connection.
        target = list(self._start)

        # Apply ONLY approved J2-J6 arm relative displacements.
        for joint, delta in (
            command.target_delta_deg.items()
        ):
            idx = self.INDEX[joint]

            target[idx] = (
                self._start[idx]
                + float(delta)
            )

        # ----------------------------------------------------
        # HARD PHYSICAL INVARIANTS
        # ----------------------------------------------------

        # J1 must remain EXACTLY where it was captured.
        target[0] = self._frozen_j1

        # J7 is relative to the physically captured gripper start.
        #
        # This physical boundary independently enforces the currently
        # proven gripper envelope even if an upstream bug constructs
        # a malicious BridgeCommand.
        gripper_delta = float(
            command.gripper_delta_deg
        )

        if not 0.0 <= gripper_delta <= 40.0:
            raise PhysicalAdapterSafetyError(
                "J7_DELTA_OUTSIDE_PHYSICAL_ENVELOPE:"
                f"{gripper_delta:+.6f}"
            )

        target[6] = (
            self._captured_j7
            + gripper_delta
        )

        if target[0] != self._frozen_j1:
            raise PhysicalAdapterSafetyError(
                "J1_INVARIANT_VIOLATED"
            )

        return PhysicalTarget(
            angles_deg=tuple(target),
            action=command.action.value,
        )

    def assert_j1_invariant(
        self,
        target: Sequence[float],
    ) -> None:

        if len(target) != 7:
            raise PhysicalAdapterSafetyError(
                "BAD_TARGET_LENGTH"
            )

        if abs(
            float(target[0])
            - self._frozen_j1
        ) > 1e-9:
            raise PhysicalAdapterSafetyError(
                (
                    "J1_TARGET_CHANGED:"
                    f"captured={self._frozen_j1:+.6f},"
                    f" target={float(target[0]):+.6f}"
                )
            )
