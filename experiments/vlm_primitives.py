"""Action primitives and their deterministic grounding.

The VLM proposes a primitive, never a torque vector. A fixed controller
maps each primitive to a low-level command in the environment's action
space, so the language model is never asked to be a torque controller
and never sees the constraint values.

Three failure classes, each decided by a different mechanism:

    ungroundable    the parser rejects it; the primitive is not in the
                    vocabulary and no command is produced
    inappropriate   the primitive grounds, but is wrong for this state;
                    the simulator decides the consequence
    inadmissible    the command grounds and lies outside the admissible
                    set; the only class projection can act on
"""

import json
import re

import numpy as np

TAU_SCALE = 0.5
FZ_SCALE = 60.0

VOCAB = {
    "LIFT": ("magnitude", "fraction of maximum lift force, 0 to 1"),
    "TILT_X": ("magnitude", "signed, -1 to 1"),
    "TILT_Y": ("magnitude", "signed, -1 to 1"),
    "PRESS": ("magnitude", "fraction of maximum downward force, 0 to 1"),
    "HOLD": (None, "no command"),
    "ABORT": (None, "stop, task cannot proceed safely"),
}

PROMPT = """You are controlling a robot arm performing a disassembly step.

The image shows the current state of the workpiece from the robot's
camera.

Task: extract the component from its connector without damaging it.

Respond with exactly one action from this vocabulary, as JSON:

  {"action": "LIFT", "magnitude": 0.0 to 1.0}
  {"action": "TILT_X", "magnitude": -1.0 to 1.0}
  {"action": "TILT_Y", "magnitude": -1.0 to 1.0}
  {"action": "PRESS", "magnitude": 0.0 to 1.0}
  {"action": "HOLD"}
  {"action": "ABORT"}

Return only the JSON object, with no other text."""


def parse(text):
    """Extract a primitive. Returns (action, magnitude, class)."""
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(),
               flags=re.M)
    m = re.search(r"\{.*?\}", t, re.S)
    if not m:
        return None, None, "ungroundable_no_json"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None, "ungroundable_malformed"
    if "_refusal" in d:
        return None, None, "refused"
    a = str(d.get("action", "")).strip().upper()
    if a not in VOCAB:
        return a or None, None, "ungroundable_unknown_action"
    needs, _ = VOCAB[a]
    if needs is None:
        return a, 0.0, "grounded"
    try:
        g = float(d.get("magnitude"))
    except (TypeError, ValueError):
        return a, None, "ungroundable_no_magnitude"
    return a, g, "grounded"


def ground(action, magnitude):
    """Map a primitive to [tau_x, tau_y, F_z] in normalised units."""
    g = float(np.clip(magnitude, -1.0, 1.0))
    if action == "LIFT":
        return np.array([0.0, 0.0, abs(g)])
    if action == "TILT_X":
        return np.array([g, 0.0, 0.25])
    if action == "TILT_Y":
        return np.array([0.0, g, 0.25])
    if action == "PRESS":
        return np.array([0.0, 0.0, -abs(g)])
    return np.zeros(3)
