#!/usr/bin/env python3
"""Export the six recorded panel states in the renders/tasks manifest format."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from workspace_environment import prepare_environment
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, ROOT

TASKS = ("SCREW", "PCB", "SNAP", "CRANK", "BATTERY", "PRY")
TRACE_KEYS = (
    "trace_file", "trace_sha256", "evaluation_arm", "training_seed",
    "checkpoint_step", "episode_seed", "trace_step",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_panel(task, panel, entry, binding):
    """Reject stale or unrelated panel state before composing a publication figure."""
    if panel["task"] != task or panel["state"] != entry["frame"]["state"]:
        raise ValueError(f"{task}: rendered panel does not match the recorded state")
    if panel["observation"] != entry["frame"]["obs"]:
        raise ValueError(f"{task}: rendered observation does not match the recorded observation")
    for key in TRACE_KEYS:
        if key != "evaluation_arm" and panel[key] != entry[key]:
            raise ValueError(f"{task}: trace provenance differs at {key}")
    visual = panel["visualization"]
    if visual["environment"] != binding:
        raise ValueError(f"{task}: environment binding differs from the recorded state")
    state = binding["state_variables"]
    fields = {
        "SCREW": (("theta_rad", "theta_rad"), ("lift_m", "z_m")),
        "PCB": (("lift_m", "lift_m"), ("tilt_x_rad", "tilt_x_rad"), ("tilt_y_rad", "tilt_y_rad")),
        "SNAP": (("delta_m", "delta_m"), ("z_m", "z_m")),
        "CRANK": (("theta_raw_rad", "theta_rad"), ("z_m", "z_m")),
        "BATTERY": (("lift_m", "z_m"),),
        "PRY": (("gap_m", "gap_m"), ("insertion_depth_m", "insertion_depth_m")),
    }[task]
    for visual_key, state_key in fields:
        if not math.isclose(visual[visual_key], state[state_key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{task}: rendered {visual_key} differs from environment {state_key}")
    if task == "PRY" and not math.isclose(
        visual["tool_axis_angle_rad"], math.remainder(state["theta_rad"], math.tau),
        rel_tol=0, abs_tol=1e-12,
    ):
        raise ValueError("PRY: rendered angle differs from recorded orientation")


def build_manifest(workspace_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE):
    workspace_dir, provenance_path = Path(workspace_dir), Path(provenance_path)
    try:
        source = provenance_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source = provenance_path.name
    result = dict(
        purpose="Representative state-conditioned DISASM-Bench workspace visualization",
        note="Workpiece poses come from recorded analytical-environment states. Values in states use the renders/tasks/state_manifest.json field convention; full unwrapped environment fields and observations are preserved below. No experiment is recomputed.",
        states={},
        provenance_source=source,
        provenance_sha256=sha256(provenance_path),
        trace_provenance={},
        visualization_scope="PyBullet renders the recorded analytical-benchmark state. B601 grasps follow the workpiece or tool frame. Static CAD dimensions are fixed; articulated arm and gripper contact dynamics are not simulated.",
        units=dict(length="m", angle="rad", angular_velocity="rad/s", linear_velocity="m/s", temperature="deg C"),
        state_conventions=dict(
            screw_crank_theta="orientation modulo 2*pi in [0, 2*pi); unwrapped theta is retained in environment_states",
            pry_theta="orientation as remainder(theta, 2*pi) in [-pi, pi]; rigid tool applied as Ry(-theta)",
            pry_gap="free-edge vertical opening; opposite lid edge supported; lid angle derived as asin(gap / fixed span), toe height follows lid underside at recorded insertion",
            battery_grasp="fixed folded extraction tab and its opposed-face grasp translate with cell z",
            pcb_tilt="Euclidean norm of tilt_x and tilt_y; components map to environment theta[0] and theta[1]",
            battery_short="boolean short flag; continuous severity is retained in environment_states.short_state",
        ),
        environment_states={},
        recorded_observations={},
        environment_provenance={},
        panels={},
    )
    for letter, task in zip("abcdef", TASKS):
        name = task.lower()
        panel_path = workspace_dir / f"{name}_workspace_manifest.json"
        panel = json.loads(panel_path.read_text())
        env, model, entry, binding = prepare_environment(task, provenance_path)
        check_panel(task, panel, entry, binding)
        if task == "SCREW":
            state = dict(theta=float(env.theta) % math.tau, z=float(env.z), engagement=float(env.engagement))
        elif task == "PCB":
            state = dict(lift=float(env.z), tilt_x=float(env.theta[0]), tilt_y=float(env.theta[1]),
                         tilt=float(np.linalg.norm(env.theta)),
                         condition="fractured" if env.fractured else "damaged" if env.damage > 0 else "normal")
        elif task == "SNAP":
            state = dict(delta=float(env.delta), z=float(env.z), released=bool(env.released), broken=bool(env.latch_broken))
        elif task == "CRANK":
            state = dict(theta=float(env.theta) % math.tau, z=float(env.z))
        elif task == "BATTERY":
            state = dict(z=float(env.z), deform=float(env.deform), short=bool(env.short > 0), temperature=float(env.temperature))
        else:
            state = dict(theta=math.remainder(float(env.state.theta), math.tau), gap=float(env.state.position[2]),
                         insertion_depth=float(env.state.position[0]), crack=float(env.crack))
        result["states"][task] = state
        result["trace_provenance"][task] = {key: entry[key] for key in TRACE_KEYS}
        result["environment_states"][task] = binding["state_variables"]
        result["recorded_observations"][task] = entry["frame"]["obs"]
        result["environment_provenance"][task] = {
            key: binding[key] for key in ("environment_class", "environment_source_sha256", "constraint_hash")
        }
        script = f"scripts/render_{name}_workspace_b601.py"
        result["panels"][task] = dict(
            panel=letter,
            image=f"{name}_workspace.png", image_sha256=sha256(workspace_dir/f"{name}_workspace.png"),
            clean_image=f"{name}_workspace_clean.png", clean_image_sha256=sha256(workspace_dir/f"{name}_workspace_clean.png"),
            render_manifest=panel_path.name, render_manifest_sha256=sha256(panel_path),
            renderer_script=script, renderer_script_sha256=sha256(ROOT/script),
        )
    return result


def write_manifest(workspace_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, *, check=False):
    data = build_manifest(workspace_dir, provenance_path)
    path = Path(workspace_dir)/"state_manifest.json"
    if check:
        if json.loads(path.read_text()) != data:
            raise ValueError(f"{path}: saved manifest differs from current states, source files or images")
    else:
        path.write_text(json.dumps(data, indent=2)+"\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--check", action="store_true", help="verify the saved manifest without writing")
    args = parser.parse_args()
    print(write_manifest(args.workspace_dir, args.state_file, check=args.check))
