#!/usr/bin/env python3
"""Render the archived BATTERY peel state as a recognizable foil pouch cell."""

import json
import tempfile
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from b601_render_common import black_meshes, fingertip_geometry
from PIL import Image, ImageDraw, ImageFont
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_render_utils import font_directory, rounded_solid


def make_label(path):
    """Draw the cell label texture."""
    image = Image.new("RGB", (1400, 800), (28, 31, 34))
    draw = ImageDraw.Draw(image)
    fontdir = font_directory()

    def font(size, bold=False):
        return ImageFont.truetype(
            str(fontdir / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size
        )

    ink = (238, 239, 236)
    draw.text((85, 65), "Li-ion", font=font(185, True), fill=ink)
    draw.text((90, 315), "RECHARGEABLE", font=font(64, True), fill=ink)
    draw.text((92, 419), "POUCH CELL", font=font(52), fill=(195, 200, 202))
    draw.line((90, 535, 1130, 535), fill=(133, 141, 146), width=4)
    # A simple cell symbol makes the label recognizable at panel size.
    draw.rounded_rectangle((95, 610, 305, 706), radius=12, outline=ink, width=8)
    draw.rectangle((306, 638, 326, 678), fill=ink)
    for x in (118, 175, 232):
        draw.rectangle((x, 631, x + 35, 686), fill=ink)
    draw.text((370, 621), "BATTERY", font=font(63, True), fill=ink)
    draw.text((1225, 90), "+", font=font(98, True), fill=ink)
    draw.text((1225, 540), "−", font=font(98, True), fill=ink)
    image.save(path)


def textured_face(cid, path, half_x, half_y, pos):
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=[
            [-half_x, -half_y, 0],
            [half_x, -half_y, 0],
            [half_x, half_y, 0],
            [-half_x, half_y, 0],
        ],
        indices=[0, 1, 2, 0, 2, 3],
        uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
        normals=[[0, 0, 1]] * 4,
        rgbaColor=[1, 1, 1, 1],
        physicsClientId=cid,
    )
    body = p.createMultiBody(0, -1, visual, pos, physicsClientId=cid)
    texture = p.loadTexture(str(path), physicsClientId=cid)
    p.changeVisualShape(body, -1, textureUniqueId=texture, physicsClientId=cid)


def place_gripper(cid, center, tab_width):
    scale, _ = b601.choose_visual_scale()
    robot = b601.load_robot(cid, scale)
    upper = [float(p.getJointInfo(robot, j, physicsClientId=cid)[9]) for j in (7, 8)]

    def opening(fraction):
        for j, limit in zip((7, 8), upper):
            p.resetJointState(robot, j, fraction * limit, physicsClientId=cid)

    opening(1)
    ref_pos, ref_quat, _, local_frame = b601.local_tool_frame(robot, cid)
    world_frame = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    b601.orient_robot(robot, cid, world_frame @ local_frame.T, ref_pos, ref_quat)
    meshes = black_meshes()
    front, back = fingertip_geometry(robot, cid, scale, meshes)
    max_gap = back["inner_y"] - front["inner_y"]
    opening(0)
    front, back = fingertip_geometry(robot, cid, scale, meshes)
    min_gap = back["inner_y"] - front["inner_y"]
    fraction = (tab_width - min_gap) / (max_gap - min_gap)
    if not 0 <= fraction <= 1:
        raise RuntimeError("Pull-tab width exceeds the B601 jaw range")
    opening(fraction)
    front, back = fingertip_geometry(robot, cid, scale, meshes)
    contact = np.array(
        [
            0.5 * (front["center_x"] + back["center_x"]),
            0.5 * (front["inner_y"] + back["inner_y"]),
            0.5 * (front["bottom_z"] + back["bottom_z"]),
        ]
    )
    b601.translate_robot(robot, cid, np.asarray(center) - contact)
    front, back = fingertip_geometry(robot, cid, scale, meshes)
    gap = back["inner_y"] - front["inner_y"]
    if not (
        abs(gap - tab_width) < 0.0001
        and abs(front["inner_y"] + tab_width / 2) < 0.0001
        and abs(back["inner_y"] - tab_width / 2) < 0.0001
        and all(abs(t["center_x"] - center[0]) < 0.0001 for t in (front, back))
    ):
        raise RuntimeError("B601 tips do not meet the pull-tab edges")
    lo, hi = b601.visible_bounds(robot, cid)
    ax, ay = 0.5 * (lo[:2] + hi[:2])
    b601.make_cyl(cid, 0.011, 0.008, [ax, ay, hi[2] + 0.004], [0.48, 0.50, 0.53, 1])
    b601.make_box(cid, [0.010, 0.010, 0.015], [ax, ay, hi[2] + 0.023], [0.14, 0.16, 0.19, 1])
    return {
        "visual_scale": scale,
        "opening_fraction": fraction,
        "gap_m": gap,
        "fingertips": [front, back],
    }


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    entry = json.loads(provenance_path.read_text())["states"]["BATTERY"]
    state = entry["frame"]["state"]
    if entry["trace_step"] != 329:
        raise RuntimeError("Expected the frozen BATTERY frame at step 329")
    if state["deformation"] != 0 or state["short_state"] != 0 or state["short_flag"]:
        raise RuntimeError("This intact pouch geometry requires the undamaged archived state")
    lift = float(state["z_m"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cid = p.connect(p.DIRECT)
    try:
        with tempfile.TemporaryDirectory(prefix="trix_battery_") as tempdir:
            # Shallow device tray and two discrete adhesive strips.
            b601.make_box(cid, [0.080, 0.054, 0.005], [0, 0, -0.005], [0.17, 0.18, 0.20, 1])
            tray = [0.065, 0.073, 0.085, 1]
            b601.make_box(cid, [0.069, 0.040, 0.0007], [0, 0, 0.0007], tray)
            for y in (-0.041, 0.041):
                b601.make_box(cid, [0.071, 0.002, 0.002], [0, y, 0.002], tray)
            b601.make_box(cid, [0.002, 0.039, 0.002], [0.070, 0, 0.002], tray)
            for y in (-0.027, 0.027):
                b601.make_box(cid, [0.002, 0.014, 0.002], [-0.070, y, 0.002], tray)
            adhesive_top = 0.0018
            for y in (-0.018, 0.018):
                b601.make_box(
                    cid, [0.055, 0.006, 0.0002], [-0.003, y, 0.0016], [0.74, 0.58, 0.27, 1]
                )

            # The complete, flat cell translates by the archived peel displacement.
            # No bend, swelling, or damage is invented for deformation == 0.
            cell_bottom = adhesive_top + lift
            thickness = 0.0055
            cell_top = cell_bottom + thickness
            foil = [0.68, 0.71, 0.73, 1]
            rounded_solid(
                cid,
                0.0575,
                0.0355,
                0.0004,
                0.0045,
                0.0001,
                [0, 0, cell_bottom + 0.0010],
                [0.50, 0.53, 0.55, 1],
            )
            rounded_solid(
                cid,
                0.054,
                0.032,
                thickness,
                0.0038,
                0.0008,
                [0, 0, cell_bottom + thickness / 2],
                foil,
            )
            # Fine crimp lines on the exposed heat-sealed foil margin.
            for x in np.linspace(-0.047, 0.047, 37):
                for y in (-0.034, 0.034):
                    b601.make_box(
                        cid,
                        [0.00014, 0.0008, 0.00004],
                        [float(x), y, cell_bottom + 0.00124],
                        [0.37, 0.40, 0.42, 1],
                    )
            label = Path(tempdir) / "pouch_label.png"
            make_label(label)
            textured_face(cid, label, 0.050, 0.0278, [0, 0, cell_top + 0.00004])

            # Electrical terminals share one end, with an insulating root seal.
            b601.make_box(
                cid,
                [0.003, 0.026, 0.00035],
                [0.055, 0, cell_bottom + 0.0016],
                [0.73, 0.38, 0.08, 1],
            )
            for y, color in ((0.013, [0.78, 0.80, 0.81, 1]), (-0.013, [0.63, 0.67, 0.70, 1])):
                b601.make_box(cid, [0.007, 0.0045, 0.0003], [0.062, y, cell_bottom + 0.0017], color)

            # Wide extraction tab bonded underneath the opposite cell end.
            tab_width = 0.018
            tab_z = cell_bottom + 0.0008
            rounded_solid(
                cid,
                0.020,
                tab_width / 2,
                0.0007,
                0.0014,
                0.00015,
                [-0.061, 0, tab_z],
                [0.91, 0.89, 0.81, 1],
            )
            grasp = place_gripper(cid, [-0.071, 0, tab_z - 0.0007], tab_width)

            width, height = 1800, 1400
            camera = {
                "target": [-0.015, 0, 0.030],
                "distance": 0.350,
                "yaw": 34,
                "pitch": -30,
                "fov": 36,
            }
            view = p.computeViewMatrixFromYawPitchRoll(
                camera["target"],
                camera["distance"],
                camera["yaw"],
                camera["pitch"],
                0,
                2,
                physicsClientId=cid,
            )
            projection = p.computeProjectionMatrixFOV(
                camera["fov"], width / height, 0.003, 3, physicsClientId=cid
            )
            _, _, rgba, depth, _ = p.getCameraImage(
                width,
                height,
                view,
                projection,
                shadow=1,
                lightDirection=[-0.45, -0.60, 1.8],
                lightColor=[1, 0.98, 0.95],
                lightAmbientCoeff=0.44,
                lightDiffuseCoeff=0.64,
                lightSpecularCoeff=0.20,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=cid,
            )
            image = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            from workspace_render_utils import (
                add_coordinate_reference,
                coordinate_metadata,
            )

            Image.fromarray(image).save(output_dir / "battery_workspace_clean.png", dpi=(300, 300))
            add_coordinate_reference(
                image, depth, view, projection, -0.0101, [-0.115, 0.105, -0.09, 0.09]
            ).save(output_dir / "battery_workspace.png", dpi=(300, 300))

        manifest = {
            "task": "BATTERY",
            **{
                key: entry[key]
                for key in (
                    "trace_file",
                    "trace_sha256",
                    "trace_step",
                    "episode_seed",
                    "training_seed",
                    "checkpoint_step",
                )
            },
            "state": state,
            "interpretation": "intact lithium-ion pouch cell lifted from adhesive using an extraction tab",
            "visualization": {
                "lift_m": lift,
                "cell_thickness_m": thickness,
                "gripper": grasp,
                "camera": camera,
                **coordinate_metadata(),
                "arm_kinematics_simulated": False,
                "geometry_is_illustrative": True,
            },
        }
        (output_dir / "battery_workspace_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print("BATTERY peel lift mm:", lift * 1000)
        print("Deformation:", state["deformation"], "temperature C:", state["temperature_C"])
        print("Pull-tab grasp gap mm:", grasp["gap_m"] * 1000)
        print("WROTE:", output_dir / "battery_workspace.png")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "BATTERY", debug=False)
