#!/usr/bin/env python3
"""Render the archived SNAP frame and its deflected enclosure latch."""

import json
import struct
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from PIL import Image
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_wrench import annotate, force, metadata

DELTA_DISENGAGE = 0.002


def make_box_between(cid, p0, p1, half_y, half_z, rgba):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    d = p1 - p0
    L = np.linalg.norm(d)

    if L < 1e-9:
        raise ValueError("zero-length segment")

    x_axis = b601.normalize(d)

    helper = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(x_axis, helper)) > 0.95:
        helper = np.array([0.0, 1.0, 0.0])

    z_axis = b601.normalize(np.cross(x_axis, helper))

    y_axis = b601.normalize(np.cross(z_axis, x_axis))

    R = np.column_stack(
        [
            x_axis,
            y_axis,
            z_axis,
        ]
    )

    q = b601.quat_from_matrix(R)

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            L / 2.0,
            half_y,
            half_z,
        ],
        rgbaColor=rgba,
        specularColor=[0.22, 0.22, 0.22],
        physicsClientId=cid,
    )

    return p.createMultiBody(
        0,
        -1,
        visual,
        ((p0 + p1) / 2.0).tolist(),
        q,
        physicsClientId=cid,
    )


def fingertip_geometry(robot, cid, scale):
    """Measure only the downward black PLA tips, not their wide CNC carriers."""
    tips = []

    for link, mesh_name in ((7, "pla_left.STL"), (8, "pla_right.STL")):
        mesh = (b601.MESH_DIR / mesh_name).read_bytes()
        triangles = struct.unpack_from("<I", mesh, 80)[0]
        vertices = np.ndarray(
            (triangles, 3, 3),
            dtype="<f4",
            buffer=mesh,
            offset=96,
            strides=(50, 12, 4),
        ).reshape(-1, 3)
        pos, quat = p.getLinkState(
            robot,
            link,
            computeForwardKinematics=True,
            physicsClientId=cid,
        )[4:6]
        rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        world = (vertices * scale) @ rotation.T + pos
        bottom = world[:, 2].min()
        tip = world[world[:, 2] <= bottom + 0.008]
        tips.append(
            {
                "inner_y": tip[:, 1].max() if link == 7 else tip[:, 1].min(),
                "center_x": 0.5 * (tip[:, 0].min() + tip[:, 0].max()),
                "bottom_z": bottom,
            }
        )

    return tips[0], tips[1]


def open_and_orient_gripper(robot, cid):
    for joint in (7, 8):
        upper = float(p.getJointInfo(robot, joint, physicsClientId=cid)[9])
        p.resetJointState(robot, joint, upper, physicsClientId=cid)

    ref_pos, ref_quat, _, local_frame = b601.local_tool_frame(robot, cid)
    # Close along Y across the lid's shorter front/back span.
    world_frame = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # The local approach axis maps upward, leaving the physical tips downward.
    b601.orient_robot(
        robot,
        cid,
        world_frame @ local_frame.T,
        ref_pos,
        ref_quat,
    )


# ============================================================
# Main
# ============================================================


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    provenance = json.loads(provenance_path.read_text())

    entry = provenance["states"]["SNAP"]

    if entry["trace_step"] != 5:
        raise RuntimeError(f"Expected SNAP step 5, got {entry['trace_step']}")

    st = entry["frame"]["state"]

    delta = float(st["delta_m"])
    z = float(st["z_m"])
    released = bool(st["released"])
    broken = bool(st["latch_broken"])

    if delta < DELTA_DISENGAGE:
        raise RuntimeError("Selected frame is not disengaged")

    if released:
        raise RuntimeError("Selected frame should have released=False")

    if broken:
        raise RuntimeError("Selected frame should have intact latch")

    scale, _ = b601.choose_visual_scale()

    cid = p.connect(p.DIRECT)
    try:

        # Colors

        FLOOR = [0.17, 0.18, 0.20, 1.0]

        BOX = [0.045, 0.050, 0.060, 1.0]
        BOX_EDGE = [0.075, 0.080, 0.095, 1.0]

        LID = [0.06, 0.34, 0.10, 1.0]
        LID_HI = [0.09, 0.48, 0.15, 1.0]

        LATCH = [0.65, 0.67, 0.70, 1.0]

        # Table/base

        b601.make_box(
            cid,
            [0.075, 0.060, 0.007],
            [0, 0, -0.007],
            FLOOR,
        )

        # ABS enclosure

        BOX_HALF_X = 0.050
        BOX_HALF_Y = 0.033
        BOX_HALF_Z = 0.014

        box_center_z = 0.014

        # Solid lower body.
        b601.make_box(
            cid,
            [
                BOX_HALF_X,
                BOX_HALF_Y,
                BOX_HALF_Z,
            ],
            [
                0.0,
                0.0,
                box_center_z,
            ],
            BOX,
        )

        # Raised top rim around opening.
        rim_z = box_center_z + BOX_HALF_Z + 0.003

        # Front rim
        b601.make_box(
            cid,
            [0.050, 0.003, 0.004],
            [0.0, -0.030, rim_z],
            BOX_EDGE,
        )

        # Rear rim
        b601.make_box(
            cid,
            [0.050, 0.003, 0.004],
            [0.0, 0.030, rim_z],
            BOX_EDGE,
        )

        # Left rim
        b601.make_box(
            cid,
            [0.003, 0.027, 0.004],
            [-0.047, 0.0, rim_z],
            BOX_EDGE,
        )

        # Right rim, with front-side break for latch
        b601.make_box(
            cid,
            [0.003, 0.013, 0.004],
            [0.047, 0.015, rim_z],
            BOX_EDGE,
        )

        # GREEN REMOVABLE SNAP-FIT LID

        LID_HALF_X = 0.047
        LID_HALF_Y = 0.030
        LID_HALF_Z = 0.0025

        # Frozen extraction displacement only.
        lid_center_z = rim_z + 0.005 + z

        lid_top_z = lid_center_z + LID_HALF_Z

        # Main thin lid.
        b601.make_box(
            cid,
            [
                LID_HALF_X,
                LID_HALF_Y,
                LID_HALF_Z,
            ],
            [
                0.0,
                0.0,
                lid_center_z,
            ],
            LID,
        )

        # Thin front skirt.
        b601.make_box(
            cid,
            [0.045, 0.0025, 0.005],
            [
                0.0,
                -0.0275,
                lid_center_z - 0.004,
            ],
            LID,
        )

        # Rear skirt.
        b601.make_box(
            cid,
            [0.045, 0.0025, 0.005],
            [
                0.0,
                0.0275,
                lid_center_z - 0.004,
            ],
            LID,
        )

        # Side skirts.
        b601.make_box(
            cid,
            [0.0025, 0.025, 0.005],
            [
                -0.0445,
                0.0,
                lid_center_z - 0.004,
            ],
            LID,
        )

        b601.make_box(
            cid,
            [0.0025, 0.025, 0.005],
            [
                0.0445,
                0.0,
                lid_center_z - 0.004,
            ],
            LID,
        )

        # SNAP INTERFACE
        #
        # Green lid has a retaining tooth.
        # Gray latch belongs to black box.

        catch_y = -0.019

        catch_x = 0.049

        catch_z = lid_center_z - 0.006

        # Green molded tab neck.
        b601.make_box(
            cid,
            [0.0045, 0.006, 0.003],
            [
                0.045,
                catch_y,
                catch_z + 0.002,
            ],
            LID_HI,
        )

        # Green retaining tooth.
        b601.make_box(
            cid,
            [0.005, 0.0065, 0.0025],
            [
                catch_x,
                catch_y,
                catch_z,
            ],
            LID_HI,
        )

        # Gray housing-integral cantilever

        latch_root = np.array([0.052, catch_y - 0.005, 0.022])

        # Small box-wall anchor; the slender gray arm remains fully exposed.
        b601.make_box(
            cid,
            [0.0025, 0.004, 0.003],
            [0.051, catch_y - 0.005, 0.022],
            BOX_EDGE,
        )

        # Engaged hook position.
        nominal_hook = np.array(
            [
                catch_x + 0.0035,
                catch_y - 0.005,
                catch_z + 0.006,
            ]
        )

        # Current frozen state: hook displaced outward.
        hook_world = nominal_hook + np.array(
            [
                delta,
                0.0,
                0.0,
            ]
        )

        # Flexible cantilever arm.
        make_box_between(
            cid,
            latch_root,
            hook_world,
            half_y=0.0018,
            half_z=0.0011,
            rgba=LATCH,
        )

        # Inward hook lip faces the green retaining tooth across the
        # clearance created by the recorded 5.13 mm outward deflection.
        hook_lip = np.array(
            [
                hook_world[0] - 0.0008,
                catch_y - 0.003,
                catch_z + 0.002,
            ]
        )

        make_box_between(
            cid,
            hook_world,
            hook_lip,
            half_y=0.0018,
            half_z=0.0011,
            rgba=LATCH,
        )

        b601.make_box(
            cid,
            [0.0019, 0.0035, 0.0014],
            hook_lip.tolist(),
            LATCH,
        )

        # B601 GRIPPER
        #
        # Grip the actual lid edges.

        robot = b601.load_robot(
            cid,
            scale,
        )
        open_and_orient_gripper(robot, cid)
        front, back = fingertip_geometry(robot, cid, scale)
        initial_gap = back["inner_y"] - front["inner_y"]
        target_gap = 2.0 * LID_HALF_Y

        # Scale this SNAP instance so the real black fingertip faces span
        # the 60 mm front/back edges with the B601 jaws fully open.
        p.removeBody(robot, physicsClientId=cid)
        scale *= target_gap / initial_gap
        robot = b601.load_robot(cid, scale)
        open_and_orient_gripper(robot, cid)
        front, back = fingertip_geometry(robot, cid, scale)
        gap = back["inner_y"] - front["inner_y"]

        if abs(gap - target_gap) > 0.0005:
            raise RuntimeError(f"B601 fingertip gap {gap:.5f} misses lid edges")

        current_contact = np.array(
            [
                0.5 * (front["center_x"] + back["center_x"]),
                0.5 * (front["inner_y"] + back["inner_y"]),
                0.5 * (front["bottom_z"] + back["bottom_z"]),
            ]
        )
        target_contact = np.array([0.0, 0.0, lid_top_z - 0.003])
        b601.translate_robot(robot, cid, target_contact - current_contact)
        front, back = fingertip_geometry(robot, cid, scale)

        if (
            abs(front["inner_y"] + LID_HALF_Y) > 0.0005
            or abs(back["inner_y"] - LID_HALF_Y) > 0.0005
            or abs(front["center_x"]) > 0.0005
            or abs(back["center_x"]) > 0.0005
        ):
            raise RuntimeError("B601 fingertips are not centered on the lid edges")

        # Short wrist stub

        lo, hi = b601.visible_bounds(
            robot,
            cid,
        )

        ax = 0.5 * (lo[0] + hi[0])

        ay = 0.5 * (lo[1] + hi[1])

        b601.make_cyl(
            cid,
            0.011,
            0.008,
            [
                ax,
                ay,
                hi[2] + 0.004,
            ],
            [0.48, 0.50, 0.53, 1.0],
        )

        b601.make_box(
            cid,
            [0.010, 0.010, 0.020],
            [
                ax,
                ay,
                hi[2] + 0.028,
            ],
            [0.14, 0.16, 0.19, 1.0],
        )

        # Render

        width = 1800
        height = 1400

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[
                0.004,
                -0.002,
                0.026,
            ],
            distance=0.285,
            yaw=34,
            pitch=-24,
            roll=0,
            upAxisIndex=2,
            physicsClientId=cid,
        )

        projection = p.computeProjectionMatrixFOV(
            fov=36,
            aspect=width / height,
            nearVal=0.003,
            farVal=3.0,
            physicsClientId=cid,
        )

        _, _, rgba, depth, _ = p.getCameraImage(
            width,
            height,
            view,
            projection,
            shadow=1,
            lightDirection=[
                -0.45,
                -0.60,
                1.8,
            ],
            lightColor=[
                1.0,
                0.98,
                0.95,
            ],
            lightAmbientCoeff=0.44,
            lightDiffuseCoeff=0.64,
            lightSpecularCoeff=0.20,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=cid,
        )

        image = np.asarray(
            rgba,
            dtype=np.uint8,
        ).reshape(
            height,
            width,
            4,
        )[:, :, :3]

        from workspace_render_utils import add_coordinate_reference, coordinate_metadata

        wrenches = [
            force([0, 0, lid_top_z], [0, 0, 1], offset=[-0.039, -0.027, 0.015], label="F_pull"),
            force(
                hook_world,
                [1, 0, 0],
                offset=[0.004, -0.005, 0.002],
                length=0.020,
                label="F_latch",
                label_offset=(18, -5),
            ),
        ]
        Image.fromarray(image).save(output_dir / "snap_workspace_clean.png", dpi=(300, 300))
        annotated = add_coordinate_reference(
            image, depth, view, projection, -0.0141, [-0.10, 0.10, -0.09, 0.09]
        )
        annotate(annotated, view, projection, wrenches).save(
            output_dir / "snap_workspace.png", dpi=(300, 300)
        )

        manifest = {
            "task": "SNAP",
            "trace_file": entry["trace_file"],
            "trace_sha256": entry["trace_sha256"],
            "observation": entry["frame"]["obs"],
            "trace_step": entry["trace_step"],
            "episode_seed": entry["episode_seed"],
            "training_seed": entry["training_seed"],
            "checkpoint_step": entry["checkpoint_step"],
            "state": st,
            "interpretation": (
                "snap-fit removable electronics enclosure lid "
                "retained by a housing-mounted cantilever latch"
            ),
            "visualization": {
                "delta_m": delta,
                "z_m": z,
                "disengage_threshold_m": DELTA_DISENGAGE,
                "released": released,
                "latch_broken": broken,
                "B601_visual_scale": scale,
                "grasp_axis": "Y (front/back lid edges)",
                "fingertip_gap_m": gap,
                **coordinate_metadata(),
                **metadata(wrenches),
                "arm_kinematics_simulated": False,
            },
        }

        (output_dir / "snap_workspace_manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
            )
            + "\n"
        )

        print("SNAP archived frame")
        print("delta mm       :", delta * 1000)
        print("extraction mm  :", z * 1000)
        print("released       :", released)
        print("broken         :", broken)
        print("jaw gap m      :", gap)
        print("B601 scale     :", scale)

        print(
            "\nWROTE:",
            output_dir / "snap_workspace.png",
        )

    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "SNAP", debug=False)
