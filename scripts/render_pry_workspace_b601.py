#!/usr/bin/env python3
"""Visualize the frozen PRY state with an enclosure and a B601-held lever."""

import json
import math
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from b601_render_common import black_meshes, fingertip_geometry
from PIL import Image
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_render_utils import rounded_outline, rounded_solid


def mesh_body(cid, vertices, indices, color):
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=color,
        specularColor=[0.35, 0.35, 0.35],
        physicsClientId=cid,
    )
    return p.createMultiBody(0, -1, visual, physicsClientId=cid)


def hollow_housing(cid, half_x, half_y, wall_top):
    """Closed wall mesh around an open cavity, with a visible top rim."""
    outer = rounded_outline(half_x, half_y, 0.005)
    inner = rounded_outline(half_x - 0.004, half_y - 0.004, 0.0015)
    n = len(outer)
    vertices = []
    for contour, z in ((outer, 0.002), (outer, wall_top), (inner, 0.002), (inner, wall_top)):
        vertices.extend([[x, y, z] for x, y in contour])
    indices = []

    def quad(a, b, c, d):
        indices.extend([a, b, c, a, c, d])

    for i in range(n):
        j = (i + 1) % n
        quad(i, j, n + j, n + i)
        quad(2 * n + j, 2 * n + i, 3 * n + i, 3 * n + j)
        quad(n + i, n + j, 3 * n + j, 3 * n + i)
        quad(j, i, 2 * n + i, 2 * n + j)
    body = mesh_body(cid, vertices, indices, [0.17, 0.20, 0.23, 1])
    rounded_solid(cid, half_x, half_y, 0.002, 0.005, 0.0004, [0, 0, 0.001], [0.13, 0.15, 0.18, 1])
    return body


def blade_mesh(cid, stations, tool_y):
    """A continuous thin steel blade, including the upturned working tip.

    Each station is x, center z, width, vertical thickness.  The short
    bend gives the toe and heel distinct contacts; the straight shank
    beyond the heel follows the recorded tool orientation.
    """
    vertices = []
    for x, z, width, thickness in stations:
        vertices.extend(
            [
                [x, tool_y - width / 2, z - thickness / 2],
                [x, tool_y + width / 2, z - thickness / 2],
                [x, tool_y + width / 2, z + thickness / 2],
                [x, tool_y - width / 2, z + thickness / 2],
            ]
        )
    indices = [0, 2, 1, 0, 3, 2]
    for ring in range(len(stations) - 1):
        for j in range(4):
            k = (j + 1) % 4
            a, b, c, d = 4 * ring + j, 4 * ring + k, 4 * (ring + 1) + k, 4 * (ring + 1) + j
            indices.extend([a, b, c, a, c, d])
    end = 4 * (len(stations) - 1)
    indices.extend([end, end + 1, end + 2, end, end + 2, end + 3])
    return mesh_body(cid, vertices, indices, [0.71, 0.75, 0.78, 1])


def place_gripper(cid, center, handle_width):
    scale, _ = b601.choose_visual_scale()
    robot = b601.load_robot(cid, scale)
    limits = [float(p.getJointInfo(robot, j, physicsClientId=cid)[9]) for j in (7, 8)]

    def opening(fraction):
        for joint, limit in zip((7, 8), limits):
            p.resetJointState(robot, joint, fraction * limit, physicsClientId=cid)

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
    fraction = (handle_width - min_gap) / (max_gap - min_gap)
    if not 0 <= fraction <= 1:
        raise RuntimeError("Tool handle exceeds the B601 jaw range")
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
        abs(gap - handle_width) < 0.0001
        and abs(front["inner_y"] - center[1] + handle_width / 2) < 0.0001
        and abs(back["inner_y"] - center[1] - handle_width / 2) < 0.0001
        and all(abs(t["center_x"] - center[0]) < 0.0001 for t in (front, back))
    ):
        raise RuntimeError("B601 fingertips do not meet both handle sides")
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
    entry = json.loads(provenance_path.read_text())["states"]["PRY"]
    state = entry["frame"]["state"]
    if entry["trace_step"] != 576 or state["crack"] != 0:
        raise RuntimeError("Expected the intact frozen PRY frame at step 576")
    gap = float(state["gap_m"])
    insertion = float(state["insertion_depth_m"])
    # theta is an unbounded accumulated rotation in the reduced-order model.
    # Modulo 2*pi preserves its exact physical orientation.
    theta = math.remainder(float(state["theta_rad"]), 2 * math.pi)
    half_x, half_y, wall_top = 0.054, 0.038, 0.020
    tool_y = -0.017
    thickness = 0.0045
    if not (0 < insertion < 0.010 and 0 < gap < 2 * half_x):
        raise RuntimeError("Archived seam state is outside this illustrative tool geometry")

    # The scalar gap is shown at the lifted lid edge.  The opposite edge
    # remains in contact with the housing; this tilt is derived geometry,
    # not an extra angle read from the benchmark.
    lid_angle = math.asin(gap / (2 * half_x))
    lid_q = p.getQuaternionFromEuler([0, -lid_angle, 0])
    lid_rotation = np.asarray(p.getMatrixFromQuaternion(lid_q)).reshape(3, 3)
    pivot = np.array([-half_x, 0, wall_top])

    def lid_point(local):
        return pivot + lid_rotation @ (np.asarray(local) + [half_x, 0, 0])

    def underside_z(x):
        return wall_top + (x + half_x) * math.tan(lid_angle)

    output_dir.mkdir(parents=True, exist_ok=True)
    cid = p.connect(p.DIRECT)
    try:
        b601.make_box(cid, [0.068, 0.051, 0.005], [0, 0, -0.005], [0.17, 0.18, 0.20, 1])
        hollow_housing(cid, half_x, half_y, wall_top)
        # Empty fastening bosses and a recessed cavity identify the lower shell.
        for x in (-0.044, 0.044):
            for y in (-0.028, 0.028):
                b601.make_cyl(cid, 0.003, 0.014, [x, y, 0.009], [0.22, 0.25, 0.28, 1])
                b601.make_cyl(cid, 0.0012, 0.0002, [x, y, 0.0161], [0.025, 0.03, 0.04, 1])

        lid_center = lid_point([0, 0, thickness / 2])
        lid = rounded_solid(
            cid,
            half_x,
            half_y,
            thickness,
            0.005,
            0.0008,
            lid_center.tolist(),
            [0.43, 0.49, 0.54, 1],
        )
        p.resetBasePositionAndOrientation(lid, lid_center, lid_q, physicsClientId=cid)
        # Shallow dark vent inlays and empty fastener recesses on the same lid.
        for x in np.linspace(-0.032, -0.008, 7):
            pos = lid_point([float(x), 0, thickness + 0.00005])
            vent = rounded_solid(
                cid, 0.00065, 0.013, 0.0001, 0.0005, 0.00002, pos.tolist(), [0.035, 0.045, 0.055, 1]
            )
            p.resetBasePositionAndOrientation(vent, pos, lid_q, physicsClientId=cid)
        for x in (-0.044, 0.044):
            for y in (-0.028, 0.028):
                pos = lid_point([x, y, thickness + 0.00007])
                hole = b601.make_cyl(cid, 0.0017, 0.00014, pos.tolist(), [0.06, 0.07, 0.08, 1])
                p.resetBasePositionAndOrientation(hole, pos, lid_q, physicsClientId=cid)

        # The blade toe contacts the lid underside at the exact insertion;
        # the bend's heel rests on the outer housing rim.
        tip_x = half_x - insertion
        toe_thickness, heel_thickness = 0.0006, 0.0014
        toe_contact = np.array([tip_x, tool_y, underside_z(tip_x)])
        heel = np.array([half_x, tool_y, wall_top + heel_thickness / 2])
        axis = np.array([math.cos(theta), 0, math.sin(theta)])
        shoulder = heel + 0.010 * axis
        tang_end = heel + 0.047 * axis
        stations = [
            [tip_x, toe_contact[2] - toe_thickness / 2, 0.008, toe_thickness],
            [tip_x + 0.0012, toe_contact[2] - 0.0009, 0.009, 0.0008],
            [heel[0], heel[2], 0.010, heel_thickness],
            [shoulder[0], shoulder[2], 0.009, 0.0014],
            [tang_end[0], tang_end[2], 0.008, 0.0014],
        ]
        # Check the entire inserted ribbon against the housing and lid planes.
        for first, second in zip(stations, stations[1:]):
            for t in np.linspace(0, 1, 101):
                x, z, _, height = (1 - t) * np.asarray(first) + t * np.asarray(second)
                if x <= half_x:
                    if z - height / 2 < wall_top - 1e-9 or z + height / 2 > underside_z(x) + 1e-9:
                        raise RuntimeError("Inserted blade intersects the housing or lid")
        blade_mesh(cid, stations, tool_y)

        handle_center = heel + 0.053 * axis
        handle_width = 0.018
        tool_q = p.getQuaternionFromEuler([0, -theta, 0])
        tool_rotation = np.asarray(p.getMatrixFromQuaternion(tool_q)).reshape(3, 3)
        handle = rounded_solid(
            cid,
            0.019,
            handle_width / 2,
            0.008,
            0.004,
            0.0007,
            handle_center.tolist(),
            [0.77, 0.29, 0.075, 1],
        )
        p.resetBasePositionAndOrientation(handle, handle_center, tool_q, physicsClientId=cid)
        for x in (-0.008, -0.004, 0, 0.004, 0.008):
            pos = handle_center + tool_rotation @ [x, 0, 0.00405]
            b601.make_box(
                cid,
                [0.0004, 0.007, 0.0001],
                pos.tolist(),
                [0.29, 0.13, 0.06, 1],
                orn=(0, -theta, 0),
            )
        grasp = place_gripper(cid, handle_center - [0, 0, 0.0025], handle_width)

        # Geometric invariants use the same coordinates as the rendered mesh.
        edge = lid_point([half_x, 0, 0])
        if not (
            abs(edge[2] - wall_top - gap) < 1e-9
            and abs(half_x - tip_x - insertion) < 1e-9
            and abs(stations[2][1] - heel_thickness / 2 - wall_top) < 1e-9
        ):
            raise RuntimeError("PRY contacts no longer represent the frozen state")

        width, height = 1800, 1400
        camera = {
            "target": [0.030, -0.004, 0.048],
            "distance": 0.330,
            "yaw": 35,
            "pitch": -28,
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
        from workspace_render_utils import add_coordinate_reference, coordinate_metadata

        Image.fromarray(image).save(output_dir / "pry_workspace_clean.png", dpi=(300, 300))
        add_coordinate_reference(
            image, depth, view, projection, -0.0101, [-0.09, 0.16, -0.08, 0.095]
        ).save(output_dir / "pry_workspace.png", dpi=(300, 300))
        manifest = {
            "task": "PRY",
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
            "interpretation": "B601-held bent pry blade lifting a vented enclosure lid at its seam",
            "visualization": {
                "gap_m": gap,
                "gap_reference": "vertical separation at raised lid edge",
                "insertion_depth_m": insertion,
                "insertion_reference": "housing outer edge to blade tip along X",
                "tool_axis_angle_rad": theta,
                "lid_tilt_rad": lid_angle,
                "lid_tilt_source": "illustrative tilt derived from recorded gap and enclosure width",
                "toe_contact_m": toe_contact.tolist(),
                "heel_contact_m": [float(heel[0]), tool_y, wall_top],
                "gripper": grasp,
                "camera": camera,
                **coordinate_metadata(),
                "arm_kinematics_simulated": False,
                "geometry_is_illustrative": True,
            },
        }
        (output_dir / "pry_workspace_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print("PRY gap mm:", gap * 1000)
        print("Insertion mm:", insertion * 1000)
        print("Tool angle degrees (modulo 360):", math.degrees(theta))
        print("Handle grasp gap mm:", grasp["gap_m"] * 1000)
        print("WROTE:", output_dir / "pry_workspace.png")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "PRY", debug=False)
