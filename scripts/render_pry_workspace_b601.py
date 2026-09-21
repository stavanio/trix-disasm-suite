#!/usr/bin/env python3
"""Visualize the frozen PRY state with an enclosure and a B601-held lever."""

import json
import math
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from PIL import Image
from workspace_environment import prepare_environment, body_geometry
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_render_utils import grasp_b601, rounded_outline, rounded_solid
from workspace_wrench import annotate, metadata, torque


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


def blade_mesh(cid, stations, origin, quaternion):
    """One rigid local-space steel mesh; state changes only its body transform."""
    vertices = []
    for x, z, width, thickness in stations:
        vertices.extend(
            [
                [x, -width / 2, z - thickness / 2],
                [x, width / 2, z - thickness / 2],
                [x, width / 2, z + thickness / 2],
                [x, -width / 2, z + thickness / 2],
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
    body = mesh_body(cid, vertices, indices, [0.71, 0.75, 0.78, 1])
    p.resetBasePositionAndOrientation(body, origin, quaternion, physicsClientId=cid)
    return body, vertices


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, environment=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    env, model, entry, binding = prepare_environment("PRY", provenance_path, environment)
    state = entry["frame"]["state"]
    gap, insertion = float(env.state.position[2]), float(env.state.position[0])
    theta = math.remainder(float(env.state.theta), 2 * math.pi)
    half_x, half_y, wall_top = 0.054, 0.038, 0.020
    tool_y, thickness = -0.017, 0.0045
    # The recorded gap is the free-edge opening; the opposite edge stays
    # supported on the rim. This CAD support constraint derives the lid angle
    # from gap and fixed width, without adding an independent state variable.
    if not 0 <= gap < 2*half_x:
        raise ValueError("PRY gap exceeds the fixed lid span")
    lid_angle = math.asin(gap/(2*half_x))
    lid_q = p.getQuaternionFromEuler([0,-lid_angle,0])
    lid_rotation = np.asarray(p.getMatrixFromQuaternion(lid_q)).reshape(3,3)
    support = np.array([-half_x,0,wall_top])
    def lid_point(local):
        return support + lid_rotation @ (np.asarray(local)+[half_x,0,0])

    output_dir.mkdir(parents=True, exist_ok=True)
    cid = p.connect(p.DIRECT)
    try:
        fixture = hollow_housing(cid, half_x, half_y, wall_top)
        # Two retaining ledges at the supported end; no four-corner bolt pattern.
        for y in (-.024,.024):
            b601.make_box(cid,[.003,.005,.001],[-half_x+.003,y,wall_top-.001],[.24,.27,.30,1])

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
        # State-to-CAD convention: toe X follows insertion, toe Z follows
        # the underside of the supported lid at that X coordinate;
        # positive model angle rotates the shank upward about world -Y.
        # The model has no rigid-contact/fulcrum constraint. Never reshape the
        # blade or move the housing to create contact absent from these states.
        toe_x = half_x-insertion
        toe_z = wall_top + (toe_x+half_x)*math.tan(lid_angle)
        toe_origin = np.array([toe_x,tool_y,toe_z])
        tool_q = p.getQuaternionFromEuler([0, -theta, 0])
        tool_rotation = np.asarray(p.getMatrixFromQuaternion(tool_q)).reshape(3, 3)
        stations = [
            [0, -0.0003, 0.008, 0.0006],
            [0.006, -0.003, 0.010, 0.0014],
            [0.014, -0.009, 0.010, 0.0014],
            [model.LEVER_LENGTH - 0.027, -0.009, 0.008, 0.0014],
        ]
        blade, blade_local_vertices = blade_mesh(cid, stations, toe_origin, tool_q)
        handle_center = toe_origin + tool_rotation @ [model.LEVER_LENGTH - 0.019, 0, -0.009]
        handle_width = 0.018
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
        # Close across the handle, with the finger approach normal to its
        # upper face. The tool angle remains the archived rotation; only the
        # gripper pose changes to follow it. Use actual STL surface contacts.
        robot, grasp = grasp_b601(
            cid,
            handle_center,
            handle_width,
            closing=tool_rotation[:, 1],
            approach=tool_rotation[:, 2],
            tip_depth=0.002,
        )
        handle_contacts = (np.asarray(grasp["contacts_m"]) - handle_center) @ tool_rotation
        handle_side_errors = np.abs(np.abs(handle_contacts[:, 1]) - handle_width / 2)
        if not (
            np.max(handle_side_errors) < 0.00005
            and np.max(np.abs(handle_contacts[:, 0])) < 0.015
            and np.max(np.abs(handle_contacts[:, 2])) < 0.0033
            and np.dot(grasp["approach_axis"], tool_rotation[:, 2]) > 1 - 1e-9
        ):
            raise RuntimeError("PRY fingertips do not contact the handle's flat side faces")

        width, height = 1800, 1400
        camera = {
            "target": [0.065, -0.004, 0.058],
            "distance": 0.490,
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

        wrenches = [
            torque(
                toe_origin,
                [0, -1, 0],
                radius=0.026,
                start_deg=-45,
                sweep_deg=210,
                offset=[0.040, -0.019, 0.025],
                label="τ_pry",
                label_offset=(-70, -40),
            )
        ]
        Image.fromarray(image).save(output_dir / "pry_workspace_clean.png", dpi=(300, 300))
        annotated = add_coordinate_reference(
            image, depth, view, projection, -0.0001, [-0.09, 0.24, -0.10, 0.12]
        )
        notes = [dict(
            anchor_world_m=(handle_center + tool_rotation @ [0.017, 0, 0.004]).tolist(),
            position_px=[1470, 1140],
            text="Tool grip",
        )]
        annotate(annotated, view, projection, wrenches, notes).save(
            output_dir / "pry_workspace.png", dpi=(300, 300)
        )
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
            "observation": entry["frame"]["obs"],
            "interpretation": "B601-held bent pry blade lifting a vented enclosure lid at its seam",
            "visualization": {
                "environment": binding,
                "rendered_bodies": body_geometry(cid, fixture=fixture, lid=lid, blade=blade, handle=handle, gripper=robot),
                "gap_m": gap,
                "gap_reference": "free-edge vertical opening: env.state.position[2]",
                "lid_support_edge_m": support.tolist(),
                "lid_free_edge_m": lid_point([half_x,0,0]).tolist(),
                "lid_angle_rad": lid_angle,
                "lid_angle_source": "asin(env.state.position[2] / fixed lid span); opposite edge supported",
                "lid_span_m": 2*half_x,
                "fixture_material": "retained-edge polymer enclosure without mounting plinth",
                "insertion_depth_m": insertion,
                "insertion_reference": "housing outer edge to tool origin along X: env.state.position[0]; toe height follows lid underside",
                "tool_axis_angle_rad": theta,
                "tool_rotation_source": "Ry(-env.state.theta)",
                "toe_origin_m": toe_origin.tolist(),
                "blade_local_vertices_m": blade_local_vertices,
                "lever_length_m": model.LEVER_LENGTH,
                "contact_constraints_in_environment": False,
                "grasp": grasp,
                "handle_center_m": handle_center.tolist(),
                "handle_dimensions_m": [0.038, handle_width, 0.008],
                "handle_rotation_world": tool_rotation.tolist(),
                "handle_contacts_local_m": handle_contacts.tolist(),
                "handle_side_contact_error_m": handle_side_errors.tolist(),
                "handle_material": "orange polymer grip on the continuous steel pry tool",
                "gripper_alignment": "closing across handle width; approach normal to handle top",
                "geometry_annotations": notes,
                "camera": camera,
                **coordinate_metadata(),
                **metadata(wrenches),
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
        print("Handle grasp gap mm:", grasp["actual_gap_m"] * 1000)
        print("WROTE:", output_dir / "pry_workspace.png")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "PRY", debug=False)
