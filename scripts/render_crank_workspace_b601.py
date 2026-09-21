#!/usr/bin/env python3
"""Frozen CRANK trace with a bearing fixture, rotary handle and real B601."""
import math
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from workspace_environment import prepare_environment, body_geometry
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_render_utils import (
    annulus,
    camera_image,
    grasp_b601,
    rounded_box,
    write_manifest,
)
from workspace_wrench import force, metadata, torque


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, environment=None, debug=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    env, model, entry, binding = prepare_environment("CRANK", provenance_path, environment)
    theta_raw = float(env.theta)
    theta = theta_raw % (2 * math.pi)
    z = float(env.z)
    radius = model.RADIUS
    radial = np.array([math.cos(theta), math.sin(theta), 0])
    tangent = np.array([-math.sin(theta), math.cos(theta), 0])
    handle_xy = radius * radial
    hub_z = 0.031 + z
    q = p.getQuaternionFromEuler([0, 0, theta])
    cid = p.connect(p.DIRECT)
    try:
        fixture = rounded_box(cid, [0.061, 0.052, 0.008], [0, 0, -0.008], [0.22, 0.24, 0.27, 1], radius=0.007)
        for x in (-0.047, 0.047):
            for y in (-0.038, 0.038):
                annulus(cid, 0.002, 0.004, 0.0005, [x, y, 0.00025], [0.53, 0.56, 0.60, 1])
                b601.make_cyl(cid, 0.002, 0.00025, [x, y, 0.0001], [0.06, 0.07, 0.085, 1])
        rounded_box(cid, [0.027, 0.023, 0.005], [0, 0, 0.005], [0.11, 0.13, 0.16, 1], radius=0.006)
        annulus(cid, 0.0095, 0.018, 0.014, [0, 0, 0.017], [0.15, 0.175, 0.21, 1])
        annulus(cid, 0.009, 0.0155, 0.0015, [0, 0, 0.0245], [0.55, 0.59, 0.64, 1])
        b601.make_cyl(cid, 0.0087, 0.028, [0, 0, 0.014], [0.63, 0.67, 0.72, 1])
        # The complete arm assembly follows the archived axial displacement.
        annulus(cid, 0.0087, 0.016, 0.006, [0, 0, hub_z], [0.49, 0.53, 0.59, 1])
        arm_center = radius * 0.5 * radial + np.array([0, 0, hub_z + 0.002])
        rounded_box(
            cid,
            [radius * 0.5 + 0.008, 0.006, 0.003],
            arm_center,
            [0.80, 0.61, 0.16, 1],
            radius=0.0058,
            bevel=0.0006,
            quaternion=q,
        )
        # Flat top fastening cap with a real hexagonal socket recess.
        from workspace_render_utils import mesh_body

        vertices = []
        indices = []
        n = 96
        for ring in range(4):
            for i in range(n):
                a = i * 2 * math.pi / n
                # Intersect a ray with a regular hexagon's six flat sides.
                r = (
                    0.007
                    if ring < 2
                    else 0.0025 / math.cos((a + math.pi / 6) % (math.pi / 3) - math.pi / 6)
                )
                height = hub_z + (0.005 if ring % 2 == 0 else 0.008)
                vertices.append([r * math.cos(a + theta), r * math.sin(a + theta), height])
        for i in range(n):
            j = (i + 1) % n
            for a, b, c, d in (
                (i, j, n + j, n + i),
                (n + i, n + j, 3 * n + j, 3 * n + i),
                (2 * n + j, 2 * n + i, 3 * n + i, 3 * n + j),
            ):
                indices.extend([a, b, c, a, c, d])
        mesh_body(cid, vertices, indices, [0.63, 0.67, 0.72, 1], flat=True)
        b601.make_cyl(cid, 0.003, 0.0002, [0, 0, hub_z + 0.0051], [0.055, 0.075, 0.095, 1])
        stem = handle_xy + np.array([0, 0, hub_z + 0.007])
        b601.make_cyl(cid, 0.0032, 0.012, stem.tolist(), [0.66, 0.70, 0.75, 1])
        grip_z = hub_z + 0.0205
        grip = handle_xy + np.array([0, 0, grip_z])
        handle = annulus(cid, 0, 0.007, 0.027, grip.tolist(), [0.86, 0.87, 0.84, 1])
        for zz in (grip_z - 0.0125, grip_z + 0.0125):
            annulus(cid, 0.0032, 0.00715, 0.001, [*handle_xy[:2], zz], [0.61, 0.65, 0.67, 1])
        b601.make_cyl(cid, 0.0028, 0.0005, [*handle_xy[:2], grip_z + 0.0137], [0.45, 0.49, 0.53, 1])
        target = grip
        robot, grasp = grasp_b601(cid, target, 0.014, closing=tangent, tip_depth=0.005, debug=debug)
        contact_radii = [np.linalg.norm((np.asarray(pt) - grip)[:2]) for pt in grasp["contacts_m"]]
        if max(abs(r - 0.007) for r in contact_radii) > 0.0001:
            raise RuntimeError("CRANK fingertips do not contact the handle surface")
        wrenches = [
            force(grip, -tangent, offset=[0.021, 0.019, 0.018], label="F_t", label_offset=(16, 2)),
            torque(
                [0, 0, hub_z],
                [0, 0, -1],
                radius=0.030,
                start_deg=15,
                sweep_deg=235,
                offset=[0, 0, 0.013],
                label="τ_z",
                label_offset=(-50, 10),
            ),
        ]
        camera = dict(
            target=[float(handle_xy[0] * 0.45), float(handle_xy[1] * 0.45), 0.059],
            distance=0.395,
            yaw=42,
            pitch=-28,
        )
        suffix = "_debug" if debug else ""
        output = output_dir / f"crank_workspace{suffix}.png"
        camera_image(
            cid,
            camera,
            -0.0161,
            [-0.090, 0.125, -0.080, 0.155],
            output,
            output_dir / "crank_workspace_clean.png" if not debug else None,
            wrenches=wrenches,
        )
        if not debug:
            write_manifest(
                output_dir / "crank_workspace_manifest.json",
                "CRANK",
                entry,
                environment=binding,
                rendered_bodies=body_geometry(cid, fixture=fixture, handle=handle, gripper=robot),
                camera=camera,
                grasp=grasp,
                **metadata(wrenches),
                handle_geometry="explicit radius mesh",
                theta_raw_rad=theta_raw,
                theta_display_rad=theta,
                z_m=z,
                crank_radius_m=radius,
                handle_center_m=grip.tolist(),
                handle_diameter_m=0.014,
                contact_radial_error_m=[r - 0.007 for r in contact_radii],
            )
        print(
            f'CRANK: theta={math.degrees(theta):.6f} deg; axial z={z*1000:.6f} mm; fingertip gap={grasp["actual_gap_m"]*1000:.4f} mm; {output}'
        )
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "CRANK", debug=True)
