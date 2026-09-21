#!/usr/bin/env python3
"""Render the archived SCREW frame with a threaded fastener and B601 grasp."""
import math
from pathlib import Path

import numpy as np
import pybullet as p
from b601_render_common import make_box
from workspace_environment import prepare_environment, body_geometry
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_wrench import force, metadata, torque


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, environment=None, debug=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    from workspace_render_utils import (
        annulus,
        camera_image,
        grasp_b601,
        mesh_body,
        rounded_box,
        write_manifest,
    )

    env, model, entry, binding = prepare_environment("SCREW", provenance_path, environment)
    theta, lift = float(env.theta), float(env.z)
    pitch = model.PITCH
    crest_radius, root_radius = model.RADIUS, 0.85 * model.RADIUS
    fixture_top = 0.014
    head_z = fixture_top + lift + 0.003
    cid = p.connect(p.DIRECT)
    try:
        # Fixed metal fixture for the modeled metric machine thread.
        fixture = rounded_box(
            cid, [.060, .042, .015], [0, 0, fixture_top - .015],
            [.46, .51, .57, 1], radius=.002, bevel=.0006,
        )
        # Machined bore mouth; the visible screw helix uses the model pitch.
        annulus(cid, crest_radius+.0003, .0065, .0006,
                [0, 0, fixture_top+.0003], [.68, .72, .77, 1])
        annulus(cid, 0, crest_radius+.0003, .00005,
                [0, 0, fixture_top+.00006], [.08, .10, .13, 1])

        # A continuous 1.25 mm helical ridge around the screw's root.
        bottom, top = head_z - 0.029, head_z - 0.003
        # Use an explicit mesh: the TinyRenderer cylinder primitive expands
        # its silhouette beyond the nominal radius and hides the thread flanks.
        annulus(cid, 0, root_radius, top - bottom, [0, 0, (top + bottom) / 2], [0.24, 0.28, 0.33, 1])
        vertices, indices = [], []
        count = int((top - bottom) / pitch * 72)
        for angle in np.linspace(0, (top - bottom) / pitch * 2 * math.pi, count):
            a = angle + theta
            z = bottom + angle * pitch / (2 * math.pi)
            for radius, dz in ((root_radius, -pitch * 0.30), (crest_radius, 0), (root_radius, pitch * 0.30)):
                vertices.append([radius * math.cos(a), radius * math.sin(a), z + dz])
        for i in range(count - 1):
            for j in range(2):
                a = 3 * i + j
                b = a + 3
                indices.extend([a, b, b + 1, a, b + 1, a + 1])
        mesh_body(cid, vertices, indices, [0.69, 0.73, 0.78, 1], flat=True)

        # Recessed slot: two metal half-discs above a lower screw-head body.
        head = annulus(cid, 0, 0.008, 0.005, [0, 0, head_z - 0.0005], [0.61, 0.65, 0.71, 1])
        make_box(
            cid,
            [0.0078, 0.0012, 0.00004],
            [0, 0, head_z + 0.00204],
            [0.13, 0.16, 0.20, 1],
            orn=[0, 0, theta],
        )
        radius, slot_half = 0.0079, 0.0012
        a0 = math.asin(slot_half / radius)
        for side in (-1, 1):
            arc = np.linspace(a0, math.pi - a0, 72)
            polygon = np.column_stack([radius * np.cos(arc), side * radius * np.sin(arc)])
            if side < 0:
                polygon = polygon[::-1]
            verts = [[x, y, z] for z in (0.002, 0.003) for x, y in polygon]
            n = len(polygon)
            faces = []
            for i in range(1, n - 1):
                faces.extend([0, i + 1, i, n, n + i, n + i + 1])
            for i in range(n):
                j = (i + 1) % n
                faces.extend([i, j, n + j, i, n + j, n + i])
            mesh_body(
                cid,
                verts,
                faces,
                [0.70, 0.74, 0.79, 1],
                [0, 0, head_z],
                p.getQuaternionFromEuler([0, 0, theta]),
                flat=True,
            )
        robot, grasp = grasp_b601(cid, [0, 0, head_z], 0.016, closing=[math.cos(theta), math.sin(theta), 0], tip_depth=0.002, debug=debug)
        contact_radii = [np.linalg.norm(np.asarray(pt)[:2]) for pt in grasp["contacts_m"]]
        if max(abs(r - 0.008) for r in contact_radii) > 0.0001:
            raise RuntimeError("SCREW fingertips do not contact the head surface")
        wrenches = [
            force([0, 0, head_z], [0, 0, 1], offset=[0.028, 0.028, 0.018], label="F_z", show_leader=False),
            torque(
                [0, 0, head_z],
                [0, 0, 1],
                radius=0.029,
                start_deg=140,
                sweep_deg=240,
                offset=[0, 0, 0.035],
                label="τ_z",
                show_leader=False,
                label_offset=(20, 8),
            ),
        ]
        notes = [
            dict(
                anchor_world_m=[crest_radius, 0, fixture_top + lift * 0.45],
                position_px=[1290, 900],
                text=f"p = {pitch*1000:.2f} mm",
            )
        ]
        camera = dict(target=[0, 0, 0.048], distance=0.350, yaw=42, pitch=-19)
        suffix = "_debug" if debug else ""
        output = output_dir / f"screw_workspace{suffix}.png"
        camera_image(
            cid,
            camera,
            -0.0161,
            [-0.092, 0.092, -0.080, 0.080],
            output,
            output_dir / "screw_workspace_clean.png" if not debug else None,
            wrenches=wrenches,
            notes=notes,
        )
        if not debug:
            write_manifest(
                output_dir / "screw_workspace_manifest.json",
                "SCREW",
                entry,
                environment=binding,
                rendered_bodies=body_geometry(cid, fixture=fixture, head=head, gripper=robot),
                camera=camera,
                grasp=grasp,
                **metadata(wrenches),
                fixture_material="machined metal block with M8 tapped bore",
                fixture_dimensions_m=[.120,.084,.030],
                thread_capacity_N=model.F_THREAD_CAPACITY,
                thread_pitch_m=pitch,
                thread_core_geometry="explicit radius mesh",
                head_diameter_m=0.016,
                contact_radial_error_m=[r - 0.008 for r in contact_radii],
                thread_root_radius_m=root_radius,
                thread_crest_radius_m=crest_radius,
                head_bottom_above_fixture_m=lift,
                fixture_top_m=fixture_top,
                theta_rad=theta,
                lift_m=lift,
            )
        print(
            f'SCREW: lift={lift*1000:.6f} mm; fingertip gap={grasp["actual_gap_m"]*1000:.4f} mm; {output}'
        )
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "SCREW", debug=True)
