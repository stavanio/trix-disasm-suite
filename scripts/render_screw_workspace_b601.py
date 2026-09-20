#!/usr/bin/env python3
"""Render the archived SCREW frame with a threaded fastener and B601 grasp."""
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
from b601_render_common import make_box, make_cyl
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, debug=False):
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

    entry = json.loads(provenance_path.read_text())["states"]["SCREW"]
    if entry["trace_step"] != 188:
        raise ValueError("Expected SCREW trace step 188")
    state = entry["frame"]["state"]
    theta, lift = float(state["theta_rad"]), float(state["z_m"])
    # The archived helical state is unchanged. Dimensions below describe
    # the illustrative fixture; pitch matches screw_env_v3.PITCH.
    pitch = 0.00125
    fixture_top = 0.014
    head_z = fixture_top + lift + 0.003
    cid = p.connect(p.DIRECT)
    try:
        rounded_box(cid, [0.060, 0.048, 0.008], [0, 0, -0.008], [0.22, 0.24, 0.27, 1], radius=0.007)
        for x in (-0.047, 0.047):
            for y in (-0.035, 0.035):
                annulus(cid, 0.0022, 0.0041, 0.0006, [x, y, 0.0003], [0.51, 0.54, 0.58, 1])
                make_cyl(cid, 0.0022, 0.0003, [x, y, 0.00012], [0.065, 0.075, 0.09, 1])
        annulus(cid, 0.0057, 0.017, 0.012, [0, 0, 0.006], [0.095, 0.11, 0.135, 1])
        annulus(cid, 0.0055, 0.010, 0.002, [0, 0, 0.013], [0.70, 0.58, 0.29, 1])

        # A continuous 1.25 mm helical ridge around the screw's root.
        bottom, top = head_z - 0.029, head_z - 0.003
        make_cyl(cid, 0.00425, top - bottom, [0, 0, (top + bottom) / 2], [0.50, 0.54, 0.60, 1])
        vertices, indices = [], []
        count = int((top - bottom) / pitch * 72)
        for angle in np.linspace(0, (top - bottom) / pitch * 2 * math.pi, count):
            a = angle + theta
            z = bottom + angle * pitch / (2 * math.pi)
            for radius, dz in ((0.00423, -pitch * 0.30), (0.0052, 0), (0.00423, pitch * 0.30)):
                vertices.append([radius * math.cos(a), radius * math.sin(a), z + dz])
        for i in range(count - 1):
            for j in range(2):
                a = 3 * i + j
                b = a + 3
                indices.extend([a, b, b + 1, a, b + 1, a + 1])
        mesh_body(cid, vertices, indices, [0.73, 0.77, 0.82, 1], flat=True)

        # Recessed slot: two metal half-discs above a lower screw-head body.
        make_cyl(cid, 0.012, 0.005, [0, 0, head_z - 0.0005], [0.61, 0.65, 0.71, 1])
        make_box(
            cid,
            [0.0118, 0.0012, 0.00004],
            [0, 0, head_z + 0.00204],
            [0.13, 0.16, 0.20, 1],
            orn=[0, 0, theta],
        )
        radius, slot_half = 0.0119, 0.0012
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
        robot, grasp = grasp_b601(cid, [0, 0, head_z], 0.024, tip_depth=0.002, debug=debug)
        camera = dict(target=[0, 0, 0.048], distance=0.350, yaw=42, pitch=-29)
        suffix = "_debug" if debug else ""
        output = output_dir / f"screw_workspace{suffix}.png"
        camera_image(
            cid,
            camera,
            -0.0161,
            [-0.092, 0.092, -0.080, 0.080],
            output,
            output_dir / "screw_workspace_clean.png" if not debug else None,
        )
        if not debug:
            write_manifest(
                output_dir / "screw_workspace_manifest.json",
                "SCREW",
                entry,
                camera=camera,
                grasp=grasp,
                thread_pitch_m=pitch,
                thread_root_radius_m=0.00425,
                thread_crest_radius_m=0.0052,
                head_bottom_above_insert_m=lift,
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
