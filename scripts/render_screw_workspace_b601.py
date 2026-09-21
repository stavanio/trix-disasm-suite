#!/usr/bin/env python3
"""Render the archived SCREW frame with a threaded fastener and B601 grasp."""
import math
from pathlib import Path

import numpy as np
import pybullet as p
from b601_render_common import make_box, make_cyl
from workspace_environment import prepare_environment, body_geometry
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_wrench import force, metadata, torque


def wooden_fixture(cid, top_z):
    """Fixed wood blank with deterministic procedural grain and a flush insert."""
    from PIL import Image
    import tempfile

    size = 512
    yy, xx = np.mgrid[0:1:complex(size), 0:1:complex(size)]
    rng = np.random.default_rng(1701)
    atlas = np.empty((2 * size, 3 * size, 3), dtype=np.uint8)
    for face in range(6):
        if face in (2, 3):  # End grain across the cut ends.
            coordinate = np.sqrt((yy + 0.6)**2 + (0.6 * xx + 0.2)**2)
        else:
            coordinate = yy + 0.035*np.sin(5*xx + face) + 0.011*np.sin(13*xx+2*yy)
        growth = np.sin(2*np.pi*(7*coordinate + 0.10*np.sin(9*coordinate)))
        fibers = np.maximum(0, np.cos(2*np.pi*77*coordinate))**12
        variation = 7*growth - 16*fibers + rng.normal(0, 1.1, xx.shape)
        base = np.array([195, 151, 98]) * (0.91 if face in (2, 3) else 1)
        tile = np.clip(base + variation[..., None]*[1, .85, .6], 0, 255).astype(np.uint8)
        row, col = divmod(face, 3)
        atlas[row*size:(row+1)*size, col*size:(col+1)*size] = tile

    hx, hy, hz = .060, .042, (top_z + .016)/2
    # Four vertices per face retain flat shading and independent grain UVs.
    faces = [
        ([[-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]], [0,0,1]),
        ([[-hx,hy,-hz],[hx,hy,-hz],[hx,-hy,-hz],[-hx,-hy,-hz]], [0,0,-1]),
        ([[hx,-hy,-hz],[hx,hy,-hz],[hx,hy,hz],[hx,-hy,hz]], [1,0,0]),
        ([[-hx,hy,-hz],[-hx,-hy,-hz],[-hx,-hy,hz],[-hx,hy,hz]], [-1,0,0]),
        ([[-hx,-hy,-hz],[hx,-hy,-hz],[hx,-hy,hz],[-hx,-hy,hz]], [0,-1,0]),
        ([[hx,hy,-hz],[-hx,hy,-hz],[-hx,hy,hz],[hx,hy,hz]], [0,1,0]),
    ]
    vertices, indices, normals, uvs = [], [], [], []
    for face, (corners, normal) in enumerate(faces):
        first = len(vertices); vertices.extend(corners); normals.extend([normal]*4)
        indices.extend([first, first+1, first+2, first, first+2, first+3])
        row, col = divmod(face, 3)
        u0, u1 = (col+.003)/3, (col+.997)/3
        v0, v1 = 1-(row+.997)/2, 1-(row+.003)/2
        uvs.extend([[u0,v0],[u1,v0],[u1,v1],[u0,v1]])
    visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices,
        normals=normals, uvs=uvs, rgbaColor=[1,1,1,1], specularColor=[.05]*3, physicsClientId=cid)
    body = p.createMultiBody(0,-1,visual,[0,0,(top_z-.016)/2],physicsClientId=cid)
    with tempfile.TemporaryDirectory(prefix='trix-wood-') as directory:
        texture_path = Path(directory)/'grain.png'
        Image.fromarray(atlas).save(texture_path)
        texture = p.loadTexture(str(texture_path),physicsClientId=cid)
    p.changeVisualShape(body,-1,textureUniqueId=texture,physicsClientId=cid)
    return body


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
        fixture = wooden_fixture(cid, fixture_top)
        # A flush M8 threaded insert preserves the modeled 1.25 mm thread.
        annulus(cid, crest_radius+.0003, .0065, .0006, [0,0,fixture_top+.0003], [.65,.51,.26,1])
        annulus(cid, 0, crest_radius+.0003, .00005, [0,0,fixture_top+.00006], [.11,.09,.065,1])

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
                fixture_material="wood blank with flush M8 threaded insert",
                fixture_dimensions_m=[.120,.084,.030],
                wood_grain_source="deterministic procedural texture in renderer",
                thread_pitch_m=pitch,
                thread_core_geometry="explicit radius mesh",
                head_diameter_m=0.016,
                contact_radial_error_m=[r - 0.008 for r in contact_radii],
                thread_root_radius_m=root_radius,
                thread_crest_radius_m=crest_radius,
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
