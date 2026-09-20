#!/usr/bin/env python3
"""Frozen PCB trace rendered as a partially extracted DIMM and real B601."""
import json
import math
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_render_utils import (
    annulus,
    camera_image,
    grasp_b601,
    rounded_box,
    write_manifest,
)
from workspace_wrench import force, metadata

L, T, H, Z0 = 0.0665, 0.001, 0.0155, 0.003


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, debug=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    entry = json.loads(provenance_path.read_text())["states"]["PCB"]
    if entry["trace_step"] != 133:
        raise ValueError("Expected PCB trace step 133")
    # pcb_env_v2._obs: lift, velocity, tilt_x, tilt_y, ... . The historical
    # state's named lift/tilt fields are null; preserve them and use obs.
    obs = entry["frame"]["obs"]
    lift, tx, ty = float(obs[0]), float(obs[2]), float(obs[3])
    quat = p.getQuaternionFromEuler([tx, ty, 0])
    R = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    center = np.array([0, 0, Z0 + H + lift])
    cid = p.connect(p.DIRECT)
    try:

        def board_box(half, local, color):
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid
            )
            return p.createMultiBody(
                0, -1, visual, (center + R @ np.asarray(local)).tolist(), quat, physicsClientId=cid
            )

        rounded_box(cid, [0.094, 0.059, 0.004], [0, 0, -0.004], [0.22, 0.24, 0.27, 1], radius=0.007)
        rounded_box(
            cid,
            [0.087, 0.053, 0.0015],
            [0, 0, 0.0015],
            [0.055, 0.21, 0.145, 1],
            radius=0.004,
            bevel=0.0003,
        )
        for x in (-0.079, 0.079):
            for y in (-0.045, 0.045):
                annulus(cid, 0.0018, 0.0036, 0.0003, [x, y, 0.00315], [0.66, 0.59, 0.31, 1])
                b601.make_cyl(cid, 0.0018, 0.0001, [x, y, 0.00315], [0.055, 0.075, 0.075, 1])
        # Routed copper paths, solder pads, and restrained support parts.
        for sign in (-1, 1):
            for x in np.linspace(-0.058, 0.058, 14):
                length = 0.012 + 0.002 * (int(abs(x) * 1000) % 4)
                b601.make_box(
                    cid,
                    [0.0002, length / 2, 0.000035],
                    [float(x), sign * (0.007 + length / 2), 0.00305],
                    [0.15, 0.37, 0.24, 1],
                )
                b601.make_box(
                    cid,
                    [0.001, 0.0014, 0.0001],
                    [float(x), sign * (0.008 + length), 0.00315],
                    [0.60, 0.57, 0.35, 1],
                )
            for x in (-0.064, -0.045, 0.045, 0.064):
                rounded_box(
                    cid,
                    [0.003, 0.0016, 0.0009],
                    [x, sign * 0.035, 0.0039],
                    [0.12, 0.14, 0.16, 1],
                    radius=0.0005,
                    bevel=0.0001,
                )
                for dx in (-0.0031, 0.0031):
                    b601.make_box(
                        cid,
                        [0.0005, 0.00165, 0.00085],
                        [x + dx, sign * 0.035, 0.00385],
                        [0.58, 0.60, 0.60, 1],
                    )
        # Open socket channel: separate rails leave the board slot visible.
        rounded_box(
            cid,
            [L + 0.004, 0.0048, 0.0008],
            [0, 0, Z0 + 0.0008],
            [0.13, 0.15, 0.18, 1],
            radius=0.0015,
            bevel=0.0002,
        )
        for y in (-0.0024, 0.0024):
            rounded_box(
                cid,
                [L + 0.004, 0.001, 0.0041],
                [0, y, Z0 + 0.0057],
                [0.18, 0.20, 0.23, 1],
                radius=0.0007,
                bevel=0.00015,
            )
            for x in np.linspace(-0.063, 0.063, 32):
                b601.make_box(
                    cid,
                    [0.00035, 0.0015, 0.00013],
                    [float(x), np.sign(y) * 0.0045, Z0 + 0.0003],
                    [0.63, 0.65, 0.62, 1],
                )
        # Retainers pivot outward at both ends; the board has been released.
        for sign in (-1, 1):
            pivot = np.array([sign * (L + 0.004), 0, Z0 + 0.0025])
            q = p.getQuaternionFromEuler([0, sign * math.radians(30), 0])
            Q = np.asarray(p.getMatrixFromQuaternion(q)).reshape(3, 3)
            rounded_box(
                cid,
                [0.0024, 0.0043, 0.006],
                pivot + Q @ np.array([0, 0, 0.005]),
                [0.58, 0.61, 0.64, 1],
                radius=0.0012,
                bevel=0.0003,
                quaternion=q,
            )
            rounded_box(
                cid,
                [0.0035, 0.0043, 0.0013],
                pivot + Q @ np.array([sign * 0.001, 0, 0.011]),
                [0.68, 0.70, 0.72, 1],
                radius=0.001,
                bevel=0.0002,
                quaternion=q,
            )
        # Keyed FR-4 outline. The bottom 3.2 mm is split around the notch.
        green = [0.09, 0.46, 0.23, 1]
        notch = -0.008
        nh = 0.0012
        nd = 0.0032
        board_box([L, T, H - nd / 2], [0, 0, nd / 2], green)
        left = -L
        right = notch - nh
        board_box([(right - left) / 2, T, nd / 2], [(left + right) / 2, 0, -H + nd / 2], green)
        left = notch + nh
        right = L
        board_box([(right - left) / 2, T, nd / 2], [(left + right) / 2, 0, -H + nd / 2], green)
        # Separate gold contacts on both faces, with a gap at the key.
        for x in np.arange(-0.063, 0.0631, 0.002):
            if abs(x - notch) < 0.002:
                continue
            for face in (-1, 1):
                board_box(
                    [0.00062, 0.000065, 0.00135],
                    [float(x), face * (T + 0.000065), -H + 0.0016],
                    [0.83, 0.67, 0.22, 1],
                )
        for face in (-1, 1):
            for x in (-0.052, -0.033, -0.014, 0.014, 0.033, 0.052):
                board_box(
                    [0.0066, 0.00085, 0.0046],
                    [x, face * (T + 0.00085), 0.001],
                    [0.055, 0.066, 0.078, 1],
                )
                # Subtle moulded upper surface and pin-one mark.
                board_box(
                    [0.0058, 0.000055, 0.0038],
                    [x, face * (T + 0.00175), 0.001],
                    [0.10, 0.115, 0.13, 1],
                )
                board_box(
                    [0.00055, 0.00006, 0.00055],
                    [x - 0.0048, face * (T + 0.00182), 0.0037],
                    [0.27, 0.30, 0.31, 1],
                )
            for x in (-0.060, -0.041, -0.023, 0, 0.023, 0.041, 0.060):
                board_box(
                    [0.0010, 0.00022, 0.0006],
                    [x, face * (T + 0.00022), -0.0065],
                    [0.64, 0.61, 0.45, 1],
                )
        # The raised board still overlaps the connector mouth. The reference
        # seat is illustrative; extraction and tilt remain the recorded state.
        socket_top = Z0 + 0.0098
        edge = np.array([center + R @ [x, 0, -H] for x in (-L, L)])
        socket_overlap = socket_top - edge[:, 2]
        if np.min(socket_overlap) <= 0:
            raise RuntimeError("PCB bottom edge has cleared the connector")
        target = center + R @ np.array([0, 0, H - 0.002])
        robot, grasp = grasp_b601(
            cid, target, 2 * T, closing=R[:, 1], approach=R[:, 2], tip_depth=0.0015, debug=debug
        )
        wrenches = [force(target, [0, 0, 1], offset=[0.033, 0.029, 0.013], label="F_z")]
        camera = dict(target=[0, 0, 0.050], distance=0.400, yaw=36, pitch=-25)
        suffix = "_debug" if debug else ""
        output = output_dir / f"pcb_workspace{suffix}.png"
        camera_image(
            cid,
            camera,
            -0.0081,
            [-0.120, 0.120, -0.087, 0.087],
            output,
            output_dir / "pcb_workspace_clean.png" if not debug else None,
            wrenches=wrenches,
        )
        if not debug:
            write_manifest(
                output_dir / "pcb_workspace_manifest.json",
                "PCB",
                entry,
                camera=camera,
                grasp=grasp,
                **metadata(wrenches),
                lift_m=lift,
                tilt_x_rad=tx,
                tilt_y_rad=ty,
                observation_mapping="pcb_env_v2: [lift, velocity, tilt_x, tilt_y, ...]",
                board_center_m=center.tolist(),
                board_dimensions_m=[2 * L, 2 * T, 2 * H],
                retainers_released=True,
                socket_top_m=socket_top,
                board_socket_overlap_m=socket_overlap.tolist(),
            )
        print(
            f'PCB: lift={lift*1000:.6f} mm; fingertip gap={grasp["actual_gap_m"]*1000:.4f} mm; {output}'
        )
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "PCB", debug=True)
