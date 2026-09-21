#!/usr/bin/env python3
"""Render PCB lift and both tilt components directly from PCBEnvV2."""
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
from workspace_wrench import force, metadata

L, T, H, Z0 = 0.0665, 0.001, 0.0155, 0.003
# Fixed CAD dimensions. These do not change to accommodate a selected state.
SEATED_BOTTOM_Z_M = Z0
SOCKET_TOP_M = Z0 + 0.0098
SOCKET_INNER_HALF_WIDTH_M = 0.0016


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, environment=None, debug=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    env, model, entry, binding = prepare_environment("PCB", provenance_path, environment)
    lift, tx, ty = float(env.z), float(env.theta[0]), float(env.theta[1])
    retainers_released = env.z >= model.Z_CLIP
    quat = p.getQuaternionFromEuler([tx, ty, 0])
    R = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    # Tilt about the lower long edge so it remains centered inside the slot.
    # Lift is measured from the seated board's lower edge at the channel floor.
    lower_edge_center = np.array([0, 0, SEATED_BOTTOM_Z_M + lift])
    center = lower_edge_center + R @ np.array([0, 0, H])
    cid = p.connect(p.DIRECT)
    try:

        def board_box(half, local, color):
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid
            )
            return p.createMultiBody(
                0, -1, visual, (center + R @ np.asarray(local)).tolist(), quat, physicsClientId=cid
            )

        # Bare motherboard on three short spacers, without a generic plinth.
        fixture = rounded_box(cid, [.087,.053,.0015], [0,0,.0015], [.055,.21,.145,1], radius=.004, bevel=.0003)
        for x,y in ((-.078,-.044),(.078,-.040),(0,.044)):
            b601.make_cyl(cid,.003,.006,[x,y,-.003],[.55,.58,.60,1])
            annulus(cid,.0015,.003,.0002,[x,y,.0031],[.62,.58,.34,1])
            b601.make_cyl(cid,.0015,.0001,[x,y,.0031],[.035,.08,.065,1])
        # Sparse routed traces with 45-degree corners, leading to component groups.
        # These are fixed CAD surface details, not plotted state values.
        routes = [
            [(-.058,-.005),(-.058,-.020),(-.046,-.032)],
            [(-.054,-.005),(-.054,-.018),(-.043,-.029)],
            [(.022,-.005),(.022,-.016),(.034,-.028),(.047,-.028)],
            [(.026,-.005),(.026,-.014),(.037,-.025),(.047,-.025)],
            [(-.025,.005),(-.025,.019),(-.013,.031),(.005,.031)],
            [(.047,.005),(.047,.018),(.060,.031)],
        ]
        for route in routes:
            for a,b in zip(route,route[1:]):
                a,b = np.asarray(a),np.asarray(b); delta=b-a; midpoint=(a+b)/2
                b601.make_box(cid,[float(np.linalg.norm(delta))/2,.00016,.000025],
                    [*midpoint,.00305],[.10,.30,.20,1],orn=[0,0,math.atan2(delta[1],delta[0])])
        for x,y,hx,hy in ((-.044,-.032,.008,.005),(.053,-.027,.007,.005),(.010,.030,.006,.004)):
            rounded_box(cid,[hx,hy,.0012],[x,y,.0042],[.075,.09,.105,1],radius=.0005,bevel=.0001)
            for sign in (-1,1):
                for dx in (-.0045,-.0015,.0015,.0045):
                    b601.make_box(cid,[.00035,.001,.0002],[x+dx,y+sign*(hy+.0006),.0034],[.58,.61,.59,1])
        for x,y in ((-.023,-.034),(-.017,-.034),(.022,.032),(.028,.032),(.063,.022)):
            b601.make_box(cid,[.0013,.0008,.0007],[x,y,.0037],[.48,.40,.23,1])
            for dx in (-.0013,.0013):
                b601.make_box(cid,[.00035,.00085,.0007],[x+dx,y,.0037],[.65,.67,.64,1])
        # Open socket channel: separate rails leave the board slot visible.
        socket = rounded_box(
            cid,
            [L + 0.004, 0.0048, 0.0008],
            [0, 0, SEATED_BOTTOM_Z_M - 0.0008],
            [0.13, 0.15, 0.18, 1],
            radius=0.0015,
            bevel=0.0002,
        )
        rail_half_height = (SOCKET_TOP_M - SEATED_BOTTOM_Z_M) / 2
        rail_center_z = (SOCKET_TOP_M + SEATED_BOTTOM_Z_M) / 2
        rail_center_y = SOCKET_INNER_HALF_WIDTH_M + 0.001
        for y in (-rail_center_y, rail_center_y):
            rounded_box(
                cid,
                [L + 0.004, 0.001, rail_half_height],
                [0, y, rail_center_z],
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
        # Retainer release follows the environment clip-clearance phase.
        for sign in (-1, 1):
            pivot = np.array([sign * (L + 0.004), 0, Z0 + 0.0025])
            q = p.getQuaternionFromEuler([0, sign * (math.radians(30) if retainers_released else 0), 0])
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
        board = board_box([L, T, H - nd / 2], [0, 0, nd / 2], green)
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
        # Check both tilted board faces, not only the bottom centerline.
        socket_top = SOCKET_TOP_M
        edge = np.array([center + R @ [x, 0, -H] for x in (-L, L)])
        socket_overlap = socket_top - edge[:, 2]
        bottom_corners = np.array([center + R @ [x, y, -H] for x in (-L, L) for y in (-T, T)])
        corner_overlap = socket_top - bottom_corners[:, 2]
        # At the mouth, the gold outer faces are the widest inserted section.
        mouth_y = []
        gold_z = []
        for face in (-1, 1):
            local_y = face * (T + 0.00013)
            local_z = (socket_top - center[2] - R[2, 1] * local_y) / R[2, 2]
            mouth_y.append(float((center + R @ [0, local_y, local_z])[1]))
            gold_z.append([(center + R @ [0, local_y, -H + h])[2] for h in (0.00025, 0.00295)])
        slot_clearance = SOCKET_INNER_HALF_WIDTH_M - max(abs(y) for y in mouth_y)
        gold_z = np.asarray(gold_z)
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
            -0.0061,
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
                environment=binding,
                rendered_bodies=body_geometry(cid, fixture=fixture, socket=socket, board=board, gripper=robot),
                fixture_material="bare motherboard on three spacers",
                board_surface_detail="sparse routed traces and component groups",
                pose_source="PCBEnvV2.z and PCBEnvV2.theta",
                lift_m=lift,
                tilt_x_rad=tx,
                tilt_y_rad=ty,
                observation_mapping="[z, v_z, theta[0], theta[1], omega[0], omega[1], damage, fractured, 0, 0]",
                tilt_axis_world=[1, 0, 0],
                tilt_axis_description="board long axis, through the lower edge",
                seated_bottom_z_m=SEATED_BOTTOM_Z_M,
                lower_edge_center_m=lower_edge_center.tolist(),
                board_center_m=center.tolist(),
                board_dimensions_m=[2 * L, 2 * T, 2 * H],
                retainers_released=bool(retainers_released),
                socket_top_m=socket_top,
                board_socket_overlap_m=socket_overlap.tolist(),
                board_bottom_corner_overlap_m=corner_overlap.tolist(),
                socket_inner_width_m=2 * SOCKET_INNER_HALF_WIDTH_M,
                socket_side_clearance_m=slot_clearance,
                gold_contact_lower_upper_z_m=gold_z.tolist(),
                gold_contact_buried_height_m=(socket_top - gold_z[:, 0]).tolist(),
                gold_contact_exposed_height_m=(gold_z[:, 1] - socket_top).tolist(),
            )
        print(
            f'PCB: lift={lift*1000:.6f} mm; fingertip gap={grasp["actual_gap_m"]*1000:.4f} mm; {output}'
        )
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "PCB", debug=True)
