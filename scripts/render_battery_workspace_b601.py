#!/usr/bin/env python3
"""Render the archived BATTERY peel state as a recognizable foil pouch cell."""

import json
import tempfile
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
from workspace_environment import prepare_environment, body_geometry
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, run_renderer
from workspace_render_utils import (
    font_directory, rounded_solid, mesh_body, grasp_b601,
    _mesh_world, point_surface_distance,
)
from workspace_wrench import annotate, force, metadata


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


def pull_tab(cid, origin, width=.018, thickness=.0007):
    """A fixed folded polymer tab; its whole frame translates with battery z."""
    stations = [(.028,0,0,1),(.001,0,0,1)]
    for angle in np.linspace(-np.pi/2,-np.pi,17)[1:]:
        x,z = .001+.001*np.cos(angle), .001+.001*np.sin(angle)
        stations.append((x,z,-np.cos(angle),-np.sin(angle)))
    stations.append((0,.021,1,0))
    vertices,indices = [],[]
    for x,z,nx,nz in stations:
        a=(x-nx*thickness/2,z-nz*thickness/2)
        b=(x+nx*thickness/2,z+nz*thickness/2)
        vertices.extend([[a[0],-width/2,a[1]],[a[0],width/2,a[1]],
                         [b[0],width/2,b[1]],[b[0],-width/2,b[1]]])
    indices.extend([0,2,1,0,3,2])
    for ring in range(len(stations)-1):
        for j in range(4):
            k=(j+1)%4; a,b,c,d=4*ring+j,4*ring+k,4*(ring+1)+k,4*(ring+1)+j
            indices.extend([a,b,c,a,c,d])
    end=4*(len(stations)-1)
    indices.extend([end,end+1,end+2,end,end+2,end+3])
    return mesh_body(cid,vertices,indices,[.91,.89,.81,1],origin),vertices


def render(*, output_dir=DEFAULT_OUTPUT_DIR, provenance_path=DEFAULT_PROVENANCE, environment=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = Path(provenance_path)
    env, model, entry, binding = prepare_environment("BATTERY", provenance_path, environment)
    state = entry["frame"]["state"]
    lift = float(env.z)
    output_dir.mkdir(parents=True, exist_ok=True)
    cid = p.connect(p.DIRECT)
    try:
        with tempfile.TemporaryDirectory(prefix="trix_battery_") as tempdir:
            # Shallow device tray and two discrete adhesive strips.
            fixture = rounded_solid(cid,.078,.049,.004,.008,.0006,[0,0,-.002],[.58,.62,.66,1])
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
            cell = rounded_solid(
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

            # Grip the broad faces of an upright pull tab, not its thin edges.
            tab_width,tab_thickness=.018,.0007
            tab_z=cell_bottom+.0008
            tab_origin=np.array([-.071,0,tab_z])
            tab,tab_vertices=pull_tab(cid,tab_origin,tab_width,tab_thickness)
            contact_center=tab_origin+[0,0,.016]
            robot,grasp=grasp_b601(cid,contact_center,tab_thickness,
                                  closing=[1,0,0],approach=[0,0,1],tip_depth=.003)
            tab_contacts=np.asarray(grasp["contacts_m"])-tab_origin
            face_errors=np.abs(np.abs(tab_contacts[:,0])-tab_thickness/2)
            if (max(face_errors)>.00005 or np.any(np.abs(tab_contacts[:,1])>tab_width/2)
                or np.any(tab_contacts[:,2]<.002) or np.any(tab_contacts[:,2]>.020)):
                raise RuntimeError("BATTERY fingertips must meet the upright tab's two broad faces")

            # Check a 2 x 2 mm patch on each tab face against the real STL,
            # so a correct gap alone cannot hide a point-only or hovering grasp.
            tip_surfaces=[_mesh_world(robot,cid,grasp["visual_scale"],link,name)
                          for link,name in ((7,"pla_left.STL"),(8,"pla_right.STL"))]
            patch_errors=[]
            for contact in grasp["contacts_m"]:
                distances=[]
                for dy,dz in ((-.001,-.001),(-.001,.001),(.001,-.001),(.001,.001)):
                    point=np.asarray(contact)+[0,dy,dz]
                    distances.append(min(point_surface_distance(point,surface) for surface in tip_surfaces))
                patch_errors.append(distances)
            if np.max(patch_errors)>.00005:
                raise RuntimeError("BATTERY tab face contact patch is outside the fingertip surfaces")

            width, height = 1800, 1400
            camera = {
                "target": [-0.015, 0, 0.040],
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

            wrenches = [
                force(contact_center, [0, 0, 1], offset=[-0.030, -0.027, 0.014], label="F_peel")
            ]
            Image.fromarray(image).save(output_dir / "battery_workspace_clean.png", dpi=(300, 300))
            annotated = add_coordinate_reference(
                image, depth, view, projection, -0.0041, [-0.115, 0.105, -0.09, 0.09]
            )
            annotate(annotated, view, projection, wrenches).save(
                output_dir / "battery_workspace.png", dpi=(300, 300)
            )

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
            "observation": entry["frame"]["obs"],
            "interpretation": "intact lithium-ion pouch cell lifted from adhesive using an extraction tab",
            "visualization": {
                "environment": binding,
                "rendered_bodies": body_geometry(cid, fixture=fixture, cell=cell, tab=tab, gripper=robot),
                "lift_m": lift,
                "cell_thickness_m": thickness,
                "grasp": grasp,
                "tab_width_m": tab_width,
                "tab_thickness_m": tab_thickness,
                "tab_local_vertices_m": tab_vertices,
                "tab_contacts_local_m": tab_contacts.tolist(),
                "tab_face_contact_error_m": face_errors.tolist(),
                "tab_contact_patch_dimensions_m": [.002,.002],
                "tab_contact_patch_surface_error_m": patch_errors,
                "grasp_interface": "opposed broad faces of upright extraction tab",
                "fixture_material": "thin light-metal device tray",
                "camera": camera,
                **coordinate_metadata(),
                **metadata(wrenches),
                "arm_kinematics_simulated": False,
                "geometry_is_illustrative": True,
            },
        }
        (output_dir / "battery_workspace_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print("BATTERY peel lift mm:", lift * 1000)
        print("Deformation:", state["deformation"], "temperature C:", state["temperature_C"])
        print("Pull-tab grasp gap mm:", grasp["actual_gap_m"] * 1000)
        print("WROTE:", output_dir / "battery_workspace.png")
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    run_renderer(render, "BATTERY", debug=False)
