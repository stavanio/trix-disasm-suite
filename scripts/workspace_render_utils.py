"""Shared geometry, measured B601 grasps, and camera-registered frame graphics."""

import json
import math
import os
import struct
from functools import lru_cache
from pathlib import Path

import b601_render_common as b601
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont


def mesh_body(
    cid, vertices, indices, color, position=(0, 0, 0), quaternion=(0, 0, 0, 1), flat=False
):
    vertices = np.asarray(vertices)
    extra = {}
    if flat:
        triangles = vertices[np.asarray(indices).reshape(-1, 3)]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        vertices = triangles.reshape(-1, 3)
        indices = list(range(len(vertices)))
        extra["normals"] = np.repeat(normals, 3, axis=0).tolist()
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=vertices.tolist(),
        indices=indices,
        rgbaColor=color,
        specularColor=[0.28, 0.28, 0.28],
        physicsClientId=cid,
        **extra,
    )
    return p.createMultiBody(0, -1, visual, position, quaternion, physicsClientId=cid)


def rounded_box(cid, half, position, color, radius=0.004, bevel=0.0006, quaternion=(0, 0, 0, 1)):
    hx, hy, hz = half
    bevel = min(bevel, hz * 0.45, radius * 0.45)
    vertices = []
    for z, inset in ((-hz, bevel), (-hz + bevel, 0), (hz - bevel, 0), (hz, bevel)):
        x, y, r = hx - inset, hy - inset, radius - inset
        for cx, cy, start in (
            (x - r, y - r, 0),
            (-x + r, y - r, 90),
            (-x + r, -y + r, 180),
            (x - r, -y + r, 270),
        ):
            for angle in np.linspace(start, start + 90, 9):
                a = math.radians(angle)
                vertices.append([cx + r * math.cos(a), cy + r * math.sin(a), z])
    n = len(vertices) // 4
    indices = []
    for ring in range(3):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = ring * n + i, ring * n + j, (ring + 1) * n + j, (ring + 1) * n + i
            indices.extend([a, b, c, a, c, d])
    for i in range(1, n - 1):
        indices.extend([0, i + 1, i, 3 * n, 3 * n + i, 3 * n + i + 1])
    return mesh_body(cid, vertices, indices, color, position, quaternion)


def annulus(cid, inner, outer, height, center, color, segments=96):
    vertices = []
    for r, z in (
        (outer, -height / 2),
        (outer, height / 2),
        (inner, -height / 2),
        (inner, height / 2),
    ):
        vertices.extend(
            [
                [r * math.cos(a), r * math.sin(a), z]
                for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)
            ]
        )
    indices = []

    def quad(a, b, c, d):
        indices.extend([a, b, c, a, c, d])

    n = segments
    for i in range(n):
        j = (i + 1) % n
        quad(i, j, n + j, n + i)
        quad(2 * n + j, 2 * n + i, 3 * n + i, 3 * n + j)
        quad(n + i, n + j, 3 * n + j, 3 * n + i)
        quad(j, i, 2 * n + i, 2 * n + j)
    return mesh_body(cid, vertices, indices, color, center)


@lru_cache(maxsize=8)
def mesh_vertices(name):
    raw = (b601.MESH_DIR / name).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    return (
        np.ndarray((count, 3, 3), dtype="<f4", buffer=raw, offset=96, strides=(50, 12, 4))
        .reshape(-1, 3)
        .astype(float)
    )


def _mesh_world(robot, cid, scale, link, name):
    pos, quat = p.getLinkState(robot, link, computeForwardKinematics=True, physicsClientId=cid)[4:6]
    rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return (mesh_vertices(name) * scale) @ rotation.T + pos


def fingertip_section(points, height):
    """Intersect the triangle surface with the actual contact-height plane.

    An extremum over the bottom 8 mm can select a rib above the workpiece.
    Intersecting at the contact height instead puts the contact on the STL.
    """
    triangles = np.asarray(points).reshape(-1, 3, 3)
    edges = np.stack((triangles, np.roll(triangles, -1, axis=1)), axis=2).reshape(-1, 2, 3)
    dz = edges[:, 1, 2] - edges[:, 0, 2]
    valid = np.abs(dz) > 1e-10
    edges, dz = edges[valid], dz[valid]
    t = (height - edges[:, 0, 2]) / dz
    valid = (t >= 0) & (t <= 1)
    section = edges[valid, 0] + t[valid, None] * (edges[valid, 1] - edges[valid, 0])
    if len(section) < 3:
        raise RuntimeError("Contact plane does not intersect the black fingertip")
    return section


def point_surface_distance(point, vertices):
    """Euclidean distance to the STL triangles, including edges and vertices."""
    tri = np.asarray(vertices).reshape(-1, 3, 3)
    ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    normal = np.cross(ab, ac)
    norm2 = np.sum(normal * normal, axis=1)
    valid = norm2 > 1e-24
    tri, ab, ac, normal, norm2 = tri[valid], ab[valid], ac[valid], normal[valid], norm2[valid]
    ap = np.asarray(point) - tri[:, 0]
    signed = np.sum(ap * normal, axis=1)
    projected = ap - normal * (signed / norm2)[:, None]
    d00, d01, d11 = np.sum(ab * ab, axis=1), np.sum(ab * ac, axis=1), np.sum(ac * ac, axis=1)
    d20, d21 = np.sum(projected * ab, axis=1), np.sum(projected * ac, axis=1)
    denominator = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    inside = (v >= -1e-9) & (w >= -1e-9) & (v + w <= 1 + 1e-9)
    best = np.min(signed[inside] ** 2 / norm2[inside]) if np.any(inside) else np.inf
    for i in range(3):
        a, b = tri[:, i], tri[:, (i + 1) % 3]
        edge = b - a
        length2 = np.sum(edge * edge, axis=1)
        t = np.clip(np.sum((point - a) * edge, axis=1) / length2, 0, 1)
        delta = point - (a + t[:, None] * edge)
        best = min(best, np.min(np.sum(delta * delta, axis=1)))
    return float(np.sqrt(best))


def grasp_b601(
    cid, target, width, closing=(1, 0, 0), approach=(0, 0, 1), tip_depth=0.002, debug=False
):
    """Place the actual black fingertip faces in the grasp's own 3-D frame.

    target is the midpoint of the two contact faces. tip_depth is the
    distance those contacts sit above the physical ends of the fingers.
    """
    c = b601.normalize(closing)
    a = b601.normalize(np.asarray(approach) - c * np.dot(approach, c))
    s = b601.normalize(np.cross(a, c))
    frame = np.column_stack([c, s, a])
    scale, _ = b601.choose_visual_scale()
    robot = b601.load_robot(cid, scale)
    limits = [float(p.getJointInfo(robot, j, physicsClientId=cid)[9]) for j in (7, 8)]

    def opening(f):
        for j, limit in zip((7, 8), limits):
            p.resetJointState(robot, j, f * limit, physicsClientId=cid)

    opening(1)
    pos, quat, _, local = b601.local_tool_frame(robot, cid)
    b601.orient_robot(robot, cid, frame @ local.T, pos, quat)

    def tips():
        pairs = []
        for link, name in ((7, "pla_left.STL"), (8, "pla_right.STL")):
            pts = _mesh_world(robot, cid, scale, link, name) @ frame
            bottom = pts[:, 2].min()
            section = fingertip_section(pts, bottom + tip_depth)
            pairs.append((section, bottom))
        pairs.sort(key=lambda item: item[0][:, 0].mean())
        result = []
        for i, (section, bottom) in enumerate(pairs):
            inner = section[:, 0].max() if i == 0 else section[:, 0].min()
            face = section[np.abs(section[:, 0] - inner) < 1e-7]
            result.append(
                dict(
                    inner=float(inner),
                    side=float((face[:, 1].min() + face[:, 1].max()) / 2),
                    bottom=float(bottom),
                )
            )
        return result

    neg, pos = tips()
    maximum = pos["inner"] - neg["inner"]
    opening(0)
    neg, pos = tips()
    minimum = pos["inner"] - neg["inner"]
    fraction = (width - minimum) / (maximum - minimum)
    if not 0 <= fraction <= 1:
        raise RuntimeError(f"Grasp width {width} is outside the B601 range")
    opening(fraction)
    neg, pos = tips()
    current = frame @ np.array(
        [
            (neg["inner"] + pos["inner"]) / 2,
            (neg["side"] + pos["side"]) / 2,
            (neg["bottom"] + pos["bottom"]) / 2 + tip_depth,
        ]
    )
    b601.translate_robot(robot, cid, np.asarray(target) - current)
    neg, pos = tips()
    contacts = [
        frame @ np.array([v["inner"], v["side"], v["bottom"] + tip_depth]) for v in (neg, pos)
    ]
    actual_gap = pos["inner"] - neg["inner"]
    if (
        abs(actual_gap - width) > 0.0001
        or np.linalg.norm(np.mean(contacts, axis=0) - target) > 0.0001
    ):
        raise RuntimeError("B601 mesh contact alignment failed")

    surfaces = [
        _mesh_world(robot, cid, scale, link, name)
        for link, name in ((7, "pla_left.STL"), (8, "pla_right.STL"))
    ]
    surface_errors = [
        min(point_surface_distance(pt, surface) for surface in surfaces) for pt in contacts
    ]
    if max(surface_errors) > 0.00005:
        raise RuntimeError("A B601 contact point is not on the fingertip STL surface")

    # Attach the wrist to the motor, not the asymmetric combined link AABB.
    motor = _mesh_world(robot, cid, scale, 6, "motor_7.STL") @ frame
    mount = np.array(
        [
            (motor[:, 0].min() + motor[:, 0].max()) / 2,
            (motor[:, 1].min() + motor[:, 1].max()) / 2,
            motor[:, 2].max(),
        ]
    )
    q = b601.quat_from_matrix(frame)
    pos = frame @ (mount + [0, 0, 0.004])
    wrist = b601.make_cyl(cid, 0.011, 0.008, pos.tolist(), [0.48, 0.50, 0.53, 1])
    p.resetBasePositionAndOrientation(wrist, pos, q, physicsClientId=cid)
    rounded_box(
        cid,
        [0.010, 0.010, 0.014],
        (frame @ (mount + [0, 0, 0.022])).tolist(),
        [0.14, 0.16, 0.19, 1],
        radius=0.0015,
        quaternion=q,
    )
    if debug:
        for point, color in zip(contacts, ([1, 0.1, 0.1, 1], [0.1, 0.3, 1, 1])):
            b601.make_sphere(cid, 0.0012, point.tolist(), color)
    return robot, dict(
        visual_scale=scale,
        opening_fraction=fraction,
        actual_gap_m=actual_gap,
        target_m=np.asarray(target).tolist(),
        contacts_m=[v.tolist() for v in contacts],
        closing_axis=c.tolist(),
        approach_axis=a.tolist(),
        tip_depth_m=tip_depth,
        contact_surface_distance_m=surface_errors,
        contact_measurement="black STL triangle intersections at contact height",
    )


_LAST_COORDINATE_REFERENCE = {}


def coordinate_metadata():
    """Describe the reference graphics separately from the physical scene."""
    return dict(_LAST_COORDINATE_REFERENCE)


def add_coordinate_reference(rgb, depth, view, projection, floor_z, xy_bounds, spacing=0.020):
    """B601/Isaac-style world-aligned display frame on the metric floor.

    The display origin is offset to a clear foreground grid intersection;
    its world position is recorded in the manifest. All three directions
    are the true world directions. This is a reference graphic, not a
    change to the simulation origin or a physical object in the task.
    """
    h, w = rgb.shape[:2]
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth).reshape(h, w)
    base = Image.fromarray(rgb).convert("RGBA")
    V = np.asarray(view).reshape(4, 4, order="F")
    P = np.asarray(projection).reshape(4, 4, order="F")
    VP = P @ V
    xmin, xmax, ymin, ymax = xy_bounds
    floor_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, floor_z, 1])
    camera_depth = abs((V @ floor_center)[2])
    # Viewport gizmos keep a near-constant apparent size as the camera zooms.
    length = 180 * 2 * camera_depth / (h * P[1, 1])
    axes = np.eye(3)
    # RGB follows the B601 world-frame convention, with enough contrast
    # for the manuscript's white background.
    colors = [(232, 66, 55), (54, 172, 82), (51, 128, 240)]
    label_colors = [(194, 42, 36), (32, 134, 58), (35, 100, 210)]

    def project(points):
        points = np.asarray(points)
        clip = np.c_[points.reshape(-1, 3), np.ones(points.size // 3)] @ VP.T
        ndc = clip[:, :3] / clip[:, 3:4]
        return np.column_stack([(ndc[:, 0] + 1) * w / 2, (1 - ndc[:, 1]) * h / 2])

    # Find an actual floor-grid intersection whose entire gizmo and text
    # clear the fixture. This avoids arbitrary arrows drawn over contacts.
    samples = [np.zeros(3)]
    for axis in axes:
        samples.extend(axis * t * length for t in np.linspace(0, 1.18, 24))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        samples.extend(
            length * (u * axes[i] + v * axes[j])
            for u in np.linspace(0.23, 0.49, 5)
            for v in np.linspace(0.23, 0.49, 5)
        )
    samples = np.asarray(samples)
    xmin, xmax, ymin, ymax = xy_bounds
    target = np.array([w * 0.12, h * 0.84])
    best = None
    for x in np.arange(math.floor((xmin - 0.06) / spacing) * spacing, xmax, spacing):
        for y in np.arange(math.floor((ymin - 0.06) / spacing) * spacing, ymax, spacing):
            origin = np.array([x, y, floor_z])
            points = project(origin + samples)
            start = points[0]
            if not (65 < start[0] < w * 0.26 and h * 0.64 < start[1] < h - 115):
                continue
            # Clearance includes cone widths and billboard labels.
            pixels = np.rint(points).astype(int)
            if (
                np.any(pixels[:, 0] < 35)
                or np.any(pixels[:, 0] >= w - 35)
                or np.any(pixels[:, 1] < 60)
                or np.any(pixels[:, 1] >= h - 65)
            ):
                continue
            clear = all(
                np.all(depth[pixels[:, 1] + dy, pixels[:, 0] + dx] > 0.99999)
                for dx in (-22, 0, 22)
                for dy in (-22, 0, 22)
            )
            if clear:
                score = np.linalg.norm(start - target)
                if best is None or score < best[0]:
                    best = (score, origin)
    if best is None:
        raise RuntimeError("No clear metric-grid position for the coordinate gizmo")
    origin = best[1]

    # Keep the established 20 mm metric grid and extend it under the frame.
    xmin = min(xmin, origin[0] - spacing)
    ymin = min(ymin, origin[1] - spacing)
    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid)
    for x in np.arange(math.ceil(xmin / spacing) * spacing, xmax + 1e-8, spacing):
        draw.line(
            [tuple(v) for v in project([[x, ymin, floor_z], [x, ymax, floor_z]])],
            fill=(137, 151, 165, 80),
            width=2,
        )
    for y in np.arange(math.ceil(ymin / spacing) * spacing, ymax + 1e-8, spacing):
        draw.line(
            [tuple(v) for v in project([[xmin, y, floor_z], [xmax, y, floor_z]])],
            fill=(137, 151, 165, 80),
            width=2,
        )
    layer = np.asarray(grid).copy()
    layer[:, :, 3] = (layer[:, :, 3] * (depth > 0.99999)).astype(np.uint8)
    base = Image.alpha_composite(base, Image.fromarray(layer))

    # Isaac-style square plane handles: XY blue, XZ green, YZ red.
    planes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(planes)
    for i, j, normal in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        corners = [
            origin + length * (u * axes[i] + v * axes[j])
            for u, v in ((0.23, 0.23), (0.49, 0.23), (0.49, 0.49), (0.23, 0.49))
        ]
        polygon = [tuple(v) for v in project(corners)]
        color = colors[normal]
        draw.polygon(polygon, fill=(*color, 58))
        draw.line(polygon + [polygon[0]], fill=(*color, 210), width=3, joint="curve")
    layer = np.asarray(planes).copy()
    layer[:, :, 3] = (layer[:, :, 3] * (depth > 0.99999)).astype(np.uint8)
    base = Image.alpha_composite(base, Image.fromarray(layer))

    # Render cylindrical shafts and cone tips in a separate PyBullet
    # graphics pass. The mechanism render, shadows and camera stay intact.
    cid = p.connect(p.DIRECT)
    try:
        for axis, color in zip(axes, colors):
            color = [c / 255 for c in color] + [1]
            side = np.cross(axis, [0, 0, 1] if axis[2] == 0 else [0, 1, 0])
            side /= np.linalg.norm(side)
            other = np.cross(axis, side)
            orientation = b601.quat_from_matrix(np.column_stack([side, other, axis]))
            shaft_length = length - 0.0062
            body = b601.make_cyl(
                cid, 0.00048, shaft_length, (origin + axis * shaft_length / 2).tolist(), color
            )
            p.resetBasePositionAndOrientation(
                body, origin + axis * shaft_length / 2, orientation, physicsClientId=cid
            )
            vertices = [
                [0.0021 * math.cos(t), 0.0021 * math.sin(t), shaft_length]
                for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)
            ]
            vertices += [[0, 0, length], [0, 0, shaft_length]]
            indices = []
            for i in range(16):
                j = (i + 1) % 16
                indices.extend([i, j, 16, j, i, 17])
            mesh_body(cid, vertices, indices, color, origin, orientation, flat=True)
        b601.make_sphere(cid, 0.00085, origin.tolist(), [0.82, 0.85, 0.89, 1])
        _, _, rgba, gizmo_depth, seg = p.getCameraImage(
            w,
            h,
            view,
            projection,
            renderer=p.ER_TINY_RENDERER,
            shadow=0,
            lightDirection=[-1, -2, 3],
            lightAmbientCoeff=0.60,
            lightDiffuseCoeff=0.62,
            lightSpecularCoeff=0.12,
            physicsClientId=cid,
        )
        layer = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4).copy()
        visible = (np.asarray(seg).reshape(h, w) >= 0) & (
            np.asarray(gizmo_depth).reshape(h, w) < depth
        )
        layer[:, :, 3] = visible.astype(np.uint8) * 255
        base = Image.alpha_composite(base, Image.fromarray(layer))
    finally:
        p.disconnect(cid)

    labels = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(labels)
    fontdir = font_directory()
    font = ImageFont.truetype(str(fontdir / "DejaVuSans-Bold.ttf"), 40)
    small = ImageFont.truetype(str(fontdir / "DejaVuSans.ttf"), 34)
    start = project([origin])[0]
    for letter, axis, color in zip("XYZ", axes, label_colors):
        end = project([origin + length * axis])[0]
        direction = (end - start) / np.linalg.norm(end - start)
        draw.text(
            tuple(end + direction * 24),
            letter,
            font=font,
            fill=(*color, 255),
            anchor="mm",
            stroke_width=2,
            stroke_fill=(255, 255, 255, 235),
        )
    draw.text((40, h - 45), "20 mm grid", font=small, fill=(112, 124, 138, 255), anchor="lt")
    global _LAST_COORDINATE_REFERENCE
    _LAST_COORDINATE_REFERENCE = dict(
        coordinate_reference="world-aligned simulator display frame; camera-projected XY grid",
        coordinate_style="B601-inspired RGB arrows with Isaac-style square plane handles",
        display_frame_origin_world_m=origin.tolist(),
        display_frame_axes_world=axes.tolist(),
        display_frame_axis_length_m=length,
        display_frame_is_simulation_origin=False,
        grid_spacing_m=spacing,
    )
    return Image.alpha_composite(base, labels).convert("RGB")


def camera_image(cid, camera, floor_z, bounds, output, clean_output=None, *, wrenches=(), notes=()):
    width, height = 1800, 1400
    view = p.computeViewMatrixFromYawPitchRoll(
        camera["target"],
        camera["distance"],
        camera["yaw"],
        camera["pitch"],
        0,
        2,
        physicsClientId=cid,
    )
    projection = p.computeProjectionMatrixFOV(36, width / height, 0.003, 3.0, physicsClientId=cid)
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
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    if clean_output:
        Image.fromarray(rgb).save(clean_output, dpi=(300, 300))
    from workspace_wrench import annotate

    annotated = add_coordinate_reference(rgb, depth, view, projection, floor_z, bounds)
    annotate(annotated, view, projection, wrenches, notes).save(output, dpi=(300, 300))


def write_manifest(output, task, entry, **visualization):
    data = dict(
        task=task,
        **{
            k: entry[k]
            for k in (
                "trace_file",
                "trace_sha256",
                "trace_step",
                "episode_seed",
                "training_seed",
                "checkpoint_step",
            )
        },
        state=entry["frame"]["state"],
        observation=entry["frame"]["obs"],
        visualization=dict(
            **visualization,
            **coordinate_metadata(),
            arm_kinematics_simulated=False,
            geometry_is_illustrative=True,
        ),
    )
    Path(output).write_text(json.dumps(data, indent=2) + "\n")


def rounded_outline(half_x, half_y, radius, segments=8):
    points = []
    for cx, cy, start in (
        (half_x - radius, half_y - radius, 0),
        (-half_x + radius, half_y - radius, 90),
        (-half_x + radius, -half_y + radius, 180),
        (half_x - radius, -half_y + radius, 270),
    ):
        for angle in np.linspace(start, start + 90, segments):
            a = math.radians(angle)
            points.append([cx + radius * math.cos(a), cy + radius * math.sin(a)])
    return points


def rounded_solid(cid, half_x, half_y, height, radius, bevel, center, color):
    """A single closed pouch surface with a softly bevelled perimeter."""
    rings = (
        (-height / 2, half_x - bevel, half_y - bevel, radius - bevel),
        (-height / 2 + bevel, half_x, half_y, radius),
        (height / 2 - bevel, half_x, half_y, radius),
        (height / 2, half_x - bevel, half_y - bevel, radius - bevel),
    )
    vertices = []
    for z, hx, hy, r in rings:
        vertices.extend([[x, y, z] for x, y in rounded_outline(hx, hy, r)])
    n = len(vertices) // len(rings)
    indices = []
    for ring in range(3):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = ring * n + i, ring * n + j, (ring + 1) * n + j, (ring + 1) * n + i
            indices.extend([a, b, c, a, c, d])
    for i in range(1, n - 1):
        indices.extend([0, i + 1, i, 3 * n, 3 * n + i, 3 * n + i + 1])
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=color,
        specularColor=[0.32, 0.32, 0.32],
        physicsClientId=cid,
    )
    return p.createMultiBody(0, -1, visual, center, physicsClientId=cid)


def font_directory():
    """Locate the DejaVu fonts used for the printed label and axis text."""
    configured = os.environ.get("TRIX_FONT_DIR")
    candidates = [Path(configured)] if configured else [Path("/usr/share/fonts/truetype/dejavu")]
    for directory in candidates:
        if all((directory / name).is_file() for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")):
            return directory
    if not configured:
        import matplotlib

        directory = Path(matplotlib.get_data_path()) / "fonts/ttf"
        if all((directory / name).is_file() for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")):
            return directory
    raise FileNotFoundError(
        "DejaVu Sans fonts were not found; set TRIX_FONT_DIR to their directory."
    )
