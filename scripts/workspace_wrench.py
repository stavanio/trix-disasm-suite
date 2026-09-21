"""Camera-projected, schematic wrench directions for the workspace figures.

These annotations identify load application and direction, not measured loads.
The frozen input records do not contain force/torque samples or robot dynamics.
"""

import math

import numpy as np
from PIL import ImageDraw, ImageFont
from workspace_render_utils import font_directory

INK = (126, 55, 151)


def force(anchor, direction, *, offset, length=0.034, label="F", label_offset=(22, -16), show_leader=True):
    anchor, direction, offset = map(
        lambda v: np.asarray(v, dtype=float), (anchor, direction, offset)
    )
    direction /= np.linalg.norm(direction)
    start = anchor + offset
    return dict(
        kind="force",
        label=label,
        anchor_world_m=anchor.tolist(),
        direction_world=direction.tolist(),
        path_world_m=[start.tolist(), (start + direction * length).tolist()],
        label_offset_px=list(label_offset),
        **({"show_leader": False} if not show_leader else {}),
    )


def torque(
    anchor,
    axis,
    *,
    radius=0.025,
    start_deg=0,
    sweep_deg=240,
    offset=(0, 0, 0),
    label="τ",
    label_offset=(22, -16),
    show_leader=True,
):
    anchor, axis, offset = map(lambda v: np.asarray(v, dtype=float), (anchor, axis, offset))
    axis /= np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1, 0])
    u = helper - axis * np.dot(helper, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.linspace(math.radians(start_deg), math.radians(start_deg + sweep_deg), 80)
    path = anchor + offset + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    return dict(
        kind="torque",
        label=label,
        anchor_world_m=anchor.tolist(),
        axis_world=axis.tolist(),
        path_world_m=path.tolist(),
        label_offset_px=list(label_offset),
        **({"show_leader": False} if not show_leader else {}),
    )


def metadata(wrenches):
    return dict(
        wrench_annotations=list(wrenches),
        wrench_values_measured=False,
        wrench_arrow_scale="schematic; lengths do not encode magnitude",
    )


def annotate(image, view, projection, wrenches, notes=()):
    """Draw from world coordinates using the scene's unchanged camera."""
    image = image.copy().convert("RGB")
    width, height = image.size
    vp = np.asarray(projection).reshape(4, 4, order="F") @ np.asarray(view).reshape(4, 4, order="F")
    draw = ImageDraw.Draw(image)
    fontdir = font_directory()
    font = ImageFont.truetype(str(fontdir / "DejaVuSans.ttf"), 38)
    sub = ImageFont.truetype(str(fontdir / "DejaVuSans.ttf"), 26)
    small = ImageFont.truetype(str(fontdir / "DejaVuSans.ttf"), 30)

    def project(points):
        points = np.asarray(points)
        clip = np.c_[points, np.ones(len(points))] @ vp.T
        ndc = clip[:, :3] / clip[:, 3:4]
        return np.column_stack([(ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2])

    def label(position, text, color=INK):
        base, _, suffix = text.partition("_")
        draw.text(tuple(position), base, font=font, fill=color, stroke_width=2, stroke_fill="white")
        if suffix:
            shift = draw.textlength(base, font=font)
            draw.text(
                tuple(position + [shift + 1, 19]),
                suffix,
                font=sub,
                fill=color,
                stroke_width=2,
                stroke_fill="white",
            )

    for item in wrenches:
        points = project(item["path_world_m"])
        anchor = project([item["anchor_world_m"]])[0]
        # A fine dotted leader identifies the application point without
        # shifting the load direction or obscuring the contact geometry.
        start = points[0]
        span = np.linalg.norm(start - anchor)
        if item.get("show_leader", True) and span > 14:
            for t in np.arange(8, span, 14):
                a = anchor + (start - anchor) * t / span
                b = anchor + (start - anchor) * min(t + 6, span) / span
                draw.line([tuple(a), tuple(b)], fill=INK, width=3)
        line = [tuple(p) for p in points]
        draw.line(line, fill="white", width=12, joint="curve")
        draw.line(line, fill=INK, width=7, joint="curve")
        direction = points[-1] - points[-3 if len(points) > 2 else 0]
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        end = points[-1]
        head = [end, end - 24 * direction + 10 * normal, end - 24 * direction - 10 * normal]
        draw.polygon([tuple(p) for p in head], fill=INK)
        label(end + item["label_offset_px"], item["label"])
    for note in notes:
        anchor = project([note["anchor_world_m"]])[0]
        pos = np.asarray(note["position_px"], dtype=float)
        end = pos + [0, 36]
        draw.line([tuple(anchor), tuple(end)], fill=(90, 101, 114), width=2)
        box = draw.textbbox(tuple(pos), note["text"], font=small)
        draw.rounded_rectangle(
            (box[0] - 7, box[1] - 5, box[2] + 7, box[3] + 5), radius=5, fill="white"
        )
        draw.text(tuple(pos), note["text"], font=small, fill=(70, 80, 93))
    if wrenches:
        draw.text(
            (width - 40, height - 43),
            "Wrench directions (schematic)",
            font=small,
            fill=INK,
            anchor="rt",
        )
    return image
