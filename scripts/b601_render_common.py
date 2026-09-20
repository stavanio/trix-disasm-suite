"""B601 mesh loading, visual scale and fingertip-frame geometry.

Model assets are supplied by the upstream B601 description package. All
positions and lengths are in metres; PyBullet quaternions use xyzw order.
"""

import atexit
import hashlib
import json
import math
import os
import struct
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = ROOT / "assets/workspaces/b601_assets.json"
DEFAULT_DESCRIPTION = (
    Path.home() / "ReBot_Arm_DigitalTwin_RS/rebotarm_ros2_RS/src/rebotarm_bringup/description"
)
B601_DESC = Path(os.environ.get("TRIX_B601_DESCRIPTION", DEFAULT_DESCRIPTION)).expanduser()
MESH_DIR = B601_DESC / "meshes_rs"
TARGET_GRIPPER_SPAN = 0.090


def configure_assets(description=None):
    """Select a description directory containing urdf/ and meshes_rs/."""
    global B601_DESC, MESH_DIR
    if description is not None:
        B601_DESC = Path(description).expanduser().resolve()
        MESH_DIR = B601_DESC / "meshes_rs"
        prepare_urdf.cache_clear()


def verify_assets():
    """Check the URDF and mesh content used for the committed renders."""
    manifest = json.loads(ASSET_MANIFEST.read_text())
    for relative, expected in manifest["sha256"].items():
        path = B601_DESC / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing B601 asset: {path}. See docs/workspace_rendering.md "
                "or pass --b601-description."
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"B601 asset differs from the reference model: {path}")
    return manifest


@lru_cache(maxsize=1)
def prepare_urdf():
    """Resolve ROS package mesh paths in a process-local temporary URDF."""
    verify_assets()
    source = B601_DESC / "urdf/ReBot_Arm_RS.urdf"
    text = source.read_text().replace(
        "package://rebotarm_bringup/description/meshes_rs/",
        str(MESH_DIR.resolve()) + "/",
    )
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="trix-b601-", suffix=".urdf", delete=False
    ) as handle:
        handle.write(text)
        path = Path(handle.name)
    atexit.register(path.unlink, missing_ok=True)
    return path


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def quat_from_matrix(R):
    R = np.asarray(R, float)
    tr = np.trace(R)

    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], float)
    return (q / np.linalg.norm(q)).tolist()


def make_box(cid, half, pos, rgba, orn=(0, 0, 0)):
    v = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half,
        rgbaColor=rgba,
        physicsClientId=cid,
    )

    return p.createMultiBody(
        0,
        -1,
        v,
        pos,
        p.getQuaternionFromEuler(orn),
        physicsClientId=cid,
    )


def make_cyl(cid, r, h, pos, rgba):
    v = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=r,
        length=h,
        rgbaColor=rgba,
        physicsClientId=cid,
    )

    return p.createMultiBody(
        0,
        -1,
        v,
        pos,
        physicsClientId=cid,
    )


def make_sphere(cid, r, pos, rgba):
    v = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=r,
        rgbaColor=rgba,
        physicsClientId=cid,
    )

    return p.createMultiBody(
        0,
        -1,
        v,
        pos,
        physicsClientId=cid,
    )


def load_robot(cid, scale):
    robot = p.loadURDF(
        str(prepare_urdf()),
        useFixedBase=True,
        globalScaling=scale,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=cid,
    )

    for j in range(6):
        p.resetJointState(robot, j, 0.0, physicsClientId=cid)

    # Hide base through link6. Keep gripper_end + both fingers.
    p.changeVisualShape(
        robot,
        -1,
        rgbaColor=[1, 1, 1, 0],
        physicsClientId=cid,
    )

    for link in range(0, 6):
        p.changeVisualShape(
            robot,
            link,
            rgbaColor=[1, 1, 1, 0],
            physicsClientId=cid,
        )

    return robot


def visible_bounds(robot, cid):
    mins, maxs = [], []

    for link in (6, 7, 8):
        lo, hi = p.getAABB(
            robot,
            link,
            physicsClientId=cid,
        )

        mins.append(np.asarray(lo))
        maxs.append(np.asarray(hi))

    return (
        np.min(np.stack(mins), axis=0),
        np.max(np.stack(maxs), axis=0),
    )


def choose_visual_scale():
    cid = p.connect(p.DIRECT)

    robot = load_robot(cid, 1.0)

    # Mid-open finger state for measuring overall native size.
    p.resetJointState(robot, 7, 0.020, physicsClientId=cid)
    p.resetJointState(robot, 8, 0.028, physicsClientId=cid)

    lo, hi = visible_bounds(robot, cid)

    native_span = float(np.max(hi - lo))
    scale = TARGET_GRIPPER_SPAN / native_span

    p.disconnect(cid)

    return scale, native_span


def local_tool_frame(robot, cid):
    ref = p.getLinkState(
        robot,
        6,
        computeForwardKinematics=True,
        physicsClientId=cid,
    )

    ref_pos = ref[4]
    ref_q = ref[5]

    inv_pos, inv_q = p.invertTransform(
        ref_pos,
        ref_q,
    )

    local = {}

    for name, link in (("left", 7), ("right", 8)):
        state = p.getLinkState(
            robot,
            link,
            computeForwardKinematics=True,
            physicsClientId=cid,
        )

        pos_local, _ = p.multiplyTransforms(
            inv_pos,
            inv_q,
            state[4],
            [0, 0, 0, 1],
        )

        local[name] = np.asarray(pos_local, float)

    midpoint = 0.5 * (local["left"] + local["right"])

    closing = normalize(local["right"] - local["left"])

    # gripper_end origin -> finger-frame midpoint
    approach = midpoint.copy()
    approach -= closing * np.dot(approach, closing)
    approach = normalize(approach)

    side = normalize(np.cross(approach, closing))

    closing = normalize(np.cross(side, approach))

    L = np.column_stack(
        [
            closing,
            side,
            approach,
        ]
    )

    return ref_pos, ref_q, midpoint, L


def orient_robot(robot, cid, target_R, ref_pos0, ref_q0):
    target_q = quat_from_matrix(target_R)

    inv_ref_pos, inv_ref_q = p.invertTransform(
        ref_pos0,
        ref_q0,
    )

    base_pos, base_q = p.multiplyTransforms(
        [0, 0, 0],
        target_q,
        inv_ref_pos,
        inv_ref_q,
    )

    p.resetBasePositionAndOrientation(
        robot,
        base_pos,
        base_q,
        physicsClientId=cid,
    )


def translate_robot(robot, cid, delta):
    pos, q = p.getBasePositionAndOrientation(
        robot,
        physicsClientId=cid,
    )

    p.resetBasePositionAndOrientation(
        robot,
        (np.asarray(pos) + delta).tolist(),
        q,
        physicsClientId=cid,
    )


def black_meshes():
    meshes = {}
    for link, name in ((7, "pla_left.STL"), (8, "pla_right.STL")):
        data = (MESH_DIR / name).read_bytes()
        count = struct.unpack_from("<I", data, 80)[0]
        meshes[link] = (
            np.ndarray(
                (count, 3, 3),
                dtype="<f4",
                buffer=data,
                offset=96,
                strides=(50, 12, 4),
            )
            .reshape(-1, 3)
            .astype(float)
        )
    return meshes


def fingertip_geometry(robot, cid, scale, meshes):
    tips = []
    for link, vertices in meshes.items():
        pos, quat = p.getLinkState(robot, link, computeForwardKinematics=True, physicsClientId=cid)[
            4:6
        ]
        rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        world = (vertices * scale) @ rotation.T + pos
        bottom = world[:, 2].min()
        tip = world[world[:, 2] <= bottom + 0.008]
        tips.append(
            {
                "inner_y": float(tip[:, 1].max() if link == 7 else tip[:, 1].min()),
                "center_x": float(0.5 * (tip[:, 0].min() + tip[:, 0].max())),
                "bottom_z": float(bottom),
            }
        )
    return tips
