"""PyBullet rendering of benchmark states.

Physics stays in the numpy environments. This draws their state so the
image a model sees comes from the state that then receives its action.
"""

import math
import os

import numpy as np
import pybullet as p

WIDTH, HEIGHT = 768, 576

GREEN = [0.08, 0.30, 0.13, 1.0]
GOLD = [0.85, 0.70, 0.25, 1.0]
CHIP = [0.09, 0.09, 0.10, 1.0]
SLOT = [0.06, 0.06, 0.07, 1.0]
MOBO = [0.10, 0.16, 0.11, 1.0]
CLIP = [0.88, 0.88, 0.90, 1.0]
CAP = [0.20, 0.20, 0.55, 1.0]
CAP_TOP = [0.70, 0.70, 0.72, 1.0]
HEATSINK = [0.62, 0.63, 0.66, 1.0]
MAT = [0.20, 0.20, 0.22, 1.0]

L, T, H = 0.0665, 0.0010, 0.0155
NOTCH_X = 0.008


class Scene:

    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                   physicsClientId=self.cid)
        self.bodies = []

    def box(self, half, pos, rgba, orn=(0, 0, 0), spec=(0.1, 0.1, 0.1)):
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba,
                                specularColor=list(spec),
                                physicsClientId=self.cid)
        b = p.createMultiBody(0, -1, v, pos,
                              p.getQuaternionFromEuler(orn),
                              physicsClientId=self.cid)
        self.bodies.append(b)
        return b

    def cyl(self, r, h, pos, rgba, orn=(0, 0, 0)):
        v = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                rgbaColor=rgba, physicsClientId=self.cid)
        b = p.createMultiBody(0, -1, v, pos,
                              p.getQuaternionFromEuler(orn),
                              physicsClientId=self.cid)
        self.bodies.append(b)
        return b

    def shot(self, rng, target, dist, yaw, pitch):
        view = p.computeViewMatrixFromYawPitchRoll(
            target, dist, yaw, pitch, 0, 2, physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(
            42, WIDTH / HEIGHT, 0.005, 2.0, physicsClientId=self.cid)
        _, _, rgb, _, _ = p.getCameraImage(
            WIDTH, HEIGHT, view, proj, shadow=1,
            lightDirection=[rng.uniform(-0.6, 0.6), rng.uniform(-0.9, -0.3),
                            rng.uniform(1.4, 2.2)],
            lightColor=[1.0, 0.98, 0.94],
            lightDistance=1.4, lightAmbientCoeff=0.42,
            lightDiffuseCoeff=0.62, lightSpecularCoeff=0.18,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid)
        img = np.reshape(np.array(rgb, dtype=np.uint8),
                         (HEIGHT, WIDTH, 4))[:, :, :3].astype(float)
        img += rng.normal(0, 2.0, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def close(self):
        p.disconnect(physicsClientId=self.cid)


def _dimm(s, x0, z0, tilt, lift, rng):
    zc = z0 + H + lift
    orn = (tilt, 0.0, 0.0)
    s.box([L, T, H], [x0, 0, zc], GREEN, orn=orn)

    for sgn in (-1, 1):
        w = (L - NOTCH_X) / 2.0
        cx = x0 + sgn * (NOTCH_X + w)
        s.box([w, T * 1.25, 0.0022], [cx, 0, zc - H + 0.0024], GOLD, orn=orn,
              spec=(0.5, 0.45, 0.2))

    for i in range(4):
        for sgn in (-1, 1):
            cx = x0 + sgn * (0.014 + i * 0.0135)
            s.box([0.0055, T * 1.9, 0.0038], [cx, 0, zc + 0.002], CHIP,
                  orn=orn)


def _slot(s, x0, z0, populated=True):
    s.box([L + 0.004, 0.0045, 0.0045], [x0, 0, z0 + 0.0045], SLOT)
    s.box([L + 0.004, 0.0014, 0.0016], [x0, 0, z0 + 0.0095],
          SLOT if populated else [0.03, 0.03, 0.04, 1.0])


def _clips(s, x0, z0, open_left=False, open_right=False):
    for sgn, is_open in ((-1, open_left), (1, open_right)):
        cx = x0 + sgn * (L + 0.0075)
        tilt = 0.85 if is_open else 0.0
        s.box([0.0035, 0.006, 0.010], [cx, 0, z0 + 0.012], CLIP,
              orn=(tilt * sgn, 0, 0), spec=(0.4, 0.4, 0.4))


def _caps(s, rng, n=7):
    for _ in range(n):
        x = rng.uniform(-0.105, 0.105)
        y = rng.choice([rng.uniform(-0.075, -0.030), rng.uniform(0.030, 0.075)])
        r = rng.uniform(0.0028, 0.0045)
        h = rng.uniform(0.005, 0.010)
        s.cyl(r, h, [x, y, 0.003 + h / 2], CAP)
        s.cyl(r * 0.95, 0.0006, [x, y, 0.003 + h], CAP_TOP)


def build_pcb(s, tilt=0.0, lift=0.0, condition="normal", rng=None):
    rng = rng or np.random.default_rng(0)
    s.box([0.30, 0.24, 0.004], [0, 0, -0.004], MAT)
    s.box([0.115, 0.085, 0.0015], [0, 0, 0.0015], MOBO)

    _slot(s, 0.0, 0.003, populated=True)
    _dimm(s, 0.0, 0.003, tilt, lift, rng)
    if lift > 0.0005:
        s.box([L * 0.96, 0.0016, 0.0009], [0, 0, 0.0122], GOLD,
              spec=(0.5, 0.45, 0.2))

    if condition == "misleading":
        s.box([L + 0.004, 0.0045, 0.0045], [0, 0.032, 0.0075], SLOT)
        s.box([L + 0.004, 0.0014, 0.0016], [0, 0.032, 0.0125], SLOT)
        zc2 = 0.003 + H
        s.box([L, T, H], [0, 0.032, zc2], GREEN)
        for i in range(4):
            for sgn in (-1, 1):
                s.box([0.0055, T * 1.9, 0.0038],
                      [sgn * (0.014 + i * 0.0135), 0.032, zc2 + 0.002], CHIP)

    _caps(s, rng)

    if condition == "occluded":
        for i in range(9):
            s.box([0.030, 0.0012, 0.016],
                  [0.045, -0.016 + i * 0.0035, 0.022], HEATSINK,
                  orn=(0, 0, 0.22))


def render_pcb(state, condition="normal", seed=0):
    rng = np.random.default_rng(seed)
    s = Scene()
    build_pcb(s, tilt=state.get("tilt", 0.0), lift=state.get("lift", 0.0),
              condition=condition, rng=rng)
    yaw = rng.uniform(38, 52) + (90.0 if condition == "rotated" else 0.0)
    img = s.shot(rng, target=(0.0, 0.0, 0.012), dist=0.175, yaw=yaw,
                 pitch=rng.uniform(-32, -22))
    s.close()
    return img


if __name__ == "__main__":
    from PIL import Image
    os.makedirs("renders", exist_ok=True)
    states = {"normal": {"tilt": 0.0, "lift": 0.0},
              "occluded": {"tilt": 0.0, "lift": 0.0},
              "rotated": {"tilt": 0.0, "lift": 0.0},
              "partial": {"tilt": 0.16, "lift": 0.011},
              "misleading": {"tilt": 0.0, "lift": 0.0}}
    for i, (cond, st) in enumerate(states.items()):
        Image.fromarray(render_pcb(st, condition=cond, seed=i)).save(
            f"renders/pcb_{cond}.png")
        print("wrote", f"renders/pcb_{cond}.png")
