"""Bind a renderer to an actual analytical environment's observable state.

Archived observations restore only the fields observable in that frame. They are
not full checkpoints: unobserved episode history must never be resumed from them.
Passing environment=env renders the caller's live object without changing it.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENVIRONMENTS = {
    "SCREW": ("screw_env_v3", "ScrewEnvV3"),
    "PCB": ("pcb_env_v2", "PCBEnvV2"),
    "SNAP": ("snap_env_v2", "SnapEnvV2"),
    "CRANK": ("crank_env_v2", "CrankEnvV2"),
    "BATTERY": ("battery_env_v2", "BatteryEnvV2"),
    "PRY": ("pry_env_v2", "PryEnvV2"),
}


def verify_environment_sources():
    expected = json.loads((ROOT / "assets/workspaces/environment_sources.json").read_text())
    for path, digest in expected["files"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Environment source changed: {path}; review and update the source record")


def state_variables(task, env):
    """Full-precision render inputs read from the live environment, never a display pose."""
    if task == "SCREW":
        return dict(theta_rad=float(env.theta), z_m=float(env.z), omega_rad_s=float(env.omega),
                    v_z_m_s=float(env.v_z), engagement=float(env.engagement), helix_error=env.helix_error())
    if task == "PCB":
        return dict(lift_m=float(env.z), tilt_x_rad=float(env.theta[0]), tilt_y_rad=float(env.theta[1]),
                    v_z_m_s=float(env.v_z), omega_rad_s=env.omega.tolist(),
                    damage=float(env.damage), fractured=bool(env.fractured))
    if task == "SNAP":
        return dict(delta_m=float(env.delta), delta_dot_m_s=float(env.delta_dot), z_m=float(env.z),
                    v_z_m_s=float(env.v_z), released=bool(env.released), latch_damage=float(env.latch_damage),
                    latch_broken=bool(env.latch_broken), phase_release_or_later=env.disengaged())
    if task == "CRANK":
        return dict(theta_rad=float(env.theta), omega_rad_s=float(env.omega), z_m=float(env.z),
                    v_z_m_s=float(env.v_z), damage=float(env.damage), rotation_complete=env.rotation_complete())
    if task == "BATTERY":
        return dict(z_m=float(env.z), v_z_m_s=float(env.v_z), deformation=float(env.deform),
                    short_state=float(env.short), temperature_C=float(env.temperature), short_flag=bool(env.short > 0))
    return dict(theta_rad=float(env.state.theta), omega_rad_s=float(env.state.omega),
                gap_m=float(env.state.position[2]), insertion_depth_m=float(env.state.position[0]),
                friction_state=float(env.state.friction.value), crack=float(env.crack))


def restore_observation(task, env, obs, module):
    """Inverse of the environment's _get_obs for its observable fields only."""
    o = np.asarray(obs, dtype=float)
    if o.shape != (10,) or not np.all(np.isfinite(o)):
        raise ValueError(f"{task}: expected ten finite observation values")
    if task == "SCREW":
        env.theta, env.z, env.omega, env.v_z, env.engagement = map(float, o[:5])
    elif task == "PCB":
        env.z, env.v_z = map(float, o[:2])
        env.theta, env.omega = o[2:4].copy(), o[4:6].copy()
        env.damage, env.fractured = float(o[6]), bool(o[7])
    elif task == "SNAP":
        env.delta, env.delta_dot, env.z, env.v_z = map(float, o[:4])
        env.released, env.latch_damage, env.latch_broken = bool(o[4]), float(o[5]), bool(o[6])
    elif task == "CRANK":
        env.theta, env.omega, env.z, env.v_z, env.damage = map(float, o[[0, 1, 4, 5, 6]])
        env.damaged = env.damage >= module.DAMAGE_LIMIT
    elif task == "BATTERY":
        env.z, env.v_z, env.deform, env.short = map(float, o[:4])
        env.temperature = float(o[4]) * 100.0
        env.peak_temperature = env.temperature  # historical peak is not observed
    elif task == "PRY":
        from benchmark.disasm_bench import FrictionState
        env.state.theta, env.state.omega = map(float, o[:2])
        env.state.position[2], env.state.position[0] = o[2], o[4]
        env.state.friction = FrictionState(int(o[3]))
        env.crack = float(o[6])
        env.cracked = env.crack >= module.CRACK_LIMIT
    reconstructed = np.asarray(env._get_obs(), dtype=float)
    # Derived fields were rounded independently in the archived float32 vector.
    if not np.allclose(reconstructed, o, rtol=1e-6, atol=5e-7):
        raise ValueError(f"{task}: observation disagrees with environment-derived fields: {reconstructed - o}")
    return float(np.max(np.abs(reconstructed - o)))


def validate_geometry_state(task, env, module):
    """Do not silently render an intact body for a failed/inconsistent state."""
    if not np.all(np.isfinite(env._get_obs())):
        raise ValueError(f"{task}: non-finite environment state")
    if task == "PCB" and not env.fractured and (env.curvature() > module.THETA_FRAC or env.damage >= module.DAMAGE_FRAC):
        raise ValueError(f"PCB: tilt {env.curvature():.6g} rad exceeds the {module.THETA_FRAC} rad fracture threshold; cannot label it intact")
    unsupported = {
        "SCREW": task == "SCREW" and env.engagement < module.ENGAGE_INTACT,
        "PCB": task == "PCB" and env.fractured,
        "SNAP": task == "SNAP" and (env.latch_broken or env.delta > module.DELTA_MAX or env.latch_damage >= module.DAMAGE_BREAK),
        "CRANK": task == "CRANK" and env.damaged,
        "BATTERY": task == "BATTERY" and (env.deform > module.D_INTACT or env.short > 0 or env.temperature >= module.T_CRIT),
        "PRY": task == "PRY" and env.cracked,
    }
    if unsupported[task]:
        raise ValueError(f"{task}: damaged state has no damage-shape model; intact mesh rendering is refused")


def prepare_environment(task, provenance_path, environment=None):
    verify_environment_sources()
    name, class_name = ENVIRONMENTS[task]
    module = importlib.import_module(f"envs.{name}")
    cls = getattr(module, class_name)
    error = None
    if environment is None:
        entry = json.loads(Path(provenance_path).read_text())["states"][task]
        rng = np.random.get_state()
        try:
            environment = cls()
            environment.reset(seed=entry["episode_seed"])
        finally:
            np.random.set_state(rng)
        error = restore_observation(task, environment, entry["frame"]["obs"], module)
        # Redundant named fields must not disagree with the authoritative observation.
        resolved = state_variables(task, environment)
        for key, value in entry["frame"]["state"].items():
            if key in resolved and value is not None and not np.allclose(value, resolved[key], rtol=1e-6, atol=5e-7):
                raise ValueError(f"{task}: named state field {key} disagrees with observation")
        kind = "archived_observation_restored_to_environment"
    else:
        if not isinstance(environment, cls):
            raise TypeError(f"{task} requires {class_name}, got {type(environment).__name__}")
        entry = {key: None for key in ("trace_file", "trace_sha256", "trace_step", "episode_seed", "training_seed", "checkpoint_step")}
        entry["frame"] = dict(obs=environment._get_obs().tolist(), state=state_variables(task, environment))
        kind = "live_environment"
    validate_geometry_state(task, environment, module)
    binding = dict(
        source_kind=kind,
        environment_class=f"{module.__name__}.{class_name}",
        environment_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        constraint_hash=module.CONSTRAINT_HASH,
        state_variables=state_variables(task, environment),
        max_observation_roundtrip_error=error,
        resumable_episode_checkpoint=False,
        pose_overrides=False,
    )
    return environment, module, entry, binding


def body_geometry(cid, **bodies):
    """Read back actual PyBullet bodies used in the image, for state-mapping tests."""
    import pybullet as p
    result = {}
    for name, body in bodies.items():
        position, quaternion = p.getBasePositionAndOrientation(body, physicsClientId=cid)
        shapes = p.getVisualShapeData(body, physicsClientId=cid)
        result[name] = dict(position_m=list(position), quaternion_xyzw=list(quaternion),
                            shape_dimensions=[list(shape[3]) for shape in shapes])
    return result
