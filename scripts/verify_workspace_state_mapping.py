#!/usr/bin/env python3
"""Regression checks on actual PyBullet bodies across changed environment states.

Synthetic perturbations below test the rendering map, not the physics. Each task
also renders a fresh environment after real step() calls. No archived frame is
used as a resumable episode checkpoint.
"""
import argparse
import copy
import importlib
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p

from workspace_environment import ENVIRONMENTS, prepare_environment, validate_geometry_state
from workspace_render_cli import DEFAULT_PROVENANCE

TASKS = tuple(ENVIRONMENTS)


def near(actual, expected, message, atol=1e-8):
    if not np.allclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"{message}: actual={actual}, expected={expected}")


def rotation(q):
    return np.asarray(p.getMatrixFromQuaternion(q)).reshape(3, 3)


def check_snapshot(task, env, model, manifest):
    v = manifest['visualization']
    bodies = v['rendered_bodies']
    pos = lambda name: np.asarray(bodies[name]['position_m'])
    near(manifest['observation'], env._get_obs(), 'saved observation')
    assert v['environment']['source_kind'] == 'live_environment'
    assert not v['environment']['pose_overrides']
    assert manifest['trace_step'] is None and manifest['trace_sha256'] is None
    if task == 'SCREW':
        near(pos('head'), [0, 0, .014 + env.z + .0025], 'screw head displacement')
        near(v['thread_crest_radius_m'], model.RADIUS, 'screw radius')
        near(v['thread_pitch_m'], model.PITCH, 'screw pitch')
        near(v['grasp']['closing_axis'], [np.cos(env.theta), np.sin(env.theta), 0], 'screw grasp rotation')
    elif task == 'PCB':
        R = rotation(p.getQuaternionFromEuler([*env.theta, 0]))
        near(pos('board'), np.array([0, 0, .003 + env.z]) + R @ [0, 0, .0171], 'PCB board body')
        near(rotation(bodies['board']['quaternion_xyzw']), R, 'PCB both tilt axes')
        near(v['grasp']['approach_axis'], R[:, 2], 'PCB gripper follows tilt')
        near(v['grasp']['target_m'], np.array([0, 0, .003 + env.z]) + R @ [0, 0, .029], 'PCB grasp follows board')
        assert v['retainers_released'] == (env.z >= model.Z_CLIP)
    elif task == 'SNAP':
        near(pos('lid'), [0, 0, .036 + env.z], 'SNAP lid')
        near(pos('tooth'), [.049, -.019, .030 + env.z], 'SNAP tooth')
        near(v['hook_tooth_horizontal_clearance_m'], env.delta-model.DELTA_DISENGAGE, 'SNAP clearance matches threshold')
        near(pos('hook'), [.0559 - model.DELTA_DISENGAGE + env.delta, -.022, .032], 'SNAP housing hook independent of lid')
    elif task == 'CRANK':
        near(pos('handle'), [model.RADIUS*np.cos(env.theta), model.RADIUS*np.sin(env.theta), .0515 + env.z], 'CRANK handle')
        near(v['grasp']['target_m'], pos('handle'), 'CRANK grasp')
    elif task == 'BATTERY':
        near(pos('cell'), [0, 0, .0018 + env.z + .00275], 'BATTERY cell')
        near(pos('tab'), [-.071, 0, .0018 + env.z + .0008], 'BATTERY pull tab')
        near(v['grasp']['target_m'], pos('tab')+[0,0,.016], 'BATTERY gripper follows tab')
        contacts=np.asarray(v['grasp']['contacts_m'])-pos('tab')
        near(np.abs(contacts[:,0]), [v['tab_thickness_m']/2]*2, 'BATTERY broad face contacts', atol=5e-8)
        assert np.max(np.abs(contacts[:,1]))<v['tab_width_m']/2
        assert np.all((contacts[:,2]>.002)&(contacts[:,2]<.020))
        assert np.max(v['tab_contact_patch_surface_error_m'])<.00005
    elif task == 'PRY':
        R = rotation(p.getQuaternionFromEuler([0, -env.state.theta, 0]))
        lid_R=rotation(bodies['lid']['quaternion_xyzw'])
        supported_edge=pos('lid')+lid_R@[-.054,0,-.00225]
        free_edge=pos('lid')+lid_R@[.054,0,-.00225]
        near(supported_edge,[-.054,0,.020],'PRY opposite edge stays supported')
        near(free_edge[2]-supported_edge[2],env.state.position[2],'PRY free edge opens by recorded gap')
        near(np.linalg.norm(free_edge-supported_edge),.108,'PRY lid remains rigid')
        toe=pos('blade')
        near(toe[:2],[.054-env.state.position[0],-.017],'PRY insertion')
        # Actual blade origin must lie on the actual lid underside plane.
        near(np.dot(toe-supported_edge,lid_R[:,2]),0,'PRY toe follows lid underside')
        near(rotation(bodies['blade']['quaternion_xyzw']), R, 'PRY rotation')
        near(pos('handle'), toe + R @ [model.LEVER_LENGTH-.019, 0, -.009], 'PRY fixed tool length')
        near(v['grasp']['approach_axis'], R[:, 2], 'PRY tool-relative grasp')
    return bodies


def perturb(task, env, which):
    if task == 'SCREW':
        if which == 0: env.z += .002
        else: env.theta += .37
    elif task == 'PCB':
        if which == 0: env.z += .003
        else: env.theta[:] = [.032, -.014]
    elif task == 'SNAP':
        if which == 0: env.z += .004
        else: env.delta -= .002
    elif task == 'CRANK':
        if which == 0: env.z += .003
        else: env.theta += .41
    elif task == 'BATTERY':
        env.z += .002 if which == 0 else -.003
    elif task == 'PRY':
        if which == 0:
            env.state.position[2] += .003
            env.state.position[0] += .002
        else:
            env.state.theta += .19


def verify(task, directory):
    renderer = importlib.import_module(f'render_{task.lower()}_workspace_b601')
    env, model, _, binding = prepare_environment(task, DEFAULT_PROVENANCE)
    baseline = copy.deepcopy(env)
    pictures, manifests = [], []
    for index in range(4):
        if index < 3:
            env = copy.deepcopy(baseline)
            if index: perturb(task, env, index-1)
        else:
            env = getattr(model, ENVIRONMENTS[task][1])(noise_mult=0)
            env.reset(seed=123)
            action = {
                'SCREW': [.05, .20943951024, 0], 'PCB': [.01, -.01, .55],
                'SNAP': [.4, .02, 0], 'CRANK': [.3, 0, 0],
                'BATTERY': [.3, 0, 0], 'PRY': [.1, .1, 0],
            }[task]
            for _ in range(12): env.step(np.asarray(action))
            assert np.any(env._get_obs() != 0), 'step produced empty state'
        output = directory / task.lower() / str(index)
        before = pickle.dumps(env.__dict__)
        renderer.render(output_dir=output, environment=env)
        assert before == pickle.dumps(env.__dict__), 'renderer mutated its input environment'
        manifest = json.loads((output/f'{task.lower()}_workspace_manifest.json').read_text())
        check_snapshot(task, env, model, manifest)
        manifests.append(manifest['visualization'])
        pictures.append((output/f'{task.lower()}_workspace_clean.png').read_bytes())
    for index in range(1, 4):
        assert pictures[index] != pictures[0], f'{task}: state changed but rendered pixels did not'
        assert manifests[index]['rendered_bodies']['fixture'] == manifests[0]['rendered_bodies']['fixture'], 'fixture changed with state'
        if task == 'PCB':
            assert manifests[index]['rendered_bodies']['socket'] == manifests[0]['rendered_bodies']['socket'], 'socket changed with state'
            assert manifests[index]['socket_top_m'] == manifests[0]['socket_top_m']
        if task == 'BATTERY':
            assert manifests[index]['tab_local_vertices_m'] == manifests[0]['tab_local_vertices_m'], 'BATTERY pull tab changed shape'
        if task == 'PRY':
            assert manifests[index]['blade_local_vertices_m'] == manifests[0]['blade_local_vertices_m'], 'PRY mesh morphed with state'
    if task == 'SNAP':
        base, lift, delta = (m['rendered_bodies'] for m in manifests[:3])
        assert lift['hook'] == base['hook'], 'lid lift moved housing hook'
        assert delta['lid'] == base['lid'], 'latch deflection moved lid'
        near(np.subtract(lift['gripper']['position_m'],base['gripper']['position_m']),[0,0,.004],'SNAP gripper follows lid', atol=2e-7)
    if task == 'PCB':
        env = copy.deepcopy(baseline)
        env.z, env.theta[:] = .011, [.16, 0]
        try:
            validate_geometry_state(task, env, model)
        except ValueError as error:
            assert 'fracture threshold' in str(error)
        else:
            raise AssertionError('0.16 rad falsely accepted as intact PCB')
    return dict(task=task, status='PASS', rendered_states=4,
                checks=['actual body transforms', 'fixed fixtures', 'changed pixels', 'input object unchanged', 'live step() state'],
                archive_observation_error=binding['max_observation_roundtrip_error'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='trix-state-map-') as temp:
        output = args.output_dir or Path(temp)
        results = [verify(task, output) for task in args.tasks]
        print(json.dumps(results, indent=2))
        if args.output_dir:
            (output/'state_mapping_verification.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
