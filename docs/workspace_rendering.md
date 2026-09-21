# B601 workspace figures

The six workspace panels are generated from Python geometry, B601 STL meshes,
and archived DISASM-Bench state frames. Each panel has its own executable script.
PyBullet's CPU TinyRenderer produces the images without a display server or GPU.

All six panels derive moving geometry from the actual analytical environment
objects. The CLI restores observable fields from the archived frame, verifies the
observation against the environment's own `_get_obs()`, and passes that object to
the renderer. There are no display-pose overrides. Static CAD dimensions,
materials, and camera settings are fixed visual assets. Gripper attachments follow
the moving workpiece or tool frame.

The six environment source files and their local dependencies are included;
[environment_sources.json](../assets/workspaces/environment_sources.json) pins
their SHA-256 hashes. These sources are copied without changing their physics.
PyBullet renders the resulting geometry; the frozen analytical models provide
the task dynamics. The viewer does not simulate articulated arm control or
B601 contact dynamics.

## Setup

From the repository root, create an environment with Python 3.10–3.12:

```bash
python3 -m venv .venv-renders
.venv-renders/bin/python -m pip install -r requirements-workspace-renders.txt
```

The B601 model is an external dependency. Obtain the reference revision from
[ReBot Arm Digital Twin RS](https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_RS):

```bash
git clone https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_RS.git ../ReBot_Arm_DigitalTwin_RS
git -C ../ReBot_Arm_DigitalTwin_RS checkout dadefb0d0681501c41e6311ccc09045f836decd6
export TRIX_B601_DESCRIPTION="$PWD/../ReBot_Arm_DigitalTwin_RS/rebotarm_ros2_RS/src/rebotarm_bringup/description"
```

An existing checkout can be used instead. Set `TRIX_B601_DESCRIPTION` to its
`description` directory, or pass `--b601-description` to a renderer. The directory
must contain `urdf/ReBot_Arm_RS.urdf` and `meshes_rs/`. The loader checks the model
against [b601_assets.json](../assets/workspaces/b601_assets.json) before rendering.
The upstream model files are not duplicated in this repository.

Labels use DejaVu Sans and DejaVu Sans Bold. On Linux, the renderer first checks
`/usr/share/fonts/truetype/dejavu`; otherwise it uses Matplotlib's bundled fonts.
Use `--font-dir` or `TRIX_FONT_DIR` to select a directory explicitly. Exact pixel
reproduction requires the reference font files and compatible runtime versions;
these are recorded in [reference_environment.json](../assets/workspaces/reference_environment.json).

## Render each panel

Run these commands from the repository root:

```bash
.venv-renders/bin/python scripts/render_screw_workspace_b601.py
.venv-renders/bin/python scripts/render_pcb_workspace_b601.py
.venv-renders/bin/python scripts/render_snap_workspace_b601.py
.venv-renders/bin/python scripts/render_crank_workspace_b601.py
.venv-renders/bin/python scripts/render_battery_workspace_b601.py
.venv-renders/bin/python scripts/render_pry_workspace_b601.py
```

Each command writes three files into `manuscript/figures/workspaces/`:

- `<task>_workspace.png`: the panel with coordinate graphics and schematic wrench arrows.
- `<task>_workspace_clean.png`: the same scene before coordinate and wrench graphics.
- `<task>_workspace_manifest.json`: archived state, trace identifiers, and visual parameters.

The default input is [frozen_frames.json](../assets/workspaces/frozen_frames.json).
It preserves the original archived frame JSON, including observations, checkpoint
identifiers, and trace SHA-256 values. Original episode traces and training
checkpoints are not required to reproduce these selected views.

| Task | Script | Trace step | Recorded variables shown |
| --- | --- | ---: | --- |
| SCREW | [render_screw_workspace_b601.py](../scripts/render_screw_workspace_b601.py) | 188 | 8.270673 mm extraction; recorded screw angle; 1.25 mm thread pitch |
| PCB | [render_pcb_workspace_b601.py](../scripts/render_pcb_workspace_b601.py) | 133 | 8.312500 mm lift; 0.006588202 rad about long X axis; 0.003539883 rad about Y |
| SNAP | [render_snap_workspace_b601.py](../scripts/render_snap_workspace_b601.py) | 5 | 5.133796 mm latch deflection; 0.808487 mm lid extraction |
| CRANK | [render_crank_workspace_b601.py](../scripts/render_crank_workspace_b601.py) | 199 | Recorded crank angle; −2.000000 mm axial displacement; 100 mm arm radius |
| BATTERY | [render_battery_workspace_b601.py](../scripts/render_battery_workspace_b601.py) | 329 | 6.753832 mm peel lift; intact cell state |
| PRY | [render_pry_workspace_b601.py](../scripts/render_pry_workspace_b601.py) | 576 | 7.237697 mm seam gap; 4.303334 mm insertion; recorded tool angle |

Archived values in this table are rounded for display; calculations use the full
precision in the input JSON. PCB's historical named fields are null, so its
observation components 0, 2 and 3 restore `env.z` and `env.theta`. The earlier
11 mm / 0.16 rad display override has been removed. `PCBEnvV2.THETA_FRAC` is
0.075 rad: an intact 0.16 rad state is rejected, without changing that threshold.

### Render an environment object directly

```python
import sys
sys.path.insert(0, "scripts")
from workspace_environment import prepare_environment
from envs.pcb_env_v2 import PCBEnvV2
from render_pcb_workspace_b601 import render

env = PCBEnvV2(noise_mult=0)
env.reset(seed=123)
for _ in range(12):
    env.step([0.01, -0.01, 0.55])
render(environment=env, output_dir="/tmp/pcb-live")
```

The renderer reads the live object without modifying it. Such manifests use
`source_kind: live_environment` and do not claim an archived trace identity.
The default archive path restores **observable state only**, not missing episode
history or hidden velocities. Do not resume simulation from an archived display
frame. Its state is sufficient for these geometric degrees of freedom.

| Task | Environment fields controlling geometry |
| --- | --- |
| SCREW | `theta`, `z`; model `PITCH` and `RADIUS` |
| PCB | `z`, both `theta` components; clip release from `z >= Z_CLIP` |
| SNAP | `delta` moves the housing hook outward; `z` moves lid/tooth/grasp vertically |
| CRANK | `theta`, `z`; model `RADIUS` |
| BATTERY | `z` moves cell, terminals, extraction tab and grasp together |
| PRY | `state.position[0]` sets insertion, `[2]` sets gap, `state.theta` rotates a rigid tool; model `LEVER_LENGTH` |

Manifests record source hashes, resolved state variables, actual PyBullet body
transforms and observation-roundtrip error. Damaged states with no corresponding
damage-shape model are refused instead of silently showing intact geometry.

All six scripts accept `--output-dir`, `--state-file`, `--b601-description`, and
`--font-dir`. The SCREW, PCB, and CRANK scripts also accept `--debug` to mark their
measured fingertip contact points. For example:

```bash
.venv-renders/bin/python scripts/render_snap_workspace_b601.py --output-dir /tmp/snap-render
.venv-renders/bin/python scripts/render_screw_workspace_b601.py --debug --output-dir /tmp/screw-contact
```

## Compose the six-panel figure

After rendering the individual panels:

```bash
.venv-renders/bin/python scripts/compose_b601_workspaces.py
```

This writes `manuscript/figures/disasm_bench_workspaces.png` and
`manuscript/figures/disasm_bench_workspaces.pdf`. The compositor reads the six
existing panels; it does not regenerate scene geometry. Alternative directories
can be selected with `--workspace-dir` and `--output-dir`.

## Per-panel state manifest

Composition also writes `manuscript/figures/workspaces/state_manifest.json`.
Its `states` and `trace_provenance` fields follow the existing
`renders/tasks/state_manifest.json` format: one entry for each of the six tasks.
Lengths are metres, angles are radians, and temperatures are degrees Celsius.
The PCB curvature is the norm of its two recorded tilt components:
0.007478982481916388 rad (about 0.429 degrees).

SCREW/CRANK `theta` stores orientation modulo 2π; PRY uses the signed remainder.
The full accumulated angles, continuous damage fields and raw observations are
retained in `environment_states` and `recorded_observations`. Each panel records
its letter, renderer path, environment fingerprint, trace seed/step identifiers,
and SHA-256 hashes of its PNGs, renderer and detailed manifest. The input archive
path and hash identify the recorded source.

The compositor regenerates this manifest and rejects discrepancies between the
recorded observation, environment binding, and rendered pose fields. To verify
an existing figure's manifest without modifying it:

```bash
.venv-renders/bin/python scripts/workspace_state_manifest.py --check
```

The manifest can also be regenerated independently with the same command without
`--check`. The caption reports the small PCB curvature numerically; its geometry
is unchanged. SNAP's latch-force label uses the common label styling in a clear
white margin above the base.

## Coordinate graphics

The world frame is right-handed with Z up. X, Y, and Z use red, green, and blue
arrows; the corresponding plane handles show XY, XZ, and YZ. The 20 mm floor grid
is projected with each scene's actual camera and masked behind the geometry.

The reference gizmo is placed at a clear grid intersection beside the fixture.
Its directions are world-aligned; its display origin is offset for visibility and
is **not the simulation origin**. Every manifest records the display origin in
world metres, the three world directions, and the displayed axis length. The
arrows are rendered in a separate PyBullet pass and do not affect scene contacts
or shadows.

The graphic convention follows the B601 simulator's RGB direction arrows and
[Omniverse's translation manipulator](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_viewport/transform-manipulator.html)
with square plane handles.

## Mechanical details and wrench annotations

The PCB bottom edge remains inside the socket mouth at the archived lift. Both
end overlaps are recorded in its manifest. The SCREW root is an explicit radius
mesh, leaving its continuous 1.25 mm helix exposed; its illustrative head diameter
is 16 mm. CRANK uses an explicit 14 mm handle mesh with the grasp at mid-height.
The SCREW, PCB, CRANK, BATTERY, and PRY contacts are found by intersecting the real black
fingertip triangles at the contact height, with a separate point-to-triangle
surface-distance check. CRANK also checks contact against the handle radius.
PRY closes across the orange polymer tool grip with its approach normal to the
handle top, following the unchanged 14.266993-degree tool inclination. Both
contacts are checked against the handle's flat side faces and recorded in its
manifest. The orange element is part of the pry tool, not a support or fixture.

PCB uses a fixed socket mouth at 12.8 mm world height, with its seated lower
edge at 3 mm. Socket geometry does not follow lift or tilt. Engagement and gold
contact exposure are measured outputs; an extracted state is allowed to clear
the connector. SNAP's housing hook stays at its fixed height when the lid moves;
its horizontal overlap clears at the environment's 2 mm deflection threshold.

PRY uses one fixed 150 mm tool rotated by `Ry(-env.state.theta)`. The recorded gap
is the vertical opening of the free lid edge. The opposite edge stays on the
housing rim. With fixed 108 mm lid span, its opening angle is derived as
`asin(gap / 0.108)`; it is not an additional environment variable or display
angle. Toe X follows the recorded insertion and toe Z follows the resulting lid
underside plane. The blade's local mesh never changes shape.
The reduced environment does not constrain a rigid tool against a housing
fulcrum. The renderer therefore does not warp the blade or force heel contact.
That missing contact constraint remains a model limitation.

The fixed fixtures are task-specific: SCREW has a machined metal block with an
M8 tapped bore, matching the model's 1.25 mm pitch and 35 N thread capacity;
PCB is a bare motherboard on three
spacers with sparse routed traces; CRANK has a circular bearing pedestal;
BATTERY sits in a thin device tray; PRY uses its enclosure shell directly.
These CAD materials and mounts do not modify the frozen analytical dynamics.

BATTERY's extraction tab is a fixed folded strip. The gripper closes across its
0.7 mm thickness and contacts the two broad upright faces. Each contact includes
a 2 x 2 mm patch checked against the actual black fingertip STL surfaces. The
cell, tab and gripper all translate by the same `env.z`. The manifest records
contact patch errors, tab-local contacts and the unchanging tab mesh.

SCREW step 188 remains partially withdrawn: the head underside is 8.270673 mm
above the tapped metal fixture, with the remaining shaft still engaged; this was accepted
as equivalent to the requested 8.25 mm presentation.

Purple arrows show schematic wrench directions. Dotted leaders locate their
application points where visible; the SCREW leaders are omitted to keep the
fastener clear. BATTERY's F_peel label sits to the left of the force arrow, clear
of the jaw bracket. Arrow lengths do not represent magnitudes. They are projected
with the scene camera and do not add objects or forces to the archived state.
No measured force or torque samples are available in the saved frames.

| Panel | Schematic load |
| --- | --- |
| SCREW | Axial extraction force Fz and positive axial torque τz |
| PCB | Upward extraction force Fz |
| SNAP | Upward lid pull and outward latch-deflection force |
| CRANK | Tangential handle force Ft and corresponding negative axial torque τz |
| BATTERY | Upward force at the extraction tab |
| PRY | Torque about world −Y, matching the positive model-angle convention |

These load directions illustrate the mechanism and reduced task inputs. They do
not assert that one displayed gripper supplies every independent input, or that
a full robot arm, controller, or contact dynamics have been simulated.

## Verify reproduction

```bash
.venv-renders/bin/python scripts/verify_workspace_renders.py
```

The check renders each panel into a temporary directory and compares both PNGs
pixel-for-pixel with the committed references. It also checks the archived state,
trace step, coordinate convention, wrench metadata, direct PCB observation-to-pose
mapping, and the CRANK contact invariants. It leaves the reference files untouched.
To check one panel, use `--tasks snap`; model and font path options are supported.
Pixel differences on another platform should be investigated against the recorded
runtime, model, and font versions before updating a reference image.

## Verify state-to-geometry behavior

```bash
.venv-renders/bin/python scripts/verify_workspace_state_mapping.py
```

This renders four states per task: the restored frame passed as a live object,
two independent state perturbations, and a fresh environment after real `step()`
calls. It checks actual PyBullet body transforms, unchanged fixture geometry,
changed pixels, and that rendering never mutates the environment. It also checks
SNAP hook/lid independence, a rigid PRY mesh, PCB's two tilt axes, gripper following,
and rejection of the inconsistent intact 0.16 rad PCB state. Synthetic
perturbations are mapping tests, not new benchmark rollouts or paper results.

## Source layout

- `render_<task>_workspace_b601.py`: task geometry, state mapping, camera and grasp.
- `b601_render_common.py`: model verification, temporary URDF resolution, B601 frames and mesh measurements.
- `workspace_render_utils.py`: geometry primitives, measured grasps, fonts and coordinate graphics.
- `workspace_wrench.py`: camera-projected force and torque annotations.
- `workspace_render_cli.py`: shared paths and command-line options.
- `workspace_environment.py`: environment restoration, source checks and state validation.
- `workspace_state_manifest.py`: consolidated states, provenance and file integrity checks.
- `verify_workspace_state_mapping.py`: tests over multiple states and actual rendered bodies.
- `envs/*_env_v*.py`: unmodified analytical environments.
- `compose_b601_workspaces.py`: the publication figure layout.
- `verify_workspace_renders.py`: reproduction checks against saved panels.

All scene lengths use metres, angles use radians unless stated otherwise, and
PyBullet quaternions use `(x, y, z, w)` ordering.
