# B601 workspace figures

The six workspace panels are generated from Python geometry, B601 STL meshes,
and archived DISASM-Bench state frames. Each panel has its own executable script.
PyBullet's CPU TinyRenderer produces the images without a display server or GPU.

These scripts visualize recorded states. They position the fixtures and gripper
geometrically; they do not run a policy, solve arm kinematics, or advance a contact
dynamics simulation. Fixture dimensions and materials are illustrative. Recorded
task variables are retained in the accompanying manifests.

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
| PCB | [render_pcb_workspace_b601.py](../scripts/render_pcb_workspace_b601.py) | 133 | 8.312500 mm lift; tilt from observation entries 2 and 3 |
| SNAP | [render_snap_workspace_b601.py](../scripts/render_snap_workspace_b601.py) | 5 | 5.133796 mm latch deflection; 0.808487 mm lid extraction |
| CRANK | [render_crank_workspace_b601.py](../scripts/render_crank_workspace_b601.py) | 199 | Recorded crank angle; −2.000000 mm axial displacement; 100 mm arm radius |
| BATTERY | [render_battery_workspace_b601.py](../scripts/render_battery_workspace_b601.py) | 329 | 6.753832 mm peel lift; intact cell state |
| PRY | [render_pry_workspace_b601.py](../scripts/render_pry_workspace_b601.py) | 576 | 7.237697 mm seam gap; 4.303334 mm insertion; recorded tool angle |

Values in this table are rounded for display. Calculations use the full precision
in the archived JSON. PCB's named lift and tilt fields are null in that archive;
the renderer uses the stored observation according to `pcb_env_v2`'s ordering.
PRY's lid tilt is derived from the recorded gap and the illustrated enclosure width.

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
The SCREW, PCB, and CRANK contacts are found by intersecting the real black
fingertip triangles at the contact height, with a separate point-to-triangle
surface-distance check. CRANK also checks contact against the handle radius.

Purple arrows show schematic wrench directions. Dotted leaders locate their
application points; arrow lengths do not represent magnitudes. They are projected
with the scene camera and do not add objects or forces to the archived state.
No measured force or torque samples are available in the saved frames.

| Panel | Schematic load |
| --- | --- |
| SCREW | Axial extraction force Fz and positive axial torque τz |
| PCB | Upward extraction force Fz |
| SNAP | Upward lid pull and outward latch-deflection force |
| CRANK | Tangential handle force Ft and corresponding negative axial torque τz |
| BATTERY | Upward force at the extraction tab |
| PRY | Torque about the lever's transverse axis, raising its inward toe |

These load directions illustrate the mechanism and reduced task inputs. They do
not assert that one displayed gripper supplies every independent input, or that
a full robot arm, controller, or contact dynamics have been simulated.

## Verify reproduction

```bash
.venv-renders/bin/python scripts/verify_workspace_renders.py
```

The check renders each panel into a temporary directory and compares both PNGs
pixel-for-pixel with the committed references. It also checks the archived state,
trace step, coordinate convention, wrench metadata, and the PCB/CRANK contact invariants. It leaves the reference files untouched.
To check one panel, use `--tasks snap`; model and font path options are supported.
Pixel differences on another platform should be investigated against the recorded
runtime, model, and font versions before updating a reference image.

## Source layout

- `render_<task>_workspace_b601.py`: task geometry, state mapping, camera and grasp.
- `b601_render_common.py`: model verification, temporary URDF resolution, B601 frames and mesh measurements.
- `workspace_render_utils.py`: geometry primitives, measured grasps, fonts and coordinate graphics.
- `workspace_wrench.py`: camera-projected force and torque annotations.
- `workspace_render_cli.py`: shared paths and command-line options.
- `compose_b601_workspaces.py`: the publication figure layout.
- `verify_workspace_renders.py`: reproduction checks against saved panels.

All scene lengths use metres, angles use radians unless stated otherwise, and
PyBullet quaternions use `(x, y, z, w)` ordering.
