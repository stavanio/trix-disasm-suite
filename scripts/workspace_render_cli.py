"""Command-line options shared by the six workspace renderers."""

import argparse
import os
from pathlib import Path

import b601_render_common as b601

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "manuscript/figures/workspaces"
DEFAULT_PROVENANCE = ROOT / "assets/workspaces/frozen_frames.json"


def run_renderer(render, task, *, debug=False):
    parser = argparse.ArgumentParser(
        description=f"Render the archived {task} workspace with the B601 gripper."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the PNG and state manifest",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_PROVENANCE,
        help="archived workspace frame JSON",
    )
    parser.add_argument(
        "--b601-description",
        type=Path,
        help="B601 description directory containing urdf/ and meshes_rs/; "
        "defaults to TRIX_B601_DESCRIPTION or ~/ReBot_Arm_DigitalTwin_RS",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        help="directory containing DejaVuSans.ttf and DejaVuSans-Bold.ttf",
    )
    if debug:
        parser.add_argument("--debug", action="store_true", help="mark fingertip contacts")
    args = parser.parse_args()
    if not args.state_file.is_file():
        parser.error(f"state file does not exist: {args.state_file}")
    b601.configure_assets(args.b601_description)
    if args.font_dir:
        os.environ["TRIX_FONT_DIR"] = str(args.font_dir.expanduser().resolve())
    options = dict(
        output_dir=args.output_dir.expanduser().resolve(),
        provenance_path=args.state_file.expanduser().resolve(),
    )
    if debug:
        options["debug"] = args.debug
    try:
        render(**options)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{task}: {error}\n")
