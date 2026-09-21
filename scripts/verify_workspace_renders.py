#!/usr/bin/env python3
"""Regenerate workspace panels and compare them with the committed images."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import b601_render_common as b601
import numpy as np
from PIL import Image
from workspace_render_cli import DEFAULT_OUTPUT_DIR, DEFAULT_PROVENANCE, ROOT

TASKS = ("screw", "pcb", "snap", "crank", "battery", "pry")


def verify(tasks, output_dir, asset_dir=None, font_dir=None):
    frames = json.loads(DEFAULT_PROVENANCE.read_text())["states"]
    failures = []
    b601.configure_assets(asset_dir)
    b601.verify_assets()
    with tempfile.TemporaryDirectory(prefix="trix-render-check-") as directory:
        temporary = Path(directory)
        for task in tasks:
            command = [
                sys.executable,
                str(ROOT / f"scripts/render_{task}_workspace_b601.py"),
                "--output-dir",
                str(temporary),
                "--state-file",
                str(DEFAULT_PROVENANCE),
            ]
            if asset_dir:
                command += ["--b601-description", str(asset_dir)]
            if font_dir:
                command += ["--font-dir", str(font_dir)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                failures.append(f"{task}: renderer failed\n{result.stdout}{result.stderr}")
                continue
            for suffix in (".png", "_clean.png"):
                filename = f"{task}_workspace{suffix}"
                with Image.open(output_dir / filename) as saved:
                    expected = np.asarray(saved.convert("RGB"))
                with Image.open(temporary / filename) as fresh:
                    actual = np.asarray(fresh.convert("RGB"))
                if expected.shape != actual.shape or not np.array_equal(expected, actual):
                    failures.append(f"{task}: image differs from reference: {filename}")
            manifest = json.loads((temporary / f"{task}_workspace_manifest.json").read_text())
            frame = frames[task.upper()]
            if (
                manifest["state"] != frame["frame"]["state"]
                or manifest["trace_step"] != frame["trace_step"]
            ):
                failures.append(f"{task}: archived state or trace step changed")
            coordinates = manifest["visualization"]
            binding = coordinates["environment"]
            if binding["source_kind"] != "archived_observation_restored_to_environment" or binding["pose_overrides"]:
                failures.append(f"{task}: expected environment state with no pose override")
            if (
                coordinates["grid_spacing_m"] != 0.020
                or coordinates["display_frame_axes_world"] != np.eye(3).tolist()
                or coordinates["display_frame_is_simulation_origin"]
            ):
                failures.append(f"{task}: coordinate-frame convention changed")
            if (
                not coordinates.get("wrench_annotations")
                or coordinates.get("wrench_values_measured") is not False
            ):
                failures.append(f"{task}: missing schematic wrench metadata")
            grasp = coordinates.get("grasp", {})
            if grasp and max(grasp["contact_surface_distance_m"]) > 0.00005:
                failures.append(f"{task}: contact is outside the fingertip STL")
            if task == "pcb":
                pose = [coordinates["lift_m"], coordinates["tilt_x_rad"], coordinates["tilt_y_rad"]]
                obs = frame["frame"]["obs"]
                if not np.allclose(pose, [obs[0], obs[2], obs[3]], rtol=0, atol=1e-12):
                    failures.append("pcb: rendered lift/tilts disagree with the environment observation")
            if task == "crank":
                center = np.asarray(coordinates["handle_center_m"])
                radius = coordinates["handle_diameter_m"] / 2
                contacts = np.asarray(grasp["contacts_m"])
                if (
                    np.max(np.abs(np.linalg.norm((contacts - center)[:, :2], axis=1) - radius))
                    > 0.0001
                ):
                    failures.append("crank: fingers do not meet the handle circumference")
            if task == "screw" and coordinates["thread_pitch_m"] != 0.00125:
                failures.append("screw: thread pitch changed")
            if not any(message.startswith(f"{task}:") for message in failures):
                print(
                    f"PASS {task.upper()}: annotated PNG, clean PNG, archived state, XYZ frame, wrench annotations, contacts"
                )
    if failures:
        raise RuntimeError("\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--b601-description", type=Path)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    verify(args.tasks, args.reference_dir, args.b601_description, args.font_dir)


if __name__ == "__main__":
    main()
