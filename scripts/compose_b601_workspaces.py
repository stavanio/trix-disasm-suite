#!/usr/bin/env python3
"""Compose the six existing B601 PNGs without re-rendering archived scenes.

Run each render_<task>_workspace_b601.py first when its scene changes.
This compositor deliberately reads the dedicated B601 output directory.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trix-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from workspace_state_manifest import write_manifest
from workspace_render_cli import DEFAULT_PROVENANCE

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "manuscript" / "figures"
TASKS = ("SCREW", "PCB", "SNAP", "CRANK", "BATTERY", "PRY")


def compose(workspace_dir=FIGURES / "workspaces", output_dir=FIGURES, provenance_path=DEFAULT_PROVENANCE):
    workspace_dir = Path(workspace_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(workspace_dir, provenance_path)
    print(manifest_path)
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), facecolor="white")
    for letter, task, axis in zip("abcdef", TASKS, axes.flat):
        path = workspace_dir / f"{task.lower()}_workspace.png"
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != (1800, 1400):
                raise ValueError(f"Unexpected source size for {task}: {image.size}")
            axis.imshow(image, interpolation="lanczos")
        axis.set_title(task, fontsize=11, fontweight="bold", pad=5)
        axis.text(
            0.025,
            0.955,
            f"({letter})",
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
        )
        axis.axis("off")
    figure.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.7)
    for extension in ("png", "pdf"):
        output = output_dir / f"disasm_bench_workspaces.{extension}"
        kwargs = {}
        if extension == "pdf":
            kwargs["metadata"] = dict(
                Title="DISASM-Bench simulation workspace snapshots",
                Subject="Benchmark workspaces in PyBullet with B601 grippers; geometry driven by archived analytical environment states",
                Creator="TRiX B601 workspace compositor",
            )
        figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white", **kwargs)
        print(output)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", type=Path, default=FIGURES / "workspaces")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()
    compose(args.workspace_dir, args.output_dir, args.state_file)
