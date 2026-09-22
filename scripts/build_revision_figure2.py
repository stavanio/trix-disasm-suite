#!/usr/bin/env python3
"""Rebuild Figure 2 from the frozen SCREW v3 constants, without a simulator run.

The projection metric is Euclidean in normalized policy-action coordinates.
The enclosing box is an illustrative representation, not an experimental arm.
Dependencies: numpy and matplotlib. Run from any directory.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trix-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def constants(path):
    required = {"PITCH", "V_CMD_MAX", "OMEGA_CMD_MAX", "EPS_HELIX"}
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in required:
                    values[target.id] = float(ast.literal_eval(node.value))
    if values.keys() != required:
        raise ValueError(f"Missing SCREW constants: {required - values.keys()}")
    return values


def clip_halfplane(polygon, normal, bound):
    output = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        dp, dq = normal @ p - bound, normal @ q - bound
        if dp <= 0:
            output.append(p)
        if (dp <= 0) != (dq <= 0):
            output.append(p + dp / (dp - dq) * (q - p))
    return output


def project_polygon(proposal, polygon, normal, bound):
    if np.all(np.abs(proposal) <= 1) and abs(normal @ proposal) <= bound:
        return proposal.copy()
    candidates = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        edge = q - p
        t = np.clip(np.dot(proposal - p, edge) / np.dot(edge, edge), 0, 1)
        candidates.append(p + t * edge)
    return min(candidates, key=lambda q: np.dot(q - proposal, q - proposal))


def build(environment_file, output_dir):
    c = constants(environment_file)
    k = c["PITCH"] / (2 * np.pi)
    # Display order is (angular action, axial action); policy order is reversed.
    normal = np.array([-k * c["OMEGA_CMD_MAX"], c["V_CMD_MAX"]])
    slope = -normal[0] / normal[1]
    half_width = c["EPS_HELIX"] / c["V_CMD_MAX"]
    enclosing_half_height = slope + half_width
    # This matches screw_filters.INTERIOR, preserving numerical output tolerance.
    target = c["EPS_HELIX"] * (1 - 1e-6)
    polygon = [np.array(v, dtype=float) for v in
               [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    polygon = clip_halfplane(polygon, normal, target)
    polygon = clip_halfplane(polygon, -normal, target)
    proposal = np.array([22 / c["OMEGA_CMD_MAX"], 0.032 / c["V_CMD_MAX"]])
    clipped = np.clip(proposal, [-1, -enclosing_half_height],
                      [1, enclosing_half_height])
    projected = project_polygon(proposal, polygon, normal, target)
    closed_form = proposal - ((normal @ proposal - target) /
                              (normal @ normal)) * normal
    assert np.allclose(projected, closed_form, atol=1e-12)
    assert abs(normal @ clipped) > c["EPS_HELIX"]
    assert abs(normal @ projected) <= target + 1e-14
    assert np.all(np.abs(projected) <= 1)

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "mathtext.fontset": "dejavuserif", "axes.labelsize": 9,
                         "axes.titlesize": 10, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "trix-figure2-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.95))
    teal, orange, ink = "#14645e", "#ac4c16", "#222b34"
    angular = np.linspace(-1, 1, 500)
    for ax in axes:
        ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#f3f4f5",
                               edgecolor="#89929a", linewidth=0.8, zorder=0))
        ax.fill_between(angular, slope * angular - half_width,
                        slope * angular + half_width, color=teal, alpha=0.30,
                        linewidth=0, zorder=2)
        ax.plot(angular, slope * angular, color=teal, linewidth=1.2, zorder=3)
        ax.add_patch(Rectangle((-1, -enclosing_half_height), 2,
                               2 * enclosing_half_height, fill=False,
                               edgecolor=orange, linestyle=(0, (4, 2)),
                               linewidth=1.25, zorder=4))
        ax.axhline(0, color="#c6cbd0", linewidth=0.5, zorder=1)
        ax.axvline(0, color="#c6cbd0", linewidth=0.5, zorder=1)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"Angular command $a_\omega=\omega_z^c/{c['OMEGA_CMD_MAX']:g}$")
        ax.set_ylabel(rf"Axial command $a_v=v_z^c/{c['V_CMD_MAX']:g}$")
        for spine in ax.spines.values():
            spine.set_color("#a8afb5")
        ax.tick_params(width=0.5, length=3, color="#89929a")

    ax = axes[0]
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05),
           xticks=[-1, -0.5, 0, 0.5, 1], yticks=[-1, -0.5, 0, 0.5, 1])
    ax.set_title("(a) Coupling within the command box", loc="left", pad=10)
    ax.text(-0.9, 0.82, rf"$|{normal[1]:g}a_v-{-normal[0]:.5f}a_\omega|\leq{c['EPS_HELIX']:g}$",
            fontsize=8.2, color=ink)
    ax.text(-0.9, -0.84, f"Admissible strip: {100 * half_width:.1f}% of the slice", fontsize=8.2,
            color=teal)
    ax.annotate("Independent bounds\nadmit points outside the strip",
                xy=(-0.74, enclosing_half_height), xytext=(-0.9, 0.56), fontsize=7.9,
                color=orange, arrowprops={"arrowstyle": "-", "lw": 0.8,
                                         "color": orange})

    ax = axes[1]
    ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xticks=np.arange(0, 0.81, 0.2),
           yticks=np.arange(0, 0.81, 0.2))
    ax.set_title("(b) One proposal, two corrections", loc="left", pad=10)
    ax.plot(*proposal, "o", color=ink, markersize=5, zorder=7)
    ax.text(proposal[0] + 0.035, proposal[1] + 0.035, "Proposal", fontsize=8,
            color=ink)
    for endpoint, color in [(clipped, orange), (projected, teal)]:
        ax.annotate("", xy=endpoint, xytext=proposal,
                    arrowprops={"arrowstyle": "->", "color": color,
                                "lw": 1.4, "shrinkA": 5, "shrinkB": 5}, zorder=6)
    ax.plot(*clipped, "s", markersize=5, markerfacecolor="white",
            markeredgecolor=orange, markeredgewidth=1.3, zorder=7)
    ax.plot(*projected, "o", markersize=5, markerfacecolor="white",
            markeredgecolor=teal, markeredgewidth=1.3, zorder=7)
    ax.annotate("Box clip\nstill inadmissible", xy=clipped, xytext=(0.025, 0.39),
                fontsize=8, color=orange,
                arrowprops={"arrowstyle": "-", "color": orange, "lw": 0.7})
    ax.annotate("Coupled projection", xy=projected, xytext=(0.28, 0.04),
                fontsize=8, color=teal,
                arrowprops={"arrowstyle": "-", "color": teal, "lw": 0.7})
    fig.legend(handles=[Patch(facecolor="#f3f4f5", edgecolor="#89929a",
                              label="Command box"),
                        Line2D([], [], color=orange, linestyle="--",
                               label="Tightest enclosing box"),
                        Patch(facecolor=teal, alpha=0.30, label="Coupled admissible strip")],
               loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3,
               frameon=False, fontsize=8, handlelength=1.5, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.86, wspace=0.38)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        metadata = {"Creator": "TRiX Figure 2 generator"}
        if ext == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif ext == "svg":
            metadata["Date"] = None
        fig.savefig(output_dir / f"figure2_representation_gap.{ext}",
                    dpi=240, metadata=metadata, facecolor="white")
    plt.close(fig)
    report = {"environment_source": "envs/screw_env_v3.py",
              "environment_sha256": hashlib.sha256(environment_file.read_bytes()).hexdigest(),
              "constants": c, "radial_action": 0,
              "metric": "Euclidean distance in normalized (angular, axial) action coordinates",
              "illustrative_box_is_experimental_arm": False,
              "coupled_area_fraction": half_width,
              "enclosing_box_half_height": enclosing_half_height,
              "projection_target_helix_error_m_per_s": target,
              "points": {name: {"normalized_angular_axial": point.tolist(),
                         "omega_rad_per_s": float(point[0] * c["OMEGA_CMD_MAX"]),
                         "v_m_per_s": float(point[1] * c["V_CMD_MAX"]),
                         "absolute_helix_error_m_per_s": float(abs(normal @ point))}
                         for name, point in [("proposal", proposal), ("box_clip", clipped),
                                             ("coupled_projection", projected)]}}
    (output_dir / "figure2_geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-file", type=Path, default=ROOT / "envs/screw_env_v3.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript/figures")
    args = parser.parse_args()
    build(args.environment_file, args.output_dir)
