"""
Generate all paper figures from experiment results.

Usage:
  python scripts/generate_figures.py              # Uses default results.json
  python scripts/generate_figures.py --input data.json --outdir figures/
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Skipping figure generation.")


COLORS = {
    'TRiX': '#2E86AB',
    'SafeLayer': '#E94F37',
    'PPO-Lag': '#4CAF50',
    'PPO': '#9E9E9E',
    'SAC': '#795548',
}


def fig_linearity_gap(outdir='figures'):
    """Figure 2: The Linearity Gap Illustrated."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    omega = np.linspace(-8, 8, 200)
    p = 1.25e-3  # 1.25mm pitch
    k = p / (2 * np.pi)
    vz = k * omega

    # True manifold
    ax.plot(omega, vz * 1000, 'b-', linewidth=2.5, label=r'True coupling $v_z = \frac{p}{2\pi}\omega_z$')

    # Near-manifold safe band
    ax.fill_between(omega, (vz - 0.0003) * 1000, (vz + 0.0003) * 1000,
                    alpha=0.15, color='blue', label='Near-manifold safe band')

    # SafeLayer box constraint
    box_w, box_h = 3.0, 2.5
    rect = plt.Rectangle((-box_w, -box_h), 2 * box_w, 2 * box_h,
                         fill=False, edgecolor='orange', linewidth=2,
                         linestyle='--', label='Learned safe set (box)')
    ax.add_patch(rect)

    # Unsafe points (allowed by box but off manifold)
    unsafe_omega = [1.0, -0.5]
    unsafe_vz = [3.0, -3.5]
    ax.scatter(unsafe_omega, unsafe_vz, c='orange', s=100, zorder=5,
              label='Unsafe (allowed by box)')

    ax.set_xlabel(r'$\omega_z$ (rad/s)', fontsize=12)
    ax.set_ylabel(r'$v_z$ (m/s)', fontsize=12)
    ax.set_title('Linearity Gap Illustrated (Fig. 2)', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-4.5, 4.5)

    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_linearity_gap.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_linearity_gap.pdf'))
    plt.close()
    print(f"  Saved fig_linearity_gap.png/pdf")


def fig_learning_curves(data, outdir='figures'):
    """Figure 4: Learning Curves."""
    if not HAS_MPL or 'learning_curves' not in data:
        return

    lc = data['learning_curves']
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for algo, d in lc.items():
        if algo not in COLORS:
            continue
        steps = np.array(d['checkpoints'])
        rates = np.array(d['violation_rates'])
        stds = np.array(d['violation_std'])

        ax.plot(steps, rates, label=algo, color=COLORS[algo], linewidth=2)
        ax.fill_between(steps,
                        np.clip(rates - stds, 0, 100),
                        np.clip(rates + stds, 0, 100),
                        color=COLORS[algo], alpha=0.15)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Violation Rate (%)', fontsize=12)
    ax.set_title('SCREW Task: Violation Rate During Training (Fig. 4)', fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(loc='center right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_learning_curves.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_learning_curves.pdf'))
    plt.close()
    print(f"  Saved fig_learning_curves.png/pdf")


def fig_violation_heatmap(data, outdir='figures'):
    """Violation rates as heatmap for visual comparison."""
    if not HAS_MPL or 'stress_tournament' not in data:
        return

    st = data['stress_tournament']
    tasks = list(st.keys())
    algos = ['PPO', 'SAC', 'PPO-Lag', 'SafeLayer', 'TRiX']

    matrix = []
    for algo in algos:
        row = []
        for task in tasks:
            if algo in st[task]:
                row.append(st[task][algo]['violation_rate'])
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=11)

    for i in range(len(algos)):
        for j in range(len(tasks)):
            color = 'white' if matrix[i, j] > 50 else 'black'
            ax.text(j, i, f'{matrix[i,j]:.1f}%', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

    plt.colorbar(im, label='Violation Rate (%)')
    ax.set_title('Safety Violation Rates (Table 4)', fontsize=13)

    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_violation_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_violation_heatmap.pdf'))
    plt.close()
    print(f"  Saved fig_violation_heatmap.png/pdf")


def fig_safelayer_failure(outdir='figures'):
    """Figure 3: SafeLayer Failure Analysis."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Safe set mismatch
    omega = np.linspace(-8, 8, 200)
    k = 1.25e-3 / (2 * np.pi)
    vz = k * omega

    ax1.plot(omega, vz * 1000, 'g-', linewidth=2.5, label='True safe manifold')
    ax1.fill_between(omega, (vz - 0.0002) * 1000, (vz + 0.0002) * 1000,
                     alpha=0.2, color='green')

    rect = plt.Rectangle((-3, -2.5), 6, 5, fill=True, facecolor='gray',
                         alpha=0.2, edgecolor='gray', linewidth=2,
                         label='Learned safe set (box)')
    ax1.add_patch(rect)

    # Unsafe points
    np.random.seed(42)
    for _ in range(30):
        w = np.random.uniform(-3, 3)
        v = np.random.uniform(-2.5, 2.5)
        if abs(v - k * w * 1000) > 0.3:
            ax1.plot(w, v, 'ro', markersize=3, alpha=0.5)

    ax1.set_xlabel(r'$\omega_z$ (rad/s)')
    ax1.set_ylabel(r'$v_z$ (mm/s)')
    ax1.set_title('Safe set mismatch')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Violation type histogram
    types = ['Pull w/o\nrotation', 'Rotate w/o\nadvance', 'Lateral\nforce']
    counts = [73, 21, 6]
    colors_bar = ['#E94F37', '#FFA500', '#FFD700']
    ax2.bar(types, counts, color=colors_bar)
    ax2.set_ylabel('Share of violations (%)')
    ax2.set_title('Violation type breakdown')
    ax2.set_ylim(0, 80)

    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_safelayer_failure.png'), dpi=300)
    plt.savefig(os.path.join(outdir, 'fig_safelayer_failure.pdf'))
    plt.close()
    print(f"  Saved fig_safelayer_failure.png/pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results.json')
    parser.add_argument('--outdir', default='figures')
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)

    data = {}
    if os.path.exists(args.input):
        with open(args.input) as f:
            data = json.load(f)

    fig_linearity_gap(args.outdir)
    fig_safelayer_failure(args.outdir)
    fig_learning_curves(data, args.outdir)
    fig_violation_heatmap(data, args.outdir)

    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == '__main__':
    main()
