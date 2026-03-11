"""
Generate IEEE-format plots for TRiX paper.

Reads evaluation results from experiments/ and generates:
- Figure 1 (fig:success): Success rate bar chart
- Figure 2 (fig:safety): Safety violations bar chart  
- Figure 3 (fig:learning): Learning curves

Outputs both pgfplots coordinates and standalone PDFs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Tuple

# IEEE formatting
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['legend.fontsize'] = 8

# Column width for IEEE single column
COLUMN_WIDTH = 3.5  # inches
FIGURE_HEIGHT = 2.2  # inches


def load_experiment_results(experiments_dir: str = "experiments") -> Dict:
    """
    Load all experiment results.
    
    Returns dict organized by:
        results[baseline][task][entropy_level][seed] = {eval_results}
    """
    results = {}
    
    # Find all experiment directories
    for exp_dir in Path(experiments_dir).iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ['logs', 'models', 'eval']:
            continue
        
        # Load config
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        
        with open(config_path) as f:
            config = json.load(f)
        
        baseline = config['baseline']
        task = config['task']
        entropy = config['entropy_level']
        seed = config['seed']
        
        # Load evaluation logs
        eval_path = exp_dir / "logs" / "eval.json"
        if not eval_path.exists():
            print(f"Warning: No eval log for {exp_dir.name}")
            continue
        
        with open(eval_path) as f:
            eval_log = json.load(f)
        
        # Organize
        if baseline not in results:
            results[baseline] = {}
        if task not in results[baseline]:
            results[baseline][task] = {}
        if entropy not in results[baseline][task]:
            results[baseline][task][entropy] = {}
        
        results[baseline][task][entropy][seed] = eval_log
    
    return results


def aggregate_final_performance(results: Dict, task: str = "screw", entropy: float = 0.3) -> Dict:
    """
    Aggregate final performance across seeds.
    
    Returns dict: baseline -> {success_rate, safety_violations, etc.}
    """
    aggregated = {}
    
    for baseline in results:
        if task not in results[baseline]:
            continue
        if entropy not in results[baseline][task]:
            continue
        
        # Collect final evaluation across seeds
        final_evals = []
        for seed, eval_log in results[baseline][task][entropy].items():
            if eval_log:  # Get last evaluation
                final_evals.append(eval_log[-1])
        
        if not final_evals:
            continue
        
        # Aggregate metrics
        aggregated[baseline] = {
            'success_rate': np.mean([e['success_rate'] for e in final_evals]),
            'success_rate_std': np.std([e['success_rate'] for e in final_evals]),
            'safety_violations': np.mean([e['safety_violations'] for e in final_evals]),
            'safety_violations_std': np.std([e['safety_violations'] for e in final_evals]),
            'avg_reward': np.mean([e['avg_reward'] for e in final_evals]),
            'avg_reward_std': np.std([e['avg_reward'] for e in final_evals]),
        }
        
        # TRiX-specific metrics
        if baseline == 'trix' and 'avg_gating' in final_evals[0]:
            aggregated[baseline]['avg_gating'] = np.mean([e.get('avg_gating', 1.0) for e in final_evals])
            aggregated[baseline]['chatter_events'] = np.mean([e.get('chatter_events', 0) for e in final_evals])
    
    return aggregated


def extract_learning_curves(results: Dict, baseline: str, task: str = "screw", entropy: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract learning curves (steps, mean success, std).
    """
    if baseline not in results or task not in results[baseline]:
        return np.array([]), np.array([]), np.array([])
    
    if entropy not in results[baseline][task]:
        return np.array([]), np.array([]), np.array([])
    
    # Collect curves across seeds
    all_curves = []
    for seed, eval_log in results[baseline][task][entropy].items():
        steps = [e['step'] for e in eval_log]
        success = [e['success_rate'] for e in eval_log]
        all_curves.append((steps, success))
    
    if not all_curves:
        return np.array([]), np.array([]), np.array([])
    
    # Find common steps (take minimum length)
    min_len = min(len(curve[0]) for curve in all_curves)
    steps = all_curves[0][0][:min_len]
    
    # Average success rates
    success_rates = np.array([curve[1][:min_len] for curve in all_curves])
    mean_success = success_rates.mean(axis=0)
    std_success = success_rates.std(axis=0)
    
    return np.array(steps), mean_success, std_success


def plot_success_rate(aggregated: Dict, output_dir: str):
    """Generate Figure 1: Success rate bar chart."""
    # Baseline ordering (matches paper)
    baseline_order = ['sac', 'cpo', 'shield', 'cbf', 'posthoc', 'trix']
    baseline_labels = ['SAC', 'CPO', 'Shield', 'CBF+SAC', 'PostXAI', 'TRiX']
    
    # Extract data
    success_rates = []
    std_errors = []
    present_labels = []
    
    for bl in baseline_order:
        if bl in aggregated:
            success_rates.append(aggregated[bl]['success_rate'])
            std_errors.append(aggregated[bl]['success_rate_std'])
            present_labels.append(baseline_labels[baseline_order.index(bl)])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIGURE_HEIGHT))
    
    x = np.arange(len(present_labels))
    bars = ax.bar(x, success_rates, yerr=std_errors, capsize=3, 
                   color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Highlight TRiX if present
    if 'trix' in aggregated:
        trix_idx = list(aggregated.keys()).index('trix')
        if trix_idx < len(bars):
            bars[trix_idx].set_color('orange')
    
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(present_labels, rotation=25, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save PDF
    pdf_path = os.path.join(output_dir, 'fig_success.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    
    # Generate pgfplots data
    pgf_path = os.path.join(output_dir, 'fig_success_data.txt')
    with open(pgf_path, 'w') as f:
        f.write("% Replace coordinates in fig:success\n")
        f.write("\\addplot coordinates {")
        for label, rate in zip(present_labels, success_rates):
            f.write(f"({label},{rate:.3f}) ")
        f.write("};\n")
    print(f"Saved: {pgf_path}")
    
    plt.close()


def plot_safety_violations(aggregated: Dict, output_dir: str):
    """Generate Figure 2: Safety violations bar chart."""
    baseline_order = ['sac', 'cpo', 'shield', 'cbf', 'trix']
    baseline_labels = ['SAC', 'CPO', 'Shield', 'CBF+SAC', 'TRiX']
    
    violations = []
    std_errors = []
    present_labels = []
    
    for bl in baseline_order:
        if bl in aggregated:
            violations.append(aggregated[bl]['safety_violations'])
            std_errors.append(aggregated[bl]['safety_violations_std'])
            present_labels.append(baseline_labels[baseline_order.index(bl)])
    
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIGURE_HEIGHT))
    
    x = np.arange(len(present_labels))
    bars = ax.bar(x, violations, yerr=std_errors, capsize=3,
                   color='crimson', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Highlight TRiX
    if 'trix' in aggregated:
        trix_idx = list(aggregated.keys()).index('trix')
        if trix_idx < len(bars):
            bars[trix_idx].set_color('green')
            bars[trix_idx].set_alpha(0.8)
    
    ax.set_ylabel('Safety Violations per 100 Episodes')
    ax.set_xticks(x)
    ax.set_xticklabels(present_labels, rotation=25, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    pdf_path = os.path.join(output_dir, 'fig_safety.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    
    pgf_path = os.path.join(output_dir, 'fig_safety_data.txt')
    with open(pgf_path, 'w') as f:
        f.write("% Replace coordinates in fig:safety\n")
        f.write("\\addplot coordinates {")
        for label, viol in zip(present_labels, violations):
            f.write(f"({label},{viol:.1f}) ")
        f.write("};\n")
    print(f"Saved: {pgf_path}")
    
    plt.close()


def plot_learning_curves(results: Dict, output_dir: str, task: str = "screw", entropy: float = 0.3):
    """Generate Figure 3: Learning curves."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIGURE_HEIGHT))
    
    colors = {'sac': 'blue', 'trix': 'orange', 'cpo': 'green', 'shield': 'purple'}
    
    for baseline in ['sac', 'trix']:  # Focus on main comparison
        if baseline not in results:
            continue
        
        steps, mean, std = extract_learning_curves(results, baseline, task, entropy)
        
        if len(steps) == 0:
            continue
        
        # Convert steps to thousands
        steps_k = steps / 1000
        
        ax.plot(steps_k, mean, label=baseline.upper(), 
                color=colors.get(baseline, 'gray'), linewidth=1.5)
        ax.fill_between(steps_k, mean - std, mean + std, 
                        alpha=0.2, color=colors.get(baseline, 'gray'))
    
    ax.set_xlabel('Training Steps (×1000)')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    pdf_path = os.path.join(output_dir, 'fig_learning.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    
    # Generate pgfplots data
    pgf_path = os.path.join(output_dir, 'fig_learning_data.txt')
    with open(pgf_path, 'w') as f:
        f.write("% Replace coordinates in fig:learning\n")
        for baseline in ['sac', 'trix']:
            if baseline not in results:
                continue
            steps, mean, _ = extract_learning_curves(results, baseline, task, entropy)
            if len(steps) == 0:
                continue
            
            f.write(f"% {baseline.upper()}\n")
            f.write("\\addplot coordinates {")
            for s, m in zip(steps, mean):
                f.write(f"({s},{m:.3f}) ")
            f.write("};\n")
    print(f"Saved: {pgf_path}")
    
    plt.close()


def generate_latex_table(aggregated: Dict, output_dir: str):
    """Generate LaTeX table with all metrics."""
    baseline_order = ['sac', 'cpo', 'shield', 'cbf', 'posthoc', 'trix']
    baseline_labels = ['SAC', 'CPO', 'Shielded SAC', 'CBF-QP + SAC', 'Post-hoc XAI', 'TRiX']
    
    table = []
    table.append("\\begin{table}[t]")
    table.append("\\centering")
    table.append("\\caption{DISASM-Bench results on screw proxy task (entropy=0.3).}")
    table.append("\\label{tab:results}")
    table.append("\\begin{tabular}{l c c c}")
    table.append("\\toprule")
    table.append("Method & Success ↑ & Safety Violations ↓ & Avg Reward ↑ \\\\")
    table.append("\\midrule")
    
    for bl, label in zip(baseline_order, baseline_labels):
        if bl not in aggregated:
            continue
        
        data = aggregated[bl]
        success = f"{data['success_rate']:.2f} ± {data['success_rate_std']:.2f}"
        violations = f"{data['safety_violations']:.1f} ± {data['safety_violations_std']:.1f}"
        reward = f"{data['avg_reward']:.1f} ± {data['avg_reward_std']:.1f}"
        
        if bl == 'trix':
            table.append(f"\\textbf{{{label}}} & \\textbf{{{success}}} & \\textbf{{{violations}}} & \\textbf{{{reward}}} \\\\")
        else:
            table.append(f"{label} & {success} & {violations} & {reward} \\\\")
    
    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    
    latex_path = os.path.join(output_dir, 'table_results.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(table))
    print(f"Saved: {latex_path}")


def main():
    """Generate all plots and tables."""
    print("="*60)
    print("Generating IEEE-format plots for TRiX paper")
    print("="*60)
    
    # Load results
    print("\nLoading experiment results...")
    results = load_experiment_results("experiments")
    
    if not results:
        print("ERROR: No experiment results found!")
        print("Run training first: python -m training.train_all")
        return
    
    print(f"Found results for baselines: {list(results.keys())}")
    
    # Create output directory
    output_dir = "experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate performance
    print("\nAggregating final performance...")
    aggregated = aggregate_final_performance(results, task="screw", entropy=0.3)
    
    # Generate plots
    print("\nGenerating figures...")
    plot_success_rate(aggregated, output_dir)
    plot_safety_violations(aggregated, output_dir)
    plot_learning_curves(results, output_dir, task="screw", entropy=0.3)
    
    # Generate table
    print("\nGenerating LaTeX table...")
    generate_latex_table(aggregated, output_dir)
    
    print("\n" + "="*60)
    print("All plots generated!")
    print(f"Output directory: {output_dir}")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy pgf data files to your LaTeX project")
    print("2. Replace coordinates in fig:success, fig:safety, fig:learning")
    print("3. Include table_results.tex in your paper")
    print("\nFiles generated:")
    print(f"  - {output_dir}/fig_success.pdf")
    print(f"  - {output_dir}/fig_success_data.txt")
    print(f"  - {output_dir}/fig_safety.pdf")
    print(f"  - {output_dir}/fig_safety_data.txt")
    print(f"  - {output_dir}/fig_learning.pdf")
    print(f"  - {output_dir}/fig_learning_data.txt")
    print(f"  - {output_dir}/table_results.tex")


if __name__ == "__main__":
    main()
