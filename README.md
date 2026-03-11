# TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **TRiX** (Transparent Real-time eXplainable Control), a neuro-symbolic safety governor that projects neural actions onto task-specific differentiable manifolds for safe robotic disassembly.

> **Paper:** *TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly*
> Submitted to Nature Machine Intelligence (2026)

---

## Key Results

TRiX reduces safety violations by **84.4 percentage points** on helical (screw) tasks compared to linear shielding, while maintaining **98.2% task success**.

| Method | SCREW | PRY | CRANK | PCB | SNAP | BATT | Avg |
|--------|-------|-----|-------|-----|------|------|-----|
| PPO | 96.4 | 81.8 | 83.3 | 85.6 | 93.2 | 48.3 | 81.4 |
| SAC | 98.6 | 92.0 | 92.6 | 93.3 | 97.3 | 56.9 | 88.5 |
| PPO-Lag | 91.7 | 56.6 | 60.9 | 72.0 | 84.1 | 22.2 | 64.6 |
| SafeLayer | 86.8 | 3.4 | 47.6 | 74.5 | 81.6 | 29.2 | 53.9 |
| **TRiX** | **2.4** | **0.0** | **8.7** | **27.1** | **59.6** | **29.1** | **21.2** |

## Quick Start

### Requirements

```bash
pip install numpy matplotlib scipy
```

No GPU required. No external physics engine required. All simulations are self-contained NumPy physics validated against analytical solutions (Appendix G of paper).

### Run All Experiments (~5 minutes)

```bash
python -m experiments.run_all
```

### Quick Verification (~30 seconds)

```bash
python -m experiments.run_all --quick
```

### Reproduce Specific Experiments

```bash
# Table 4: Safety violation rates (stress tournament)
python -m experiments.stress_tournament --steps 100000 --seeds 3

# Figure 4: Learning curves during training
python -m experiments.learning_curves --steps 500000 --seeds 3

# Table 7: Grounding Success Rate (hallucination injection)
python -m experiments.hallucination_injection --trials 10000

# Table 6: Thermal latency analysis
python -m experiments.thermal_latency

# Table 3: Computational overhead measurement
python -m experiments.computational_overhead --iterations 100000

# Table XII (Appendix): Physics validation
python -m experiments.physics_validation

# Pitch estimation ablation study
python -m experiments.pitch_ablation --steps 100000 --seeds 3
```

## Repository Structure

```
trix-disasm-bench/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── envs/                          # DISASM-Bench environments
│   ├── __init__.py
│   ├── base_env.py                # Base environment class
│   ├── screw_env.py               # Helical constraint (M8 screw)
│   ├── battery_env.py             # Thermal constraint (Li-ion)
│   ├── pcb_env.py                 # Planar constraint (FR-4)
│   ├── snap_env.py                # Sequential constraint
│   ├── pry_env.py                 # Path constraint
│   └── crank_env.py               # Hybrid kinematic
├── trix/                          # TRiX safety governor
│   ├── __init__.py
│   ├── governor.py                # Core manifold projections
│   └── manifolds.py               # Manifold definitions
├── baselines/                     # Baseline safety methods
│   ├── __init__.py
│   ├── ppo.py
│   ├── sac.py
│   ├── ppo_lagrangian.py
│   └── safe_layer.py
├── experiments/                   # Reproducible experiments
│   ├── __init__.py
│   ├── run_all.py                 # Run complete experiment suite
│   ├── stress_tournament.py       # Table 4
│   ├── learning_curves.py         # Figure 4
│   ├── hallucination_injection.py # Table 7
│   ├── thermal_latency.py         # Table 6
│   ├── computational_overhead.py  # Table 3
│   ├── physics_validation.py      # Appendix G
│   └── pitch_ablation.py          # Ablation study
├── configs/
│   └── default.yaml               # Hyperparameters (Table 11)
├── scripts/
│   ├── generate_figures.py        # All paper figures
│   └── generate_tables.py         # LaTeX table generation
├── tests/
│   ├── test_manifolds.py          # Manifold projection unit tests
│   ├── test_envs.py               # Environment sanity checks
│   └── test_reproducibility.py    # Deterministic seed tests
└── figures/                       # Generated figures (after running)
```

## Design Philosophy

### Why Self-Contained NumPy Physics?

TRiX's core claim is about the **geometry of safety constraints**, not about high-fidelity rendering. Our simulation:

1. **Validates against analytical solutions** (helical motion <2%, thermal diffusion <1.5%)
2. **Enables exact reproducibility** (deterministic with fixed seeds, no external engine variability)
3. **Requires zero setup** (no GPU, no PyBullet compilation, no URDF assets)

Cross-simulator validation (PyBullet, Isaac Gym) confirms that while absolute violation rates differ by up to 15%, the **relative ranking of methods is preserved** (Appendix G).

### Simulation Constants

All physical parameters are derived from engineering specifications:

| Parameter | Value | Source |
|-----------|-------|--------|
| Integration Rate | 240 Hz | Contact physics standard |
| Force Noise (σ_F) | 1.235 N | ATI Mini45 + actuator RSS |
| Torque Noise (σ_τ) | 0.039 Nm | ATI Mini45 + actuator RSS |
| Static Friction (μ_s) | 0.61 | Steel/aluminum tables |
| Kinetic Friction (μ_k) | 0.47 | Steel/aluminum tables |
| Battery Puncture Limit | 20 N | Cell rupture threshold |
| Battery Temp Limit | 60°C | Thermal runaway onset |

## Citation

```bibtex
@article{dholakia2026trix,
  title={{TRiX}: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly},
  author={Dholakia, Stavan and Shukla, Shivani and Singh, Abhishek and Gazta, Aditya},
  journal={Nature Machine Intelligence},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
