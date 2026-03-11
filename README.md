# TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly

Official code for reproducing the experiments in:

> **TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly**
> S. Dholakia, S. Shukla, A. Singh, A. Gazta
> *Nature Machine Intelligence* (2026, under review)

TRiX is a neuro-symbolic safety governor that projects neural policy actions onto task-specific differentiable manifolds, closing the **Linearity Gap** between linear safety methods and nonlinear physical constraints.

---

## Reproduce Paper Results (< 5 minutes)

```bash
# Only dependency
pip install numpy

# Run the benchmark (Table 4, 100K steps × 3 seeds)
python benchmark/disasm_bench.py

# Run physics validation (Tables 3, 6, 7, Appendix G)
python benchmark/validation_suite.py
```

No GPU required. No PyBullet. No PyTorch. Fully deterministic.

### Expected Output (Table 4)

```
ALGO         |    SCREW |     SNAP |      PRY |    CRANK
---------------------------------------------------------
PPO          |   96.49% |   93.26% |   81.59% |   83.22%
SAC          |   98.62% |   97.34% |   92.07% |   92.70%
PPO-Lag      |   91.69% |   84.05% |   56.89% |   60.90%
SafeLayer    |   87.01% |   81.60% |    0.62% |   47.72%
TRiX         |    2.37% |   59.75% |    0.03% |    8.86%  <<<
```

TRiX reduces SCREW violations by **84.6 percentage points** vs SafeLayer (87.0% to 2.4%).

### Run All 6 Tasks (including BATTERY, PCB)

```bash
python -c "
from benchmark.disasm_bench import run_benchmark, TASKS_ALL
run_benchmark(steps_per_cell=100000, n_seeds=3, tasks=TASKS_ALL)
"
```

---

## Repository Structure

```
trix-disasm-bench/
├── benchmark/                      # Canonical benchmark (produces paper numbers)
│   ├── disasm_bench.py             # DISASM-Bench: 6 envs, 7 algos, self-contained
│   └── validation_suite.py         # Physics validation, overhead, GSR, thermal latency
│
├── trix/                           # PyTorch reference implementation
│   ├── governor.py                 # Neuro-symbolic controller (rules, gating, explanations)
│   ├── sac_policy.py               # SAC neural policy backbone
│   └── manifolds.py                # Closed-form manifold projections (Eq. 5-8)
│
├── envs/                           # Environment implementations
│   ├── base_env.py                 # Abstract base with entropy factors
│   ├── screw_pybullet.py           # PyBullet screw environment (Panda arm)
│   └── screw_proxy.py              # Lightweight proxy environment
│
├── checkpoints/                    # Trained model weights (3 seeds × 500K steps)
│   ├── trix_s0_step0.pt ... trix_s0_step500000.pt
│   ├── trix_s1_step0.pt ... trix_s1_step500000.pt
│   └── trix_s2_step0.pt ... trix_s2_step500000.pt
│
├── scripts/
│   ├── run_benchmark.sh            # One-liner benchmark runner
│   ├── plot_results.py             # IEEE-format figure generation
│   └── generate_figures.py         # Paper figures from results
│
├── tests/
│   └── test_manifolds.py           # Projection correctness proofs
│
└── figures/                        # Generated figures (after running scripts)
```

### What Is What

| Component | Purpose | Dependencies |
|-----------|---------|-------------|
| `benchmark/disasm_bench.py` | **Reproduce Table 4.** Self-contained NumPy simulation of 6 tasks, 7 algorithms. This is the canonical file. | `numpy` only |
| `benchmark/validation_suite.py` | **Reproduce Tables 3, 6, 7 and Appendix G.** Computational overhead, GSR, thermal latency, physics validation. | `numpy` only |
| `trix/governor.py` | **Reference TRiX implementation.** Full neuro-symbolic controller with symbolic rule library, sigmoid feasibility gating, and deterministic explanation generation. | `torch`, `numpy` |
| `trix/sac_policy.py` | SAC neural policy used as the base learner inside TRiX. | `torch` |
| `envs/screw_pybullet.py` | PyBullet-based screw environment with Panda arm, contact physics, and force/torque sensing. | `pybullet`, `gymnasium` |
| `checkpoints/` | Trained TRiX model weights from 500K-step runs across 3 seeds. | `torch` (to load) |

---

## Design Rationale

### Why Self-Contained NumPy Physics?

TRiX's core claim is about the **geometry of safety constraints**, not high-fidelity rendering. The benchmark simulation:

1. **Validates against analytical solutions** (helical motion <0.012mm error, thermal diffusion <0.001°C RMSE)
2. **Is fully deterministic** (fixed seeds, no engine variability)
3. **Requires zero setup** (no GPU, no compilation, no URDF assets)

The PyBullet environment (`envs/screw_pybullet.py`) is included for researchers who want to extend to full physics. Cross-simulator validation confirms the relative ranking of methods is preserved.

### Physical Constants

All parameters are derived from engineering specifications (see paper Appendix D):

| Parameter | Value | Source |
|-----------|-------|--------|
| Integration rate | 240 Hz | Contact physics standard |
| Force noise (σ_F) | 1.235 N | ATI Mini45 + actuator RSS |
| Torque noise (σ_τ) | 0.039 Nm | ATI Mini45 + actuator RSS |
| Static friction (μ_s) | 0.61 | Steel/aluminum tables |
| Kinetic friction (μ_k) | 0.47 | Steel/aluminum tables |
| Screw pitch (M8) | 1.25 mm/rev | ISO 261 |

---

## DISASM-Bench Tasks

| Task | Constraint Type | What TRiX Enforces |
|------|----------------|-------------------|
| **SCREW** | Helical coupling | v_z = (p/2π)ω_z manifold projection |
| **SNAP** | Unilateral | Half-space constraint (no pulling) |
| **PRY** | State-dependent | Dynamic torque limit f(insertion depth) |
| **CRANK** | Curved manifold | Tangential force projection |
| **BATTERY** | Thermodynamic | Temperature-aware force scaling |
| **PCB** | Precision | Tight lateral force bounds |

---

## Citation

```bibtex
@article{dholakia2026trix,
  title={{TRiX}: Neuro-Symbolic Safety for Foundation Model Agents
         in Robotic Disassembly},
  author={Dholakia, Stavan and Shukla, Shivani and Singh, Abhishek
          and Gazta, Aditya},
  journal={Nature Machine Intelligence},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
