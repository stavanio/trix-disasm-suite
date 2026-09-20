# TRiX: Neuro-Symbolic Safety for Foundation Model Agents in Robotic Disassembly

> S. Dholakia, S. Shukla, A. Singh, A. Gazta
> *Nature Machine Intelligence* (2026, under review)

TRiX is a neuro-symbolic safety governor that projects neural policy actions onto task-specific differentiable manifolds before execution. It closes the **Linearity Gap**: the fundamental mismatch between the linear constraint representations used by existing safety methods and the nonlinear, geometrically structured constraints imposed by physical reality.

On helical fastener extraction, TRiX reduces safety violations from **86.8% to 2.4%** compared to the best projection-based baseline (SafeLayer), while all actions are computed in closed form at **< 10 μs**.

---

## Reproduce Paper Results

For the six B601 workspace panels, see [Workspace rendering](docs/workspace_rendering.md).
Each panel has a separate PyBullet script, archived input frame, and render manifest.

```bash
pip install numpy
```

### Table 4 + Table 5 + Table 12 (one command, ~5 min)

```bash
python3 benchmark/disasm_bench.py
```

This runs the full stress tournament: 6 tasks, 7 algorithms, 100K steps, 3 seeds. Output includes violation rates (Table 4), task success rates (Table 5), and statistical significance with Welch's t-test and Cohen's d (Table 12).

### Tables 3, 6, 7, Appendix G (one command, ~30 sec)

```bash
python3 benchmark/validation_suite.py
```

Runs computational overhead measurement (Table 3/13), thermal latency analysis (Table 6), grounding success rate under synthetic hallucinations (Table 7), and physics validation against analytical solutions (Appendix G).

### Quick Smoke Test (~30 sec)

```bash
python3 -c "
from benchmark.disasm_bench import run_benchmark, TASKS
run_benchmark(steps_per_cell=10000, n_seeds=1, tasks=TASKS)
"
```

### Unit Tests

```bash
python3 tests/test_manifolds.py
```

Verifies manifold membership, projection minimality, idempotency, and constraint satisfaction for all projections.

---

## Expected Output

### Table 4: Safety Violation Rates (%)

```
ALGO         |    SCREW |     SNAP |      PRY |    CRANK
---------------------------------------------------------
PPO          |   96.4%  |   93.2%  |   81.8%  |   83.3%
SAC          |   98.6%  |   97.3%  |   92.0%  |   92.6%
PPO-Lag      |   91.7%  |   84.1%  |   56.6%  |   60.9%
SafeLayer    |   86.8%  |   81.6%  |    3.4%  |   47.6%
TRiX         |    2.4%  |   59.6%  |    0.0%  |    8.7%  <<<
```

Results are deterministic (fixed seeds). Minor variation (< 1 pp) may appear across platforms.

### BATTERY and PCB (extended benchmark)

```bash
python3 -c "
from benchmark.disasm_bench import run_benchmark, TASKS_ALL
run_benchmark(steps_per_cell=100000, n_seeds=3, tasks=TASKS_ALL)
"
```

---

## Repository Structure

```
trix-disasm-suite/
│
├── benchmark/                          # Canonical reproducibility code
│   ├── disasm_bench.py                 # THE benchmark (Tables 4, 5, 12)
│   └── validation_suite.py            # Physics validation (Tables 3, 6, 7, App G)
│
├── trix/                               # Reference implementations
│   ├── governor.py                     # PyTorch neuro-symbolic controller
│   │                                   #   Symbolic rule library, sigmoid gating,
│   │                                   #   deterministic explanation generation
│   ├── sac_policy.py                   # SAC neural policy backbone
│   └── manifolds.py                    # Closed-form manifold projections (Eq. 5-8)
│
├── envs/                               # Environment implementations
│   ├── base_env.py                     # Abstract base with entropy factors
│   ├── screw_pybullet.py              # PyBullet Panda arm environment
│   └── screw_proxy.py                 # Lightweight proxy environment
│
├── checkpoints/                        # Trained model weights
│   └── trix_s{0,1,2}_step{0..500K}.pt # 3 seeds, 500K steps each
│
├── scripts/
│   ├── generate_figures.py             # Figures 2, 3, 4
│   ├── plot_results.py                 # IEEE-format plotting utilities
│   └── run_benchmark.sh               # Shell wrapper
│
└── tests/
    └── test_manifolds.py               # Projection correctness proofs
```

---

## Paper-to-Code Mapping

Every table in the manuscript maps to runnable code:

| Table | Content | Source | Command |
|-------|---------|--------|---------|
| 1 | Paradigm comparison | Qualitative | N/A |
| 2 | Task suite | `disasm_bench.py` env classes | Specification |
| **3** | Computational overhead | `validation_suite.py` Exp 3 | `python3 benchmark/validation_suite.py` |
| **4** | **Violation rates** | `disasm_bench.py` | `python3 benchmark/disasm_bench.py` |
| **5** | **Task success rates** | `disasm_bench.py` | Same as above |
| **6** | Thermal latency | `validation_suite.py` Exp 5 | `python3 benchmark/validation_suite.py` |
| **7** | Grounding success rate | `validation_suite.py` Exp 4 | Same as above |
| 8 | Observation space | Env `_get_obs()` methods | Specification |
| 9 | Violation criteria | Env `step()` methods | Specification |
| 10 | Domain randomization | Env `reset()` methods | Specification |
| 11 | Hyperparameters | `Algorithms` class, constants | Specification |
| **12** | **Statistical significance** | `disasm_bench.py` | `python3 benchmark/disasm_bench.py` |
| 13 | Overhead (= Table 3) | `validation_suite.py` Exp 3 | Same as Table 3 |

Tables in **bold** produce computed results. The rest are specifications whose values can be traced to named constants and method signatures in the source code.

---

## DISASM-Bench Tasks

| Task | Constraint | Manifold | What TRiX Enforces |
|------|-----------|----------|-------------------|
| SCREW | Helical coupling | v_z = (p/2π)ω_z | Torque-force coordination along helix |
| SNAP | Unilateral | F·n ≥ 0 | Half-space projection (no pulling) |
| PRY | State-dependent | τ_max(depth) | Dynamic torque limit tracks insertion |
| CRANK | Curved | Tangent circle | Radial force elimination |
| BATTERY | Thermodynamic | F·v ≤ P_max(T) | Temperature-aware force scaling |
| PCB | Precision | Tight lateral bounds | Gentle lift, trace protection |

---

## Architecture

```
VLM / Neural Policy                TRiX Governor               Safe Execution
┌─────────────────┐    unsafe    ┌──────────────────┐   safe   ┌──────────────┐
│  "Remove screw" │───────────→ │  Intercept       │────────→│  Execute      │
│  Pull fast      │  u_π        │  Project onto M  │  u_safe │  Respect      │
│                 │             │  Closed-form O(1)│         │  constraints  │
└─────────────────┘             └──────────────────┘         └──────────────┘
                                  u_safe = Ψ_M(u_π)
                                  = argmin ||u - u_π||
                                    u ∈ M
```

Safety constraints in manipulation arise from the geometric and thermodynamic structure of the physical world. TRiX encodes them directly as differentiable manifolds derived from first principles, rather than learning them from data or approximating them with linear functions.

---

## Design Decisions

**Why NumPy physics instead of PyBullet/Isaac?**

The benchmark validates the *geometry of safety constraints*, not rendering fidelity. The simulation validates against analytical solutions (helical: <0.012mm error, thermal: <0.001°C RMSE) and is fully deterministic across platforms. The PyBullet environment (`envs/screw_pybullet.py`) is included for researchers who want full contact physics.

**Why random policies instead of trained RL?**

The central claim is that TRiX provides safety *by construction*, independent of policy quality. Random policies explore the full action space uniformly, maximally stressing safety mechanisms. A trained policy would produce fewer violations for all methods, obscuring the safety mechanism's contribution.

**Why 7 algorithms?**

PPO and SAC (unconstrained). PPO-Lagrangian (soft constraints). CPO and Lambda (conservative optimization). SafeLayer (learned linear projection). TRiX (geometric projection). This spans the major paradigms in safe RL.

---

## Physical Constants

All parameters trace to engineering specifications (Appendix D):

| Parameter | Value | Source |
|-----------|-------|--------|
| Integration rate | 240 Hz | Contact simulation standard |
| Force noise σ_F | 1.235 N | ATI Mini45 + actuator (RSS) |
| Torque noise σ_τ | 0.039 Nm | ATI Mini45 + actuator (RSS) |
| Static friction μ_s | 0.61 | Steel/aluminum (engineering tables) |
| Kinetic friction μ_k | 0.47 | Steel/aluminum (engineering tables) |
| Screw pitch (M8) | 1.25 mm/rev | ISO 261 |
| Battery puncture limit | 20 N | Cell rupture threshold |
| Thermal runaway onset | 60°C | Li-ion safety literature |

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

MIT. See [LICENSE](LICENSE).
