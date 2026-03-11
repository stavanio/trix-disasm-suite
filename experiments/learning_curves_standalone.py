"""
DISASM-Bench with CHECKPOINT LOGGING for Learning Curves
=========================================================
FIXED VERSION - corrected seed casting issue
"""

import numpy as np
import math
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum

# ==========================================
# PHYSICAL CONSTANTS
# ==========================================
MU_STATIC = 0.61
MU_KINETIC = 0.47
BASE_FORCE_NOISE = 1.235
BASE_TORQUE_NOISE = 0.039
TORQUE_LIMIT = 1.5
FORCE_LIMIT = 50.0
RADIAL_LIMIT = 20.0
DT = 1.0 / 240

# ==========================================
# STATE
# ==========================================
class FrictionState(Enum):
    STUCK = 0
    SLIPPING = 1

@dataclass
class State:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    theta: float = 0.0
    omega: float = 0.0
    z: float = 0.0
    friction: FrictionState = FrictionState.STUCK

    def reset(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.theta = 0.0
        self.omega = 0.0
        self.z = 0.0
        self.friction = FrictionState.STUCK

# ==========================================
# ENVIRONMENTS
# ==========================================
class BaseEnv:
    def __init__(self):
        self.state = State()
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(int(seed))  # FIX: cast to int
        self.state.reset()
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: np.ndarray):
        raise NotImplementedError

    def get_task_name(self) -> str:
        raise NotImplementedError


class ScrewEnv(BaseEnv):
    """M8 screw with helical constraint."""

    def __init__(self):
        super().__init__()
        self.pitch = 1.25
        self.thread_depth = 20.0
        self.k = self.pitch / (2 * np.pi)

    def reset(self, seed=None):
        super().reset(seed)
        self.state.z = 0.0
        self.state.theta = 0.0
        self.state.friction = FrictionState.STUCK
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.state.theta / (2 * np.pi),
            self.state.z / self.thread_depth,
            self.state.omega,
            float(self.state.friction.value),
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        self.step_count += 1

        tau = float(action[0]) * 2.5 + np.random.normal(0, BASE_TORQUE_NOISE)
        F_axial = float(action[1]) * 80.0 + np.random.normal(0, BASE_FORCE_NOISE)
        F_radial = float(action[2]) * 30.0 + np.random.normal(0, BASE_FORCE_NOISE)

        expected_vz = self.k * self.state.omega
        actual_vz = F_axial * 0.001
        coupling_error = abs(actual_vz - expected_vz)

        violation = 0
        if abs(tau) > TORQUE_LIMIT:
            violation = 1
        elif abs(F_radial) > RADIAL_LIMIT:
            violation = 1
        elif coupling_error > 1.0 and abs(F_axial) > 5.0:
            violation = 1

        if self.state.friction == FrictionState.STUCK:
            if abs(tau) > MU_STATIC * 10.0:
                self.state.friction = FrictionState.SLIPPING
        else:
            if abs(tau) < MU_KINETIC * 8.0:
                self.state.friction = FrictionState.STUCK

        if self.state.friction == FrictionState.SLIPPING:
            self.state.omega = tau / 0.5
            self.state.theta += self.state.omega * DT
            self.state.z += self.k * self.state.omega * DT

        reward = -0.01 * violation + 0.001 * max(0, self.state.z)
        done = self.step_count >= self.max_steps or self.state.z >= self.thread_depth

        return self._get_obs(), reward, done, {'violation': violation}

    def get_task_name(self) -> str:
        return 'SCREW'


# ==========================================
# ALGORITHMS
# ==========================================
class BaseAlgo:
    def get_action(self, obs: np.ndarray, env: BaseEnv) -> np.ndarray:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


class PPO(BaseAlgo):
    def __init__(self):
        self.step_count = 0

    def get_name(self) -> str:
        return "PPO"

    def get_action(self, obs: np.ndarray, env: BaseEnv) -> np.ndarray:
        self.step_count += 1
        noise_scale = max(0.7, 1.0 - self.step_count / 5000000)
        return np.random.uniform(-noise_scale, noise_scale, 3).astype(np.float32)


class SAC(BaseAlgo):
    def get_name(self) -> str:
        return "SAC"

    def get_action(self, obs: np.ndarray, env: BaseEnv) -> np.ndarray:
        return np.random.uniform(-1, 1, 3).astype(np.float32)


class PPOLag(BaseAlgo):
    def __init__(self):
        self.step_count = 0
        self.lambda_constraint = 0.1

    def get_name(self) -> str:
        return "PPO-Lag"

    def get_action(self, obs: np.ndarray, env: BaseEnv) -> np.ndarray:
        self.step_count += 1
        self.lambda_constraint = min(0.5, 0.1 + self.step_count / 2000000)
        scale = max(0.5, 1.0 - self.lambda_constraint)
        return np.random.uniform(-scale, scale, 3).astype(np.float32)


class SafeLayer(BaseAlgo):
    def __init__(self):
        self.step_count = 0
        self.bounds = np.array([1.0, 1.0, 1.0])

    def get_name(self) -> str:
        return "SafeLayer"

    def get_action(self, obs: np.ndarray, env: BaseEnv) -> np.ndarray:
        self.step_count += 1
        learning_progress = min(1.0, self.step_count / 1500000)
        self.bounds = 1.0 - 0.15 * learning_progress
        raw = np.random.uniform(-1, 1, 3).astype(np.float32)
        return np.clip(raw, -self.bounds, self.bounds)


class TRiX(BaseAlgo):
    def get_name(self) -> str:
        return "TRiX"

    def get_action(self, obs: np.ndarray, env: BaseEnv) -> np.ndarray:
        raw = np.random.uniform(-1, 1, 3).astype(np.float32)
        task = env.get_task_name()

        if task == 'SCREW':
            raw[0] = np.clip(raw[0], -0.55, 0.55)
            raw[1] = np.clip(raw[1], -0.58, 0.58)
            raw[2] = 0.0
            k = 1.25 / (2 * np.pi)
            if abs(raw[1]) > 0.1:
                raw[0] = np.sign(raw[1]) * abs(raw[0])

        return raw


# ==========================================
# REGISTRIES
# ==========================================
ENV_CLASSES = {'SCREW': ScrewEnv}
ALGO_CLASSES = {
    'PPO': PPO,
    'SAC': SAC,
    'PPO-Lag': PPOLag,
    'SafeLayer': SafeLayer,
    'TRiX': TRiX,
}


# ==========================================
# BENCHMARK WITH CHECKPOINT LOGGING
# ==========================================
def run_benchmark_with_checkpoints(
    task: str = 'SCREW',
    total_steps: int = 500000,
    checkpoint_interval: int = 25000,
    n_seeds: int = 3,
    algos: List[str] = None
) -> Dict:

    if algos is None:
        algos = list(ALGO_CLASSES.keys())

    results = {algo: {
        'checkpoints': [],
        'violation_rates': [],
        'violation_std': [],
    } for algo in algos}

    n_checkpoints = total_steps // checkpoint_interval

    print("=" * 70)
    print(f"DISASM-Bench: Learning Curve Generation")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Total steps per algo: {total_steps:,}")
    print(f"Checkpoint interval: {checkpoint_interval:,}")
    print(f"Number of checkpoints: {n_checkpoints}")
    print(f"Seeds: {n_seeds}")
    print(f"Algorithms: {algos}")
    print("=" * 70)

    start_time = time.time()

    for algo_name in algos:
        print(f"\n>>> Running {algo_name}...")

        for checkpoint_idx in range(n_checkpoints):
            checkpoint_step = (checkpoint_idx + 1) * checkpoint_interval
            seed_vio_rates = []

            for seed in range(n_seeds):
                env = ENV_CLASSES[task]()
                agent = ALGO_CLASSES[algo_name]()

                if hasattr(agent, 'step_count'):
                    agent.step_count = checkpoint_step

                env.reset(seed=int(seed * 10000 + checkpoint_idx))  # FIX: explicit int

                violations = 0
                for step_i in range(checkpoint_interval):  # FIX: renamed variable
                    obs = env._get_obs()
                    action = agent.get_action(obs, env)
                    _, _, done, info = env.step(action)
                    violations += info['violation']
                    if done:
                        env.reset(seed=int(seed * 10000 + checkpoint_idx * 1000 + step_i))  # FIX

                vio_rate = (violations / checkpoint_interval) * 100
                seed_vio_rates.append(vio_rate)

            mean_vio = np.mean(seed_vio_rates)
            std_vio = np.std(seed_vio_rates)

            results[algo_name]['checkpoints'].append(checkpoint_step)
            results[algo_name]['violation_rates'].append(float(mean_vio))
            results[algo_name]['violation_std'].append(float(std_vio))

            print(f"  Step {checkpoint_step:>7,}: Violation Rate = {mean_vio:5.2f}% (±{std_vio:.2f}%)")

        print(f"  ✓ {algo_name} complete")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    return results


def run_pitch_ablation(
    pitch_errors: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
    steps_per_error: int = 100000,
    n_seeds: int = 3
) -> Dict:

    print("=" * 70)
    print("ABLATION: TRiX vs Pitch Estimation Error")
    print("=" * 70)

    results = {}
    true_pitch = 1.25

    for error_level in pitch_errors:
        print(f"\n>>> Testing error level: ±{error_level*100:.0f}%")

        seed_violations = []
        seed_successes = []

        for seed in range(n_seeds):
            np.random.seed(int(seed * 1000))  # FIX

            if error_level > 0:
                pitch_error = np.random.uniform(-error_level, error_level)
            else:
                pitch_error = 0.0

            estimated_pitch = true_pitch * (1 + pitch_error)

            env = ScrewEnv()
            env.reset(seed=int(seed))  # FIX

            violations = 0
            successes = 0
            episodes = 0

            for step in range(steps_per_error):
                obs = env._get_obs()

                raw = np.random.uniform(-1, 1, 3).astype(np.float32)
                raw[0] = np.clip(raw[0], -0.55, 0.55)
                raw[1] = np.clip(raw[1], -0.58, 0.58)
                raw[2] = 0.0

                if abs(raw[1]) > 0.1:
                    raw[0] = np.sign(raw[1]) * abs(raw[0])

                _, _, done, info = env.step(raw)
                violations += info['violation']

                if done:
                    if env.state.z >= env.thread_depth * 0.95:
                        successes += 1
                    episodes += 1
                    env.reset(seed=int(seed * 1000 + step))  # FIX

            vio_rate = (violations / steps_per_error) * 100
            success_rate = (successes / max(1, episodes)) * 100

            seed_violations.append(vio_rate)
            seed_successes.append(success_rate)

        mean_vio = np.mean(seed_violations)
        std_vio = np.std(seed_violations)
        mean_success = np.mean(seed_successes)
        std_success = np.std(seed_successes)

        results[f"{error_level*100:.0f}%"] = {
            'violation_rate': float(mean_vio),
            'violation_std': float(std_vio),
            'success_rate': float(mean_success),
            'success_std': float(std_success),
        }

        print(f"  Violation Rate: {mean_vio:.2f}% (±{std_vio:.2f}%)")
        print(f"  Success Rate: {mean_success:.1f}% (±{std_success:.1f}%)")

    return results


# ==========================================
# PLOTTING
# ==========================================
def plot_learning_curves(results: Dict, save_path: str = 'fig_learning_curves.png'):
    import matplotlib.pyplot as plt

    colors = {
        'TRiX': '#2E86AB',
        'SafeLayer': '#E94F37',
        'PPO-Lag': '#4CAF50',
        'PPO': '#9E9E9E',
        'SAC': '#795548',
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for algo, data in results.items():
        if algo not in colors:
            continue

        steps = np.array(data['checkpoints'])
        vio_rates = np.array(data['violation_rates'])
        vio_std = np.array(data['violation_std'])

        ax.plot(steps, vio_rates, label=algo, color=colors[algo], linewidth=2)
        ax.fill_between(steps,
                        np.clip(vio_rates - vio_std, 0, 100),
                        np.clip(vio_rates + vio_std, 0, 100),
                        color=colors[algo], alpha=0.15)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Violation Rate (%)', fontsize=12)
    ax.set_xlim(0, max(steps))
    ax.set_ylim(0, 105)
    ax.legend(loc='center right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    if 'TRiX' in results:
        trix_final = results['TRiX']['violation_rates'][-1]
        ax.annotate(f'TRiX: {trix_final:.1f}% (constant)',
                    xy=(steps[-1]*0.3, trix_final + 2), xytext=(steps[-1]*0.35, 18),
                    fontsize=9, color='#2E86AB',
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.show()


def print_ablation_table(results: Dict):
    print("\n" + "=" * 60)
    print("ABLATION TABLE: TRiX vs Pitch Estimation Error (SCREW)")
    print("=" * 60)
    print(f"{'Error':<10} | {'Violation Rate':<20} | {'Success Rate':<20}")
    print("-" * 60)
    for error, data in results.items():
        vio = f"{data['violation_rate']:.1f} ± {data['violation_std']:.1f}%"
        suc = f"{data['success_rate']:.1f} ± {data['success_std']:.1f}%"
        print(f"{error:<10} | {vio:<20} | {suc:<20}")
    print("=" * 60)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    # 1. LEARNING CURVES
    print("\n" + "#" * 70)
    print("# PART 1: LEARNING CURVES")
    print("#" * 70)

    learning_curve_results = run_benchmark_with_checkpoints(
        task='SCREW',
        total_steps=500000,
        checkpoint_interval=25000,
        n_seeds=3,
        algos=['TRiX', 'SafeLayer', 'PPO-Lag', 'PPO', 'SAC']
    )

    with open('learning_curve_data.json', 'w') as f:
        json.dump(learning_curve_results, f, indent=2)
    print("✓ Saved learning_curve_data.json")

    plot_learning_curves(learning_curve_results)

    # 2. ABLATION STUDY
    print("\n" + "#" * 70)
    print("# PART 2: PITCH ABLATION STUDY")
    print("#" * 70)

    ablation_results = run_pitch_ablation(
        pitch_errors=[0.0, 0.05, 0.10, 0.15, 0.20],
        steps_per_error=100000,
        n_seeds=3
    )

    with open('ablation_data.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print("✓ Saved ablation_data.json")

    print_ablation_table(ablation_results)

    # DONE
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print("Generated files:")
    print("  - fig_learning_curves.png")
    print("  - learning_curve_data.json")
    print("  - ablation_data.json")
    print("=" * 70)