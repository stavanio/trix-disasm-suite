"""DISASM-Bench: Six disassembly task environments."""

from .base_env import BaseEnv
from .screw_env import ScrewEnv
from .battery_env import BatteryEnv
from .task_envs import PCBEnv, SnapEnv, PryEnv, CrankEnv

ENV_REGISTRY = {
    'SCREW': ScrewEnv,
    'BATTERY': BatteryEnv,
    'PCB': PCBEnv,
    'SNAP': SnapEnv,
    'PRY': PryEnv,
    'CRANK': CrankEnv,
}

# Task groupings (Table 2)
INSTANTANEOUS_TASKS = ['SCREW', 'PRY', 'CRANK', 'PCB']
LATENT_SEQUENTIAL_TASKS = ['BATTERY', 'SNAP']
ALL_TASKS = INSTANTANEOUS_TASKS + LATENT_SEQUENTIAL_TASKS


def make_env(task_name: str, **kwargs) -> BaseEnv:
    """Create an environment by name."""
    if task_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[task_name](**kwargs)
