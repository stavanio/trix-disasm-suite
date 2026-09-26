"""Train nominal policies with a pinned reference implementation.

The custom agents are retired. Policies come from Stable-Baselines3 so
the algorithm itself is not ours to defend, and every checkpoint records
the library versions that produced it.

The primary experiment is frozen-policy evaluation: one nominal policy
per task and seed, then every filter applied to the same policy. That
makes the filter comparison exactly paired, since each arm sees identical
nominal actions from identical states.

Checkpoints are written at the cadence the selection protocol expects, so
selection reads artifacts rather than re-training.
"""

import json
import os
import platform
import sys

import numpy as np
import torch

import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from benchmark.gym_adapter import make, CAPS
from benchmark.registry import constraint_hashes, get_filter
from benchmark.metrics import EpisodeRecorder
from benchmark import selection as SEL
from benchmark import margin_policy as MP
from benchmark import evaluation_distribution as ED

ALGOS = {"ppo": PPO, "sac": SAC}
NET_ARCH = [256, 256]


def fingerprint(task):
    return {
        "sb3": sb3.__version__,
        "gymnasium": gym.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "constraint_hash": constraint_hashes()[task],
        "robust_margin_hash": MP.policy_hash(task),
        "net_arch": list(NET_ARCH),
    }


class CheckpointEvery(BaseCallback):
    """Save at the selection protocol's cadence."""

    def __init__(self, every, out, prefix):
        super().__init__()
        self.every = every
        self.out = out
        self.prefix = prefix
        self.saved = []

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            path = os.path.join(self.out, f"{self.prefix}_{self.num_timesteps}")
            self.model.save(path)
            self.saved.append((self.num_timesteps, path + ".zip"))
        return True


def train(task, algo, seed, steps, out, filter_name=None):
    """Train one nominal policy. The filter is None for the primary
    frozen-policy experiment; passing one trains with it active, which is
    the secondary experiment."""
    os.makedirs(out, exist_ok=True)
    env = make(task, filter_name=filter_name)
    env.reset(seed=seed)
    cls = ALGOS[algo]
    kwargs = dict(policy="MlpPolicy", env=env, seed=seed, verbose=0,
                  device="cpu", policy_kwargs=dict(net_arch=NET_ARCH))
    model = cls(**kwargs)
    prefix = f"{task}__{algo}__{filter_name or 'none'}__seed{seed}"
    cb = CheckpointEvery(SEL.CHECKPOINT_EVERY, out, prefix)
    model.learn(total_timesteps=steps, callback=cb, progress_bar=False)
    final = os.path.join(out, prefix + "_final")
    model.save(final)
    meta = {
        "task": task, "algo": algo, "seed": seed, "train_steps": steps,
        "filter_during_training": filter_name,
        "episode_cap": CAPS[task],
        "checkpoints": cb.saved + [(steps, final + ".zip")],
        "fingerprint": fingerprint(task),
    }
    with open(os.path.join(out, prefix + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load(path, task):
    algo = "sac" if "__sac__" in os.path.basename(path) else "ppo"
    return ALGOS[algo].load(path, device="cpu")


def evaluate_frozen(model, task, filter_name, seeds, recorder,
                    evaluation_distribution=None):
    """Run one frozen policy through one filter on a fixed seed list.

    Filtering happens here rather than inside the environment so the
    nominal action is recoverable and every arm sees the same proposal.
    """
    filt = get_filter(task, filter_name)
    spec = (ED.specification(task) if evaluation_distribution is None
            else evaluation_distribution)
    digest = ED.validate(spec, task)
    if recorder.episodes:
        raise ValueError("Frozen evaluation requires an empty recorder")
    recorder.evaluation_distribution = spec
    env = make(task, evaluation_distribution=spec)
    cap = CAPS[task]
    for s in seeds:
        obs, reset_info = env.reset(seed=int(s))
        if reset_info["evaluation_distribution_hash"] != digest:
            raise ValueError("Environment used a different reset distribution")
        recorder.start_episode()
        terminated = truncated = False
        for _ in range(cap):
            nominal, _ = model.predict(obs, deterministic=True)
            action, finfo = filt(np.asarray(obs, dtype=np.float64),
                                 np.asarray(nominal, dtype=np.float64))
            obs, reward, terminated, truncated, info = env.step(action)
            recorder.step(info, reward=reward, filter_info={
                "intervention_norm": float(np.linalg.norm(
                    np.asarray(action, dtype=np.float64)
                    - np.asarray(nominal, dtype=np.float64))),
                "infeasible": bool(finfo.get("infeasible", False)),
                "recovery": bool(finfo.get("recovery", False)),
            })
            if terminated or truncated:
                break
        recorder.end_episode(env.unwrapped_task, completed=bool(terminated))
        recorder.episodes[-1]["evaluation_reset"] = reset_info
    return recorder
