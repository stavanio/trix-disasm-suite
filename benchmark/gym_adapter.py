"""Gymnasium adapter for the benchmark environments.

Wraps the six task environments so a reference reinforcement learning
implementation can train against them without any custom update code.
The environments themselves are unchanged: the adapter only translates
the interface.

Termination semantics follow Gymnasium. A finished task is `terminated`,
so the critic does not bootstrap past it. Reaching the episode cap is
`truncated`, which does bootstrap, since the task was not actually over.
The submitted pipeline conflated these, which biases value estimates on
tasks that frequently hit the cap.

The episode summary is attached to the final `info` under `episode_end`
so the outcome contract survives the wrapper.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from benchmark.registry import ENVS, get_filter


CAPS = {"SCREW": 2500, "PCB": 1500, "SNAP": 1500,
        "CRANK": 3000, "BATTERY": 3000, "PRY": 4000,
        "BAYONET_S": 2500, "BAYONET_P": 2500}


class DisasmEnv(gym.Env):
    """One benchmark task as a Gymnasium environment.

    If `filter_name` is given the filter is applied inside `step`, so the
    action recorded by an off-policy learner is the one the environment
    actually executed. A wrapper that transformed the action outside the
    environment would leave the replay buffer holding the nominal action
    against a transition produced by the filtered one.
    """

    metadata = {"render_modes": []}

    def __init__(self, task, filter_name=None, cap=None):
        super().__init__()
        if task not in ENVS:
            raise KeyError(f"unknown task {task!r}")
        self.task = task
        self.filter_name = filter_name
        self._filter = get_filter(task, filter_name) if filter_name else None
        self.cap = cap or CAPS[task]
        self._env = ENVS[task]()
        obs = self._env.reset(seed=0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset(seed=seed)
        self._steps = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        executed = a
        finfo = {}
        if self._filter is not None:
            executed, finfo = self._filter(self._last_obs(), a)
            executed = np.asarray(executed, dtype=np.float64)
        obs, reward, done, info = self._env.step(executed)
        self._steps += 1
        terminated = bool(done)
        truncated = bool(not done and self._steps >= self.cap)
        if terminated or truncated:
            info = dict(info)
            info["episode_end"] = self._env.episode_summary(terminated)
        if finfo:
            info = dict(info)
            info["filter"] = finfo
        info["executed_action"] = executed
        return (np.asarray(obs, dtype=np.float32), float(reward),
                terminated, truncated, info)

    def _last_obs(self):
        return np.asarray(self._env._get_obs(), dtype=np.float64)

    @property
    def unwrapped_task(self):
        return self._env


def make(task, filter_name=None, cap=None, seed=None):
    env = DisasmEnv(task, filter_name=filter_name, cap=cap)
    if seed is not None:
        env.reset(seed=seed)
    return env
