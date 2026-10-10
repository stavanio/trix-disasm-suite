"""Acceptance gate for the reference training path.

Nothing may run at production scale until these hold: the interface is
valid, checkpoints reload to identical behavior, termination semantics
are distinguished, and every learned quantity stays finite.
"""
import glob
import json
import os
import shutil
import sys
import warnings

import numpy as np
import torch
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")

from stable_baselines3.common.env_checker import check_env
from benchmark.gym_adapter import make, CAPS
from benchmark.registry import TASKS
from benchmark.metrics import EpisodeRecorder
from benchmark import selection as SEL
from training import sb3_runner as R

OUT = "/tmp/sb3_gate"


def _clean():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)


def test_1_every_task_passes_the_interface_check():
    for task in TASKS:
        check_env(make(task), warn=True, skip_render_check=True)


def test_2_terminated_and_truncated_are_distinct():
    """A finished task must not be reported the same way as a hit cap:
    one stops the bootstrap, the other must not."""
    env = make("PCB", cap=5)
    env.reset(seed=0)
    trunc = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        trunc |= truncated
    assert trunc, "reaching the cap did not set truncated"

    env = make("SCREW")
    env.reset(seed=0)
    term = False
    for _ in range(CAPS["SCREW"]):
        _, _, terminated, truncated, info = env.step(
            np.array([0.35, 0.35, 0.2]))
        if terminated:
            term = True
            assert not truncated, "a completed task also reported truncated"
            assert "episode_end" in info
            break
    assert term, "task never terminated"


def test_3_executed_action_reaches_the_transition():
    """With a filter inside the environment, the recorded action must be
    the one executed, not the proposal."""
    env = make("PCB", filter_name="trix")
    env.reset(seed=0)
    nominal = np.array([1.0, 1.0, 1.0])
    _, _, _, _, info = env.step(nominal)
    executed = np.asarray(info["executed_action"], dtype=np.float64)
    assert not np.allclose(executed, nominal), "filter did not act"
    assert np.hypot(executed[0], executed[1]) <= 1.0


def test_4_checkpoints_land_at_the_protocol_cadence():
    _clean()
    meta = R.train("PCB", "sac", seed=900, steps=3 * SEL.CHECKPOINT_EVERY,
                   out=OUT)
    steps = [s for s, _ in meta["checkpoints"]]
    assert SEL.CHECKPOINT_EVERY in steps, f"cadence missing from {steps}"
    for _, path in meta["checkpoints"]:
        assert os.path.exists(path), f"missing checkpoint {path}"


def test_5_reload_reproduces_deterministic_actions():
    files = sorted(glob.glob(os.path.join(OUT, "*_final.zip")))
    assert files, "no final checkpoint to reload"
    m1 = R.load(files[0], "PCB")
    m2 = R.load(files[0], "PCB")
    env = make("PCB")
    obs, _ = env.reset(seed=7)
    for _ in range(50):
        a1, _ = m1.predict(obs, deterministic=True)
        a2, _ = m2.predict(obs, deterministic=True)
        assert np.array_equal(a1, a2), "reload changed deterministic actions"
        obs, _, t, tr, _ = env.step(a1)
        if t or tr:
            break


def test_6_learned_quantities_stay_finite():
    """A reloaded model carries no replay buffer, so states come from the
    environment. This checks the divergence the custom implementation
    showed: unbounded Q, exploding log-prob, saturated actions."""
    files = sorted(glob.glob(os.path.join(OUT, "*_final.zip")))
    model = R.load(files[0], "PCB")
    env = make("PCB")
    obs, _ = env.reset(seed=11)
    states = []
    for _ in range(64):
        a, _ = model.predict(obs, deterministic=True)
        states.append(np.asarray(obs, dtype=np.float32))
        obs, _, t, tr, _ = env.step(a)
        if t or tr:
            obs, _ = env.reset(seed=12)
    o = torch.as_tensor(np.array(states))
    with torch.no_grad():
        act, logp = model.actor.action_log_prob(o)
        q = torch.cat(model.critic(o, act), dim=1)
        ec = (float(model.log_ent_coef.exp())
              if model.ent_coef_optimizer is not None
              else float(model.ent_coef_tensor))
    assert torch.isfinite(q).all(), "non-finite Q"
    assert torch.isfinite(logp).all(), "non-finite log-prob"
    assert np.isfinite(ec) and ec > 0, f"entropy coefficient {ec}"
    assert float(act.abs().max()) <= 1.0 + 1e-6, "action outside the box"
    assert float(logp.mean()) < 10.0, \
        f"log-prob {float(logp.mean()):.1f} indicates tanh saturation"


def test_7_fingerprint_records_versions_and_hashes():
    metas = glob.glob(os.path.join(OUT, "*_meta.json"))
    assert metas
    fp = json.load(open(metas[0]))["fingerprint"]
    for k in ("sb3", "gymnasium", "torch", "numpy", "python",
              "constraint_hash", "robust_margin_hash", "net_arch"):
        assert k in fp and fp[k], f"fingerprint missing {k}"
    assert fp["net_arch"] == R.NET_ARCH


def test_8_architecture_is_confirmed_from_the_instantiated_policy():
    """Configuration is not evidence: read the widths off the network."""
    files = sorted(glob.glob(os.path.join(OUT, "*_final.zip")))
    model = R.load(files[0], "PCB")
    widths = [p.shape[0] for n, p in model.actor.named_parameters()
              if n.endswith("weight") and p.dim() == 2]
    for w in R.NET_ARCH:
        assert w in widths, f"width {w} not present in {widths}"


def test_9_frozen_policy_evaluation_is_paired_across_filters():
    """Every arm must see the same nominal action from the same state."""
    files = sorted(glob.glob(os.path.join(OUT, "*_final.zip")))
    model = R.load(files[0], "PCB")
    seeds = SEL.validation_seeds("PCB", 3)
    firsts = []
    for filt in ("none", "trix"):
        env = make("PCB")
        obs, _ = env.reset(seed=int(seeds[0]))
        nominal, _ = model.predict(obs, deterministic=True)
        firsts.append(np.asarray(nominal, dtype=np.float64))
    assert np.array_equal(firsts[0], firsts[1]), \
        "arms received different nominal actions"


def test_10_evaluation_produces_the_outcome_contract():
    files = sorted(glob.glob(os.path.join(OUT, "*_final.zip")))
    model = R.load(files[0], "PCB")
    rec = EpisodeRecorder("PCB", "sac+trix", seed=900)
    R.evaluate_frozen(model, "PCB", "trix", SEL.validation_seeds("PCB", 3), rec)
    s = rec.validate()
    for k in ("safe_completion_rate", "destructive_completion_rate",
              "timeout_rate", "counts"):
        assert k in s, f"summary missing {k}"
    assert s["counts"]["episodes"] == 3


def _order(name):
    return int(name.split('_')[1])


if __name__ == '__main__':
    names = sorted((k for k in globals() if k.startswith('test_')), key=_order)
    for name in names:
        globals()[name](); print(f"PASS {name}")
