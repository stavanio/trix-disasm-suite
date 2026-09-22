"""Frozen checkpoint-selection protocol.

Per (task, method, training seed):

    1. train to the task's predeclared budget, identical across methods
    2. checkpoint every CHECKPOINT_EVERY steps
    3. score every checkpoint on SCREEN_EPISODES validation seeds
    4. rank contiguous three-checkpoint windows by mean safe completion,
       take the centre of the best N_CANDIDATES
    5. re-evaluate those centres on CONFIRM_EXTRA further seeds
    6. choose by the lexicographic rule below
    7. evaluate the winner once on TEST_EPISODES held-out seeds

Thirty episodes carries about +-18 points of uncertainty on a rate near
one half, hence the two-stage design. Validation seeds are shared across
methods within a task and disjoint from training and test seeds.
"""

from dataclasses import dataclass

# --- frozen constants ----------------------------------------------------
CHECKPOINT_EVERY = 1000
SCREEN_EPISODES = 30
CONFIRM_EXTRA = 70
CONFIRM_EPISODES = SCREEN_EPISODES + CONFIRM_EXTRA
TEST_EPISODES = 100
N_CANDIDATES = 3
WINDOW = 3
PRODUCTION_SEEDS = tuple(range(10))
PILOT_SEEDS = (900, 901)          # budget determination only, never reported

VALIDATION_SEED_BASE = 500_000
TEST_SEED_BASE = 800_000

# Fixed offsets preserve the dominant historical Stage-1 seed schedule.
TASK_SEED_OFFSET = {
    "BATTERY": 29,
    "CRANK": 86,
    "PCB": 65,
    "PRY": 19,
    "SCREW": 99,
    "SNAP": 72,
}

def _task_seed_offset(task):
    if task in TASK_SEED_OFFSET:
        return TASK_SEED_OFFSET[task]

    # Deterministic fallback for future task names.
    import hashlib
    digest = hashlib.sha256(task.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100

# Order matters and is frozen. Safe completion leads because ranking by
# destructive rate first would let a policy that never acts, and therefore
# never damages anything, outrank one that does the task safely.
CRITERIA = (
    ("safe_completion_rate", "max"),
    ("destructive_completion_rate", "min"),
    ("progress", "max"),
    ("episodes_with_any_violation_rate", "min"),
    ("step", "min"),
)

# Normalised task progress, frozen per task so the selection code carries
# no informal judgement. Each is a fraction of the task's own goal, taken
# from the episode summary rather than recomputed here.
PROGRESS_KEY = {
    "SCREW": "progress_axial",       # z / Z_TARGET
    "PCB": "progress_lift",          # lift / target
    "SNAP": "progress_extract",      # extraction distance / target
    "CRANK": "progress_rotation",    # turns / required turns
    "BATTERY": "progress_peel",      # peel distance / target
    "PRY": "progress_gap",           # gap / Z_TARGET
    "BAYONET_S": "progress_release",
    "BAYONET_P": "progress_release",
}


def validation_seeds(task, n, offset=0):
    base = VALIDATION_SEED_BASE + 1000 * _task_seed_offset(task)
    return [base + offset + i for i in range(n)]


def test_seeds(task, n):
    base = TEST_SEED_BASE + 1000 * _task_seed_offset(task)
    return [base + i for i in range(n)]


@dataclass
class Checkpoint:
    step: int
    metrics: dict          # keys as in CRITERIA, plus 'progress'
    episodes: int

    def key(self, name):
        if name == "step":
            return self.step
        v = self.metrics.get(name)
        return 0.0 if v is None else float(v)


def candidate_windows(checkpoints, window=WINDOW, n=N_CANDIDATES):
    """Centre checkpoints of the best contiguous windows by mean safe
    completion. Windows overlap, so centres are deduplicated and the
    ranking is stable in step order for ties."""
    if len(checkpoints) < window:
        return sorted(checkpoints,
                      key=lambda c: (-c.key("safe_completion_rate"), c.step)
                      )[:n]
    scored = []
    half = window // 2
    for i in range(len(checkpoints) - window + 1):
        w = checkpoints[i:i + window]
        mean = sum(c.key("safe_completion_rate") for c in w) / window
        scored.append((mean, w[half].step, w[half]))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out, seen = [], set()
    for _, step, c in scored:
        if step in seen:
            continue
        seen.add(step)
        out.append(c)
        if len(out) == n:
            break
    return out


def _lex_key(c):
    parts = []
    for name, sense in CRITERIA:
        v = c.key(name)
        parts.append(-v if sense == "max" else v)
    return tuple(parts)


def select(candidates):
    """Apply the frozen lexicographic rule to confirmed candidates."""
    if not candidates:
        return None
    for c in candidates:
        if c.episodes < CONFIRM_EPISODES:
            raise ValueError(
                f"checkpoint at step {c.step} was scored on {c.episodes} "
                f"episodes; selection requires {CONFIRM_EPISODES}")
    return min(candidates, key=_lex_key)


def describe():
    return {
        "checkpoint_every": CHECKPOINT_EVERY,
        "screen_episodes": SCREEN_EPISODES,
        "confirm_episodes": CONFIRM_EPISODES,
        "test_episodes": TEST_EPISODES,
        "n_candidates": N_CANDIDATES,
        "window": WINDOW,
        "production_seeds": list(PRODUCTION_SEEDS),
        "pilot_seeds": list(PILOT_SEEDS),
        "criteria": [f"{s}:{n}" for n, s in CRITERIA],
        "progress_key": dict(PROGRESS_KEY),
    }
