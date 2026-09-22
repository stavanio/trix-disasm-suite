"""Registry of tasks, environments and safety filter arms.

One place decides which environment version and which filter each cell of
the sweep uses, so no experiment can silently mix versions. Every entry
records the environment's constraint hash at import, and run_cell writes
it into the result.

Filters share one signature, (obs, action) -> (action, info), and are
applied to the same nominal action within a cell.
"""

from envs.screw_env_v3 import ScrewEnvV3
from envs.pcb_env_v2 import PCBEnvV2
from envs.snap_env_v2 import SnapEnvV2
from envs.crank_env_v2 import CrankEnvV2
from envs.battery_env_v2 import BatteryEnvV2
from envs.pry_env_v2 import PryEnvV2
from envs.bayonet_env import BayonetS, BayonetP

from baselines import screw_solvers as SS
from baselines import screw_filters as SF
from baselines import pcb_filters as PF
from baselines import phase_filters as PH
from baselines import task_filters as TF
from baselines import bayonet_filters as BF

ENVS = {
    "SCREW": ScrewEnvV3,
    "PCB": PCBEnvV2,
    "SNAP": SnapEnvV2,
    "CRANK": CrankEnvV2,
    "BATTERY": BatteryEnvV2,
    "PRY": PryEnvV2,
    "BAYONET_S": BayonetS,
    "BAYONET_P": BayonetP,
}

TASKS = tuple(ENVS)


def _passthrough(obs, a):
    return a, {}


def _pcb_margin():
    """PCB's exact projector aims at the robust interior disk, so the
    oracle must aim there too: c = |tau| - TAU_N, and the target is
    TAU_ROBUST, hence a negative margin of the difference."""
    from baselines.pcb_filters import TAU_N, TAU_ROBUST, INTERIOR
    return -(TAU_N - TAU_ROBUST * (1.0 - INTERIOR))


def _task_bounds(task):
    """The action box each oracle correction must also respect.

    A halfspace step handles the coupled constraint but can leave the
    box; without these the arm satisfies the manifold and violates the
    per-axis limits, which is not what the oracle is meant to represent.
    """
    if task == "PCB":
        # The ROBUST interior, matching what trix_project targets. Giving
        # the oracle the nominal limits would let it aim at a larger set
        # than the exact projector, so the comparison would measure the
        # margin rather than the method.
        # Tilt is bounded by the DISK, which the cost function enforces.
        # Clipping each tilt axis to the disk radius would be wrong twice
        # over: a per-axis box of half-width r admits points of norm
        # r*sqrt(2), and it truncates legitimate interior points. Only the
        # action box and the separable lift axis are clipped here.
        from baselines.pcb_filters import FZ_ROBUST
        return [(-1.0, 1.0), (-1.0, 1.0), (-FZ_ROBUST, FZ_ROBUST)]
    if task == "SCREW":
        from baselines.screw_filters import RAD_LIM_N
        return [(-1.0, 1.0), (-1.0, 1.0), (-RAD_LIM_N, RAD_LIM_N)]
    return [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]


def _oracle_arm(cost_fn, bounds, iters, margin=-1e-9):
    """A per-action oracle tangent arm.

    The halfspace form corrects one constraint per step, so a set with k
    violated constraints needs k iterations to match a simultaneous
    projection. Iterations are set per task from the number of families.
    """
    from baselines.learned_cost import halfspace_correct

    def arm(obs, a):
        # The margin must match the interior the exact projector aims at,
        # or the comparison measures the margin rather than the method,
        # and a correction landing on the boundary falls outside it under
        # float32 conversion.
        x = halfspace_correct(cost_fn, obs, a, bounds, margin=margin,
                              iters=iters)
        return x.astype("float32"), {}
    return arm


# Per task: the unfiltered arm, an axis-aligned baseline, and the exact
# structured projection. The axis-aligned arms are NOT one method: each is
# named for what it actually does, since a static componentwise clip and a
# phase-blind bound are different baselines.
FILTERS = {
    "SCREW": {
        "none": _passthrough,
        # 12 iterations, not 2: the slab and the action box are not
        # separable, so correcting one constraint at a time is alternating
        # projection and converges geometrically toward their intersection.
        # Feasible 97% at 1 to 5 iterations, 100% by 10. The exact
        # projector reaches the intersection polygon in one step.
        "oracle_tangent": _oracle_arm(SF.safelayer_oracle_cost,
                                      _task_bounds("SCREW"), 12),
        "trix": SS.trix_scalar,
        "qp_matched": SS.qp_quadprog,
    },
    "PCB": {
        "none": _passthrough,
        "oracle_tangent": _oracle_arm(PF.safelayer_oracle_cost, _task_bounds("PCB"), 2,
                                      margin=_pcb_margin()),
        "box_clip": PF.box_clip,
        "trix": PF.trix_project,
    },
    "SNAP": {
        "none": _passthrough,
        "oracle_tangent": _oracle_arm(TF.snap_oracle_cost, _task_bounds("SNAP"), 3),
        "static_clip": PH.snap_static_clip,
        "trix": PH.snap_project,
    },
    "CRANK": {
        "none": _passthrough,
        "oracle_tangent": _oracle_arm(PH.crank_oracle_cost, _task_bounds("CRANK"), 4),
        "static_clip": PH.crank_static_clip,
        "trix": PH.crank_project,
    },
    "BATTERY": {
        "none": _passthrough,
        "oracle_tangent": _oracle_arm(TF.battery_oracle_cost, _task_bounds("BATTERY"), 3),
        "box_clip": TF.battery_box_clip,
        "box_clip_preventive": TF.battery_box_clip_preventive,
        "trix": TF.battery_project_flagged,
        "trix_preventive": TF.battery_project_preventive,
    },
    "BAYONET_S": {
        "none": _passthrough,
        "box_clip": BF.box_static,
        "trix": BF.projection_static,
    },
    "BAYONET_P": {
        "none": _passthrough,
        "box_blind_conservative": BF.box_blind_conservative,
        "box_blind_permissive": BF.box_blind_permissive,
        "box_phase_aware": BF.box_phase_aware,
        "trix": BF.projection_phase_aware,
    },
    "PRY": {
        "none": _passthrough,
        "oracle_tangent": _oracle_arm(TF.pry_oracle_cost, _task_bounds("PRY"), 3),
        "static_clip": TF.pry_static_clip,
        "trix": TF.pry_project,
    },
}


def constraint_hashes():
    out = {}
    for name, cls in ENVS.items():
        out[name] = cls().episode_summary(False)["constraint_hash"]
    return out


def make_env(task):
    return ENVS[task]()


def get_filter(task, method):
    fs = FILTERS[task]
    if method not in fs:
        raise KeyError(f"no filter '{method}' registered for {task}; "
                       f"available: {sorted(fs)}")
    return fs[method]
