"""
metrics.py -- one metric definition shared by every v2 environment.

The submitted manuscript reported a single headline number per cell,
"violation rate" as violating timesteps divided by total timesteps. That
statistic is not sufficient on its own:

  * it is distorted by episode length. One violation in a one-step
    episode reads as 100%; several violations across a 700-step episode
    read as under 1%. A method that terminates quickly and destructively
    is scored more favourably than a slow careful one.
  * it says nothing about whether the task was completed, or completed
    without damage.
  * it conflates a sustained near-miss with a single catastrophic event.

DESIGN RULES
  1. Episode-weighted and timestep-weighted statistics are kept separate,
     and every rate is reported alongside the raw numerator and
     denominator it came from. Per-seed percentages are never averaged
     without the counts that produced them.
  2. Timeout is not failure, and completion is not safe completion. The
     terminal categories partition the episodes exactly and are carried
     separately all the way into table generation.
  3. This module consumes ONLY the shared info contract emitted by the
     environments. It contains no task-specific interpretation: violation
     families are discovered from the "vio_*" keys present in info,
     severity from the matching "exceed_*" keys, and terminal
     classification comes from benchmark.outcome.classify(). Environments
     define outcomes; the evaluator aggregates them.

TERMINAL PARTITION
Categories come from benchmark.outcome.classify(), which is the single
implementation of the rule and is shared with the environments, so the
evaluator and the tasks cannot disagree about what an episode was. Five
categories partition the episodes exactly:

    safe_completion, unsafe_completion_intact, destructive_completion,
    mechanical_failure, timeout

Completed episodes are separated by two independent bits, whether a
constraint fired and whether damage resulted, because neither substitutes
for the other: a sustained near-miss is not safe, and damage without any
violation is reachable under continuous wear models and is still
destructive.

AGGREGATION LAYERS
Two layers answer different questions and both are produced.
  * POOLED sums raw counts across seeds, giving exact numerators and
    denominators for invariant checking and event incidence.
  * SEED-LEVEL computes each metric within a seed and reports the mean
    and dispersion across seeds. The independent experimental unit is the
    TRAINING SEED, not the evaluation episode: three seeds with 100
    episodes each is n = 3 trained policies, not n = 300. Evaluation
    episodes quantify within-policy stochasticity; seeds quantify
    algorithm variability.

EXCEEDANCE CONVENTION
Environments emit exceed_<family> alongside each vio_<family> flag, as

    e = max(0, (x - s) / s)

where x is the constrained quantity and s its documented threshold. So e
is the FRACTIONAL EXCESS ABOVE the threshold, not the ratio to it:

    e = 0.00  ->  at or below the limit
    e = 0.13  ->  13% above the limit   (x = 1.13 s)
    e = 3.24  -> 324% above the limit   (x = 4.24 s)

Prose and captions must use "above the limit", never "of the limit".
Normalising by s makes magnitudes comparable across newtons,
newton-metres, velocities and temperatures, which raw units are not. The
normalisation rule and every scale s are recorded in each environment's
constraint hash, so changing a divisor invalidates stale comparisons
rather than silently shifting the numbers.

Severity is reported both over all timesteps and conditional on
violating timesteps. The conditional figure is the meaningful one: long
safe stretches dilute the all-steps mean toward zero, understating real
excursions by several fold.

SAFETY-FILTER STATISTICS
Intervention magnitude, projection latency, infeasibility and recovery
events describe the safety mechanism rather than the environment, so
they are supplied by the caller through step(filter_info=...), with keys
    intervention_norm, latency_us, infeasible, recovery
"""

import numpy as np

from benchmark.outcome import CATEGORIES, classify
from benchmark import taxonomy as TAX

TERMINAL_FIELDS = ("completion", "damage_outcome", "safe_completion",
                   "episode_with_violation")


def _family_exceedance(eps, vio_key):
    """Per-family exceedance, keyed from the matching exceed_* field."""
    key = "exceed_" + vio_key[len("vio_"):]
    tot = {"sum": 0.0, "n": 0, "max": 0.0, "sum_vio": 0.0, "n_vio": 0}
    for e in eps:
        f = e["exceed_by_family"].get(key)
        if not f:
            continue
        tot["sum"] += f["sum"]; tot["n"] += f["n"]
        tot["max"] = max(tot["max"], f["max"])
        tot["sum_vio"] += f["sum_vio"]; tot["n_vio"] += f["n_vio"]
    if tot["n"] == 0:
        return {}
    return {
        "exceedance_mean_all_steps": tot["sum"] / tot["n"],
        "exceedance_mean_violating_steps":
            tot["sum_vio"] / tot["n_vio"] if tot["n_vio"] else None,
        "exceedance_max": tot["max"],
    }


class EpisodeRecorder:
    """Accumulates per-step and per-episode statistics for one cell."""

    def __init__(self, task, method, seed=None):
        self.task = task
        self.method = method
        self.seed = seed
        self.episodes = []
        self._cur = None
        self.constraint_hash = None

    def start_episode(self):
        self._cur = {
            "steps": 0,
            "violating_steps": 0,
            "violations_by_family": {},
            "first_violation_step": None,
            "exceedance_max": 0.0,
            "exceedance_sum": 0.0,
            "exceedance_n": 0,
            "exceedance_sum_vio": 0.0,
            "exceedance_n_vio": 0,
            "exceedance_cumulative": 0.0,
            "exceed_by_family": {},
            "return": 0.0,
            "intervention_norms": [],
            "latencies_us": [],
            "infeasible_steps": 0,
            "recovery_events": 0,
        }

    def step(self, info, reward=0.0, filter_info=None):
        c = self._cur
        c["steps"] += 1
        c["return"] += float(reward)
        if self.constraint_hash is None:
            self.constraint_hash = info.get("constraint_hash")

        if int(info.get("vio", 0)):
            c["violating_steps"] += 1
            if c["first_violation_step"] is None:
                c["first_violation_step"] = c["steps"]

        # Layer separation: only families declared outcome-affecting may
        # convert an intact completion into an unsafe one. Diagnostic
        # families are counted and reported alongside.
        out_fams, diag_fams = TAX.split(self.task, info)
        if out_fams:
            c["outcome_violating_steps"] = c.get("outcome_violating_steps", 0) + 1
            c["any_outcome_violation"] = True
        if diag_fams:
            c["diagnostic_steps"] = c.get("diagnostic_steps", 0) + 1
            c["any_diagnostic"] = True

        # violation families discovered from the contract, not hardcoded
        for k, v in info.items():
            if k.startswith("vio_") and int(v):
                c["violations_by_family"][k] = \
                    c["violations_by_family"].get(k, 0) + 1

        # normalized exceedance per family, discovered from the contract
        step_exceed = 0.0
        for k, v in info.items():
            if k.startswith("exceed_"):
                e = float(v)
                fam = c["exceed_by_family"].setdefault(
                    k, {"sum": 0.0, "n": 0, "max": 0.0,
                        "sum_vio": 0.0, "n_vio": 0})
                fam["sum"] += e
                fam["n"] += 1
                fam["max"] = max(fam["max"], e)
                if e > 0.0:
                    fam["sum_vio"] += e
                    fam["n_vio"] += 1
                step_exceed = max(step_exceed, e)
        c["exceedance_sum"] += step_exceed
        c["exceedance_n"] += 1
        c["exceedance_cumulative"] += step_exceed
        c["exceedance_max"] = max(c["exceedance_max"], step_exceed)
        if step_exceed > 0.0:
            c["exceedance_sum_vio"] += step_exceed
            c["exceedance_n_vio"] += 1

        if filter_info:
            if "intervention_norm" in filter_info:
                c["intervention_norms"].append(
                    float(filter_info["intervention_norm"]))
            if "latency_us" in filter_info:
                c["latencies_us"].append(float(filter_info["latency_us"]))
            if filter_info.get("infeasible"):
                c["infeasible_steps"] += 1
            if filter_info.get("recovery"):
                c["recovery_events"] += 1

    def end_episode(self, env, completed):
        term = env.episode_summary(bool(completed))
        missing = [f for f in TERMINAL_FIELDS if f not in term]
        if missing:
            raise ValueError(
                f"{self.task}: environment episode_summary is missing "
                f"{missing}; the evaluator does not infer terminal state")
        rec = dict(self._cur)
        rec.update(term)
        rec.setdefault("any_outcome_violation", False)
        rec.setdefault("any_diagnostic", False)
        rec.setdefault("outcome_violating_steps", 0)
        rec.setdefault("diagnostic_steps", 0)
        # The environment ORs every family into episode_with_violation; the
        # taxonomy decides which of those actually bear on the outcome.
        rec["episode_with_violation_any"] = term["episode_with_violation"]
        rec["episode_with_violation"] = bool(rec["any_outcome_violation"])
        rec["taxonomy_hash"] = TAX.taxonomy_hash(self.task)
        rec["category"] = classify(term["completion"], term["damage_outcome"],
                                   rec["episode_with_violation"])
        expected = classify(term["completion"], term["damage_outcome"],
                            term["episode_with_violation"])
        if "category" in term and term["category"] != expected:
            raise ValueError(
                f"{self.task}: environment category {term['category']!r} "
                f"disagrees with classify() {expected!r}")
        self.episodes.append(rec)
        self._cur = None

    # ------------------------------------------------------------- summary
    def summary(self):
        eps = self.episodes
        n = len(eps)
        if n == 0:
            return {"task": self.task, "method": self.method, "episodes": 0}

        def cnt(pred):
            return sum(1 for e in eps if pred(e))

        by_cat = {c: cnt(lambda e, c=c: e["category"] == c) for c in CATEGORIES}
        safe = by_cat["safe_completion"]
        unsafe_intact = by_cat["unsafe_completion_intact"]
        destructive = by_cat["destructive_completion"]
        mech = by_cat["mechanical_failure"]
        timeouts = by_cat["timeout"]
        completed = safe + unsafe_intact + destructive
        with_vio = cnt(lambda e: e["episode_with_violation"])
        with_any = cnt(lambda e: e.get("episode_with_violation_any", False))
        with_diag = cnt(lambda e: e.get("any_diagnostic", False))

        total_steps = sum(e["steps"] for e in eps)
        total_vio_steps = sum(e["violating_steps"] for e in eps)

        dmg = {}
        for e in eps:
            dmg[e["damage_outcome"]] = dmg.get(e["damage_outcome"], 0) + 1

        fam_steps, fam_eps = {}, {}
        for e in eps:
            for k, v in e["violations_by_family"].items():
                fam_steps[k] = fam_steps.get(k, 0) + v
                fam_eps[k] = fam_eps.get(k, 0) + 1

        firsts = [e["first_violation_step"] for e in eps
                  if e["first_violation_step"] is not None]
        comp_lens = [e["steps"] for e in eps if e["completion"] == "completed"]
        rets = [e["return"] for e in eps]
        interv = [x for e in eps for x in e["intervention_norms"]]
        lat = [x for e in eps for x in e["latencies_us"]]
        exc_sum = sum(e["exceedance_sum"] for e in eps)
        exc_n = sum(e["exceedance_n"] for e in eps)

        def pct(num, den):
            return 100.0 * num / den if den else 0.0

        return {
            "task": self.task,
            "method": self.method,
            "seed": self.seed,
            "constraint_hash": self.constraint_hash,
            "episodes": n,

            # --- terminal partition (episode-weighted) -------------------
            "completion_rate": pct(completed, n),
            "safe_completion_rate": pct(safe, n),
            "unsafe_completion_intact_rate": pct(unsafe_intact, n),
            "destructive_completion_rate": pct(destructive, n),
            "mechanical_failure_rate": pct(mech, n),
            "timeout_rate": pct(timeouts, n),
            "damage_breakdown_rate": {k: pct(v, n) for k, v in dmg.items()},

            # --- violations: episode- and timestep-weighted, kept apart --
            "episodes_with_any_violation_rate": pct(with_vio, n),
            "violations_per_episode_mean": total_vio_steps / n,
            "violating_timestep_rate": pct(total_vio_steps, total_steps),
            # Reported alongside the outcome measure so the layer split
            # conceals nothing: a reader sees both.
            "episodes_with_any_flag_rate": pct(with_any, n),
            "episodes_with_diagnostic_rate": pct(with_diag, n),
            "diagnostic_timestep_rate": pct(
                sum(e.get("diagnostic_steps", 0) for e in eps), total_steps),
            "first_violation_step_mean":
                float(np.mean(firsts)) if firsts else None,
            "exceedance_mean_all_steps": exc_sum / exc_n if exc_n else None,
            "exceedance_mean_violating_steps":
                (sum(e["exceedance_sum_vio"] for e in eps)
                 / sum(e["exceedance_n_vio"] for e in eps))
                if sum(e["exceedance_n_vio"] for e in eps) else None,
            "exceedance_max":
                max(e["exceedance_max"] for e in eps) if exc_n else None,
            "cumulative_exceedance_per_episode_mean":
                sum(e["exceedance_cumulative"] for e in eps) / n,
            "violations_by_constraint": {
                k: {"violating_steps": fam_steps[k],
                    "episodes_affected": fam_eps[k],
                    "violating_timestep_rate": pct(fam_steps[k], total_steps),
                    "episode_rate": pct(fam_eps[k], n),
                    **_family_exceedance(eps, k)}
                for k in sorted(fam_steps)},

            # --- effort and return ---------------------------------------
            "completion_steps_median":
                float(np.median(comp_lens)) if comp_lens else None,
            "completion_steps_p90":
                float(np.percentile(comp_lens, 90)) if comp_lens else None,
            "return_mean": float(np.mean(rets)),
            "return_std": float(np.std(rets)),

            # --- safety mechanism ----------------------------------------
            "intervention_norm_mean":
                float(np.mean(interv)) if interv else None,
            "intervention_norm_p95":
                float(np.percentile(interv, 95)) if interv else None,
            "projection_latency_us_p50":
                float(np.percentile(lat, 50)) if lat else None,
            "projection_latency_us_p95":
                float(np.percentile(lat, 95)) if lat else None,
            "projection_latency_us_p99":
                float(np.percentile(lat, 99)) if lat else None,
            "infeasibility_rate":
                pct(sum(e["infeasible_steps"] for e in eps), total_steps),
            "recovery_event_rate":
                pct(sum(e["recovery_events"] for e in eps), total_steps),

            # --- raw counts, so rates are never averaged without them ----
            "counts": {
                "episodes": n,
                "safe_completions": safe,
                "unsafe_completions_intact": unsafe_intact,
                "destructive_completions": destructive,
                "mechanical_failures": mech,
                "timeouts": timeouts,
                "episodes_with_violation": with_vio,
                "total_steps": total_steps,
                "violating_steps": total_vio_steps,
            },
        }

    # ----------------------------------------------------------- invariants
    def validate(self):
        """Assert the internal consistency the tables depend on."""
        s = self.summary()
        if s["episodes"] == 0:
            return s
        c = s["counts"]
        assert (c["safe_completions"] + c["unsafe_completions_intact"]
                + c["destructive_completions"] + c["mechanical_failures"]
                + c["timeouts"]) == c["episodes"], \
            "terminal categories do not partition the episodes"
        assert s["safe_completion_rate"] <= s["completion_rate"] + 1e-9
        assert s["destructive_completion_rate"] <= s["completion_rate"] + 1e-9
        assert c["violating_steps"] <= c["total_steps"]
        for e in self.episodes:
            cat = e["category"]
            assert cat in CATEGORIES
            # exactly one category, by construction of classify()
            if e["completion"] == "completed" and e["episode_with_violation"]:
                assert cat != "safe_completion", \
                    "a completed episode with a violation was counted safe"
            if e["damage_outcome"] != "intact" and e["completion"] == "completed":
                assert cat == "destructive_completion"
        return s


def format_summary(s) -> str:
    """One-line rendering for sweep logs."""
    if s.get("episodes", 0) == 0:
        return f"{s['task']:8s} {s['method']:12s}  (no episodes)"
    return (f"{s['task']:8s} {s['method']:14s} "
            f"safe {s['safe_completion_rate']:5.1f}%  "
            f"destr {s['destructive_completion_rate']:5.1f}%  "
            f"timeout {s['timeout_rate']:5.1f}%  "
            f"ep_vio {s['episodes_with_any_violation_rate']:5.1f}%  "
            f"step_vio {s['violating_timestep_rate']:5.1f}%")


class SeedAggregator:
    """Combines per-seed summaries for one (task, method) cell.

    Produces two layers, because they answer different questions and one
    cannot be recovered from the other:

      per_seed      each seed's full summary, unmodified
      seed_summary  mean and dispersion ACROSS SEEDS for each scalar
                    metric. The independent experimental unit is the
                    training seed, so this is what belongs in the primary
                    comparative table. With few seeds the standard
                    deviation is itself unstable, so the seed count is
                    always reported beside it.
      pooled_counts raw numerators and denominators summed across seeds,
                    for exact incidence and invariant checking. Rates
                    derived here are pooled, NOT the mean of per-seed
                    rates: those differ whenever seeds have unequal
                    episode or step counts.

    Skewed quantities (latency, completion time) also carry median and
    interquartile range, which describe them better than mean and SD.
    """

    SKEWED = ("projection_latency_us_p50", "projection_latency_us_p95",
              "projection_latency_us_p99", "completion_steps_median",
              "completion_steps_p90")

    def __init__(self, task, method):
        self.task = task
        self.method = method
        self.per_seed = []

    def add(self, summary):
        if summary.get("episodes", 0) == 0:
            return
        self.per_seed.append(summary)

    def result(self):
        seeds = self.per_seed
        if not seeds:
            return {"task": self.task, "method": self.method, "seeds": 0}

        hashes = {s.get("constraint_hash") for s in seeds}
        if len(hashes) > 1:
            raise ValueError(
                f"{self.task}/{self.method}: seeds used different constraint "
                f"definitions {hashes}; results are not comparable")

        scalar_keys = [k for k, v in seeds[0].items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]
        seed_summary = {}
        for k in scalar_keys:
            vals = [s[k] for s in seeds if isinstance(s.get(k), (int, float))]
            if not vals:
                continue
            entry = {"mean": float(np.mean(vals)),
                     "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                     "n_seeds": len(vals)}
            if k in self.SKEWED:
                entry["median"] = float(np.median(vals))
                entry["iqr"] = [float(np.percentile(vals, 25)),
                                float(np.percentile(vals, 75))]
            seed_summary[k] = entry

        pooled = {}
        for s in seeds:
            for k, v in s.get("counts", {}).items():
                pooled[k] = pooled.get(k, 0) + v

        def pct(num, den):
            return 100.0 * pooled.get(num, 0) / pooled[den] if pooled.get(den) else 0.0

        pooled_rates = {
            "safe_completion_rate": pct("safe_completions", "episodes"),
            "unsafe_completion_intact_rate":
                pct("unsafe_completions_intact", "episodes"),
            "destructive_completion_rate": pct("destructive_completions", "episodes"),
            "mechanical_failure_rate": pct("mechanical_failures", "episodes"),
            "timeout_rate": pct("timeouts", "episodes"),
            "episodes_with_any_violation_rate":
                pct("episodes_with_violation", "episodes"),
            "violating_timestep_rate": pct("violating_steps", "total_steps"),
        }

        return {
            "task": self.task,
            "method": self.method,
            "seeds": len(seeds),
            "constraint_hash": seeds[0].get("constraint_hash"),
            "per_seed": seeds,
            "seed_summary": seed_summary,
            "pooled_counts": pooled,
            "pooled_rates": pooled_rates,
        }

    def validate(self):
        r = self.result()
        if r.get("seeds", 0) == 0:
            return r
        p = r["pooled_counts"]
        assert (p["safe_completions"] + p["unsafe_completions_intact"]
                + p["destructive_completions"] + p["mechanical_failures"]
                + p["timeouts"]) == p["episodes"], \
            "pooled terminal categories do not partition the episodes"
        assert p["violating_steps"] <= p["total_steps"]
        return r


def format_seed_result(r) -> str:
    """Primary-table rendering: mean +- SD across training seeds."""
    if r.get("seeds", 0) == 0:
        return f"{r['task']:8s} {r['method']:14s}  (no seeds)"
    ss = r["seed_summary"]

    def ms(key):
        e = ss.get(key)
        if not e:
            return "   n/a"
        sd = f" +-{e['sd']:4.1f}" if e["sd"] is not None else "      "
        return f"{e['mean']:5.1f}{sd}"

    p = r["pooled_counts"]
    return (f"{r['task']:8s} {r['method']:14s} "
            f"safe {ms('safe_completion_rate')}  "
            f"unsafe-intact {ms('unsafe_completion_intact_rate')}  "
            f"destr {ms('destructive_completion_rate')}  "
            f"[n={r['seeds']} seeds, pooled "
            f"{p['safe_completions']}/{p['episodes']}]")
