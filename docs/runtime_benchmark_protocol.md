# Runtime benchmark protocol

Defined on 25 September 2026 before the recorded timing runs, for E.4 and R1.6.
This is a new implementation benchmark. It does not recover or reuse the
unsupported historical latency figures.

## Measurement boundary

Time the unmodified `(obs, action) -> (action, info)` callable, including its
array conversion/allocation, per-call constraint construction, projection and
diagnostics. Exclude input generation, imports, policy/VLM inference, simulator
steps, communication, measurement storage and independent output validation.
Report OSQP object construction separately. Reuse that object, with its existing
warm-start/polishing settings, for subsequent calls. These are filter-call
latencies on one host, not end-to-end robot deadlines or worst-case guarantees.

## Inputs and methods

- Six tasks, each with 2,048 saved float32 observation/action pairs and seed
  20260925. Observations use the environments' own observation encoders, with
  no simulation rollout: reset states for state-independent filters; equal
  SNAP engaged/disengaged cases; CRANK angles before/after the rotation
  threshold; PRY insertion depths sampled over its declared range.
- Half of each action bank is uniform in the normalized cube `[-1,1]^3`.
  One quarter is obtained by scaling the TRiX projection by a factor in
  `[0.1,0.9]`. One quarter perturbs that projection toward the original proposal
  by `1e-4` normalized units, clipped to the action cube. These are deliberately
  constructed workload strata, not policy visitation frequencies or damage
  outcomes. Record the observed intervention fraction rather than assume the
  constructed strata are perfectly classified.
- Time every registered non-passthrough filter for these tasks. For SCREW also
  include the existing NumPy polygon implementation and reused OSQP. The
  scalar polygon implementation is the actual registered production TRiX arm.
- Only SCREW's analytical, active-set QP and OSQP calls are a matched
  projection comparison. Componentwise and oracle-tangent controls on the
  other tasks have different representations or specifications; their timing
  differences do not establish equivalent safety or a solver speedup.

## Timing protocol

- Five sequential, fresh Python processes on one allowed CPU core, with native
  numerical thread limits set to one before import. Record the CPU, OS, Python,
  package versions, affinity, observed process thread count and frequency policy.
  Do not alter system frequency controls or discard scheduler outliers.
- Each process warms every callable for 256 calls. Disable cyclic garbage
  collection during the measurement loop and record this choice. Reference
  counting and array allocation remain part of each call.
- Use `perf_counter_ns`, retaining every duration. Randomize the task and arm
  order within blocks of 64 inputs. Each arm receives the same ordered inputs
  for its task within a process; use a new seeded permutation per process.
- Keep worker stdout/stderr in a log. Any diagnostic work emitted internally
  by a solver remains within the measured call. The harness writes its raw
  measurement files after timing. Measure an empty timer-pair distribution,
  report it separately and do not subtract it from call durations.
- Primary summaries are median and 95th percentile of all retained calls,
  accompanied by per-process medians, maxima and stratum summaries. These are
  descriptive repeated-call statistics on one machine, not independent robot
  trials. Keep the raw integer nanoseconds, outputs, flags and call order.

## Correctness and provenance

Retain finite-output and recovery/failure counts for every arm. For the matched
SCREW implementations, independently check the slab, actuator and radial
inequalities and compare returned actions with the analytical projection.
Predeclare `1e-5` normalized units as the maximum allowed infinity-norm
difference and `1e-7` as the numerical feasibility tolerance. Report failures;
do not silently omit them or claim a speedup between inequivalent solutions.

Save input, code, protocol and raw-file SHA-256 hashes. The analysis command
must recreate the numerical summary and TeX table from the saved arrays without
rerunning timers. A fresh timing run will naturally have different durations.
The frozen training/VLM/hardware evidence archive is not modified.

## Commands

```bash
python3 -m pip install -r requirements-runtime.txt
python3 -B experiments/runtime_benchmark.py run --out results/runtime
python3 -B experiments/runtime_benchmark.py check --out results/runtime
python3 -B experiments/runtime_benchmark.py table --out results/runtime
```

`run` refuses to overwrite an existing output directory. Use a new `--out`
directory for an independent replication. No training or provider calls run.
