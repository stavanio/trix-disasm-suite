# Recorded runtime results: E.4 and R1.6

Measured on 25 September 2026 under the [predefined protocol](runtime_benchmark_protocol.md).
The new measurements replace the unsupported historical timing claim.

## Main observations

22 existing implementations, 2,048 inputs per task and five fresh processes
produce 225,280 recorded complete-filter calls. Each method receives the same
input sequence as the other methods for that task in each process. Inputs,
outputs, integer nanosecond durations, recovery flags, execution order and
environment metadata are saved under [results/runtime](../results/runtime/summary.json).

| Matched SCREW implementation | Median, microseconds | 95th percentile, microseconds |
|---|---:|---:|
| Production TRiX, scalar polygon | 1.255 | 1.556 |
| Active-set QP | 12.578 | 15.534 |
| Reused OSQP | 14.924 | 19.895 |
| Alternative NumPy polygon | 18.163 | 20.898 |

All matched SCREW arms pass the predefined action-agreement tolerance of 1e-5
and constraint-residual tolerance of 1e-7. Active-set and NumPy outputs equal
the scalar outputs numerically in float32 on the bank; the maximum OSQP
action-coordinate difference is 1.0132789611816406e-6. No matched-arm call
returns a recovery output. This finite-bank check is not a universal proof.

Registered TRiX medians across the six tasks range from 1.123 microseconds
(PCB) to 11.770 microseconds (CRANK); BATTERY preventive takes 3.122
microseconds. All implementation/stratum/process summaries, maxima and
reported infeasible/recovery counts are retained in the machine-readable
summary. The script regenerates the manuscript's Table 14 from these records.

The scalar implementation is faster than the matched QP implementations on
this host and workload. The NumPy implementation of the same analytical
geometry is slower than both QP implementations by median. The result is
therefore specific to these implementations, rather than evidence that
analytical projection universally outperforms numerical optimization.
Other task controls implement different restrictions; overhead comparisons
there do not establish equivalent safety. CRANK's frame-dependent TRiX call
has higher median cost than its static clip, and both are reported.

## Scope and repeatability

- Intel Core Ultra 9 275HX, CPU 0, one observed worker thread, CPython 3.12.3.
  Exact packages and host settings are recorded for every process.
- The callable includes allocation, constraint work and diagnostics. Imports,
  policy/VLM inference, simulation, communication and measurement storage are
  outside the interval. OSQP construction is reported separately; its problem
  is reused, with warm starting and the existing polishing configuration.
- Empty timer-pair median: 0.025 microseconds, not subtracted. All scheduler
  outliers are retained. Largest recorded call: 352.557 microseconds (OSQP);
  largest production TRiX call: 335.812 microseconds (CRANK).
- CPU frequency scaling remains enabled. These are measured distributions
  from one machine, not worst-case bounds, a new episode-success experiment,
  an end-to-end robot benchmark or independent hardware replicates.
- Workload fractions are deliberately constructed, not measured policy
  visitation frequencies. No policy is trained and no environment is stepped.
- One initial attempt stopped after the first worker's measurements because
  provenance collection encountered a third-party virtual module filename.
  The file-existence guard was corrected before the complete five-process run.
  That incomplete attempt is kept locally in `work/runtime-benchmark/aborted_provenance_capture`
  outside the repository and is not included in the reported results. No
  latency-based run selection or exclusion was performed.

## Traceability

- [Experiment and saved-record checker](../experiments/runtime_benchmark.py)
- [Pinned measurement environment](../requirements-runtime.txt)
- [Input/source/raw hashes](../results/runtime/manifest.json)
- [Full numerical summary](../results/runtime/summary.json)
- [Generated TeX table](../manuscript/tables/runtime_latency.tex)

`python3 -B experiments/runtime_benchmark.py check --out results/runtime`
verifies the hashes, schedules, counts and recalculated summaries. It does not
rerun timing. `table` regenerates the table; `run --out NEW_DIRECTORY` performs
a new timing replication. Duration bytes will naturally differ between runs.

E.4 and R1.6 now have measured evidence and revised responses. Verification
against the original reviewer reports remains a separate editorial task.
