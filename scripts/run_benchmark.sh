#!/bin/bash
# Run the canonical DISASM-Bench benchmark.
# Produces Table 4 from the paper.
#
# Usage:
#   ./scripts/run_benchmark.sh              # 4 core tasks (default)
#   ./scripts/run_benchmark.sh --all        # All 6 tasks
#   ./scripts/run_benchmark.sh --quick      # Quick test (10K steps)

set -e
cd "$(dirname "$0")/.."

if [ "$1" = "--all" ]; then
    python3 -c "
from benchmark.disasm_bench import run_benchmark, TASKS_ALL
run_benchmark(steps_per_cell=100000, n_seeds=3, tasks=TASKS_ALL)
"
elif [ "$1" = "--quick" ]; then
    python3 -c "
from benchmark.disasm_bench import run_benchmark, TASKS
run_benchmark(steps_per_cell=10000, n_seeds=1, tasks=TASKS)
"
else
    python3 benchmark/disasm_bench.py
fi
