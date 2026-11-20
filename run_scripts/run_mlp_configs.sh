#!/bin/bash
# run_mlp_configs.sh — quick runner for the small_mlp case (batch 1, fixed
# 2x2 mesh): runs bin/small_mlp_Acc on inst_gen/MLP_small_mlp/B1/ and prints
# the testbench result. For debugging only; run_scripts/score_mlp.py does
# the full check (case files + run + output comparison).
#
# Prerequisites:
#   - the provided tensors under new_code/DRAM_sim/DRAM_data/;
#   - your case folder inst_gen/MLP_small_mlp/B1/ (core_0..3_insts.txt,
#     dram_map.yaml), see USER_GUIDE.md §9-§12;
#   - the binary: run_scripts/gen_acc_from_case.sh, then new_code/build.sh.
#
# The output file is filled with NaN before the run, so values the run
# doesn't write can't pass the testbench's finite-value check as stale
# results from an earlier run.
#
# Usage:
#   run_scripts/run_mlp_configs.sh
#
# Logs -> $REPO_ROOT/sim_logs/mlp_small_mlp_B1.{stdout,stderr}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CFG=small_mlp
B=1
NC=10
BIN="new_code/build/bin/${CFG}_Acc"
IDIR="inst_gen/MLP_${CFG}/B${B}"
OUTFILE="new_code/DRAM_sim/DRAM_data/out_tensor/mlp/${CFG}_out_B${B}.bin"
LOG_DIR="sim_logs"
TAG="${CFG}_B${B}"

case "${1:-}" in
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    "") ;;
    *) echo "This script takes no arguments." >&2; exit 1 ;;
esac

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN missing -- run 'run_scripts/gen_acc_from_case.sh' then 'new_code/build.sh' first" >&2
    exit 1
fi
if [ ! -f "$IDIR/core_0_insts.txt" ]; then
    echo "ERROR: $IDIR has no core_0_insts.txt -- write your case files first (USER_GUIDE.md §9-§12)" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$OUTFILE")"
python3 -c "import numpy as np; np.full(($B, $NC), np.nan, dtype=np.float32).tofile('$OUTFILE')"

echo "Running $TAG ..."
set +e
"$BIN" "$IDIR" "$OUTFILE" "$B" "$NC" > "$LOG_DIR/mlp_${TAG}.stdout" 2> "$LOG_DIR/mlp_${TAG}.stderr"
rc=$?
set -e
grep -E '^\[TB\] All .* done at|^\[TB\] .* checks,|WATCHDOG|^\[FAIL\]' "$LOG_DIR/mlp_${TAG}.stderr" || true
grep -E '^Fatal:' "$LOG_DIR/mlp_${TAG}.stdout" || true
if [ $rc -eq 0 ]; then
    echo "DONE (exit 0). Logs: $LOG_DIR/mlp_${TAG}.stderr"
else
    echo "FAILED (exit $rc). Logs: $LOG_DIR/mlp_${TAG}.stderr / .stdout"
fi
exit $rc
