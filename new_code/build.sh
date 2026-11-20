#!/usr/bin/env bash
# build.sh — configure and build the small_mlp testbench (small_mlp_Acc).
#
# Generate the testbench source first with run_scripts/gen_acc_from_case.sh,
# which writes new_code/test/small_mlp_Acc.cpp. This script then re-runs
# `cmake ..` (so a newly generated *_Acc.cpp is picked up --
# test/CMakeLists.txt discovers it with file(GLOB)) and builds the target in
# new_code/build/.
#
# Lives at new_code/build.sh (NOT new_code/build/build.sh) because
# new_code/build/ is the CMake out-of-source build directory itself
# (gitignored), so a hand-written script belongs one level up.
#
# Usage (from any directory):
#   new_code/build.sh            # build small_mlp_Acc
#   new_code/build.sh -j8        # pass a parallelism flag through
#
# See USER_GUIDE.md §14 and new_code/README.md §2.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

TARGET=small_mlp_Acc

usage() {
    echo "Usage: $0 [-jN]"
    exit 1
}

jobs=""
for arg in "$@"; do
    case "$arg" in
        -j*) jobs="$arg" ;;
        *)   usage ;;
    esac
done

if [[ ! -f "test/${TARGET}.cpp" ]]; then
    echo "ERROR: new_code/test/${TARGET}.cpp does not exist yet." \
         "Run 'run_scripts/gen_acc_from_case.sh' from the repo root first." >&2
    exit 1
fi

mkdir -p build
cd build
cmake ..
echo "[build.sh] Building: ${TARGET}"
exec cmake --build . --target "${TARGET}" ${jobs:+"$jobs"}
