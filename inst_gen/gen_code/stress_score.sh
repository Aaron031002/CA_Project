#!/bin/bash
# Run the official score_mlp.py on extra input images, the way the TA note describes:
# "put it (784 float32 values) in place of in_B1.bin in a copy of the repository and run
#  score_mlp.py there."  Your own repository is not changed.
#
# Usage (host terminal, from anywhere):
#   bash inst_gen/gen_code/stress_score.sh <copy dir> <image dir> [softmax variant]
#     <copy dir>   where to put the repository copy (created/overwritten)
#     <image dir>  folder of *.bin images (e.g. inst_gen/stress_inputs from stress_inputs.py)
#     variant      optional: regenerate the copy's case with gen_small_mlp.py --softmax <variant>
#                  (default: test the case files exactly as they are in this repository)
# Prints one line per image: the RESULT line of score_mlp.py (in_B1 = that image).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COPY="$1"; IMGS="$(cd "$2" && pwd)"; VARIANT="${3:-}"
IN=new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin

mkdir -p "$COPY"
rsync -a --delete --exclude docs --exclude sim_logs --exclude inst_gen/stress_inputs "$REPO"/ "$COPY"/
if [ -n "$VARIANT" ]; then
    bash "$COPY/start_env.sh" python3 inst_gen/gen_code/gen_small_mlp.py --softmax "$VARIANT" | tail -1
fi
for img in "$IMGS"/*.bin; do
    name="$(basename "$img" .bin)"
    cp "$img" "$COPY/$IN"
    res="$({ bash "$COPY/start_env.sh" python3 run_scripts/score_mlp.py 2>&1 || true; } | grep '^RESULT' || echo 'RESULT: (no result line)')"
    printf '%-16s %s\n' "$name" "$res"
done
