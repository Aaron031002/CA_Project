"""Run one case folder in the simulator and report latency (+ optional correctness).

Usage (inside the container, from the repository root):
    python3 inst_gen/gen_code/run_case.py <case_dir> [--check] [--input <bin>]

This is a quick development helper only; the official check is
run_scripts/score_mlp.py.
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "new_code/DRAM_sim/DRAM_data/out_tensor/mlp/small_mlp_out_B1.bin"
BIN = "new_code/build/bin/small_mlp_Acc"


def run(case_dir, verbose=False, env=None):
    os.chdir(ROOT)
    np.full((1, 10), np.nan, dtype=np.float32).tofile(OUT)
    p = subprocess.run([BIN, case_dir, OUT, "1", "10"], capture_output=True, text=True,
                       timeout=600, env=env)
    m = re.search(r"done at .* \((\d+) cycles\)", p.stderr)
    lat = int(m.group(1)) if m else None
    if verbose or lat is None or p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-25:])
        print(tail)
    return lat, p


def check(input_file=None):
    sys.path.insert(0, os.path.join(ROOT, "run_scripts"))
    args = [sys.executable, "run_scripts/compare_output_mlp.py"]
    if input_file:
        args += ["--input", input_file]
    p = subprocess.run(args, capture_output=True, text=True)
    return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    lat, p = run(a.case_dir, verbose=a.v)
    print(f"latency={lat} rc={p.returncode}")
    if a.check:
        print(check())
