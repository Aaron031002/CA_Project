"""Timeline tracer (development helper).

For every data id an instruction produces, re-runs the simulator with
DBG_PIN_ID=<id> (a built-in Cache debug switch) and records the cycle at
which that id is committed to the core's cache ([PIN-WRITE]).  Prints, per
core, every instruction with the cycle its result became available.

Usage (inside the container, repo root):
    python3 inst_gen/gen_code/trace.py <case_dir>
"""
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = "new_code/build/bin/small_mlp_Acc"


def produced_id(tok):
    if not tok:
        return None
    op = tok[0].upper()
    if op == "DMA" and tok[1] == "DRAM" and tok[2] == "read":
        return int(tok[9])
    if op == "DMA" and tok[1] == "NoC" and tok[2] == "recv":
        return int(tok[4])
    if op == "ARRAY":
        i = tok.index("stride") + 2 if "stride" in tok else 8
        return int(tok[i + 1])
    if op == "VECTOR":
        return int(tok[5])
    if op == "TMU":
        return int(tok[tok.index("out") + 1])
    return None


def load(case_dir):
    progs = {}
    for c in range(4):
        rows = []
        with open(os.path.join(case_dir, f"core_{c}_insts.txt")) as f:
            for line in f:
                s = re.split(r"(?:^|(?<=\s))#", line, maxsplit=1)[0].strip()
                if not s:
                    continue
                rows.append((s, produced_id(s.split())))
        progs[c] = rows
    return progs


def pin_time(case_dir, core, data_id):
    env = dict(os.environ, DBG_PIN_ID=str(data_id), DBG_PIN_CORE=f"Core_{core}.")
    p = subprocess.run([BIN, case_dir], capture_output=True, text=True, env=env, cwd=ROOT)
    m = re.search(r"\[PIN-WRITE\] @(\d+) ns", p.stdout)
    return int(m.group(1)) if m else None


def main(case_dir):
    progs = load(case_dir)
    jobs = [(c, did) for c, rows in progs.items() for _, did in rows if did]
    with ThreadPoolExecutor(max_workers=8) as ex:
        times = dict(zip(jobs, ex.map(lambda j: pin_time(case_dir, *j), jobs)))
    p = subprocess.run([BIN, case_dir], capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"done at .* \((\d+) cycles\)", p.stderr)
    for c, rows in progs.items():
        print(f"===== core {c}")
        for s, did in rows:
            t = times.get((c, did)) if did else None
            short = s.replace("new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp/", "")
            short = short.replace("new_code/DRAM_sim/DRAM_data/", "")
            print(f"{'' if t is None else t:>6}  {short}")
    print("TOTAL", m.group(1) if m else None)


if __name__ == "__main__":
    main(sys.argv[1])
