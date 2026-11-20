"""score_mlp.py — full check of the small_mlp case (batch 1, fixed 2x2 mesh).

This script:
  1. checks the case folder inst_gen/MLP_small_mlp/B1/ (exactly one
     core_<i>_insts.txt per core of the fixed 2x2 mesh, dram_map.yaml
     num_cores), and the assignment's rules: instructions read only the
     provided input/weight files and write only the output file,
     dram_map.yaml has an entry (owners among cores 0..3) for every file
     the instructions name and uses only path/owner/owners/kind, and no Vector mode
     from the not-allowed list (MoE routing, quantization) is used;
  2. runs new_code/build/bin/small_mlp_Acc on it and reads the latency
     (cycles) from the testbench's "done at ... (N cycles)" line;
  3. checks that the output file (reset to NaN before the run) was fully
     written by this run and passes run_scripts/compare_output_mlp.py;
  4. repeats 2-3 with other input images in place of in_B1.bin: a second
     provided MNIST image and a random image generated for this check. The
     program has to be correct for any input, and its latency must be the
     same for all of them.

It prints PASS with the latency in cycles, or FAIL with the reason.

Usage:
    python3 run_scripts/score_mlp.py
    python3 run_scripts/score_mlp.py --inst-root /path/to/inst_gen   # case folder elsewhere
    python3 run_scripts/score_mlp.py --seed 1234   # repeat a run's random image (seed in score.json)
"""
import argparse
import datetime
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO_ROOT, "new_code", "DRAM_sim", "DRAM_data")

CONFIG = "small_mlp"
BATCH = 1
MESH_W, MESH_H = 2, 2          # fixed accelerator, built into the testbench
IN_DIM = 784
NUM_CLASSES = 10

# The testbench's own lines start at the beginning of a line; the last one counts.
_DONE_RE = re.compile(r"^\[TB\] All (\d+) cores .* done at .* \((\d+) cycles\)$", re.M)
_CHECKS_RE = re.compile(r"^\[TB\] (\d+) checks, (\d+) failures$", re.M)
_CORE_FILE_RE = re.compile(r"^core_(\d+)_insts\.txt$")
# Simulator lines that say why a run stopped. "Fatal: ..." / "Error: ..." (SystemC, e.g.
# MainCtrl/DECODE, NoC destination out of range, an unreadable DRAM file) and the testbench's
# "[FAIL] ..." / "[ERROR] ..." lines, plus unit messages such as "[CACHE-WREQ-MISMATCH] ...".
_FAIL_PREFIXES = ("[FAIL] ", "[ERROR] ", "Fatal: ", "Error: ")
_UNIT_ERROR_RE = re.compile(r"^\[[A-Z0-9_-]*(MISMATCH|ERROR|FATAL)[A-Z0-9_-]*\]")
_SIGNAL_HINTS = {
    "SIGFPE": "arithmetic error in the simulator, usually an instruction with a size or dim of 0",
    "SIGSEGV": "the simulator crashed, usually on an out-of-range size or index",
}
# What some unit messages usually mean in this assignment.
_MESSAGE_HINTS = {
    "CACHE-WREQ-MISMATCH": "usually a DMA NoC recv whose size differs from the size of the matching send",
}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_output_mlp  # noqa: E402  (same folder; provides the PyTorch reference)

# Vector modes the simulator accepts (they serve other workloads) but this assignment does not
# allow: MoE routing (TOP_K, SPARSE_GATE) and the quantization family. Their timing model does
# not apply to this MLP, so a mapping using them would not be comparable with the others.
NOT_ALLOWED_VECTOR_MODES = ("TOP_K", "SPARSE_GATE", "REDUCE_MAX_ABS", "QUANTIZE_DEQUANT_Q8_0",
                            "QUANTIZE_DEQUANT_Q8_0_EXT_SCALE", "QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX")
# dram_map.yaml keys this assignment allows (the simulator parses more, for other workloads).
ALLOWED_MAP_KEYS = ("version", "num_cores", "entries")
ALLOWED_ENTRY_KEYS = ("path", "owner", "owners", "kind")


def inst_tokens(line):
    """The tokens of one instruction line exactly as the simulator decodes them (MainCtrl.cpp):
    a '#' at the start of the line or after whitespace starts a comment, and tokens are
    separated by ASCII whitespace. Comment text must never count as part of the instruction."""
    line = re.split(r"(?:^|(?<=[ \t\n\v\f\r]))#", line, maxsplit=1)[0]
    return [t for t in re.split(r"[ \t\n\v\f\r]+", line) if t]


def array_reads_dram(tok):
    """An Array instruction reads its layer_name file unless it takes the weight from the cache
    (`wsrc cache ...`, a keyword after the coordinates)."""
    return "wsrc" not in tok[3:]


def layer_dims(config):
    path = os.path.join(_DATA, "model_weight", "5mlp_weight", config, "config.json")
    with open(path) as f:
        hidden = json.load(f)[config]["hidden_layers_dims"]
    return [IN_DIM] + hidden + [NUM_CLASSES]


def output_path(config, batch):
    return os.path.join(_DATA, "out_tensor", "mlp", f"{config}_out_B{batch}.bin")


def provided_files(config, batch):
    """Files a case may read without writing them first."""
    files = {os.path.join(_DATA, "in_tensor", "mlp", f"in_B{batch}.bin")}
    weight_dir = os.path.join(_DATA, "model_weight", "mlp_bin", config)
    for n in range(len(layer_dims(config)) - 1):
        for kind in ("weight", "bias"):
            files.add(os.path.join(weight_dir, f"layer{n}_{kind}.bin"))
    return {os.path.realpath(f) for f in files}


# The second provided image the case is checked with (the provided in_B1.bin is the first).
SECOND_IMAGE = os.path.join(_DATA, "in_tensor", "mlp", "score_inputs", "mnist_2.bin")
# A random image must not have a near-tie between its two most likely classes: the check also
# compares the predicted class, which a near-tie would make depend on rounding.
MIN_TOP2_MARGIN = 1e-3


def reference_files(config):
    """Provided files the check scripts themselves need (not read by the simulator)."""
    return [os.path.join(_DATA, "model_weight", "5mlp_weight", config, f"{config}_mlp_model.pth"),
            SECOND_IMAGE]


def check_provided_data(config, batch):
    """The provided DRAM_data files must be in place before anything else can work."""
    missing = sorted(f for f in list(provided_files(config, batch)) + reference_files(config)
                     if not os.path.isfile(f))
    if missing:
        rel = [os.path.relpath(f, _REPO_ROOT) for f in missing]
        return (f"provided data file(s) missing: {rel[:3]}{' ...' if len(rel) > 3 else ''} -- "
                f"they are part of the repository archive: extract it again into another folder "
                f"(USER_GUIDE.md §2) and copy its new_code/DRAM_sim/DRAM_data/ over yours")
    return None


def check_path_token(path, where):
    """DRAM file paths must be repository-root-relative and use '/'."""
    if "\\" in path:
        return (f"{where}: path '{path}' contains '\\'. Use '/' as the separator (on Windows, "
                f"build paths with '/' or posixpath.join, not os.path.join)")
    if os.path.isabs(path) or re.match(r"^[A-Za-z]:", path):
        return (f"{where}: path '{path}' is absolute. Write it relative to the repository root, "
                f"e.g. new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin")
    return None


def check_case_files(config, batch, case_dir):
    """Static checks. Returns (num_cores, error message or None)."""
    if not os.path.isdir(case_dir):
        root = os.path.dirname(os.path.dirname(case_dir))
        hint = (" (--inst-root takes the folder that contains MLP_small_mlp/B1/, not the B1 folder "
                "itself)" if os.path.isfile(os.path.join(root, "core_0_insts.txt")) else "")
        return None, f"case folder missing: {case_dir}{hint}"
    w, h = MESH_W, MESH_H
    num_cores = w * h

    core_ids = set()
    for name in os.listdir(case_dir):
        m = _CORE_FILE_RE.match(name)
        if m:
            core_ids.add(int(m.group(1)))
    if core_ids != set(range(num_cores)):
        missing = sorted(set(range(num_cores)) - core_ids)
        extra = sorted(core_ids - set(range(num_cores)))
        return num_cores, (f"core_<i>_insts.txt must be exactly i=0..{num_cores - 1} "
                           f"for a {w}x{h} mesh (missing {missing[:5]}, extra {extra[:5]})")

    # Instruction paths first: a Windows backslash path also breaks the YAML
    # parse of dram_map.yaml below, with a much less helpful message.
    path_tokens = []
    for i in range(num_cores):
        name = f"core_{i}_insts.txt"
        with open(os.path.join(case_dir, name), "rb") as f:
            if f.read(3) == b"\xef\xbb\xbf":
                return num_cores, (f"{name} starts with a UTF-8 byte-order mark (BOM); save it as "
                                   f"plain UTF-8 without BOM (in Python: encoding=\"utf-8\")")
        with open(os.path.join(case_dir, name), encoding="utf-8", errors="replace") as f:
            for ln, line in enumerate(f, 1):
                tok = inst_tokens(line)
                if not tok:
                    continue
                if tok[0] == "DMA" and len(tok) > 3 and tok[1] == "DRAM":
                    path_tokens.append((tok[3], f"{name}:{ln}"))
                elif tok[0].upper() == "ARRAY" and len(tok) > 2 and array_reads_dram(tok):
                    path_tokens.append((tok[2], f"{name}:{ln}"))
    for path, where in path_tokens:
        err = check_path_token(path, where)
        if err:
            return num_cores, err

    try:
        with open(os.path.join(case_dir, "dram_map.yaml"), encoding="utf-8", errors="replace") as f:
            text = f.read()
        dram_map = yaml.safe_load(text) or {}
    except OSError as e:
        return num_cores, f"dram_map.yaml: {e}"
    except yaml.YAMLError as e:
        hint = (" (it contains '\\': use '/' as the path separator)" if "\\" in text else "")
        return num_cores, f"dram_map.yaml is not valid YAML{hint}: {e}"
    if not isinstance(dram_map, dict):
        return num_cores, "dram_map.yaml must be a mapping (version/num_cores/entries)"
    if dram_map.get("num_cores") != num_cores:
        return num_cores, (f"dram_map.yaml num_cores={dram_map.get('num_cores')} "
                           f"!= {num_cores} (the mesh is fixed at {w}x{h})")
    extra = sorted(str(k) for k in dram_map if k not in ALLOWED_MAP_KEYS)
    if extra:
        return num_cores, (f"dram_map.yaml: key(s) {extra} not allowed "
                           f"(only {', '.join(ALLOWED_MAP_KEYS)}; see USER_GUIDE.md §12.2)")
    for e in dram_map.get("entries") or []:
        if not isinstance(e, dict):
            continue
        extra = sorted(str(k) for k in e if k not in ALLOWED_ENTRY_KEYS)
        if extra:
            return num_cores, (f"dram_map.yaml entry {e.get('path')!r}: field(s) {extra} not allowed "
                               f"(only {', '.join(ALLOWED_ENTRY_KEYS)}; see USER_GUIDE.md §12.2)")
        if "kind" in e and e["kind"] != "output":
            return num_cores, (f"dram_map.yaml entry {e.get('path')!r}: kind {e['kind']!r} not allowed "
                               f"(the only kind is 'output', for the output file)")
    for e in dram_map.get("entries") or []:
        if isinstance(e, dict) and isinstance(e.get("path"), str):
            err = check_path_token(e["path"], "dram_map.yaml")
            if err:
                return num_cores, err
    entry_paths = set()
    for e in dram_map.get("entries") or []:
        if not isinstance(e, dict) or not isinstance(e.get("path"), str):
            return num_cores, f"dram_map.yaml: every entry needs a path string (got {e!r})"
        entry_paths.add(e["path"])
        owners = [e["owner"]] if "owner" in e else e.get("owners")
        if "owner" in e and "owners" in e:
            return num_cores, f"dram_map.yaml entry {e['path']!r}: use either owner or owners, not both"
        if (not isinstance(owners, list) or not owners
                or any(type(o) is not int or not 0 <= o < num_cores for o in owners)):
            return num_cores, (f"dram_map.yaml entry {e['path']!r}: needs owner: <core> or "
                               f"owners: [<core>, ...] with cores 0..{num_cores - 1}")
    for path, where in path_tokens:
        if path not in entry_paths:
            return num_cores, (f"{where}: '{path}' has no entry in dram_map.yaml (every file named in "
                               f"a DMA DRAM or Array instruction needs one, with exactly the same "
                               f"path string; see USER_GUIDE.md §12.2)")

    reads, writes = set(), set()
    for i in range(num_cores):
        # Only the ASCII instruction tokens matter here; comments may be in any
        # encoding (e.g. Big5/CP950 from a Windows editor), which the simulator
        # ignores too, so undecodable bytes must not abort the check.
        with open(os.path.join(case_dir, f"core_{i}_insts.txt"), encoding="utf-8",
                  errors="replace") as f:
            for line in f:
                tok = inst_tokens(line)
                if not tok:
                    continue
                if (tok[0].upper() == "VECTOR" and len(tok) > 1
                        and tok[1].upper() in NOT_ALLOWED_VECTOR_MODES):
                    return num_cores, (f"core_{i}_insts.txt: Vector {tok[1]} is not allowed in this "
                                       f"assignment (see USER_GUIDE.md §12.1)")
                if tok[0] == "DMA" and len(tok) > 3 and tok[1] == "DRAM":
                    path = os.path.realpath(os.path.join(_REPO_ROOT, tok[3]))
                    (reads if tok[2] == "read" else writes).add(path)
                elif tok[0].upper() == "ARRAY" and len(tok) > 2 and array_reads_dram(tok):
                    # With `wsrc cache`, the weight comes from the cache and the
                    # simulator never opens layer_name, so it may be a placeholder.
                    reads.add(os.path.realpath(os.path.join(_REPO_ROOT, tok[2])))

    allowed = provided_files(config, batch)
    out = os.path.realpath(output_path(config, batch))
    rel = lambda paths: [os.path.relpath(x, _REPO_ROOT) for x in paths[:3]]
    bad_reads = sorted(reads - allowed)
    if bad_reads:
        return num_cores, (f"reads file(s) other than the provided input/weights: {rel(bad_reads)} "
                           f"(intermediate results move between cores over the NoC, not through DRAM files)")
    bad_writes = sorted(writes & allowed)
    if bad_writes:
        return num_cores, f"overwrites provided input/weight files: {rel(bad_writes)}"
    bad_writes = sorted(writes - {out})
    if bad_writes:
        return num_cores, (f"writes file(s) other than the output file: {rel(bad_writes)} "
                           f"(the only DMA DRAM write target is {os.path.relpath(out, _REPO_ROOT)})")
    if out not in writes:
        return num_cores, f"no DMA DRAM write to the required output file {output_path(config, batch)}"
    return num_cores, None


def run_case(config, batch, inst_root, log_dir, timeout, seed):
    result = {"config": config, "batch": batch, "cores": None, "latency": None,
              "status": "FAIL", "reason": None, "seed": seed}
    # Anything unexpected in one case (a malformed file, ...) fails that case
    # only, instead of aborting the whole check with a traceback.
    try:
        return _run_case(config, batch, inst_root, log_dir, timeout, seed, result)
    except Exception as e:
        result["status"] = "FAIL"
        result["reason"] = f"could not check this case: {type(e).__name__}: {e}"
        return result


def _run_case(config, batch, inst_root, log_dir, timeout, seed, result):
    case_dir = os.path.join(inst_root, f"MLP_{config}", f"B{batch}")

    err = check_provided_data(config, batch)
    if err:
        result["reason"] = err
        return result

    num_cores, err = check_case_files(config, batch, case_dir)
    result["cores"] = num_cores
    if err:
        result["reason"] = err
        return result

    binary = os.path.join(_REPO_ROOT, "new_code", "build", "bin", f"{config}_Acc")
    if not os.access(binary, os.X_OK):
        result["reason"] = (f"{binary} not built (run_scripts/gen_acc_from_case.sh, "
                            f"then build {config}_Acc)")
        return result

    # The inputs: the provided image, a second provided image, and a random one. Copies go to
    # the log folder, so the run can be repeated from there.
    in_dir = os.path.join(log_dir, "inputs")
    os.makedirs(in_dir)
    inputs = [("in_B1", os.path.join(_DATA, "in_tensor", "mlp", f"in_B{batch}.bin")),
              ("mnist_2", SECOND_IMAGE)]
    for label, src in inputs:
        shutil.copyfile(src, os.path.join(in_dir, f"{label}.bin"))
    write_random_image(config, batch, seed, os.path.join(in_dir, "random.bin"))
    inputs = [(label, os.path.join(in_dir, f"{label}.bin"))
              for label in ("in_B1", "mnist_2", "random")]
    result["inputs"] = [label for label, _ in inputs]

    latencies = {}
    for k, (label, in_file) in enumerate(inputs):
        if k == 0:
            run_dir = case_dir            # the case as it is (it reads in_B1.bin)
        else:
            run_dir = os.path.join(log_dir, f"case_{label}")
            copy_case_with_input(case_dir, run_dir, num_cores, in_file)
        latency, reason = run_once(config, batch, binary, run_dir, in_file, num_cores,
                                   os.path.join(log_dir, f"{config}_B{batch}_{label}"), timeout)
        if reason:
            result["reason"] = (reason if k == 0 else
                                f"with input {label} ({os.path.relpath(in_file, _REPO_ROOT)}): {reason}")
            return result
        latencies[label] = latency
    result["latencies"] = latencies
    if len(set(latencies.values())) != 1:
        result["reason"] = (f"the latency depends on the input values ({latencies}); the program "
                            f"must do the same work for every input image")
        return result
    result["latency"] = latencies["in_B1"]
    result["status"] = "PASS"
    return result


def write_random_image(config, batch, seed, path):
    """A random [batch, 784] image with values in [0, 1), like the provided images but with every
    pixel nonzero. Drawn again until its two most likely classes are clearly apart."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        x = rng.random((batch, IN_DIM), dtype=np.float32)
        p = np.sort(compare_output_mlp.pytorch_reference(config, batch, x=x), axis=1)
        if np.all(p[:, -1] - p[:, -2] >= MIN_TOP2_MARGIN):
            break
    x.tofile(path)


def copy_case_with_input(case_dir, dst, num_cores, in_file):
    """Copy the case folder to dst, with every reference to in_B1.bin (in the instructions and in
    dram_map.yaml) pointing at in_file instead. Nothing else changes."""
    orig = os.path.realpath(os.path.join(_DATA, "in_tensor", "mlp", f"in_B{BATCH}.bin"))
    new = os.path.relpath(in_file, _REPO_ROOT).replace(os.sep, "/")
    is_input = lambda p: os.path.realpath(os.path.join(_REPO_ROOT, p)) == orig
    os.makedirs(dst)
    for i in range(num_cores):
        name = f"core_{i}_insts.txt"
        with open(os.path.join(case_dir, name), "rb") as f:
            lines = f.read().split(b"\n")
        for n, raw in enumerate(lines):
            tok = inst_tokens(raw.decode("utf-8", errors="replace"))
            if not tok:
                continue
            pos = None
            if tok[0] == "DMA" and len(tok) > 3 and tok[1] == "DRAM":
                pos = 3
            elif tok[0].upper() == "ARRAY" and len(tok) > 2:
                pos = 2
            if pos is not None and is_input(tok[pos]):
                # Rebuild the line from its tokens (the path token is replaced; the
                # simulator splits on whitespace, so the spacing doesn't matter, and
                # the comment is dropped).
                tok[pos] = new
                lines[n] = " ".join(tok).encode()
        with open(os.path.join(dst, name), "wb") as f:
            f.write(b"\n".join(lines))
    with open(os.path.join(case_dir, "dram_map.yaml"), encoding="utf-8", errors="replace") as f:
        dram_map = yaml.safe_load(f)
    for e in dram_map.get("entries") or []:
        if is_input(e["path"]):
            e["path"] = new
    with open(os.path.join(dst, "dram_map.yaml"), "w") as f:
        yaml.safe_dump(dram_map, f, sort_keys=False)


def failure_details(rc, out_log, err_log):
    """Why a run stopped: the simulator's error lines, else the signal that killed it."""
    lines = (out_log + "\n" + err_log).splitlines()
    msgs = [l[7:] if l.startswith("[FAIL] ") else l for l in lines
            if l.startswith(_FAIL_PREFIXES) or _UNIT_ERROR_RE.match(l)]
    if rc < 0:
        try:
            name = signal.Signals(-rc).name
        except ValueError:
            name = f"signal {-rc}"
        if name in _SIGNAL_HINTS:
            msgs.append(f"{name}: {_SIGNAL_HINTS[name]}")
        elif not msgs:
            msgs.append(name)
    msgs += [f"({hint})" for tag, hint in _MESSAGE_HINTS.items() if any(tag in m for m in msgs)]
    return "; ".join(dict.fromkeys(m.strip() for m in msgs))


def run_once(config, batch, binary, case_dir, in_file, num_cores, stem, timeout):
    """One simulation of case_dir plus the output check. Returns (latency, None) or (None, reason)."""
    out_file = output_path(config, batch)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # Reset the output to NaN so that any element this run does not write
    # (left over from an earlier run) fails the finite/compare checks.
    np.full((batch, NUM_CLASSES), np.nan, dtype=np.float32).tofile(out_file)
    logs = f"see {stem}.stdout / .stderr"
    start = time.time()
    try:
        with open(stem + ".stdout", "w") as so, open(stem + ".stderr", "w") as se:
            proc = subprocess.run([binary, case_dir, out_file, str(batch), str(NUM_CLASSES)],
                                  cwd=_REPO_ROOT, stdout=so, stderr=se, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"wall-clock timeout after {timeout}s ({logs})"
    with open(stem + ".stderr", encoding="utf-8", errors="replace") as f:
        log = f.read()

    if "WATCHDOG EXPIRED" in log:
        return None, f"simulation watchdog expired (likely deadlock, {logs})"
    done = ([None] + list(_DONE_RE.finditer(log)))[-1]
    checks = ([None] + list(_CHECKS_RE.finditer(log)))[-1]
    if proc.returncode != 0 or not done or not checks or checks.group(2) != "0":
        with open(stem + ".stdout", encoding="utf-8", errors="replace") as f:
            out_log = f.read()
        details = failure_details(proc.returncode, out_log, log)
        return None, (f"simulation failed (exit {proc.returncode}"
                      f"{': ' + details if details else ''}, {logs})")
    if int(done.group(1)) != num_cores:
        return None, f"simulated {done.group(1)} cores, expected {num_cores}"
    if not os.path.exists(out_file) or os.path.getmtime(out_file) < start:
        return None, "output file was not written by this run"
    shutil.copyfile(out_file, stem + ".out.bin")

    cmp = subprocess.run([sys.executable, os.path.join(_REPO_ROOT, "run_scripts", "compare_output_mlp.py"),
                          "--config", config, "--batch", str(batch), "--input", in_file],
                         cwd=_REPO_ROOT, capture_output=True, text=True)
    with open(stem + ".compare", "w") as f:
        f.write(cmp.stdout + cmp.stderr)
    if cmp.returncode != 0:
        return None, f"output does not match the PyTorch reference (see {stem}.compare)"
    return int(done.group(2)), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inst-root", default=os.path.join(_REPO_ROOT, "inst_gen"),
                    help="folder holding MLP_small_mlp/B1/ (default: inst_gen/)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="wall-clock seconds allowed for each run (default: 3600)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed of the random input image (default: a new one each time)")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # A new folder for every check, also when two start in the same second.
    for n in range(1, 1000):
        log_dir = os.path.join(_REPO_ROOT, "sim_logs", f"score_{stamp}" + (f"_{n}" if n > 1 else ""))
        try:
            os.makedirs(log_dir)
            break
        except FileExistsError:
            continue
    inst_root = os.path.abspath(args.inst_root)
    # Every run writes the same output file, so two checks at the same time would overwrite each
    # other's output: a second check waits for the first one.
    lock = open(os.path.join(_REPO_ROOT, "sim_logs", ".score.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another score_mlp.py is running in this repository (both would write the same output "
              "file). Waiting for it to finish ...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
    except OSError:
        pass                              # file system without locks: don't wait
    print(f"Checking {CONFIG} B={BATCH} ({MESH_W}x{MESH_H} mesh) from {inst_root}. Logs: {log_dir}")

    r = run_case(CONFIG, BATCH, inst_root, log_dir, args.timeout, seed)
    if r["status"] == "PASS":
        print(f"RESULT: PASS  cores={r['cores']}  latency={r['latency']} cycles  "
              f"(same on all {len(r['inputs'])} inputs: {', '.join(r['inputs'])})")
        exit_code = 0
    else:
        print(f"RESULT: FAIL  {r['reason']}")
        exit_code = 1

    summary = {"timestamp": stamp, "inst_root": inst_root, **r}
    with open(os.path.join(log_dir, "score.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {os.path.join(log_dir, 'score.json')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
