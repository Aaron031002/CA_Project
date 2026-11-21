"""Calibration micro-benchmarks: measure the latency of single operations.
Each case is written to inst_gen/exp/<name>/ (scratch, safe to delete) and run once.
(TMU concat/split were measured separately inside a dependent chain: ~37 / ~41 cycles.)
Run inside the container from the repo root:  python3 inst_gen/gen_code/exp_calib.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isa import Prog, IN_FILE, wfile, bfile  # noqa: E402
from run_case import run  # noqa: E402

EXP = "inst_gen/exp"


def case(name, build):
    p = Prog()
    build(p)
    d = os.path.join(EXP, name)
    p.write(d)
    lat, _ = run(d)
    print(f"{name:40s} {lat}")
    return lat


def empty(p):
    pass


def rd(n):
    def b(p):
        p.dram_read(0, IN_FILE, 0, 1, 0, n)
    return b


def rd_arr(k, n, sync=False):
    def b(p):
        x = p.dram_read(0, IN_FILE, 0, 1, 0, k)
        if sync:
            p.sync(0)
        p.array(0, wfile(0), 0, k, 0, n, 128, x, 1)
    return b


def xfer(n, dst, sizes=None):
    def b(p):
        x = p.dram_read(0, wfile(0), 0, 1, 0, n)
        pk = p.send(0, x, [dst])
        p.recv(dst, pk, sizes or [n])
    return b


def xfer_only_read(n):
    def b(p):
        p.dram_read(0, wfile(0), 0, 1, 0, n)
    return b


def vec(mode, n, w=True):
    def b(p):
        x = p.dram_read(0, wfile(0), 0, 1, 0, n)
        wid = p.dram_read(0, wfile(0), 1, 2, 0, n) if w else 0
        p.sync(0)
        p.vec(0, mode, n, x, wid, 1, 1 if w else 0)
    return b


def vec_base(n, w=True):
    def b(p):
        p.dram_read(0, wfile(0), 0, 1, 0, n)
        if w:
            p.dram_read(0, wfile(0), 1, 2, 0, n)
        p.sync(0)
    return b


def concat4(n):
    def b(p):
        ids = [p.dram_read(0, wfile(0), i, i + 1, 0, n) for i in range(4)]
        p.sync(0)
        p.tmu(0, "concat", [(i, 1) for i in ids], [p.new_id()], [1, 4])
    return b


def concat4_base(n):
    def b(p):
        [p.dram_read(0, wfile(0), i, i + 1, 0, n) for i in range(4)]
        p.sync(0)
    return b


def gather3(n, sizes_fn=lambda n: [n]):
    def b(p):
        for src in (1, 2, 3):
            x = p.dram_read(src, wfile(0), 0, 1, 0, n)
            pk = p.send(src, x, [0])
            p.recv(0, pk, sizes_fn(n))
    return b


if __name__ == "__main__":
    case("empty", empty)
    for n in (10, 32, 196, 392, 784):
        case(f"dram_read_{n}", rd(n))
    case("rd784+arr784x128", rd_arr(784, 128))
    case("rd784+arr784x32", rd_arr(784, 32))
    case("rd784+sync+arr784x32", rd_arr(784, 32, True))
    case("rd392+arr392x32", rd_arr(392, 32))
    case("rd392+arr392x64", rd_arr(392, 64))
    case("rd196+arr196x128", rd_arr(196, 128))
    case("rd32+arr32x64", rd_arr(32, 64))
    for n in (10, 16, 32, 64, 128):
        case(f"read_{n}", xfer_only_read(n))
        case(f"xfer_{n}_to1", xfer(n, 1))
        case(f"xfer_{n}_to3", xfer(n, 3))
    case("xfer_32_to1_sizes_1_32", xfer(32, 1, [1, 32]))
    case("xfer_64_to1_sizes_1_64", xfer(64, 1, [1, 64]))
    for n in (10, 32, 128):
        case(f"vbase_w_{n}", vec_base(n))
        case(f"vbias_{n}", vec("BIAS_ADD", n))
        case(f"vadd_{n}", vec("ADD", n))
        case(f"vbase_now_{n}", vec_base(n, False))
        case(f"vrelu_{n}", vec("RELU", n, False))
    case("vsoftmax_10", vec("SOFTMAX", 10, False))
    case("vsig_10", vec("SIG", 10, False))
    case("vrecip_10", vec("RECIP", 10, False))
    case("vaddreduce_10", vec("ADD_REDUCE", 10, False))
    case("gather3_32", gather3(32))
    case("gather3_64", gather3(64))
