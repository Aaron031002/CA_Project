"""Generator for the small_mlp case (batch 1) on the fixed 2x2 mesh.

Writes inst_gen/MLP_small_mlp/B1/{core_0..3_insts.txt, dram_map.yaml}.

    python3 inst_gen/gen_code/gen_small_mlp.py                     # submitted version (Vector SOFTMAX)
    python3 inst_gen/gen_code/gen_small_mlp.py --softmax pairwise  # 1547 cycles, valid for every input
    python3 inst_gen/gen_code/gen_small_mlp.py --softmax accurate  # sigmoid softmax: NOT valid for every input

Mesh (core_id = y*2 + x):   y=1 | 2 3        hops: 0-1, 0-2, 1-3, 2-3 = 1
                            y=0 | 0 1              0-3, 1-2           = 2

Mapping (see docs/README_zh.md for the full explanation):

  Layer 0 (784->128, ~90% of all weights): column split. Core c computes the 32
    columns 32c..32c+31 for all 784 inputs. The input is loaded in two K-chunks
    (rows 0..175, 176..783) so the Array can start on chunk 0 while chunk 1 is
    still being copied into the cache; the two partial sums are added by the Vector
    unit, then + b0 slice, ReLU  ->  h0 slice (32 values) on every core.

  Layer 1 (128->64): row split ("K split"). Core c owns the 32 rows of W1 that
    match its h0 slice and computes partial sums for all 64 outputs, as two
    Array ops: half A (outputs 0..31) and half B (outputs 32..63). Half A is
    summed on core 0, half B on core 1 (two "collectors" so each collector only
    adds 32-wide vectors). The partial that travels 2 hops is sent first and
    carries the bias b1 half (added on the sender, which is idle anyway).
    -> h1A on core 0, h1B on core 1.

  Layer 2 (64->32): row split over cores 0 and 1 (h1A rows on core 0, h1B rows
    on core 1); core 1 sends its 32-wide partial to core 0.
  Layers 3, 4 and the softmax run on core 0; core 0 writes the output.

  Softmax: must be correct for every input in [0, 1]^784 (requirements §5 rule 3 and the
    TA clarification: any 784 values in [0, 1]).
  "builtin" (1618 cycles): BIAS_ADD b4 + Vector SOFTMAX.

  "pairwise" (1547 cycles, default, the submitted version): p_i = 1 / sum_j exp(z_j - z_i). Only DIFFERENCES of logits are
    exponentiated, so nothing that matters overflows for any input:
        R[j][i] = -u_j                 one Array op: outer product u (10x1) x (-1 row), df OS
        D[j][i] = z_i - u_j            BIAS_ADD (z = u + b4 is computed while the Array works)
        H[j][i] = 1/sig(D) = 1 + exp(u_j - z_i)                  SIG + RECIP on 100 values
        S_i = sum_j exp(b4_j) H[j][i] = K + sum_j exp(z_j - z_i)  one Array op, K = sum exp(b4)
        p_i = 1 / (S_i - K)
    If p_i > 1e-30 every exponent is < ~72, so column i is exact. A column with a much larger
    logit elsewhere (gap > ~88) gets inf in H, S_i = inf and p_i = 1/inf = 0 (correct: p_i < 1e-38).
    For gaps > ~104, sig() underflows to exactly 0 and RECIP divides by 0: the simulator has an
    assert for that, compiled out in the provided Release build (-DNDEBUG), where 1/0 = inf.
  "pairwise_safe" (1589): the same, plus SIG + 2^-120 before RECIP, so it never divides by 0.

  "accurate" (kept for reference only -- it does NOT meet rule 3): no exp unit exists, but
        exp(u) = sigmoid(u) / sigmoid(-u)     (exact identity, accurate in float32)
    With u = h3 @ W4 (bias not added), p_i = exp(z_i)/sum_j exp(z_j), z = u + b4:
        T_i = sum_j exp(u_j) * exp(b4_j - b4_i)   = one Array op with a 10x10 matrix
        p_i = exp(u_i) / T_i
    The matrix W'[j][i] = exp(b4_j - b4_i) is computed from the provided bias at the
    start (outer product exp(b4) x exp(-b4)) while core 0's units are otherwise idle.
    Valid only while the largest logit stays below ~85 (sigmoid(-u) underflows in
    float32 above ~88). MNIST images give ~12-18 and uniform random images < 35, but
    inputs in [0, 1] built by gradient ascent reach ~185 -> wrong output.
  "shift": exp(u+C) = 1/sigmoid(-(u+C)) - 1 (valid for max logit < ~45; also NOT valid).

No intermediate result goes through DRAM; there is no LOCAL_SYNC/GLOBAL_SYNC: every
unit simply waits in the cache for the data id it needs (a cache read miss is retried
every cycle), so instructions start as soon as their inputs exist.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from isa import Prog, IN_FILE, OUT_FILE, wfile, bfile, DIMS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
CASE_DIR = os.path.join(ROOT, "inst_gen", "MLP_small_mlp", "B1")

COLL = {"A": 0, "B": 1}               # layer-1 collectors
HALF = {"A": (0, 32), "B": (32, 64)}  # layer-1 output columns per half


def other(h):
    return "B" if h == "A" else "A"


def build(kchunks=(0, 176, 784), softmax="pairwise", C=40):
    # Which layer-1 half each core computes first: the one that must travel
    # (and, for cores 2/3, the one that travels 2 hops) goes first.
    first = {0: "B", 1: "A", 2: "B", 3: "A"}
    # 2-hop partials (core 3 -> core 0, core 2 -> core 1) carry the b1 half.
    prebias = {(3, "A"), (2, "B")}
    # Accumulation order on each collector = expected arrival order.
    chain = {"A": [1, "own", 3, 2], "B": [0, "own", 2, 3]}

    p = Prog()
    N0 = DIMS[1] // 4                 # 32 layer-0 columns per core
    qpart, b1h = {}, {}
    for c in range(4):
        cols = (N0 * c, N0 * (c + 1))
        p.c(c, f"===== core {c}: layer 0 columns {cols[0]}..{cols[1] - 1} =====")
        # ---- DMA: input chunks first (critical), then the small bias vectors
        xs = [p.dram_read(c, IN_FILE, 0, 1, k0, k1, comment=f"input x[{k0}:{k1}]")
              for k0, k1 in zip(kchunks[:-1], kchunks[1:])]
        b0 = p.dram_read(c, bfile(0), 0, 1, cols[0], cols[1], comment="b0 slice")
        for h in "AB":
            if (c, h) in prebias:
                b1h[c, h] = p.dram_read(c, bfile(1), 0, 1, *HALF[h], comment=f"b1 half {h}")
        if c == 0:
            b2 = p.dram_read(c, bfile(2), 0, 1, 0, 32, comment="b2")
            b3 = p.dram_read(c, bfile(3), 0, 1, 0, 16, comment="b3")
            b4 = p.dram_read(c, bfile(4), 0, 1, 0, 10, comment="b4")

        # ---- layer 0: one Array op per input chunk (weights streamed from local DRAM)
        parts = [p.array(c, wfile(0), k0, k1, cols[0], cols[1], 128, x, 1,
                         comment=f"L0 partial: x[{k0}:{k1}] @ W0[{k0}:{k1}, {cols[0]}:{cols[1]}]")
                 for x, k0, k1 in zip(xs, kchunks[:-1], kchunks[1:])]
        acc = p.vec(c, "BIAS_ADD", N0, parts[0], b0, 1, 1, comment="partial0 + b0 (off the critical path)")

        if c == 0 and softmax != "builtin":
            p.c(c, "softmax constants, built while the Array is busy with layer 0")
            z10 = p.vec(c, "SUB", 10, b4, b4, 0, 0, comment="0 (exact)")
            nb4 = p.vec(c, "SUB", 10, z10, b4, 0, 0, comment="-b4")
            sb = p.vec(c, "SIG", 10, b4, 0, 0, 0, comment="sig(b4)")
            snb = p.vec(c, "SIG", 10, nb4, 0, 1, 0, comment="sig(-b4)")
            rsnb = p.vec(c, "RECIP", 10, snb, 0, 1, 0, comment="1+exp(b4)")
            Eb = p.vec(c, "MUL", 10, sb, rsnb, 1, 1, comment="exp(b4)")
            if softmax.startswith("pairwise"):
                half = p.vec(c, "SIG", 10, z10, 0, 0, 0, comment="0.5 (exact)")
                keep_half = softmax == "pairwise_safe"     # half is reused for eps
                one = p.vec(c, "ADD", 10, half, half, 0, int(not keep_half), comment="1")
                m1 = p.vec(c, "SUB", 10, z10, one, 0, 1, comment="-1 (weight row of the outer product)")
                Ksc = p.new_id()
                p.emit(c, f"Vector ADD_REDUCE 10 {Eb} 0 {Ksc} 0 0 1", "K = sum_j exp(b4_j) (one value)")
                Kv = p.vec(c, "BIAS_ADD", 1, z10, Ksc, 0, 1, comment="K broadcast to 10 values")
            else:
                Fb = p.vec(c, "RECIP", 10, Eb, 0, 0, 0, comment="exp(-b4)")
            if softmax == "shift":
                half = p.vec(c, "SIG", 10, z10, 0, 0, 0, comment="0.5 (exact)")
                one = p.vec(c, "ADD", 10, half, half, 0, 1, comment="1")
                m1 = p.vec(c, "SUB", 10, z10, one, 0, 0, comment="-1")
                pw, k, v = {1: one}, one, 1
                while v * 2 <= C:
                    k = p.vec(c, "ADD", 10, k, k, 0, 0, comment=f"{v * 2}")
                    v *= 2
                    pw[v] = k
                cval, cid = v, k
                for bit in sorted(pw, reverse=True):
                    if bit < v and cval + bit <= C:
                        cid = p.vec(c, "ADD", 10, cid, pw[bit], 0, 0, comment=f"{cval + bit}")
                        cval += bit
                assert cval == C
                nC = p.vec(c, "SUB", 10, z10, cid, 0, 0, comment=f"-{C}")

        for pp in parts[1:]:
            acc = p.vec(c, "ADD", N0, acc, pp, 1, 1, comment="+ partial1")
        h0 = p.vec(c, "RELU", N0, acc, 0, 1, 0, comment="h0 slice (32 values)")
        if c == 0 and softmax == "pairwise_safe":
            p.c(c, "eps = 2^-120 for the pairwise softmax (Vector idle until the layer-1 partials arrive)")
            pw2 = {1: half}
            for k in (2, 4, 8, 16, 32, 64):
                pw2[k] = p.vec(c, "MUL", 10, pw2[k // 2], pw2[k // 2], 0, int(k == 2), comment=f"2^-{k}")

        # ---- layer 1 partial sums for this core's 32 rows of W1
        p.c(c, f"layer 1: partial sums h0[{cols[0]}:{cols[1]}] @ W1[{cols[0]}:{cols[1]}, :]")
        for i, h in enumerate((first[c], other(first[c]))):
            qpart[c, h] = p.array(c, wfile(1), cols[0], cols[1], *HALF[h], 64, h0, int(i == 1),
                                  comment=f"L1 partial, output half {h} (cols {HALF[h][0]}..{HALF[h][1] - 1})")
            if (c, h) in prebias:
                qpart[c, h] = p.vec(c, "ADD", 32, qpart[c, h], b1h[c, h], 1, 1,
                                    comment=f"+ b1 half {h} (this partial travels 2 hops)")
        if c == 0 and softmax in ("accurate", "shift"):
            Wp = p.new_id()
            p.emit(c, f"Array NORMAL {wfile(4)} index 0 1 0 10 stride 10 {Eb} {Wp} 1 wsrc cache {Fb} 1",
                   "W'[j][i] = exp(b4_j) * exp(-b4_i): outer product, weight taken from the cache")

    # ---- NoC: every non-local layer-1 partial goes to its collector
    pk = {}
    for c in range(4):
        for h in (first[c], other(first[c])):
            if COLL[h] != c:
                pk[c, h] = p.send(c, qpart[c, h], [COLL[h]], truncate=1,
                                  comment=f"L1 partial half {h} -> core {COLL[h]}")
    h1 = {}
    for h, col in COLL.items():
        p.c(col, f"collector of layer-1 half {h}")
        rv = {s: p.recv(col, pk[s, h], [32], comment=f"L1 partial half {h} from core {s}")
              for s in chain[h] if s != "own"}
        rv["own"] = qpart[col, h]
        seq = chain[h]
        acc = p.vec(col, "ADD", 32, rv[seq[0]], rv[seq[1]], 1, 1)
        for s in seq[2:]:
            acc = p.vec(col, "ADD", 32, acc, rv[s], 1, 1)
        h1[h] = p.vec(col, "RELU", 32, acc, 0, 1, 0, comment=f"h1 half {h}")
        if col == 0 and softmax == "pairwise_safe":
            e96 = p.vec(0, "MUL", 10, pw2[64], pw2[32], 1, 1, comment="2^-96 (Vector idle during layer 2)")
            e112 = p.vec(0, "MUL", 10, e96, pw2[16], 1, 1, comment="2^-112")
            eps = p.vec(0, "MUL", 10, e112, pw2[8], 1, 1, comment="eps = 2^-120")

    # ---- layer 2: row split over the two collectors, summed on core 0
    p.c(1, "layer 2 partial for rows 32..63, sent to core 0")
    q2B = p.array(1, wfile(2), 32, 64, 0, 32, 32, h1["B"], 1, comment="h1B @ W2[32:64, :]")
    pkt = p.send(1, q2B, [0], truncate=1)
    p.c(0, "layer 2, 3, 4 and softmax")
    q2A = p.array(0, wfile(2), 0, 32, 0, 32, 32, h1["A"], 1, comment="h1A @ W2[0:32, :]")
    r2 = p.recv(0, pkt, [32], comment="h1B @ W2[32:64, :] from core 1")
    acc = p.vec(0, "BIAS_ADD", 32, q2A, b2, 1, 1)
    acc = p.vec(0, "ADD", 32, acc, r2, 1, 1)
    h = p.vec(0, "RELU", 32, acc, 0, 1, 0, comment="h2")
    y = p.array(0, wfile(3), 0, 32, 0, 16, 16, h, 1)
    y = p.vec(0, "BIAS_ADD", 16, y, b3, 1, 1)
    h = p.vec(0, "RELU", 16, y, 0, 1, 0, comment="h3")
    u = p.array(0, wfile(4), 0, 16, 0, 10, 10, h, 1,
                comment="u = h3 @ W4" + ("" if softmax == "builtin" or softmax.startswith("pairwise")
                                          else " (b4 is folded into W')"))

    if softmax == "builtin":
        z = p.vec(0, "BIAS_ADD", 10, u, b4, 1, 1, comment="logits")
        prob = p.vec(0, "SOFTMAX", 10, z, 0, 1, 0, comment="probabilities")
    elif softmax.startswith("pairwise"):
        # p_i = 1 / sum_j exp(z_j - z_i): only DIFFERENCES of logits are exponentiated, so nothing
        # that matters can overflow, whatever the input (see the module docstring).
        z = p.vec(0, "BIAS_ADD", 10, u, b4, 0, 0, comment="z = u + b4 (runs while the Array builds R)")
        R = p.new_id()
        p.emit(0, f"Array NORMAL {wfile(4)} index 0 1 0 10 stride 10 {u} {R} 0 df OS wsrc cache {m1} 1",
               "R[j][i] = -u_j: outer product u (10x1) x (-1 row), output-stationary (K = 1)")
        D = p.vec(0, "BIAS_ADD", 10, R, z, 1, 1, comment="D[j][i] = z_i - u_j  (100 values)")
        G = p.vec(0, "SIG", 100, D, 0, 1, 0, comment="sig(z_i - u_j)")
        if softmax == "pairwise_safe":
            G = p.vec(0, "BIAS_ADD", 10, G, eps, 1, 1, comment="+ 2^-120: never divide by exactly 0")
        H = p.vec(0, "RECIP", 100, G, 0, 1, 0, comment="H[j][i] = 1 + exp(u_j - z_i)")
        S = p.new_id()
        p.emit(0, f"Array NORMAL {wfile(4)} index 0 10 0 10 stride 10 {Eb} {S} 1 wsrc cache {H} 1",
               "S_i = sum_j exp(b4_j) H[j][i] = K + sum_j exp(z_j - z_i)")
        Sm = p.vec(0, "SUB", 10, S, Kv, 1, 1, comment="sum_j exp(z_j - z_i) = 1 / p_i")
        prob = p.vec(0, "RECIP", 10, Sm, 0, 1, 0, comment="probabilities")
    else:
        if softmax == "accurate":
            nu = p.vec(0, "SUB", 10, z10, u, 1, 0, comment="-u")
            s1 = p.vec(0, "SIG", 10, u, 0, 1, 0, comment="sig(u)")
            s2 = p.vec(0, "SIG", 10, nu, 0, 1, 0, comment="sig(-u)")
            r2_ = p.vec(0, "RECIP", 10, s2, 0, 1, 0, comment="1/sig(-u) = 1+exp(u)")
            e = p.vec(0, "MUL", 10, s1, r2_, 1, 1, comment="exp(u) = sig(u)/sig(-u)")
        else:
            t = p.vec(0, "SUB", 10, nC, u, 1, 1, comment=f"-(u+{C})")
            s = p.vec(0, "SIG", 10, t, 0, 1, 0)
            r = p.vec(0, "RECIP", 10, s, 0, 1, 0, comment=f"1+exp(u+{C})")
            e = p.vec(0, "ADD", 10, r, m1, 1, 1, comment=f"exp(u+{C})")
        T = p.new_id()
        p.emit(0, f"Array NORMAL {wfile(4)} index 0 10 0 10 stride 10 {e} {T} 0 wsrc cache {Wp} 1",
               "T_i = sum_j e_j * exp(b4_j - b4_i) = sum(exp(z)) / exp(b4_i)")
        rT = p.vec(0, "RECIP", 10, T, 0, 1, 0)
        prob = p.vec(0, "MUL", 10, e, rT, 1, 1, comment="p_i = exp(z_i) / sum_j exp(z_j)")
    p.dram_write(0, OUT_FILE, 0, 1, 0, 10, prob, 1, comment="the 10 class probabilities")
    return p


HEADER = [
    "Generated by inst_gen/gen_code/gen_small_mlp.py -- edit the generator, not this file.",
    "small_mlp 784-128-64-32-16-10, batch 1, 2x2 mesh. No LOCAL_SYNC: units wait on cache ids.",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--softmax", choices=("builtin", "pairwise", "pairwise_safe", "accurate", "shift"), default="pairwise")
    ap.add_argument("--out", default=CASE_DIR, help="case folder to write (default: the official one)")
    ap.add_argument("--run", action="store_true", help="also run it once (inside the container)")
    a = ap.parse_args()
    build(softmax=a.softmax).write(a.out, header=HEADER)
    print(f"wrote {os.path.relpath(a.out, ROOT)}  (softmax={a.softmax})")
    if a.run:
        from run_case import run, check
        lat, _ = run(os.path.relpath(a.out, ROOT))
        print("latency", lat, check())
