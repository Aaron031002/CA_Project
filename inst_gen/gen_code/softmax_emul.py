"""float32 emulation of sigmoid-based softmax variants vs float64 reference.
Checks the same criteria as compare_output_mlp.py (abs 1e-4, log 1e-3 for p>1e-30, argmax)."""
import numpy as np
f = np.float32
W = "new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp"
dims = [784, 128, 64, 32, 16, 10]
Ws = [np.fromfile(f"{W}/layer{i}_weight.bin", f).reshape(dims[i], dims[i+1]) for i in range(5)]
bs = [np.fromfile(f"{W}/layer{i}_bias.bin", f) for i in range(5)]
def u_of(x):
    h = x.astype(f)
    for i in range(4):
        h = np.maximum(h @ Ws[i] + bs[i], f(0))
    return (h @ Ws[4]).astype(f)
sig = lambda t: (f(1) / (f(1) + np.exp(-t.astype(f)))).astype(f)
def ref(u):
    z = u.astype(np.float64) + bs[4]; z -= z.max(); e = np.exp(z); return e / e.sum()
def accurate(u):
    b = bs[4]; z = (u + b).astype(f); nz = ((-b) - u).astype(f)
    e = (sig(z) * (f(1) / sig(nz))).astype(f)
    S = f(e.astype(f).sum(dtype=f)); return (e * (f(1) / S)).astype(f)
def shifted(u, C):
    t = ((-bs[4] - f(C)) - u).astype(f); r = (f(1) / sig(t)).astype(f); e = (r + f(-1)).astype(f)
    S = f(e.sum(dtype=f)); return (e * (f(1) / S)).astype(f)
def score(p, q):
    ok = np.abs(p - q).max() < 1e-4 and p.argmax() == q.argmax()
    m = q > 1e-30
    with np.errstate(divide="ignore"):
        lerr = np.abs(np.log(p[m].astype(np.float64)) - np.log(q[m])).max()
    return ok and lerr < 1e-3, lerr
rng = np.random.default_rng(1)
xs = [np.fromfile("new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin", f),
      np.fromfile("new_code/DRAM_sim/DRAM_data/in_tensor/mlp/score_inputs/mnist_2.bin", f)]
xs += [rng.random(784).astype(f) for _ in range(5000)]
for name, fn in [("accurate", accurate)] + [(f"shift C={C}", (lambda C: lambda u: shifted(u, C))(C)) for C in (24, 32, 40, 48)]:
    worst, fails = 0, 0
    for x in xs:
        u = u_of(x); ok, le = score(fn(u), ref(u)); worst = max(worst, le); fails += (not ok)
    print(f"{name:12s} fails={fails}/{len(xs)}  worst log err={worst:.2e}")
