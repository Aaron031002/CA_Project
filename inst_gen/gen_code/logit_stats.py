"""Offline analysis: range of the final-layer logits for various inputs (numpy, float64)."""
import numpy as np, os
W = "new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp"
dims = [784, 128, 64, 32, 16, 10]
Ws = [np.fromfile(f"{W}/layer{i}_weight.bin", np.float32).reshape(dims[i], dims[i+1]).astype(np.float64) for i in range(5)]
bs = [np.fromfile(f"{W}/layer{i}_bias.bin", np.float32).astype(np.float64) for i in range(5)]
def logits(x):
    h = x
    for i in range(4):
        h = np.maximum(h @ Ws[i] + bs[i], 0)
    return h @ Ws[4] + bs[4]
ims = {"in_B1": np.fromfile("new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin", np.float32),
       "mnist_2": np.fromfile("new_code/DRAM_sim/DRAM_data/in_tensor/mlp/score_inputs/mnist_2.bin", np.float32)}
for k, x in ims.items():
    z = logits(x.astype(np.float64)); print(k, "min/max px", x.min(), x.max(), "logits", np.round(z, 2), "zmax", z.max().round(2), "zmin", z.min().round(2))
rng = np.random.default_rng(0)
zs = np.array([logits(rng.random(784)) for _ in range(20000)])
print("random: zmax range", zs.max(1).min().round(2), zs.max(1).max().round(2), " zmin range", zs.min(1).min().round(2), zs.min(1).max().round(2))
print("random: zmax-zmin range", (zs.max(1)-zs.min(1)).min().round(2), (zs.max(1)-zs.min(1)).max().round(2))
print("weights min |w| per layer", [float(np.abs(w).min()) for w in Ws])
print("all-ones image logits max", logits(np.ones(784)).max().round(2), " all-zero", logits(np.zeros(784)).max().round(2))
