"""Stress-test images for rule 3 ("correct for any 784 values in [0, 1]").

    python3 inst_gen/gen_code/stress_inputs.py <out dir>      (inside the container: needs torch)

Writes <out dir>/<name>.bin (784 float32 in [0, 1], the in_B1.bin format) and index.json:
  zeros, ones                 constant images
  maxlogit_<T>                gradient ascent on the largest logit, snapshot when it first exceeds T
  maxlogit_final              the end of that ascent (largest logit found)
  class<k>                    gradient ascent on logit k alone (k = 0..9)
  minlogit                    gradient descent on the largest logit (all logits as small as possible)
For each image it records the float64 logits, the predicted class and the gap between the two
largest probabilities (the TA note: grading inputs have this gap >= 1e-3).
"""
import json
import os
import sys

import numpy as np
import torch

W = "new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp"
DIMS = [784, 128, 64, 32, 16, 10]
Ws = [torch.tensor(np.fromfile(f"{W}/layer{i}_weight.bin", np.float32).reshape(DIMS[i], DIMS[i + 1]),
                   dtype=torch.float64) for i in range(5)]
bs = [torch.tensor(np.fromfile(f"{W}/layer{i}_bias.bin", np.float32), dtype=torch.float64) for i in range(5)]


def net(x):
    h = x
    for i in range(4):
        h = torch.relu(h @ Ws[i] + bs[i])
    return h @ Ws[4] + bs[4]


def ascend(objective, steps=400, batch=32, seed=0, snap=None):
    """projected Adam on x in [0,1]^784; returns the best image (and snapshots by threshold)"""
    torch.manual_seed(seed)
    x = torch.rand(batch, 784, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.05)
    snaps = {}
    for _ in range(steps):
        obj = objective(net(x))
        opt.zero_grad(); (-obj.sum()).backward(); opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
            if snap:
                m = net(x).max(dim=1).values
                for t in snap:
                    if t not in snaps and (m > t).any():
                        snaps[t] = x[(m > t).nonzero()[0, 0]].clone()
    with torch.no_grad():
        best = x[objective(net(x)).argmax()].clone()
    return best, snaps


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    imgs = {"zeros": torch.zeros(784, dtype=torch.float64), "ones": torch.ones(784, dtype=torch.float64)}
    best, snaps = ascend(lambda z: z.max(dim=1).values, snap=(40, 80, 90, 120, 160))
    for t, x in sorted(snaps.items()):
        imgs[f"maxlogit_{t}"] = x
    imgs["maxlogit_final"] = best
    for k in range(10):
        imgs[f"class{k}"], _ = ascend(lambda z, k=k: z[:, k], steps=300, seed=k + 1)
    imgs["minlogit"], _ = ascend(lambda z: -z.max(dim=1).values, steps=300, seed=99)
    index = {}
    for name, x in imgs.items():
        x32 = x.to(torch.float32).clamp(0, 1)
        x32.numpy().astype(np.float32).tofile(os.path.join(out, name + ".bin"))
        with torch.no_grad():
            z = net(x32.to(torch.float64))
            p = torch.softmax(z, 0)
            top = torch.topk(p, 2).values
        index[name] = {"max_logit": round(z.max().item(), 2), "min_logit": round(z.min().item(), 2),
                       "pred": int(z.argmax()), "top2_gap": float(top[0] - top[1])}
        print(f"{name:16s} max logit {z.max().item():8.2f}  pred {int(z.argmax())}  top-2 gap {float(top[0] - top[1]):.3g}")
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
