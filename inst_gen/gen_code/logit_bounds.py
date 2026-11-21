"""Bounds on the logits z = MLP(x) over all images x in [0,1]^784.
 - IBP (interval bound propagation): guaranteed (loose) bounds.
 - PGD (projected gradient ascent/descent): achievable extremes (lower bound on the true max)."""
import numpy as np, torch
W = "new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp"
dims = [784, 128, 64, 32, 16, 10]
Ws = [torch.tensor(np.fromfile(f"{W}/layer{i}_weight.bin", np.float32).reshape(dims[i], dims[i+1]), dtype=torch.float64) for i in range(5)]
bs = [torch.tensor(np.fromfile(f"{W}/layer{i}_bias.bin", np.float32), dtype=torch.float64) for i in range(5)]
lo, hi = torch.zeros(784, dtype=torch.float64), torch.ones(784, dtype=torch.float64)
for i in range(5):
    Wp, Wn = Ws[i].clamp(min=0), Ws[i].clamp(max=0)
    lo, hi = lo @ Wp + hi @ Wn + bs[i], hi @ Wp + lo @ Wn + bs[i]
    if i < 4: lo, hi = lo.clamp(min=0), hi.clamp(min=0)
print("IBP  z upper bound:", hi.max().item(), " z lower bound:", lo.min().item())
def net(x):
    h = x
    for i in range(4): h = torch.relu(h @ Ws[i] + bs[i])
    return h @ Ws[4] + bs[4]
best_max, best_min, best_gap = -1e9, 1e9, 0
torch.manual_seed(0)
for k in range(10):
    for sign in (1, -1):
        x = torch.rand(64, 784, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([x], lr=0.05)
        for it in range(300):
            z = net(x)[:, k] * sign
            opt.zero_grad(); (-z.sum()).backward(); opt.step()
            with torch.no_grad(): x.clamp_(0, 1)
        z = net(x.detach())
        if sign == 1: best_max = max(best_max, z.max().item())
        else: best_min = min(best_min, z.min().item())
print("PGD achievable z_max:", round(best_max, 2), " achievable z_min:", round(best_min, 2))
