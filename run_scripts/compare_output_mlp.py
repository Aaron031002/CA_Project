"""compare_output_mlp.py — output check for the small_mlp case. Computes
the PyTorch reference forward pass of small_mlp on in_B1.bin and compares
it against the simulator's output file
new_code/DRAM_sim/DRAM_data/out_tensor/mlp/small_mlp_out_B1.bin
(post-softmax, [1, 10]).

Usage:
    python3 run_scripts/compare_output_mlp.py
    python3 run_scripts/compare_output_mlp.py --input other_image.bin   # reference on another
                                                                        # [1, 784] float32 input
"""
import argparse
import os

import numpy as np


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_input(batch):
    return f"{_REPO_ROOT}/new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B{batch}.bin"


def pytorch_reference(config_name, batch, input_path=None, x=None):
    """Reference output for the input file (default in_B<batch>.bin), or for the array x."""
    import torch
    import torch.nn.functional as F

    pth_path = (f"{_REPO_ROOT}/new_code/DRAM_sim/DRAM_data/model_weight/"
                f"5mlp_weight/{config_name}/{config_name}_mlp_model.pth")
    sd = torch.load(pth_path, map_location="cpu", weights_only=False)
    in_dim = 784

    if x is None:
        x = np.fromfile(input_path or default_input(batch), dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(batch, in_dim)
    h = torch.from_numpy(x)

    linear_keys = sorted((k for k in sd if k.endswith(".weight")),
                         key=lambda k: int(k.split(".")[1]))
    for i, wk in enumerate(linear_keys):
        bk = wk.replace(".weight", ".bias")
        h = F.linear(h, sd[wk], sd[bk])
        if i < len(linear_keys) - 1:
            h = F.relu(h)
    return F.softmax(h, dim=-1).detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small_mlp", choices=["small_mlp"])
    ap.add_argument("--batch", type=int, default=1, choices=[1])
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--log-tol", type=float, default=1e-3)
    ap.add_argument("--input", default=None,
                    help="input the simulator ran on (default: the provided in_B1.bin)")
    args = ap.parse_args()

    data = f"{_REPO_ROOT}/new_code/DRAM_sim/DRAM_data"
    input_path = os.path.abspath(args.input) if args.input else default_input(args.batch)
    needed = {
        "PyTorch checkpoint (provided)":
            f"{data}/model_weight/5mlp_weight/{args.config}/{args.config}_mlp_model.pth",
        ("input image (provided)" if args.input is None else "input image"): input_path,
        "simulator output (written by your run)":
            f"{data}/out_tensor/mlp/{args.config}_out_B{args.batch}.bin",
    }
    for what, path in needed.items():
        if not os.path.isfile(path):
            print(f"ERROR: {what} not found: {os.path.relpath(path, _REPO_ROOT)}")
            if "provided" in what:
                print("  It is part of the repository archive: extract the archive again into another")
                print("  folder (USER_GUIDE.md §2) and copy its new_code/DRAM_sim/DRAM_data/ over yours.")
            else:
                print("  Run the simulation first (USER_GUIDE.md §15).")
            print("FAIL")
            return 1

    if os.path.getsize(input_path) != args.batch * 784 * 4:
        print(f"ERROR: {os.path.relpath(input_path, _REPO_ROOT)} is not a [{args.batch}, 784] float32 file")
        print("FAIL")
        return 1
    ref = pytorch_reference(args.config, args.batch, input_path)
    num_classes = ref.shape[1]

    sim_path = (f"{_REPO_ROOT}/new_code/DRAM_sim/DRAM_data/out_tensor/mlp/"
                f"{args.config}_out_B{args.batch}.bin")
    sim = np.fromfile(sim_path, dtype=np.float32).reshape(args.batch, num_classes)

    labels_path = (f"{_REPO_ROOT}/new_code/DRAM_sim/DRAM_data/in_tensor/mlp/"
                   f"labels_B{args.batch}.bin")
    # The ground-truth label belongs to the provided image only.
    labels = (np.fromfile(labels_path, dtype=np.int32)
              if args.input is None and os.path.exists(labels_path) else None)

    max_diff = np.max(np.abs(ref - sim))
    # A confident output (one probability ~1, the others tiny) hides a wrong
    # computation in the absolute difference: the tiny probabilities barely move
    # even when the logits are off. So the logarithms of the probabilities are
    # compared too, for every reference probability above LOG_FLOOR (float32
    # keeps full precision there). A correct program differs by ~1e-5 or less.
    LOG_FLOOR = 1e-30
    keep = ref > LOG_FLOOR
    with np.errstate(divide="ignore", invalid="ignore"):
        log_diff = np.abs(np.log(sim[keep].astype(np.float64)) - np.log(ref[keep].astype(np.float64)))
    max_log_diff = float(np.max(log_diff)) if log_diff.size else 0.0
    if np.isnan(max_log_diff):
        max_log_diff = float("inf")
    ref_argmax = ref.argmax(axis=1)
    sim_argmax = sim.argmax(axis=1)
    argmax_match = np.array_equal(ref_argmax, sim_argmax)

    print(f"[{args.config} B={args.batch}] input: {os.path.relpath(input_path, _REPO_ROOT)}")
    print(f"[{args.config} B={args.batch}] max abs diff vs PyTorch reference: {max_diff:.3e}")
    print(f"[{args.config} B={args.batch}] max |log p_sim - log p_ref| (p_ref > {LOG_FLOOR:g}): "
          f"{max_log_diff:.3e}")
    print(f"  argmax match (sim vs PyTorch): {argmax_match} "
          f"(sim={sim_argmax.tolist()}, ref={ref_argmax.tolist()})")
    if labels is not None:
        acc = float((sim_argmax == labels).mean())
        print(f"  accuracy vs ground-truth labels: {acc:.3f} ({int(acc*args.batch)}/{args.batch})")

    ok = max_diff < args.atol and max_log_diff < args.log_tol and argmax_match
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
