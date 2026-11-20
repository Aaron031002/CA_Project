# CBI Simulator — User Guide

This guide has two parts:

- **Part 1 — Environment Setup** gets your machine ready to configure and build the simulator.
- **Part 2 — Using the Simulator** explains the files you write, where they go, and how to build,
  run, and check them.

---

## Contents

**Part 1 — Environment Setup**

1. [What you need](#1-what-you-need)
2. [Windows: install WSL 2 and extract the repository](#2-windows-install-wsl-2-and-extract-the-repository)
3. [Docker](#3-docker)
4. [Linux and macOS](#4-linux-and-macos)
5. [Verify your environment](#5-verify-your-environment)
6. [Every new terminal session](#6-every-new-terminal-session)
7. [Troubleshooting](#7-troubleshooting)

**Part 2 — Using the Simulator**

8. [Overview](#8-overview)
9. [What is provided and what you produce](#9-what-is-provided-and-what-you-produce)
10. [The case folder](#10-the-case-folder)
11. [The accelerator: a fixed 2×2 mesh](#11-the-accelerator-a-fixed-22-mesh)
12. [File formats](#12-file-formats)
13. [Tips for writing the files](#13-tips-for-writing-the-files)
14. [Generating and building the testbench](#14-generating-and-building-the-testbench)
15. [Running and checking](#15-running-and-checking)

---

# Part 1 — Environment Setup

---

## 1. What you need

You work inside a **Docker container** that the course provides as a ready-made image. It already
contains the compiler, SystemC and Python with all packages, so you don't install any of them
yourself.

| | |
|---|---|
| **Files from the course** | `ca_fp_env_v1.tar.gz` (the environment image, about 1 GB) and `CA_FP1_26.tar.gz` (this repository) |
| **Computer** | An x86_64 computer with about **5 GB** of free disk space |
| **Windows** | WSL 2 + Docker Desktop (§2, §3). This is the setup this guide describes step by step |
| **Linux / macOS** | Docker (§4) |

If you load the image once (§3.2), you can use it for both final projects: each project's
repository has its own `start_env.sh`, and both use the same image.

---

## 2. Windows: install WSL 2 and extract the repository

WSL 2 runs a Linux system (Ubuntu) inside Windows. You build and run everything from its terminal.

1. Install WSL 2 with the **Ubuntu** distribution by following Microsoft's instructions:
   <https://learn.microsoft.com/windows/wsl/install>. During the setup you choose a Linux user
   name and password.
2. Open *Ubuntu* from the Start menu. From now on, "WSL terminal" means this window.
3. Extract the repository **in the WSL terminal**, into your Linux home folder. Your Windows
   `Downloads` folder is `/mnt/c/Users/<your Windows user name>/Downloads`:
   ```bash
   tar -xzf /mnt/c/Users/<your Windows user name>/Downloads/CA_FP1_26.tar.gz -C ~
   cd ~/CA_FP1_26
   ```
   `~/CA_FP1_26` is the **repository root** that the rest of this guide refers to.

> Don't extract the archive with Windows tools (Explorer, 7-Zip), and don't work in a Windows
> folder (`/mnt/c/…`): Windows tools lose the Linux file permissions, and builds in `/mnt/c/…`
> are much slower.

---

## 3. Docker

### 3.1 Install and start Docker Desktop

1. Install **Docker Desktop** from <https://docs.docker.com/desktop/>. Keep **"Use WSL 2 instead
   of Hyper-V"** selected (the default).
2. Open Docker Desktop and wait until it shows **"Engine running"**. Docker Desktop must be
   running whenever you use the project, also after every reboot (closing its window is fine; it
   keeps running in the system tray).
3. In Docker Desktop, open **Settings → Resources → WSL integration**, turn on **Ubuntu**, and click
   **Apply & restart**.
4. Check it in a **new** WSL terminal:
   ```bash
   docker run --rm hello-world      # must print "Hello from Docker!"
   ```

### 3.2 Load the provided image (one time)

Don't extract `ca_fp_env_v1.tar.gz`. Load it into Docker, with `<your image path>` replaced by the
path of the file you downloaded (in WSL, your Windows `Downloads` folder is
`/mnt/c/Users/<your Windows user name>/Downloads`):

```bash
docker load -i <your image path>                            # ends with "Loaded image: ca_fp_env:latest"
docker run --rm ca_fp_env cat /etc/ca_fp_env_versions.txt   # checks the image: prints a list of versions
```

If either command prints an error, the download is damaged: download the file again and repeat
both commands. The image stays in Docker after a reboot, so you only do this once.

### 3.3 Start a container

From the **repository root**, in a WSL terminal:

```bash
cd ~/CA_FP1_26
bash start_env.sh
```

This opens a shell in the container, at `/workspace`, which **is your repository folder**:
everything you create or change there stays in `~/CA_FP1_26` after you leave. Anything you write
outside `/workspace` is lost when the container exits. To leave, type `exit` (or `Ctrl-D`).

The prompt looks like `(ca_fp_2026) I have no name!@…:/workspace$`, and you may see
`groups: cannot find name for group ID …`. Both are harmless.

You can also run one command without opening a shell, e.g.
`bash start_env.sh python3 run_scripts/score_mlp.py`.

**Editing with VS Code:** install the *WSL* extension, run `code .` in the repository root in a
WSL terminal, and edit there. Use VS Code's terminal (`` Ctrl+` ``) for `bash start_env.sh`.

---

## 4. Linux and macOS

Use the same image. Install Docker, then follow §3.2 and §3.3 in a normal terminal (there is no
`/mnt/c/…`: use the folder you downloaded the files to). Extract the repository with
`tar -xzf CA_FP1_26.tar.gz -C ~`.

- **Linux:** install **Docker Engine** (<https://docs.docker.com/engine/install/>) and start it
  with `sudo systemctl start docker`. To use `docker` without `sudo`, run
  `sudo usermod -aG docker "$USER"` once, then log out and back in.
- **macOS:** install **Docker Desktop** (<https://docs.docker.com/desktop/>) and keep it running.
  On Apple Silicon (M1/M2/…) Macs the image runs under emulation, so everything is slower but
  works the same.
- **No Docker available** (e.g. a shared workstation without administrator rights): we recommend
  using your own computer. A native install is possible but not supported; it must match the
  versions pinned in [`Dockerfile`](Dockerfile) exactly (Ubuntu 24.04, GCC 13.3.0, SystemC 3.0.0
  built as C++17, and the conda environment in [`environment.yml`](environment.yml)).

---

## 5. Verify your environment

Inside the container (`bash start_env.sh`), from `/workspace`:

```bash
cd new_code && mkdir -p build && cd build && cmake .. && cd ../..
```

It must end with `SystemC found: TRUE` in the summary, then `Configuring done` and
`Generating done`. Always build in `new_code/build/`, never in `new_code/` itself.

To check the whole tool chain end to end, build the provided example mapping
([§14](#14-generating-and-building-the-testbench)) and score it ([§15](#15-running-and-checking)):
it must end with `RESULT: PASS` and a latency of 3,097 cycles.


---

## 6. Every new terminal session

1. Make sure Docker is running (Docker Desktop open, or the Docker engine started on Linux).
2. In a WSL (or Linux/macOS) terminal: `cd ~/CA_FP1_26` and `bash start_env.sh`.

Nothing else is needed.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `CMakeCache.txt directory … is different`, or CMake errors after you built outside the container | Delete `new_code/build/` and run `cmake ..` again inside the container |
| `Permission denied` when you edit files in your folder | Run `bash start_env.sh --fix-owner` once, and always start the container with `bash start_env.sh` |

---

# Part 2 — Using the Simulator

---

## 8. Overview

You map the inference (forward pass) of an **MNIST digit classifier MLP** onto the simulated
many-core accelerator. The setup is fixed:

| | |
|---|---|
| **Model** | `small_mlp`: 784 → 128 → 64 → 32 → 16 → 10 (5 linear layers) |
| **Input** | One flattened 28×28 image (**batch size 1**): `in_tensor/mlp/in_B1.bin` |
| **Output** | 10 class probabilities (after softmax) |
| **Accelerator** | A **2×2 mesh**: 4 cores, XY routing ([§11](#11-the-accelerator-a-fixed-22-mesh)) |

Each hidden layer is `Linear → bias → ReLU`. The last layer is `Linear → bias` followed by a
**softmax**.

You write the program the accelerator executes:

1. **Instructions**: one instruction file per core (4 files).
2. **DRAM mapping**: which core's local DRAM holds each weight/input/output file.

> **Your program must be correct for any input image, not just `in_B1.bin`.** `score_mlp.py`
> also runs it on other images in place of `in_B1.bin`, including a random one
> ([§15](#full-check-score_mlppy)).

You can write these files by hand, but a small **script that generates them** is much easier to
change and debug ([§13](#13-tips-for-writing-the-files)). The provided `run_scripts/` helpers
turn the case into a compiled testbench, run it and check the result.

---

## 9. What is provided and what you produce

| | Item | Location |
|---|---|---|
| ✅ Provided | Pre-converted weights and biases (`float32`) | `new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp/layer<N>_{weight,bias}.bin` |
| ✅ Provided | Trained PyTorch checkpoint, used only by the check scripts | `new_code/DRAM_sim/DRAM_data/model_weight/5mlp_weight/small_mlp/` |
| ✅ Provided | The input image, plus its ground-truth label | `new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin`, `labels_B1.bin` |
| ✅ Provided | A second image, used only by `score_mlp.py` | `new_code/DRAM_sim/DRAM_data/in_tensor/mlp/score_inputs/mnist_2.bin` |
| ✅ Provided | The simulator, the testbench generator, and the run/check scripts | `new_code/`, `run_scripts/` |
| ✍️ **You produce** | `core_0_insts.txt` … `core_3_insts.txt` | `inst_gen/MLP_small_mlp/B1/` |
| ✍️ **You produce** | `dram_map.yaml` | `inst_gen/MLP_small_mlp/B1/` |
| ✍️ **You produce** (recommended) | A generator script that writes the files above | Your choice (e.g. `inst_gen/gen_code/`) |
| ⚙️ Generated by script | `new_code/test/small_mlp_Acc.cpp` testbench | From `run_scripts/gen_acc_from_case.sh` |
| ⚙️ Produced by the simulation | Output probabilities `small_mlp_out_B1.bin` | `new_code/DRAM_sim/DRAM_data/out_tensor/mlp/` |

---

## 10. The case folder

Your files go into one folder, which already exists. It comes with a **deliberately slow example
mapping**: core 0 runs all 5 layers alone, cores 1–3 do nothing, and every instruction is followed by
a `LOCAL_SYNC`. Use it to check that your setup works end to end (build, run, check:
[§14](#14-generating-and-building-the-testbench) and [§15](#15-running-and-checking)), then replace
the files with your own mapping:

```
inst_gen/MLP_small_mlp/B1/
├── dram_map.yaml            # which core owns which DRAM file
├── core_0_insts.txt         # instructions for core 0
├── core_1_insts.txt
├── core_2_insts.txt
└── core_3_insts.txt         # one file per core, exactly 4 files
```

> **Every core needs a file, even an idle one.** If a core does nothing in your mapping, still
> write its `core_<i>_insts.txt` with only comment lines (or empty). The testbench refuses to start
> if any of the 4 files is missing.

---

## 11. The accelerator: a fixed 2×2 mesh

The accelerator is a 2-D **mesh** of 4 cores connected by a NoC with **XY routing** (packets go
along x first, then y; there are no wraparound links). The size is **built into the testbench**:
you don't configure it, and an `acc_config.yaml` in the case folder is ignored.

Core IDs are assigned row by row, `core_id = y * 2 + x`:

```
 y=1 │  2   3
 y=0 │  0   1
     └────────
       x=0 x=1
```

Each core has its own Array (matrix) unit, Vector unit, TMU, DMA engine, local DRAM and cache
(see [`new_code/curr_arch.md`](new_code/curr_arch.md)). A core can only `DMA DRAM read` from its
own local DRAM ([§12.2](#122-dram_mapyaml)). Data moves between cores over the NoC
(`DMA NoC send` / `DMA NoC recv`), and the NoC distance between two cores depends on their
coordinates: core 0 → core 3 takes two hops, core 0 → core 1 takes one.

---

## 12. File formats

### 12.1 `core_<i>_insts.txt`

This is a plain-text file with **one instruction per line**, executed in order by core *i*.
`#` starts a comment, either at the start of a line or after an instruction (with a space before
the `#`), and blank lines are ignored. Use comments generously, since they make your output much
easier to debug.

**Every other line must match the grammar exactly.** A misspelled opcode or mode (`Vectr`,
`Vector RELUU`, `DMA DRAM reed`), a missing or non-numeric field, or an extra token stops the
simulation before it starts, naming the file, line and problem:

```
Fatal: MainCtrl/DECODE: core 3, inst_gen/MLP_small_mlp/B1/core_3_insts.txt:17: unknown Vector mode 'RELUU' (see inst_spec.md for the list)
    > Vector RELUU 16 500024 0 500025 0 0
```

The full grammar for every opcode is in [`new_code/inst_spec.md`](new_code/inst_spec.md). What each
unit does is described in [`new_code/curr_arch.md`](new_code/curr_arch.md). The spec covers many
more opcodes than you need. These are the ones relevant to an MLP:

| Unit | Opcodes | Used for |
|---|---|---|
| DMA | `DMA DRAM read` / `DMA DRAM write` | Loading inputs/weights from DRAM, and writing the final output |
| DMA | `DMA NoC send` / `DMA NoC recv` | Moving tensors between cores |
| Array | `Array NORMAL` | Matrix multiply (activation × weight) |
| Vector | `Vector BIAS_ADD`, `Vector RELU`, `Vector SOFTMAX` | Bias, activation, and the final softmax |
| TMU | `TMU concat` | Combining partial results (e.g. column slices) into one tensor |
| TCore | `TCore LOCAL_SYNC`, `TCore GLOBAL_SYNC` | Synchronisation within a core and across cores |

Other opcodes and modes from the spec (for example `Vector ADD`, `TMU split`, `DMA Cache copy`) may
be used too. **Not allowed** in this assignment are the Vector modes for MoE routing and
quantization: `TOP_K`, `SPARSE_GATE`, `REDUCE_MAX_ABS` and the `QUANTIZE_DEQUANT_Q8_0*` family. They
exist for other workloads; `score_mlp.py` rejects a case that uses them.

**Example: format only.** These lines are excerpts from the provided example mapping in
`inst_gen/MLP_small_mlp/B1/core_0_insts.txt`. It is complete and correct but deliberately slow, so
it shows syntax, not a good mapping. File paths are written relative to the repository root (see
[§12.4](#124-limits-and-behaviours-to-know)).

```text
# Load the input image (row 0, cols 0..784) into on-chip buffer id 500000
DMA DRAM read new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin index 0 1 0 784 500000
TCore LOCAL_SYNC

# Layer 0: 784 -> 128: load the bias, then x*W, + bias, ReLU
DMA DRAM read new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp/layer0_bias.bin index 0 1 0 128 500001
TCore LOCAL_SYNC
Array NORMAL new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp/layer0_weight.bin index 0 784 0 128 stride 128 500000 500002 1
TCore LOCAL_SYNC
Vector BIAS_ADD 128 500002 500001 500003 1 1
TCore LOCAL_SYNC
Vector RELU 128 500003 0 500004 1 0
TCore LOCAL_SYNC
…
# Layer 4: … softmax
Vector SOFTMAX 10 500019 0 500020 1 0
TCore LOCAL_SYNC

# Write the 10 class probabilities to the output file
DMA DRAM write new_code/DRAM_sim/DRAM_data/out_tensor/mlp/small_mlp_out_B1.bin index 0 1 0 10 500020 0
TCore LOCAL_SYNC
```

The example uses only one core, so it has no inter-core instructions. Their syntax, one line each
(the IDs are just numbers):

```text
# Send cache buffer 500005 to core 3 as packet 1000000 (to multicast, list several cores and end with the truncate flag: … 1 2 3 0)
DMA NoC send 1000000 500005 3
# On core 3: receive packet 1000000, a 1×32 tensor, into cache buffer 500006
DMA NoC recv 1000000 500006 1 32
# Join 4 tensors along dimension 1 (columns), in the order listed, into buffer 500013
TMU concat in 500006 0 500007 0 500008 0 500005 0 out 500013 size 1 4
# Barrier over all 4 cores
TCore GLOBAL_SYNC 0 4 0 1 2 3
```

Things that are easy to get wrong:

- **`Array NORMAL` reads the weight from DRAM by itself.** Its `layer_name` field is the weight
  file, and `index … stride` selects the block. You do **not** need a separate `DMA DRAM read` of
  the weight. Loading it with DMA as well costs extra time and fills the cache with data that is
  never used. If you do want the Array to take the weight from the cache, add
  `wsrc cache <w_id> <w_reserved>` (see `inst_spec.md`).
- **`stride` is the full row width of the file** (for layer 0: 128, its `out_dim`), not the
  width of the block you read. It matters as soon as you read only some of the columns.
- **A send and its recv must agree** on the packet ID (`1000000` above) and the size (`1 32`).
  Each sender uses its own packet ID towards a given receiver.
- **End a multicast `send` with the `truncate` flag.** After the two IDs, `DMA NoC send` reads
  every number as a destination core, except one case: if there are two or more numbers and the
  last one is `0` or `1`, that last number is the `truncate` flag (`1` = free the source buffer
  after sending), not a core. So `DMA NoC send 1900000 500000 3 2 1` sends only to cores 3 and 2
  and frees buffer 500000; core 1 never gets the packet and the run deadlocks. Whenever you list
  more than one core, end the line with the flag: `… 3 2 1 0` (keep the buffer) or `… 3 2 1 1`
  (free it). A send to a single core (`… 1`) is not affected.
- **`TMU concat … size 1 4`** joins 4 inputs along dimension 1 (the columns), in the order listed.
- **`TCore GLOBAL_SYNC 0 4 0 1 2 3`** is a barrier over all 4 cores: every core listed must
  execute a `GLOBAL_SYNC` with the same barrier ID and the same list, or the run deadlocks.

### 12.2 `dram_map.yaml`

This file tells the simulator **which core's local DRAM holds each file** that a `DMA DRAM read`,
`DMA DRAM write` or `Array` instruction touches. It contains ownership metadata only, never tensor
data.

```yaml
version: 1
num_cores: 4                        # must be 4
entries:
  - path: "new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin"
    owner: 0                        # a single owning core …
  - path: "new_code/DRAM_sim/DRAM_data/model_weight/mlp_bin/small_mlp/layer0_weight.bin"
    owners: [0, 1, 2, 3]            # … or replicated on several cores
  - path: "new_code/DRAM_sim/DRAM_data/out_tensor/mlp/small_mlp_out_B1.bin"
    owner: 3
    kind: output                    # mark the file your program writes as output
```

- **Every** file named in a `DMA DRAM read/write` or `Array` instruction needs an entry, and
  every entry needs `owner: <core>` or `owners: [<core>, …]` with cores 0–3. `score_mlp.py` checks
  both.
- **A core can only `DMA DRAM read` files from its own local DRAM.** List every core that reads a
  file under `owners`. Otherwise the simulation stops immediately with
  `Fatal: DMA/REMOTE_DRAM: core X attempted a remote read of a file owned by non-group core Y`.
  Also list cores that read a file through `Array`. Listing a file under several owners
  (replication) costs nothing.
- The `path` strings must match the paths in your instructions **exactly** (same spelling, both
  relative).
- **Only the fields shown above are allowed:** `version`, `num_cores` and `entries`, and in each entry
  `path`, `owner`/`owners` and `kind: output`. The simulator's parser knows more fields (for other
  workloads); `score_mlp.py` rejects a `dram_map.yaml` that uses them.

### 12.3 Data layout of the provided tensors

All tensors are flat, row-major **`float32`** files with no header. The coordinates you give in
`index <row_s> <row_e> <col_s> <col_e>` follow these shapes (end-exclusive):

| File | Shape | Notes |
|---|---|---|
| `in_tensor/mlp/in_B1.bin` | `[1, 784]` | One flattened image |
| `model_weight/mlp_bin/small_mlp/layer<N>_weight.bin` | `[in_dim, out_dim]` | Already transposed: `y = x · W` |
| `model_weight/mlp_bin/small_mlp/layer<N>_bias.bin` | `[1, out_dim]` | |
| `out_tensor/mlp/small_mlp_out_B1.bin` (**your output**) | `[1, 10]` | Post-softmax probabilities |
| `in_tensor/mlp/labels_B1.bin` | `[1]` (`int32`) | Ground truth. Used only by the check scripts, never by the simulator |
| `in_tensor/mlp/score_inputs/mnist_2.bin` | `[1, 784]` | A second image. Used only by `score_mlp.py` ([§15](#full-check-score_mlppy)); your instructions never name it |

The layers are:

| Layer `N` | `in_dim` → `out_dim` | After the linear layer |
|:---:|---|---|
| 0 | 784 → 128 | bias, ReLU |
| 1 | 128 → 64 | bias, ReLU |
| 2 | 64 → 32 | bias, ReLU |
| 3 | 32 → 16 | bias, ReLU |
| 4 | 16 → 10 | bias, softmax |

`model_weight/mlp_bin/small_mlp/manifest.json` lists the same dimensions.

> **The output file name is fixed.** Your program must write its final result to
> `new_code/DRAM_sim/DRAM_data/out_tensor/mlp/small_mlp_out_B1.bin`, because that is where the
> run and check scripts look for it.

### 12.4 Limits and behaviours to know

| Topic | What happens |
|---|---|
| **Cache capacity** | Each core's cache holds at most **2 MiB** (2,097,152 bytes) of resident data. Nothing is evicted automatically. Free data you no longer need with the `*_reserved` flags (`1` = drop the input after use) or the `truncate` flag of `DMA NoC send`. When the cache is full, new writes wait forever. The log shows `[CACHE-OOM]` / `[STALL0]`, and the run ends with `WATCHDOG EXPIRED` |
| **File paths** | Write paths **relative to the repository root** (`new_code/DRAM_sim/…`), with `/` as the separator, and always run from the repository root. `score_mlp.py` does this. Absolute paths break as soon as the repository is somewhere else, e.g. under `/workspace` in Docker, and `score_mlp.py` rejects them. Windows-style `\` separators can't be opened by the simulator (which runs on Linux) |
| **Core IDs** | Only cores 0–3 exist. A `DMA NoC send` to any other ID stops the simulation with a `Fatal: NoCSim: …` error |
| **IDs are not checked** | The simulator doesn't report duplicate data IDs in a core or two senders using the same packet ID for one receiver. These bugs show up as wrong results or deadlocks, so keep IDs unique by construction |
| **Partial output** | If your program writes only part of the output tensor, the rest of the file keeps whatever was there before. `score_mlp.py` fills the file with NaN before each run, so unwritten values always fail |

---

## 13. Tips for writing the files

A generator script (Python in the `ca_fp_2026` environment is recommended) that writes the whole
case folder makes it easy to try a different mapping:

```
outputs:  inst_gen/MLP_small_mlp/B1/core_<0..3>_insts.txt
          inst_gen/MLP_small_mlp/B1/dram_map.yaml
```

- **Write file paths relative to the repository root** (e.g.
  `new_code/DRAM_sim/DRAM_data/in_tensor/mlp/in_B1.bin`), never absolute paths, so your case
  folder works on any machine. To find files from inside your script, locate the root relative to
  the script's own location rather than hard-coding your home directory.
- **Use `/` in the paths you write into the files**, even if you run your script on Windows. There,
  `os.path.join` produces `\`, which the simulator can't open and which also breaks
  `dram_map.yaml`. Write the path strings with `/` yourself, or use `posixpath.join` /
  `pathlib.PurePosixPath`.
- **Run your generator script inside the container too** (or from WSL). Then it uses the same
  Python and writes the files with Linux line endings. Windows line endings (CRLF) are accepted,
  though.
- **Read the layer dimensions** from `model_weight/5mlp_weight/small_mlp/config.json` (hidden
  sizes; input is 784, output is 10), or from `mlp_bin/small_mlp/manifest.json`'s `layer_dims`.
  The manifest's `weight_root` is that folder's path relative to the repository root.
- **Generate `dram_map.yaml` from the same data structures** you use to emit the `DMA DRAM` and
  `Array` instructions, so the two can never disagree.
- **Keep IDs unique.** Buffer IDs and NoC packet IDs in the example above are just numbers, but a
  send and its matching recv must agree on the packet ID, and IDs must not collide within a core.
  See [`inst_spec.md`](new_code/inst_spec.md) for the exact rules.

---

## 14. Generating and building the testbench

The case needs a compiled simulator binary. `run_scripts/gen_acc_from_case.sh` generates its C++
source:

```bash
# from the repository root
bash run_scripts/gen_acc_from_case.sh
```

This writes **`new_code/test/small_mlp_Acc.cpp`**, a SystemC testbench for the 2×2 mesh. Don't
edit this file by hand.

Then build:

```bash
mkdir -p new_code/build && cd new_code/build
cmake ..                                     # picks up the new small_mlp_Acc.cpp
cmake --build . --target small_mlp_Acc -j4   # or: make -j4 small_mlp_Acc
cd ../..                                     # back to the repository root
```

or, from anywhere, `bash new_code/build.sh -j4`, which does the same two steps.

The first build compiles the whole simulator (about 60 files): a few minutes with `-j4`, and up
to about 15 minutes on a single core. `-j4` uses 4 CPU cores; on a shared workstation, don't
use more. Later builds only recompile what changed. One warning, `argument 1 null where non-null expected
[-Wnonnull]` from the C++ standard library, is harmless.

The binary lands in `new_code/build/bin/small_mlp_Acc`.

> You only need to do this once. Changing your instruction files or `dram_map.yaml` needs **no
> rebuild**, because both are read at run time.

---

## 15. Running and checking

### Run the case

**Run every command in this section from the repository root.** All file paths in your
instructions and `dram_map.yaml` are relative to it. Started from anywhere else (for example from
`new_code/build/`), the testbench stops with
`[ERROR] Run this from the repository root …`.

```bash
new_code/build/bin/small_mlp_Acc  inst_gen/MLP_small_mlp/B1 \
    new_code/DRAM_sim/DRAM_data/out_tensor/mlp/small_mlp_out_B1.bin  1 10
```

The arguments are `<case folder> [<output file> <batch=1> <num_classes=10>]`. The last three are
optional. When you give them, the testbench also checks that the output file exists, has the right
size, and contains only finite values.

`bash run_scripts/run_mlp_configs.sh` (no arguments) does the same run, but first fills the output file
with NaN and saves the logs to `sim_logs/mlp_small_mlp_B1.{stdout,stderr}`.

On a successful run you'll see something like:

```
[TB] Acc mesh test (small_mlp, fixed mesh: 2x2 XY)
...
[TB] All 4 cores (2x2 mesh) done at 123456 ns (123456 cycles)
...
[TB] 4 checks, 0 failures
```

Lines such as `[MC-PROG] @… inst[50]` (core 0's progress) may appear in between; they are normal.

The **cycle count** on the `done at` line is the run's **latency**: the simulated time from start
until every core has finished.

> **Watchdog:** a run that hasn't finished after **2,000,000 cycles** stops with
> `WATCHDOG EXPIRED`. This almost always means a deadlock in your instructions: a `recv` with no
> matching `send`, mismatched packet IDs, a sync list that names a core which never arrives, or a
> multicast `send` whose last core is 0 or 1 (read as the `truncate` flag, see [§12.1](#121-core_i_inststxt)).

### Check the output

```bash
python3 run_scripts/compare_output_mlp.py
```

This recomputes the same forward pass in PyTorch from the provided checkpoint and compares it with
your simulator output. It reports the maximum absolute difference, whether the predicted class
(argmax) matches, and the accuracy against the ground-truth label. It prints **PASS** or **FAIL**.
If the simulator ran on another input file, give it with `--input <file>`.

**PASS** needs all three: every probability within 1e-4 of the reference, the same predicted class,
and every logarithm of a probability within 1e-3 of the reference's (for the probabilities above
1e-30). The last check matters when the network is very sure of one class: the other probabilities
are then tiny, and a wrong result would hardly change them in absolute terms. A correct program is
far inside all three limits (float rounding gives differences around 1e-5).

> **Careful:** `compare_output_mlp.py` only reads the output file. If your run crashed, hung, or
> wrote only part of the output, the file may still hold a correct result from an earlier run, and
> the comparison will say PASS. Trust `score_mlp.py` below, which resets the output before every
> run.

### Full check: `score_mlp.py`

```bash
python3 run_scripts/score_mlp.py
```

It:

1. checks the case folder: exactly 4 `core_<i>_insts.txt` files, `num_cores: 4` in
   `dram_map.yaml`, an entry with valid owners for every file your instructions name and only the
   allowed fields ([§12.2](#122-dram_mapyaml)), the allowed Vector modes
   ([§12.1](#121-core_i_inststxt)), and the DRAM file rules below;
2. fills the output file with NaN, runs the simulator and reads the latency;
3. checks that the output file was written **by this run** and passes `compare_output_mlp.py`;
4. repeats steps 2–3 twice more, with other input images in place of `in_B1.bin`: the second
   provided image `score_inputs/mnist_2.bin`, and a **random image generated for this run** (every
   pixel a random value in [0, 1)). For these runs it uses a copy of your case folder in which
   every `in_B1.bin` path points to the other image. Your own files are not changed.

**Every run must pass, and the latency must be the same on all three inputs.** A correct program does
the same work whatever the pixel values are, so its latency doesn't change. A program that only
works for `in_B1.bin` (for example, one that skips the pixels that are 0 in `in_B1.bin`, or the
neurons that happen to be 0 for it) fails here. The reported latency is the one for `in_B1.bin`.

**DRAM file rules.** Your instructions (`DMA DRAM read` and `Array`) may read only the provided
input file `in_B1.bin` and the provided `small_mlp/layer<N>_{weight,bias}.bin` files. The only file
they may write is the output, `out_tensor/mlp/small_mlp_out_B1.bin`, and the final result must be
written there by a `DMA DRAM write`. Intermediate results stay on chip: they move between cores over
the NoC, not through files in DRAM.

It prints one line at the end:

```
RESULT: PASS  cores=4  latency=123456 cycles  (same on all 3 inputs: in_B1, mnist_2, random)
```

or `RESULT: FAIL  <reason>`. When a run with another input fails, the reason starts with
`with input mnist_2 (…)` or `with input random (…)`.

Everything is saved in a new folder `sim_logs/score_<timestamp>/`: `score.json` (the result, the
latency of each input, and the random image's `seed`), the three inputs under `inputs/`, the copies
of your case folder used for the other inputs (`case_mnist_2/`, `case_random/`), and for each input
the simulator log (`.stdout`, `.stderr`), the output (`.out.bin`) and the comparison (`.compare`). To
repeat a run with the same random image, pass its seed: `python3 run_scripts/score_mlp.py --seed <seed>`.

Every run writes the same output file, so run one check at a time. A second `score_mlp.py` started
while one is running prints `Another score_mlp.py is running …` and waits for it. Don't run the
simulator by hand (or `run_mlp_configs.sh`) during a check either.

> **Grading uses this script.** The TAs run an unmodified `score_mlp.py`, with the provided
> simulator and data, on the case folder you submit. Run it as often as you like, and organise your
> own work (scripts, extra runs, other inputs) however you want.

> **First run: the provided example.** Before you change anything, build and run the example that
> comes in the case folder ([§10](#10-the-case-folder)). `score_mlp.py` should print
> `RESULT: PASS  cores=4  latency=3097 cycles  (same on all 3 inputs: …)`. If it does, your environment, the provided data and
> the scripts all work, and any later problem is in your own files.

### Common error messages

| Message | Meaning |
|---|---|
| `bash: run_scripts/….sh: Permission denied` | The script lost its execute permission (this can happen when the archive is extracted with Windows tools). Run it as `bash run_scripts/….sh`, as written in this guide |
| `[ERROR] Run this from the repository root …` | You started the binary from another folder (e.g. `new_code/build/`). `cd` back to the repository root |
| `[ERROR] Missing instruction file: …/core_<i>_insts.txt` | One of the 4 core files is missing ([§10](#10-the-case-folder)) |
| `[ERROR] Missing DRAM map: …/dram_map.yaml` | The case folder has no `dram_map.yaml` ([§10](#10-the-case-folder)) |
| `[ERROR] Could not read the case files in … (check dram_map.yaml): …` | `dram_map.yaml` isn't valid YAML or misses an entry; the text after the colon says what and where |
| `[ERROR] … starts with a UTF-8 byte-order mark (BOM)` | The file was saved as "UTF-8 with BOM" (some Windows editors, or Python's `encoding="utf-8-sig"`). Save it as plain UTF-8 |
| `Error: DRAMSim/do_read: read_data returned 0 element(s) … unreadable: <path>` | That file can't be read: a typo in the path, a path with `\`, or the provided data isn't in `new_code/DRAM_sim/DRAM_data/` |
| `score_mlp.py`: `provided data file(s) missing: …` | Provided files under `new_code/DRAM_sim/DRAM_data/` were moved or deleted. Extract the archive again into another folder (§2) and copy its `new_code/DRAM_sim/DRAM_data/` over yours |
| `score_mlp.py`: `path '…' contains '\'` / `is absolute` | Fix the path strings your generator writes ([§13](#13-tips-for-writing-the-files)) |
| `score_mlp.py`: `Vector … is not allowed in this assignment` | The MoE-routing and quantization Vector modes are not allowed ([§12.1](#121-core_i_inststxt)) |
| `score_mlp.py`: `dram_map.yaml … field(s) […] not allowed` | Use only `path`, `owner`/`owners` and `kind: output` in entries ([§12.2](#122-dram_mapyaml)) |
| `score_mlp.py`: `reads file(s) other than …` / `writes file(s) other than the output file` | Only the provided files may be read and only the output file written; move intermediate results over the NoC ([§15](#full-check-score_mlppy)) |
| `score_mlp.py`: `'…' has no entry in dram_map.yaml` / `needs owner: <core> or owners: [...]` | Add an entry for that exact path string, with owners among cores 0–3 ([§12.2](#122-dram_mapyaml)) |
| `score_mlp.py`: `with input mnist_2 (…)` / `with input random (…)` then a reason | Your program passes on `in_B1.bin` but not on another image: it depends on the values in `in_B1.bin`, e.g. it skips pixels or neurons that are 0 only for that image. The inputs and outputs of that run are in the log folder ([§15](#full-check-score_mlppy)) |
| `score_mlp.py`: `the latency depends on the input values` | The same program took a different number of cycles on different inputs. Look for work that depends on the data |
| `Fatal: MainCtrl/DECODE: core X, …/core_X_insts.txt:<line>: …` | That line doesn't match the instruction grammar: misspelled opcode/mode, a missing or extra field. The message says which ([§12.1](#121-core_i_inststxt)) |
| `Fatal: NoCSim: node X sent packet P to node D, but this 2x2 mesh only has nodes 0..3` | A `DMA NoC send` names a core that doesn't exist. Only cores 0–3 exist ([§11](#11-the-accelerator-a-fixed-22-mesh)) |
| `Fatal: DMA/REMOTE_DRAM: core X attempted a remote read …` | Core X uses `DMA DRAM read` on a file whose `dram_map.yaml` entry doesn't list X as an owner |
| `[CACHE-WREQ-MISMATCH] … write request and data disagree` (exit -6) | Usually a `DMA NoC recv` whose size differs from the size of the matching `send` ([§12.1](#121-core_i_inststxt)) |
| `simulation failed (exit -8: SIGFPE …)` | The simulator hit a division by zero, usually from an instruction with a size or `dim` of 0 |
| `[CACHE-OOM] … Cache OOM` then `WATCHDOG EXPIRED` | A core's cache is full (2 MiB). Free data with the `*_reserved` / `truncate` flags |
| `[ARR-STUCK]`, `[VEC-STUCK]`, `[STALL0] stuck on inst[n]` then `WATCHDOG EXPIRED` | A unit is waiting for data that never arrives: a `recv` without a matching `send`, a wrong packet ID or size, or a data ID that was never produced (or was already freed). Also check every multicast `send` whose last core is 0 or 1 ([§12.1](#121-core_i_inststxt)) |
| `[FAIL] Output values are all finite` | Part of the output was never written, e.g. wrong row/column range in `DMA DRAM write`, or a Vector `dim` that's too small |
