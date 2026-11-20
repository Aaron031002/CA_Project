# CBI Simulator — dense-MLP student handout

A cycle-level, SystemC-based simulator of a many-core accelerator: a 2-D
mesh of cores (each with an Array/Vector/TMU/DMA unit, local DRAM and a
scratch cache) connected by a NoC. In this handout the accelerator is a
**fixed 2×2 mesh** (4 cores, XY routing). The workload is one MNIST
classifier MLP, `small_mlp` (784 → 128 → 64 → 32 → 16 → 10), run on a
single input image (batch size 1).

**Start with [USER_GUIDE.md](USER_GUIDE.md).** It covers:

- **Part 1 — Environment setup.** Install WSL 2 and Docker Desktop (Windows)
  or Docker (Linux/macOS), load the provided environment image, and start it
  with `bash start_env.sh`.
- **Part 2 — Using the simulator.** The files you write for the case, their
  formats, how to build and run the testbench, and how to check the result.

Reference material:

| File | Contents |
|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | Setup, file formats, build/run/check |
| [new_code/inst_spec.md](new_code/inst_spec.md) | Instruction grammar (every opcode and field) |
| [new_code/curr_arch.md](new_code/curr_arch.md) | What each hardware unit does, module wiring |
| [new_code/README.md](new_code/README.md) | Simulator internals: configuration YAML, build options, input/output data layout |

This repository is distributed as one archive, `CA_FP1_26.tar.gz`, not a
`git clone`. Everything it needs is inside: the simulator, the scripts,
yaml-cpp under `third_party/`, and the provided weight and input files under
`new_code/DRAM_sim/DRAM_data/`. The Docker environment is a separate file
(USER_GUIDE.md §3.2); start it with `bash start_env.sh`.
