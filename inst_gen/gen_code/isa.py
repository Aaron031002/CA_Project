"""Tiny instruction-emitter helpers for the CBI simulator (small_mlp, 2x2 mesh).

Every instruction is emitted as one text line exactly in the grammar of
new_code/inst_spec.md. A `Prog` collects the four per-core instruction lists
and records every DRAM file that an instruction touches, so that
dram_map.yaml is generated from the very same data (it can never disagree with
the instructions).
"""
import os
import posixpath

NUM_CORES = 4
DATA = "new_code/DRAM_sim/DRAM_data"
WROOT = posixpath.join(DATA, "model_weight/mlp_bin/small_mlp")
IN_FILE = posixpath.join(DATA, "in_tensor/mlp/in_B1.bin")
OUT_FILE = posixpath.join(DATA, "out_tensor/mlp/small_mlp_out_B1.bin")
DIMS = [784, 128, 64, 32, 16, 10]


def wfile(layer):
    return posixpath.join(WROOT, f"layer{layer}_weight.bin")


def bfile(layer):
    return posixpath.join(WROOT, f"layer{layer}_bias.bin")


class Prog:
    def __init__(self):
        self.lines = {c: [] for c in range(NUM_CORES)}
        self.readers = {}        # path -> set of cores that read it (DMA or Array)
        self.writers = {}        # path -> set of cores that write it
        self._next_id = 500000
        self._next_pkt = 1000000

    # ---- ids -------------------------------------------------------------
    def new_id(self):
        """Cache data ids are globally unique (simplest way to never collide)."""
        self._next_id += 1
        return self._next_id

    def new_pkt(self):
        self._next_pkt += 1
        return self._next_pkt

    # ---- emit helpers ----------------------------------------------------
    def c(self, core, text):
        """Comment line."""
        self.lines[core].append(f"# {text}")

    def emit(self, core, line, comment=None):
        self.lines[core].append(line if comment is None else f"{line}  # {comment}")

    def _read(self, core, path):
        self.readers.setdefault(path, set()).add(core)

    def dram_read(self, core, path, r0, r1, c0, c1, out_id=None, comment=None):
        out_id = out_id or self.new_id()
        self._read(core, path)
        self.emit(core, f"DMA DRAM read {path} index {r0} {r1} {c0} {c1} {out_id}", comment)
        return out_id

    def dram_write(self, core, path, r0, r1, c0, c1, data_id, reserved=0, comment=None):
        self.writers.setdefault(path, set()).add(core)
        self.emit(core, f"DMA DRAM write {path} index {r0} {r1} {c0} {c1} {data_id} {reserved}", comment)

    def array(self, core, wpath, r0, r1, c0, c1, stride, x_id, x_res, out_id=None, comment=None):
        """out = x[1, r1-r0] @ W[r0:r1, c0:c1]   (weight streamed from local DRAM)."""
        out_id = out_id or self.new_id()
        self._read(core, wpath)
        self.emit(core, f"Array NORMAL {wpath} index {r0} {r1} {c0} {c1} stride {stride} "
                        f"{x_id} {out_id} {int(x_res)}", comment)
        return out_id

    def vec(self, core, mode, dim, x_id, w_id, x_res, w_res, out_id=None, comment=None):
        out_id = out_id or self.new_id()
        self.emit(core, f"Vector {mode} {dim} {x_id} {w_id} {out_id} {int(x_res)} {int(w_res)}", comment)
        return out_id

    def send(self, core, data_id, dsts, truncate=0, pkt=None, comment=None):
        pkt = pkt or self.new_pkt()
        dsts = list(dsts)
        # Always end with the truncate flag: then a destination list ending in
        # core 0 or 1 can never be misread as the flag (inst_spec.md, DMA NoC).
        self.emit(core, f"DMA NoC send {pkt} {data_id} {' '.join(map(str, dsts))} {int(truncate)}", comment)
        return pkt

    def recv(self, core, pkt, sizes, out_id=None, comment=None):
        out_id = out_id or self.new_id()
        self.emit(core, f"DMA NoC recv {pkt} {out_id} {' '.join(map(str, sizes))}", comment)
        return out_id

    def tmu(self, core, mode, ins, outs, size, comment=None):
        """ins: list of (id, reserved)."""
        ins_s = " ".join(f"{i} {int(r)}" for i, r in ins)
        outs_s = " ".join(map(str, outs))
        size_s = f" size {' '.join(map(str, size))}" if size is not None else ""
        self.emit(core, f"TMU {mode} in {ins_s} out {outs_s}{size_s}", comment)

    def sync(self, core):
        self.emit(core, "TCore LOCAL_SYNC")

    # ---- output ----------------------------------------------------------
    def dram_map_text(self):
        paths = sorted(set(self.readers) | set(self.writers))
        out = ["version: 1", "num_cores: 4", "entries:"]
        for p in paths:
            owners = sorted(self.readers.get(p, set()) | self.writers.get(p, set()))
            out.append(f'  - path: "{p}"')
            if len(owners) == 1:
                out.append(f"    owner: {owners[0]}")
            else:
                out.append(f"    owners: [{', '.join(map(str, owners))}]")
            if p in self.writers:
                out.append("    kind: output")
        return "\n".join(out) + "\n"

    def write(self, case_dir, header=None):
        os.makedirs(case_dir, exist_ok=True)
        for core in range(NUM_CORES):
            with open(os.path.join(case_dir, f"core_{core}_insts.txt"), "w", encoding="utf-8", newline="\n") as f:
                if header:
                    for h in header:
                        f.write(f"# {h}\n")
                body = self.lines[core] or [f"# core {core} is idle"]
                f.write("\n".join(body) + "\n")
        with open(os.path.join(case_dir, "dram_map.yaml"), "w", encoding="utf-8", newline="\n") as f:
            f.write(self.dram_map_text())
