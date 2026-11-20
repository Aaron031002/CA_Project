# CBI Simulator — Current Hardware Architecture & Dataflow

> Describes the SystemC module hierarchy, the port-level dataflow between
> modules, and the control-flow logic of `MainCtrl`, as implemented in the
> current `new_code` tree (the dense-MLP student handout; the RPC/MoE dynamic-dispatch subsystem and its `SchedulingPolicy`
> layer that an earlier transformer-oriented version of this codebase had
> here have been physically deleted, not just disabled — see §4.4). For the
> instruction text grammar see [inst_spec.md](inst_spec.md); for build/
> config/run see [README.md](README.md).

## 1. Module hierarchy

```
Acc<T> / Acc8x8<T>                              (top level, one per simulation)
├── DataManager<T>            — single shared KV-cache/weight file I/O backend
├── NoCSim<T>                 — mesh interconnect (packet transport + latency)
├── GlobalSyncManager         — distributed barrier (TCore GLOBAL_SYNC)
├── DRAMRouter<T>              — cross-core DRAM read forwarding
└── NUM_CORES × Core<T>        — one per mesh node
    ├── MainCtrl                — instruction fetch/decode/dispatch, sync
    ├── ArrayUnit<T>             — systolic-array GEMM/QK/SV
    ├── VectorUnit<T>            — elementwise/reduction/activation ops
    ├── TMU<T>                   — tensor manipulation (concat/split/view/...)
    ├── DMA<T>                   — NoC / DRAM / Cache transfer engine
    ├── Cache<T>                 — per-core shared scratch memory, 8 ports
    ├── DRAMSim<T>                — per-core local DRAM model
    └── Router<T>                — per-core NoC packet endpoint
```

Two nearly-identical top-level implementations exist: the generic `Acc<T>`
(`Acc/Acc.h`/`.cpp`, `ACC_MESH_N×ACC_MESH_N`, mesh size a compile-time
constant) and `Acc8x8<T>` (`Acc/acc_8x8.h`/`.cpp`, what every `*_Acc.cpp`
student testbench and legacy `test_acc_8x8_*` binary runs, mesh size a
runtime constructor parameter — see [new_code/README.md](README.md) §5).
Both wire an identical per-`Core` topology, described below. `Acc8x8<T>`
used to also wire 8 off-chip DRAM nodes (one per mesh row) that `Acc<T>`
never had; that wiring assumed the compile-time default mesh size and
crashed (out-of-bounds `NoCSim` port access) for any other runtime
mesh_width/mesh_height, so it has been removed entirely — `Acc8x8<T>` now
genuinely supports any mesh shape the constructor is given, including
non-square ones like 1×N (see §5.3).

## 2. Per-`Core` wiring (`Core/Core.h`/`.cpp`)

Every channel between sub-modules inside one `Core` is a typed SystemC
channel (`sc_fifo`, custom `buffer`/`fifo`, or `sc_buffer` for
signal-like values) — modules never share pointers or call each other's
methods directly (except one explicitly-injected raw-pointer hook used only
by `MainCtrl`, see §4.4). This is a hard project convention: `sc_module`s
only ever talk through ports.

```
MainCtrl ──inst──▶ ArrayUnit ──┐
         ──inst──▶ VectorUnit ─┤
         ──inst──▶ DMA ────────┼──(8-port)──▶ Cache ◀── shared scratch memory
         ──inst──▶ TMU ────────┘
         ◀─busy── {Array,Vector,DMA,TMU}         (per-unit idle/busy signal)

DMA ◀──dram_req/read/write──▶ DRAMSim ◀──remote_req/read──▶ (DRAMRouter, cross-core)
ArrayUnit ◀──dram_req/weight_read/write──▶ DRAMSim         (2nd, independent DRAM port)

DMA ◀──noc_inst/send/recv──▶ Router ◀──packet_in/out──▶ NoCSim (mesh)
Router ──dynamic_notify──▶ MainCtrl                         (wired, permanently dead — §4.4)

MainCtrl ◀─gsync_arrive/release─▶ GlobalSyncManager (top-level, shared by all cores)
```

### 2.1 Instruction path
`MainCtrl` decodes the per-core instruction file once at start (a line that
doesn't match [inst_spec.md](inst_spec.md) exactly stops the simulation with
`Fatal: MainCtrl/DECODE`, see that file's intro) and then, one
instruction per clock cycle (strict in-order), pushes typed instruction
pointers directly onto the target unit's `sc_fifo`:
`array_inst_fifo` / `vec_inst_fifo` / `dma_inst_fifo` / `tmu_inst_fifo`
(`Core.h:108-111`, each depth 8). `TCore` instructions never go through a
FIFO — they're handled inline inside `MainCtrl::run()` (§4).

### 2.2 Busy-state path
Each unit exposes an `sc_out<{Array,Vector,DMA,TMU}BusyPtr_t>` that a small
per-type converter thread in `Core` re-casts to `BaseBusyPtr_t` before
feeding `MainCtrl`'s busy-state inputs (`Core.cpp:339-377`). `MainCtrl` uses
these purely to implement `TCore LOCAL_SYNC` (§4.3) — it never inspects unit
internals otherwise.

### 2.3 Cache: 8 ports, one per producer/consumer role
**Role:** the per-core shared scratchpad memory — the *only* hand-off point
through which intermediate results move between units. There is no other
channel by which, say, ArrayUnit's GEMM output could reach VectorUnit for a
bias-add: it must be written to Cache and re-read from Cache. It also holds
data that's cheap to keep resident across instructions (activations,
small weight tiles fetched via `wsrc cache`) without round-tripping DRAM.

`Cache<T>` (`Core/Cache/Cache.h`) is instantiated with **8 ports**
(`Core.cpp:103`); each port is an independent `(data_req, data_out,
data_in)` triple. Only 6 are load-bearing for a given unit's own reads —
ports 1/2/6/7 are read-only from that unit's perspective and are wired to
`Core`-internal stub FIFOs on the write side so the port array stays
uniform:

| Port | Owner | Traffic |
|---|---|---|
| 0 | ArrayUnit | activation read **and** write (ArrayUnit both reads its input and writes its own output back through port 0) |
| 1 | VectorUnit | input operand read only |
| 2 | VectorUnit | weight operand read only |
| 3 | VectorUnit | output write only |
| 4 | DMA | read (for NoC-send / DRAM-write sourcing) and write (for NoC-recv / DRAM-read landing, plus `DMA Cache copy`) |
| 5 | TMU | read (gather inputs) and write (scatter outputs) |
| 6 | MainCtrl (via DMA) | read only — `DMA Cache mc` issues the request through DMA, and the *result* is wired straight to `MainCtrl::cache_data_in`, bypassing DMA entirely (`Core.cpp:214-219`). **Wired but currently dead**: this was the RPC subsystem's read path for routing-table/`TOP_K` lookups; nothing in `MainCtrl.cpp` reads `cache_data_in` anymore (§4.4), and the MLP instruction streams never emit `DMA Cache mc`. |
| 7 | ArrayUnit (weight-from-cache) | read only — used only when an `Array` instruction's `wsrc cache` field selects `weight_src=CACHE` instead of DRAM |

Cache is a single shared `unordered_map<data_id, Data_t>` (`Cache.cpp`), not
banked per port.

**Arbitration** (`Cache.cpp:41-338`, config: `config/hardware.yaml`'s
`cache.arbiter_mode`):
- **`default`** (fixed-priority scan): each cycle, Phase 1 scans ports
  0..N-1 and commits the *first* write that fits under
  `CACHE_MAX_CAPACITY_BITS`; Phase 2 (skipped unless `dual_port: true`)
  scans ports 0..N-1 and serves the *first* read that hits. Misses/rejected
  writes are held as `pending` and retried every cycle (no eviction — the
  cache never evicts, only pends). An age-based anti-starvation mechanism
  (`WRITE_STARVE_THRESHOLD`/`READ_STARVE_THRESHOLD = 64` cycles) forces
  through a long-starved port to prevent priority-inversion deadlocks seen
  during prefill development.
- **`parallel`**: every port gets its own independent write-then-read pipe
  with no priority scan and no starvation possible — all ports commit and
  hit-check simultaneously every eligible cycle, at the cost of not
  modeling any real port contention.

Read/write latency is bandwidth-modeled, not fixed:
`latency_cal() = max(1, floor(data_bits(data) / CACHE_BUS_WIDTH_BITS))`
(`Cache.cpp:117-124`; `CACHE_BUS_WIDTH_BITS = 256` bits/cycle, `config/hardware.yaml`'s
`cache.bus_width_bits`).

## 3. Execution units: how each one sends/receives data

All four compute units (`ArrayUnit`, `VectorUnit`, `TMU`, and — in a
different, resource-slot style — `DMA`) share the same basic per-instruction
shape: **wait for an instruction → fetch operand(s) from Cache (blocking on
the Cache's own latency) → compute (or, if `compute_=false`, just wait the
equivalent latency with zeroed output) → push result(s) back to Cache →
toggle busy**. All run as a single `SC_THREAD` sensitive to `clk.pos()`.

### 3.1 ArrayUnit (`Core/ArrayUnit/ArrayUnit.cpp`)
**Role:** the systolic-array matrix-multiply engine — everything in a
transformer that's a big GEMM goes through here: `NORMAL` for weight
projections (Q/K/V/O projections, MoE expert gate/up/down FFN matmuls) and
`DOT_PRODUCT` for `(...,M,K)×(...,M,N)→(...,K,N)` matmuls that reduce along
the M (row/token) dimension instead of K, plus two modes fused
specifically for attention: `QK` (`scores = Q·Kᵀ`, with optional causal-mask/
sliding-window/GQA parameters baked into the instruction) and `SV`
(`context = softmax(scores)·V`, with optional attention-sink handling). It's
the only unit with a weight-stationary/output-stationary dataflow choice
(`inst_spec.md`'s `df` field) and the only compute unit with two independent
weight-fetch paths (DRAM or Cache).
- Input: `inst_in` (`ArrayInstPtr_t`).
- Reads the operand two ways depending on the instruction: from **DRAM**
  (`dram_req_out`/`weight_buf_in`, its own dedicated DRAM port — port index
  1 of `DRAMSim::req_port`, separate from DMA's port 0 so Array and DMA DRAM
  traffic never contend) for the default `weight_src=DRAM` path, or from
  **Cache** (`w_buf_req`/`weight_buf_cache_in`, Cache port 7) when the
  instruction says `wsrc cache`. The activation input always comes through
  Cache port 0 (`in_buf_req`/`in_buf_in`).
- Computes via `ArrayOp<T>` (systolic-array GEMM/QK/SV timing model,
  weight-stationary vs output-stationary dataflow — see `inst_spec.md`'s
  Array `df` field).
- Writes its output back through Cache port 0 (`out_buf_out`), and can also
  write straight to DRAM (`dram_write_out`) for the (rare) DRAM-writeback
  path.
- Drives `array_busy` true for the instruction's duration.

### 3.2 VectorUnit (`Core/VectorUnit/VectorUnit.cpp`)
**Role:** everything that is *not* a big matmul — elementwise arithmetic
(`ADD`/`SUB`/`MUL`/`DIV_CONSTANT`, used for residual adds and the like),
activations (`RELU`/`SIG`/`SILU`/`SWIGLU`), normalization (`RMS` =
RMSNorm), `SOFTMAX` (with an optional attention-sink term) and `ROPE`
(rotary position embedding, computed on-the-fly with no weight fetch),
reductions (`ADD_REDUCE`/`DOT_PRODUCT`/`REDUCE_MAX_ABS`), MoE routing
(`TOP_K` for expert selection, `SPARSE_GATE` to expand it into a dense
gate tensor), the Q8_0 quantize/dequantize family (simulating quantization
error), and `BIAS_ADD`. In short: the per-element/per-row math that sits
between ArrayUnit's matmuls in every transformer block. See `inst_spec.md`
for the full mode list and semantics.
- Input: `vector_inst_in` (`VectorInstPtr_t`).
- Fetches its input operand through Cache port 1 (`in_buf_req`/
  `input_buffer_in`) unconditionally, and — only for modes in the
  `needs_weight` set (see `inst_spec.md`'s Vector section) — its weight
  operand through Cache port 2 (`w_buf_req`/`weight_buffer_in`).
- Computes via `VectorOp<T>::execute()` (or `latency_only()` when
  `compute_=false`) — every arithmetic mode (`ADD`...`QUANTIZE_DEQUANT_*`)
  lives there.
- Writes its output(s) through Cache port 3 (`output_buffer_out`) — `TOP_K`
  is the one mode that writes **two** `Data_t` (indices + values) in one
  instruction.
- Drives `vector_busy` true for the instruction's duration.

### 3.3 TMU (`Core/TMU/TMU.cpp`)
**Role:** the tensor-manipulation unit — pure data reshaping/movement with
no arithmetic: `concat`/`split`/`scatter`/`gather`/`trans`(pose)/`view`/
`pad`. This is what stitches per-head or per-shard tensors back together
(e.g. concatenating attention output across heads, splitting a projection's
output into Q/K/V, gathering the top-k experts' rows), letting ArrayUnit and
VectorUnit each stay focused on uniform, contiguous operands.
- Input: `inst_buffer_in` (`TMUInstPtr_t`).
- Both reads and writes go through the **same** Cache port 5
  (`cache_req_fifo`/`input_fifo` for reads, `output_fifo` for writes) —
  unlike Array/Vector, TMU has no separate input/output port split, since
  its inputs and outputs are pure data reshuffling (concat/scatter/split/
  trans/view/pad/gather), not tied to distinct roles.
- Computes via `TMUOp<T>` (pure data movement/reshaping, no arithmetic
  latency model beyond bandwidth).
- Drives `tmu_busy`.

### 3.4 DMA (`Core/DMA/DMA.cpp`) — the odd one out: 6 parallel resource slots
**Role:** the core's I/O engine — it is the *only* unit that can move data
across a Cache/DRAM/NoC boundary (ArrayUnit can also touch DRAM directly,
but only for its own weight fetch, §3.1). Every `DMA NoC send/recv`, `DMA
DRAM read/write`, and `DMA Cache copy/mc` instruction goes through here, so
DMA is what makes a core's Cache contents visible to another core (via NoC)
and what pulls DRAM-resident weights/KV-cache into Cache in the first
place. `MainCtrl` never touches the NoC ports directly.

Unlike the three units above, `DMA` does not process one instruction fully
before accepting the next. It models **6 independent resource slots**
(`DMA.h:180-194`), each with its own state machine (`SubPhase` enum) and its
own `TransChannel` timer, so unrelated transfers (e.g. a DRAM read and a NoC
send) proceed concurrently within one core's DMA:

| Slot | Resource | Used by instruction |
|---|---|---|
| `DRAM_SLOT` | local DRAM read/write | `DMA DRAM read/write` |
| `NOC_RECV_SLOT` | NoC receive | `DMA NoC recv` |
| `NOC_SEND_SLOT` | NoC send | `DMA NoC send` |
| `CACHE_SLOT` | Cache read/write, incl. `Cache copy` | `DMA Cache copy` / cache-sourced sends / DRAM-write staging |
| `REMOTE_DRAM_SLOT` | cross-core DRAM read | DRAM read whose owner ≠ this core (resolved via `AddressMapper::owner_of()`); never reached in this handout, where such a read stops the simulation (§5.2) |
| `OFFCHIP_FETCH_SLOT` | off-chip DRAM fetch | synthesized internally (not instruction-driven) on a group-owner cache-miss; dead code in this handout — `Acc8x8` no longer wires any off-chip node to reach (§5.3) |

`DMA Cache mc` is the one instruction type that bypasses this slot machinery
entirely: it does not occupy `CACHE_SLOT` — DMA issues the Cache request and
is immediately free again the same cycle, while the *result* is wired
directly to `MainCtrl::cache_data_in` rather than back through DMA
(`Core.cpp:214-219`; see §2.3's Cache port 6).

`try_dispatch()` picks up the next pending instruction and routes it to the
correct slot if that slot is idle and there's FIFO room; each slot's
`tick_*()` method (`tick_dram`/`tick_remote_dram`/`tick_noc_recv`/
`tick_noc_send`/`tick_cache`/`tick_offchip_fetch`) advances that slot's own
`SubPhase` state machine one step per cycle. All transferred data flows
through an internal `data_pool_` (`unordered_map<data_id, DataPtr_t>`) —
e.g. a NoC recv stages its payload in `data_pool_` before a later
`Cache write` (or another `NoC send`) drains it back out, letting a
multi-hop forward happen without ever touching Cache for a pure
pass-through hop.

DMA's ports, by role:
- **DRAM**: `dram_req_out`/`dram_read_in`/`dram_write_out` (local),
  `remote_dram_req_out`/`remote_dram_resp_in` (cross-core, via `Core` →
  `DRAMRouter`).
- **NoC**: `noc_inst_out` (both send and recv instructions — split by
  `Core::noc_inst_dispatch()` on `src_id` into `Router::send_inst_in` vs
  `recv_inst_in`, `Core.cpp:414-425`), `noc_send_out`/`noc_recv_in` (data).
- **Cache**: `cache_inst_out`/`cache_data_out`/`cache_data_in` (port 4, DMA's
  own read/write), `cache_inst_out_mc` (port 6, `DMA Cache mc` — result
  bypasses DMA, lands straight in `MainCtrl::cache_data_in`).
- **Busy**: `dma_busy`, driven whenever any slot is non-idle.

## 4. `MainCtrl`: instruction dispatch and control flow

**Role:** the core's "brain" — its per-core program counter and instruction
sequencer. It owns the one thing none of the compute/transfer units have:
program order. It also absorbs the one piece of control-plane logic that
has no dedicated hardware unit of its own — barrier synchronization
(`LOCAL_SYNC`/`GLOBAL_SYNC`).

`Core/MainCtrl/MainCtrl.cpp` (310 lines — shrank from ~2268 when the
RPC/MoE dynamic-dispatch subsystem was deleted for this handout, see §4.4).
One `SC_THREAD run()`, sensitive to `clk.pos()`.

### 4.1 Startup
Blocks on `start` (`while (!start.read()) wait();`), then reads the whole
instruction file into `inst_queue_` via `Decoder::decode()` line-by-line
(comments/blank lines skipped; a line that doesn't decode stops the
simulation with `Fatal: MainCtrl/DECODE`, §2.1)
(`MainCtrl.cpp:75-98`).

### 4.2 Main dispatch loop (`MainCtrl::run()`, `MainCtrl.cpp:106-197`)
Each cycle:
1. Look at `inst_queue_[inst_idx]`.
   - **`TCORE`-typed**: cast to `TCoreInst_t` and handled inline —
     `LOCAL_SYNC`→`handleLocalSync()`, `GLOBAL_SYNC`→`handleGlobalSync()`,
     `WAIT`→no-op — then unconditionally `++inst_idx`.
   - **Everything else** (`Array`/`Vector`/`DMA`/`TMU`): routed through
     `PortBundle` — a small type→`(write_fn, is_full_fn)` registry built in
     the constructor (`MainCtrl.cpp:22-69`) that dynamic-casts the
     instruction to its concrete type and calls the matching unit's
     `nb_write()`. If the target FIFO is full, dispatch fails silently and
     **the same instruction is retried next cycle** — this is the whole
     stall mechanism; there is no separate scoreboard/dependency tracker.
2. When `inst_idx` reaches `inst_queue_.size()`, `done` is driven `true`
   (read by the top-level testbench's "all cores done" monitor).

Dispatch is **strictly in-order per core** — the only concurrency across
instructions comes from units being pipelined internally (each unit accepts
a new instruction into its FIFO as soon as its own compute finishes, per
§3) and DMA's 6 independent slots (§3.4). There is no equivalent anymore of
an incoming request preempting this loop — the old RPC path's "drain
dynamic requests before the next local instruction" step (and the
`dynamic_req_in`/`main_ctrl_miss_notify_in` ports it read from) was deleted
along with RPC; both ports stay wired (`Core.cpp`) but nothing ever writes
to or reads from them now (§4.4).

`run()` also carries a block of debug-only instrumentation left over from
earlier debugging sessions — periodic `[MC-PROG]` progress prints for core
0, a `MONITOR_CORES`/`stall_count` stall tracer, and a hardcoded
`inst_idx`-range `[MC-HOT-*]` tracer — none of which affects dispatch
semantics; it only prints to stdout.

### 4.3 `TCore LOCAL_SYNC` / `GLOBAL_SYNC` (`MainCtrl.cpp:220-310`)
- `LOCAL_SYNC` (`handleLocalSync()`): polls `allUnitsIdle()` (all four
  busy-state inputs read `isBusy()==false`) every cycle, requiring **two
  consecutive** idle cycles before continuing (guards against a
  busy-flag-clear/next-instruction race). Also carries a `DBG_NOC_TRACE`-
  gated `[SYNC-STUCK]` diagnostic print (see §4.4 for what it reads).
- `GLOBAL_SYNC` (`handleGlobalSync()`): writes a `GSyncArriveMsg_t{barrier_id,
  core_id, participants}` to `gsync_arrive_out`, then blocks polling
  `gsync_release_in` until `GlobalSyncManager` (one instance, shared by all
  cores, top-level) has seen an arrival from every core in `participants`
  for that `barrier_id` and released it (`gsync.h`/`.cpp`). A core can only
  have one `GLOBAL_SYNC` in flight at a time (dispatch itself blocks on it).

### 4.4 `setDMA` — the one remaining raw-pointer hook (debug-only)
`MainCtrl::setDMA(dma)` (called once by `Core`'s constructor) gives
`MainCtrl` a raw pointer to the local `DMA<T>` object — the only place left
in the codebase that breaks the "ports only" convention. Its counterpart,
`setCache(cache)`, and the `Cache<T>* cache_` member it set, were **deleted
outright** along with the RPC subsystem that was their only caller
(`handleRPCDispatch`'s synchronous routing-table/`TOP_K` reads). What's
left of `dma_` today is purely diagnostic: `handleLocalSync()`'s
`[SYNC-STUCK]` print calls `dma_->noc_recv_waiting_pkt()`/
`dma_->debug_slots()` to report what a stuck core's DMA slots are doing —
no dispatch-path logic depends on it.

This branch's RPC/MoE dynamic-dispatch subsystem — `RPCInst_t`/
`RPCAggRegisterInst_t`/`RPCAggDispatchInst_t`, `handleRPCInit/Wait/Pass`,
`handleRPCDispatch`/`handleAggRegister`/`handleAggDispatch`,
`handleNotify`/`handleGrant`, `buildExpertTopo`/`selectSchedulingTopo`,
`registerDynamicOp`/`remapInst` (kernel-invocation `data_id` rewriting),
and the standalone `SchedulingPolicy` demand-tiered-scheduling layer — has
been **physically deleted** from this code base, not gated behind a flag.
`ENABLE_DYNAMIC_DISPATCH` (default `OFF`) is still a real CMake option, but
its scope has been narrowed to only the DRAM-side group/offchip/fallback/
residency mechanisms (§5) that it also used to gate — turning it `ON` no
longer restores any RPC behavior (the types it would need no longer exist,
so `ON` doesn't even compile).

## 5. Memory subsystem: DRAM (local, cross-core, off-chip)

**Role:** models the memory hierarchy behind Cache — where model weights and
the KV-cache actually live, and what it costs in latency/bandwidth to reach
them depending on *whose* DRAM the data is in. **In this handout only a
core's own local DRAM (§5.1) is used.** No DRAM group is configured, so a
`DMA DRAM read` of a file that `dram_map.yaml` doesn't list for the reading
core stops the simulation with `Fatal: DMA/REMOTE_DRAM`; list every core
that reads a file under its `owners` instead (USER_GUIDE.md §12.2). The
cross-core tier (§5.2) and the off-chip tier (§5.3) are described below
only because the code is still in the tree.

### 5.1 Local `DRAMSim<T>` (`DRAM_sim/DRAMSim.h`/`.cpp`)
**Role:** this core's own local DRAM — the default source for a `DMA DRAM
read/write` or an `Array` instruction's weight fetch when the target file's
owner is this core. One instance per core, backed by the single shared `DataManager<T>` (actual
file I/O, §"Input: tensors" in README). Two independent local request
sources (`NUM_SRC=2`, `dma_params`-configured): `req_port[0]`↔DMA,
`req_port[1]`↔ArrayUnit — kept separate so Array's own weight-fetch traffic
never contends with DMA's DRAM traffic for the *same* core. A single
`arbiter_thread()` polls both ports every cycle into one arrival-ordered
`pending_queue_`; a `processor_thread()` drains it one request at a time
(no read/write priority — strict arrival order), so all local DRAM access
from one core is fully serialized regardless of source.

### 5.2 Cross-core: `DRAMRouter<T>` (`DRAM_sim/DRAMRouter.h`/`.cpp`)
> **Not reachable in this handout** (no DRAM group is configured, see §5):
> a cross-core read stops the simulation instead.

**Role:** lets a core read a weight/KV-cache file that physically lives in
*another* core's local DRAM (e.g. a shared weight tensor not duplicated
per-core), without ever copying the data — it forwards the request into the
owner's own `DRAMSim` and relays the answer back, at the cost of an extra
per-mesh-edge "rail" latency on top of the owner's normal DRAM latency. One
instance for the whole simulation (top-level, like
`GlobalSyncManager`). When `DMA::try_dispatch()` resolves a DRAM read's
`AddressMapper::owner_of(file)` to a *different* core (and `is_group()`
confirms cross-core access is allowed for that pair — configured via
`dram_map.yaml`/`dram_group.bin`, README §4), the request goes out
`DMA::remote_dram_req_out` → `Core::remote_dram_req_fifo` →
`DRAMRouter::req_in[requester]` instead of the local `dram_req_out`. The
router forwards it into the **owner's own `DRAMSim`** via
`owner_req_out[owner]` → that core's `dram_sim->remote_req_port` — so a
remote read consumes the owner's real local DRAM bandwidth exactly like one
of the owner's own local reads. Once the owner's `DRAMSim` responds, the
router holds the response for a per-edge "rail" latency
(`DRAM_RAIL_OVERHEAD`/`DRAM_RAIL_BANDWIDTH`, `hardware.yaml`'s `dram:`
block) — modeled as one independent bandwidth resource per unordered mesh
edge `(min(a,b), max(a,b))`, so unrelated core-pairs' remote reads never
contend — before delivering it to `resp_out[requester]` → `Core`'s
`remote_dram_resp_fifo` → `DMA`'s `REMOTE_DRAM_SLOT` (§3.4).

### 5.3 Off-chip DRAM — removed from `Acc8x8<T>`, dead code elsewhere
**Role (as originally designed):** a further, slower memory tier beyond
every core's local DRAM — modeling something like off-chip HBM/DDR behind
the mesh, reached automatically (never by an explicit instruction) when a
cross-core read misses even the group-owner's DRAM, via 8 `OffChipDRAM<T>`
nodes (one per mesh row) each with independent storage and its own
`OffChipPacketAdapter<T>` for NoC framing.

**Current status:** `Acc8x8<T>` no longer instantiates any of this. Its
off-chip node wiring used a compile-time node-id base
(`LLM_SIM_NOC::OFFCHIP_ID_BASE = NOC_WIDTH*NOC_HEIGHT`, from
`hardware.yaml`) while `NoCSim`'s own port arrays were sized from the
*runtime* `mesh_width*mesh_height` the constructor was actually given —
these only happened to agree because every existing case ships at
`hardware.yaml`'s own default size (8×8 today). Constructing `Acc8x8<T>`
at any other mesh size (verified with an explicit 1×1) walked off the end
of `NoCSim`'s `in_ports`/`out_fifos` and segfaulted. Since
this handout's dense-MLP workload never sets `dram_group_path`/
`offchip_map_path` away from empty (`is_group()` always false), this path
was already unreachable at runtime even when it didn't crash — so it was
removed outright rather than fixed, which also makes `Acc8x8<T>`'s mesh
size genuinely arbitrary (see §1). The `OffChipDRAM<T>`/
`OffChipPacketAdapter<T>`/`OffchipMapper` classes and the `OFFCHIP_FETCH_SLOT`
DMA opcode (§3.4) remain in the tree but are no longer wired to anything
this handout builds by default — reviving this tier would mean re-adding
the wiring with the node-id base fixed to track the runtime mesh size
(`NoCSim` already supports this via its own `num_offchip_nodes` constructor
parameter).

## 6. NoC: `Router<T>` (per-core endpoint) + `NoCSim<T>` (mesh)

**Role:** the inter-core communication fabric — the only way one core's
`MainCtrl`/`DMA` can exchange data with another's. Everything that crosses a
core boundary goes through it: any `DMA NoC send/recv` traffic.

`Router<T>` (`NoC_sim/Router/Router.h`/`.cpp`) is each core's packet
endpoint, with three independent threads:
- `send_thread`: pairs a `send_inst_in` (`NoCInst_t`, from `DMA` via
  `Core::noc_inst_dispatch`) with the matching `send_in_ports` data, packs a
  `Packet_t` (`data2packet()` — sets `flit_size` from the payload's real
  `elem_bits`, so e.g. Q8_0 payloads report their true 8-bit/element wire
  size), and forwards it to `send_out_packet` → `NoCSim`.
- `recv_packet_thread`: buffers every inbound `Packet_t` from `NoCSim` into
  `packet_pool` keyed by `packet_id`, unconditionally — the RPC-era
  "dynamic packet" detection that used to also emit a `DynamicReq_t` to
  `dynamic_notify_out` on this same thread has been deleted; every packet
  is now pooled the same way regardless of content, and
  `dynamic_notify_out`/`MainCtrl::dynamic_req_in` stay wired but nothing
  writes to them anymore (§4.4).
- `recv_inst_thread`: matches a `recv_inst_in` (`DMA NoC recv`) against
  `packet_pool` by `packet_id` (blocking until it arrives) and delivers the
  payload to `recv_out_ports` → `DMA`.

`NoCSim<T>` (`NoC_sim/NoC_sim.h`/`.cpp`) is the shared mesh: one
`in_ports`/`out_fifos` pair per node id (cores `0..mesh_width*mesh_height-1`,
plus `num_offchip_nodes` extra ids after that if a caller asks for any —
`Acc8x8<T>` passes 0, see §5.3), configurable topology (`"XY"` plain mesh,
default; or `"TORUS_XY"` wraparound — off-chip nodes, when any exist, never
use torus wrap, since they need a rectangular mesh edge to attach to),
congestion-aware/adaptive routing (`RouteAlgo`/`NoCLat` — per-hop
`ROUTER_DELAY`/`LINK_DELAY` from `hardware.yaml`, backpressure tracked via
per-port buffer occupancy against `buffer_depth`), and its own
utilization/stall-cycle bookkeeping — exposed as a `util_*.csv` dump by
testbenches that opt into it (the legacy `test_acc_8x8_*` targets under
`ENABLE_TRANSFORMER_TESTS`, currently unbuildable on this branch — see
[new_code/README.md](README.md); this handout's own `*_Acc.cpp` testbenches
don't write one).

## 7. Global barrier: `GlobalSyncManager`

**Role:** lets an arbitrary subset of cores agree on a synchronization
point (`TCore GLOBAL_SYNC`) — e.g. "every core writing to the KV-cache must
finish before any core starts decode," or coordinating phase transitions in
a multi-batch/multi-layer schedule — without hardcoding a fixed all-cores
barrier; the participant list is chosen per-barrier by the instruction
stream itself (`inst_spec.md`'s `TCore GLOBAL_SYNC` format).

One instance, shared by all cores (top-level, `Acc/gsync.h`/`.cpp`). A
single clocked `process()` thread polls every core's `arrive_in[i]` FIFO
each cycle; the first arrival for a given `barrier_id` establishes the
expected participant set, and once every listed core has arrived, the
barrier is released to all of them via `release_out[i]` in the same pass.
Each core may have at most one `GLOBAL_SYNC` in flight at a time — enforced
naturally by `MainCtrl` blocking dispatch between the arrive-write and the
release-read (§4.3).
