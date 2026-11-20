// acc_8x8.h ── Top-level mesh accelerator module. Defaults to an 8×8
// (64-core) mesh but the mesh size is a runtime constructor parameter
// (mesh_width/mesh_height, defaulting to hardware.yaml's noc.width/
// noc.height) -- see num_cores()/mesh_width()/mesh_height() below for how
// to read an instance's actual configured size.
//
// Acc8x8 is a variant of the generic Acc<T> module whose mesh size is a
// runtime constructor parameter instead of Acc<T>'s own still-compile-time
// ACC_MESH_N constant (Acc.h, unaffected by this class). Using a dedicated
// class avoids touching Acc<T> and lets the two configurations coexist in
// the same build.
//
// Off-chip DRAM (previously wired here under a fixed compile-time node-id
// base independent of the runtime mesh size -- broken for any mesh_width/
// mesh_height other than hardware.yaml's compiled-in default, confirmed by
// a segfault when constructed at e.g. 1x1) has been removed from this
// class entirely: CA_2026_FP's dense-MLP workload never sets
// offchip_map_path/dram_group_path away from empty, so it was already
// dead weight here. This lets num_cores() genuinely be any mesh_width *
// mesh_height the constructor is given -- NoCSim itself is now
// constructed with num_offchip_nodes=0, so its own node id range is
// exactly [0, mesh_width*mesh_height) with no fixed extra slots.
//
// Structure mirrors Acc<T> exactly; only the mesh size / off-chip wiring
// differ:
//   - num_cores() Core<T> instances indexed 0..num_cores()-1 (row-major:
//     id = y*mesh_width() + x)
//   - 1 NoCSim<T>, "XY" (plain mesh) by default -- "TORUS_XY" (wrap-around
//     torus) also works for rectangular meshes (see NoCSim's own
//     constructor comment)
//   - 1 GlobalSyncManager for all num_cores() cores
//
// Instruction files are loaded per-core from:
//   <inst_dir>/core_<i>_insts.txt
//
// Usage (testbench):
//   Acc8x8<float> dut("dut", "/path/to/inst_dir");  // default mesh size
//   dut.clk(clk_sig);
//   dut.start(start_sig);

#ifndef LLM_SIM_ACC_8X8_H
#define LLM_SIM_ACC_8X8_H

#ifndef SC_INCLUDE_DYNAMIC_PROCESSES   // also set by CMake (acc_lib/core_lib)
#define SC_INCLUDE_DYNAMIC_PROCESSES
#endif
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <systemc.h>

#include "Core/Core.h"
#include "NoC_sim/NoC_sim.h"
#include "Acc/gsync.h"
#include "DRAM_sim/DRAMRouter.h"
#include "DRAM_sim/OffchipMapper.h"
#include "utils/fifo.h"

// Research switches S1 - S4 (+ S2's fork mode) of the NoCSim Acc8x8 builds
// (see NoCSim's constructor). Every default is NoCSim's own (hardware.yaml
// `noc:`), so a caller that passes none gets exactly today's NoC.
struct AccNocSwitches {
    bool endpoint_port = LLM_SIM_NOC::ENDPOINT_PORT_RULE;
    bool mc_tree       = LLM_SIM_NOC::MULTICAST_TREE;
    bool mc_fork_async = LLM_SIM_NOC::MULTICAST_FORK_ASYNC;
    bool pf_fix        = LLM_SIM_NOC::PORT_FINISH_FIX;
    bool backfill      = LLM_SIM_NOC::BACKFILL;
};

template <typename T>
class Acc8x8 : public sc_module {
public:
    SC_HAS_PROCESS(Acc8x8);

    // MESH_N/NUM_CORES are retained as the pre-existing 8x8 defaults for
    // any code that still refers to them (e.g. array-sizing constants
    // elsewhere), but the ACTUAL mesh size an instance uses is the runtime
    // mesh_width_/mesh_height_/num_cores_ set by its constructor -- see
    // mesh_width/mesh_height constructor parameters below. Do not use
    // Acc8x8<T>::NUM_CORES to size anything that depends on a specific
    // instance's configured mesh; use that instance's num_cores() instead.
    static constexpr int MESH_N    = 8;
    static constexpr int NUM_CORES = MESH_N * MESH_N;  // 64 (default-size only)

    // inst_dir        : directory containing core_<i>_insts.txt files (i = 0..63)
    // topology        : NoC routing topology for the core mesh (default: XY,
    //                   plain 2-D mesh, no wraparound). Phase 6
    //                   switched the default away from TORUS_XY: confirmed
    //                   inst_gen's scheduling/placement has no functional
    //                   dependency on torus wraparound, and off-chip DRAM
    //                   nodes need a rectangular (non-wrapping) mesh edge to
    //                   attach to. Off-chip DRAM nodes themselves never use
    //                   this topology setting (see OffChipRouteAlgo).
    // compute         : if false, units skip arithmetic and only simulate Data_t transactions
    // dram_map_path   : optional path to dram_map.yaml (Phase 1)
    //                   describing initial DRAM ownership; empty (default)
    //                   means no mapping is loaded (AddressMapper stays empty,
    //                   owner_of() always returns OWNER_LOCAL).
    // dram_group_path : optional path to dram_group.bin (Phase 3a); empty
    //                   (default) means no group bitmap is loaded (is_group()
    //                   always false, so DMA never attempts a remote read).
    // offchip_map_path: optional path to offchip_map.yaml (Phase 6c); empty
    //                   (default) means no off-chip ownership mapping is
    //                   loaded (OffchipMapper::owner_of() always returns
    //                   OWNER_NONE). Deliberately a SEPARATE file from
    //                   dram_map_path -- never merged.
    // local_dram_capacity_bytes: Phase D;
    //                   per-core local DRAM capacity limit in bytes, forwarded
    //                   to DataManagerConfig::local_dram_capacity_bytes. 0
    //                   (default) means capacity tracking stays disabled --
    //                   every pre-Phase-D testbench that doesn't pass this
    //                   keeps its exact previous behavior (has_capacity_limit()
    //                   false for every core, is_resident() always true).
    // dram_residency_in_path: Phase G; optional
    //                   path to a residency YAML dumped by a previous chained
    //                   run's DataManager::dump_residency() (see that method
    //                   and DataManagerConfig::dram_residency_in_path for the
    //                   load-before-warm_start ordering rationale). Empty
    //                   (default) means no prior residency to load -- this
    //                   run starts from warm_start() alone, unchanged from
    //                   every pre-Phase-G caller.
    // mesh_width/mesh_height: core mesh dimensions. Both default to the
    //                   hardware.yaml-derived LLM_SIM_NOC::NOC_WIDTH/
    //                   NOC_HEIGHT (8x8 today), so every existing call site
    //                   that doesn't pass these keeps its exact current
    //                   behavior. Appended at the end of the parameter list
    //                   (not interleaved with the pre-existing ones above)
    //                   so no existing positional call site needs editing.
    //                   Rectangular meshes (mesh_width != mesh_height) are
    //                   supported under both "XY" and "TORUS_XY" topology --
    //                   see NoCSim/RouteAlgo's own constructor comments.
    // noc_switches    : the NoC research switches (AccNocSwitches above);
    //                   default = hardware.yaml's, i.e. today's NoC. Appended
    //                   after mesh_height so no existing positional call site
    //                   needs editing. Note: endpoint_port (S1) is core-to-core
    //                   only; this class builds NoCSim with num_offchip_nodes=0
    //                   so there is no off-chip traffic for it to reject.
    Acc8x8(sc_module_name name, const std::string& inst_dir,
           const std::string& topology = "XY",
           bool compute = true,
           const std::string& dram_map_path = "",
           const std::string& dram_group_path = "",
           const std::string& offchip_map_path = "",
           uint64_t local_dram_capacity_bytes = 0,
           const std::string& dram_residency_in_path = "",
           int mesh_width = LLM_SIM_NOC::NOC_WIDTH,
           int mesh_height = LLM_SIM_NOC::NOC_HEIGHT,
           const AccNocSwitches& noc_switches = AccNocSwitches());
    ~Acc8x8();

    // Runtime mesh size for this instance -- use this, not the static
    // NUM_CORES/MESH_N constants above, wherever an instance's ACTUAL
    // configured size is needed (e.g. iterating cores[]).
    int num_cores() const { return num_cores_; }
    int mesh_width() const { return mesh_width_; }
    int mesh_height() const { return mesh_height_; }

    const bool compute_;

    // Runtime mesh size backing num_cores()/mesh_width()/mesh_height()
    // above -- set once at construction from the mesh_width/mesh_height
    // constructor arguments, defaulting to LLM_SIM_NOC::NOC_WIDTH/
    // NOC_HEIGHT (hardware.yaml's noc.width/noc.height).
    const int mesh_width_;
    const int mesh_height_;
    const int num_cores_;

    // ── Clock ──────────────────────────────────────────────────────────────
    sc_in_clk  clk;

    // ── Control (broadcast to all cores) ──────────────────────────────────
    sc_in<bool> start;

    // ── Sub-modules (exposed for test access) ─────────────────────────────
    // Sized at runtime (num_cores_ = mesh_width_ * mesh_height_) in the
    // constructor -- not a fixed-size array off the static NUM_CORES
    // constant, so an instance can be built with any mesh size the
    // constructor's mesh_width/mesh_height arguments specify.
    std::vector<Core<T>*> cores;
    NoCSim<T>*         noc_sim;
    GlobalSyncManager* gsync_mgr;

    // Single DataManager for the whole simulation (Phase 2).
    // Owned here; injected as a non-owning pointer into every Core.
    DataManager<T>*    data_manager;

    // Cross-core DRAM read forwarding (Phase 3b/3c). One
    // instance for the whole simulation, wired to every core's remote-read
    // channels (requester side) and every core's DRAMSim remote port (owner
    // side).
    DRAMRouter<T>*      dram_router;

    // Off-chip ownership mapping (Phase 6c/7b): which off-chip node owns
    // which weight/KV file. Phase 7b moved this into DataManager itself
    // (DataManagerConfig::offchip_map_path) since DMA already holds a
    // data_manager_ pointer -- query via data_manager->offchip_mapper()
    // rather than a separate member here, avoiding two independent copies
    // that could silently diverge.

private:
    sc_buffer<bool> start_buf;

    // Owner-side channels: DRAMRouter <-> each core's DRAMSim remote port.
    // Not exposed on Core itself (dram_sim is a public Core member, so these
    // are bound directly core-to-router here rather than bridged through
    // Core, unlike the requester-side remote_dram_req_fifo/resp_fifo which
    // Core does expose since DMA -- not dram_sim -- is the endpoint there).
    // Runtime-sized alongside cores (see its own comment above).
    std::vector<buffer<LLM_SIM_DRAM::ReqPtr_t>*>      owner_req_bufs;
    std::vector<sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>*> owner_resp_fifos;

    void start_conv();
};

#endif // LLM_SIM_ACC_8X8_H
