// Acc.h ── Top-level accelerator module.
//
// Contains ACC_MESH_N×ACC_MESH_N Core modules wired to a single NoCSim
// that models the on-chip 2-D mesh network.
//
// Each Core[i] exposes two packet channels that are connected to the
// corresponding NoCSim port:
//   core[i]->packet_out_fifo  → noc_sim->in_ports[i]   (Core → NoC)
//   noc_sim->out_fifos[i]     → core[i]->packet_in_fifo (NoC → Core)
//
// The start sc_in is broadcast to every Core via an
// internal sc_buffer (same pattern used inside Core.h).
//
// Instruction files are loaded per-Core from:
//   <inst_dir>/core_<i>_insts.txt
//
// Change ACC_MESH_N to resize the mesh; ACC_MESH_N=2 gives a 2×2 system.

#ifndef LLM_SIM_ACC_H
#define LLM_SIM_ACC_H

#ifndef SC_INCLUDE_DYNAMIC_PROCESSES   // also set by CMake (acc_lib/core_lib)
#define SC_INCLUDE_DYNAMIC_PROCESSES
#endif
#include <memory>
#include <string>
#include <vector>
#include <systemc.h>

#include "Core/Core.h"
#include "NoC_sim/NoC_sim.h"
#include "Acc/gsync.h"
#include "DRAM_sim/DRAMRouter.h"
#include "utils/fifo.h"

// ── Mesh dimension (change this one constant to resize) ───────────────────────
static constexpr int ACC_MESH_N = 2;   // 2×2 = 4 cores, currently

template <typename T>
class Acc : public sc_module {
public:
    SC_HAS_PROCESS(Acc);

    // inst_dir        : directory that contains core_<i>_insts.txt files
    // topology        : NoC routing topology for the core mesh (default: XY,
    //                   plain 2-D mesh, no wraparound; "TORUS_XY" also
    //                   available). Phase 6 switched the
    //                   default away from TORUS_XY -- see acc_8x8.h for the
    //                   rationale (inst_gen has no dependency on wraparound;
    //                   off-chip DRAM nodes need a non-wrapping mesh edge).
    // compute         : if true (default), all units perform real arithmetic;
    //                   if false, units only simulate Data_t transactions (values zeroed)
    // dram_map_path   : optional path to dram_map.yaml (Phase 1);
    //                   empty (default) means no mapping is loaded.
    // dram_group_path : optional path to dram_group.bin (Phase 3a); empty
    //                   (default) means no group bitmap is loaded.
    Acc(sc_module_name name, const std::string& inst_dir,
        const std::string& topology = "XY",
        bool compute = true,
        const std::string& dram_map_path = "",
        const std::string& dram_group_path = "");
    ~Acc();

    // ── Clock ──────────────────────────────────────────────────────────────
    sc_in_clk clk;

    // ── Control inputs (broadcast to all cores) ────────────────────────────
    sc_in<bool> start;

    // ── Sub-modules (exposed for test access) ─────────────────────────────
    static constexpr int NUM_CORES = ACC_MESH_N * ACC_MESH_N;
    Core<T>*           cores[NUM_CORES];
    NoCSim<T>*         noc_sim;
    GlobalSyncManager* gsync_mgr;   // centralized barrier manager for GLOBAL_SYNC

    // Single DataManager for the whole simulation (Phase 2).
    // Owned here; injected as a non-owning pointer into every Core.
    DataManager<T>*    data_manager;

    // Cross-core DRAM read forwarding (Phase 3b/3c).
    DRAMRouter<T>*      dram_router;

    // Whether units should perform real computation (true) or
    // only simulate Data_t transactions with zeroed values (false).
    const bool compute_;

private:
    // Internal broadcast channels: Core sc_in → sc_buffer → each Core
    sc_buffer<bool> start_buf;

    // Owner-side channels: DRAMRouter <-> each core's DRAMSim remote port.
    buffer<LLM_SIM_DRAM::ReqPtr_t>*          owner_req_bufs[NUM_CORES];
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>*     owner_resp_fifos[NUM_CORES];

    // Converter thread: Acc sc_in → sc_buffer
    void start_seq_conv();
};

#endif // LLM_SIM_ACC_H
