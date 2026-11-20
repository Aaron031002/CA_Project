// acc_8x8.cpp ── Implementation of the mesh accelerator module (8×8 by
// default, configurable via the constructor's mesh_width/mesh_height
// arguments -- see acc_8x8.h's own comment).
//
// Wires num_cores_ (= mesh_width_ * mesh_height_) Core instances to a
// single NoCSim and a GlobalSyncManager, following the same pattern as
// Acc.cpp. Defaults to 8×8=64 cores (LLM_SIM_NOC::NOC_WIDTH/NOC_HEIGHT,
// from hardware.yaml) for every caller that doesn't pass a different size,
// but genuinely supports any mesh_width/mesh_height the constructor is
// given (see acc_8x8.h's class comment re: off-chip DRAM's removal).

#include "Acc/acc_8x8.h"
#include <sstream>

template <typename T>
Acc8x8<T>::Acc8x8(sc_module_name name, const std::string& inst_dir,
                   const std::string& topology, bool compute,
                   const std::string& dram_map_path,
                   const std::string& dram_group_path,
                   const std::string& offchip_map_path,
                   uint64_t local_dram_capacity_bytes,
                   const std::string& dram_residency_in_path,
                   int mesh_width, int mesh_height,
                   const AccNocSwitches& noc_switches)
    : sc_module(name),
      compute_(compute),
      mesh_width_(mesh_width),
      mesh_height_(mesh_height),
      num_cores_(mesh_width * mesh_height),
      cores(num_cores_),
      start_buf("start_buf"),
      owner_req_bufs(num_cores_),
      owner_resp_fifos(num_cores_)
{
    // ── Instantiate the single DataManager for this simulation ───────────
    // Must happen before any Core/DRAMSim is built so dram_map.yaml/
    // dram_group.bin are loaded (if given) before the first DRAM access can
    // occur.
    DataManagerConfig dm_cfg;
    dm_cfg.compute = compute_;
    dm_cfg.dram_map_path = dram_map_path;
    dm_cfg.dram_group_path = dram_group_path;
    // Phase 7b: offchip_map_path now loads into DataManager's own
    // OffchipMapper (data_manager->offchip_mapper()) rather than a separate
    // Acc8x8 member -- DMA already holds a data_manager_ pointer, so this
    // keeps ownership-mapping queries at a single source of truth.
    dm_cfg.offchip_map_path = offchip_map_path;
    // Phase D: 0 (default) keeps every
    // pre-Phase-D testbench's exact prior behavior (capacity tracking
    // disabled for every core).
    dm_cfg.local_dram_capacity_bytes = local_dram_capacity_bytes;
    // Phase G: empty (default) keeps every
    // pre-Phase-G caller's exact prior behavior (no residency to load).
    dm_cfg.dram_residency_in_path = dram_residency_in_path;
    data_manager = new DataManager<T>(dm_cfg);

    // ── Instantiate NoCSim with the configured mesh, no off-chip nodes ────
    // num_offchip_nodes=0: this class no longer wires physical off-chip
    // DRAM (see acc_8x8.h's class comment) -- NoCSim's own node id range is
    // therefore exactly [0, mesh_width_*mesh_height_), which scales
    // correctly with any runtime mesh size the constructor is given.
    // backpressure_enabled=true is NoCSim's own default (unchanged); the
    // NoC research switches S1 - S4 (noc_switches) default to hardware.yaml's.
    noc_sim = new NoCSim<T>("NoCSim", mesh_width_, mesh_height_, topology,
                             /*num_offchip_nodes=*/0, /*backpressure_enabled=*/true,
                             noc_switches.endpoint_port, noc_switches.mc_tree,
                             noc_switches.mc_fork_async, noc_switches.pf_fix,
                             noc_switches.backfill);
    noc_sim->clk(clk);

    // ── Instantiate GlobalSyncManager for 64 cores ────────────────────────
    gsync_mgr = new GlobalSyncManager("GlobalSyncManager", num_cores_);
    gsync_mgr->clk(clk);

    // ── Instantiate DRAMRouter for cross-core DRAM reads (Phase 3b/3c) ────
    dram_router = new DRAMRouter<T>("DRAMRouter", data_manager, num_cores_);
    dram_router->clk(clk);

    // ── Instantiate all 64 Core modules and wire them ─────────────────────
    for (int i = 0; i < num_cores_; ++i) {
        std::ostringstream cname, ipath;
        cname << "Core_" << i;
        ipath << inst_dir << "/core_" << i << "_insts.txt";

        cores[i] = new Core<T>(cname.str().c_str(), ipath.str(), /*router_id=*/i, compute_,
                                data_manager);

        cores[i]->clk(clk);
        cores[i]->start(start_buf);

        // NoC packet channels
        noc_sim->in_ports[i](cores[i]->packet_out_fifo);
        noc_sim->out_fifos[i](cores[i]->packet_in_fifo);

        // Global-sync channels
        gsync_mgr->arrive_in[i] (cores[i]->gsync_arrive_fifo);
        gsync_mgr->release_out[i](cores[i]->gsync_release_fifo);

        // ── DRAMRouter: requester-side channels (this core, as requester) ──
        dram_router->req_in[i]  (cores[i]->remote_dram_req_fifo);
        dram_router->resp_out[i](cores[i]->remote_dram_resp_fifo);

        // ── DRAMRouter: owner-side channels (this core, as owner) ──────────
        std::ostringstream orb, orf;
        orb << "owner_req_buf_" << i;
        orf << "owner_resp_fifo_" << i;
        owner_req_bufs[i]   = new buffer<LLM_SIM_DRAM::ReqPtr_t>(orb.str().c_str());
        owner_resp_fifos[i] = new sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>(orf.str().c_str());

        dram_router->owner_req_out[i]        (*owner_req_bufs[i]);
        cores[i]->dram_sim->remote_req_port  (*owner_req_bufs[i]);

        cores[i]->dram_sim->remote_read_port (*owner_resp_fifos[i]);
        dram_router->owner_resp_in[i]        (*owner_resp_fifos[i]);
    }

    // ── start converter: Acc8x8 sc_in → internal sc_buffer ───────────────
    SC_THREAD(start_conv);
}

template <typename T>
Acc8x8<T>::~Acc8x8() {
    delete noc_sim;
    delete gsync_mgr;
    delete dram_router;
    for (int i = 0; i < num_cores_; ++i) {
        delete cores[i];
        delete owner_req_bufs[i];
        delete owner_resp_fifos[i];
    }
    // Delete last: Cores hold a non-owning pointer to data_manager but never
    // dereference it after their own destruction, so this order is safe.
    delete data_manager;
}

template <typename T>
void Acc8x8<T>::start_conv() {
    while (true) {
        wait(start.default_event());
        start_buf.write(start.read());
    }
}

// ── Explicit template instantiation ──────────────────────────────────────────
template class Acc8x8<float>;
