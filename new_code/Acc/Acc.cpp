// Acc.cpp ── Implementation of the Acc top-level module.
//
// Wires ACC_MESH_N×ACC_MESH_N Core instances to a single NoCSim:
//   core[i]->packet_out_fifo  → noc_sim->in_ports[i]
//   noc_sim->out_fifos[i]     → core[i]->packet_in_fifo
//
// Each Core reads its instruction file from:
//   <inst_dir>/core_<i>_insts.txt

#include "Acc/Acc.h"
#include <sstream>

template <typename T>
Acc<T>::Acc(sc_module_name name, const std::string& inst_dir, const std::string& topology, bool compute,
             const std::string& dram_map_path, const std::string& dram_group_path)
    : sc_module(name),
      compute_(compute),
      start_buf("start_buf")
{
    // ── Instantiate the single DataManager for this simulation ───────────
    // Must happen before any Core/DRAMSim is built so dram_map.yaml/
    // dram_group.bin are loaded (if given) before the first DRAM access can
    // occur.
    DataManagerConfig dm_cfg;
    dm_cfg.compute = compute_;
    dm_cfg.dram_map_path = dram_map_path;
    dm_cfg.dram_group_path = dram_group_path;
    data_manager = new DataManager<T>(dm_cfg);

    // ── Instantiate NoCSim with ACC_MESH_N × ACC_MESH_N nodes ─────────────
    // num_offchip_nodes=0: Phase 6's off-chip DRAM id scheme (noc_param.h's
    // OFFCHIP_ID_BASE etc.) is a fixed global constant sized for the real
    // 8x8 mesh (Acc8x8), not parametrized by ACC_MESH_N -- it would be
    // meaningless for this generic (typically 2x2, test-only) mesh, so
    // off-chip capability is simply not offered here, now or later.
    noc_sim = new NoCSim<T>("NoCSim", ACC_MESH_N, /*mesh_height=*/ACC_MESH_N, topology, /*num_offchip_nodes=*/0);
    noc_sim->clk(clk);

    // ── Instantiate GlobalSyncManager ─────────────────────────────────────
    gsync_mgr = new GlobalSyncManager("GlobalSyncManager", NUM_CORES);
    gsync_mgr->clk(clk);

    // ── Instantiate DRAMRouter for cross-core DRAM reads (Phase 3b/3c) ────
    dram_router = new DRAMRouter<T>("DRAMRouter", data_manager, NUM_CORES);
    dram_router->clk(clk);

    // ── Instantiate each Core and wire to NoCSim and GlobalSyncManager ────
    for (int i = 0; i < NUM_CORES; ++i) {
        std::ostringstream core_name, inst_path;
        core_name << "Core_" << i;
        inst_path << inst_dir << "/core_" << i << "_insts.txt";

        cores[i] = new Core<T>(core_name.str().c_str(), inst_path.str(), /*router_id=*/i, compute_,
                                data_manager);

        // Clock
        cores[i]->clk(clk);

        // Control (broadcast via internal sc_buffers)
        cores[i]->start(start_buf);

        // ── NoC packet path ────────────────────────────────────────────────
        noc_sim->in_ports[i](cores[i]->packet_out_fifo);
        noc_sim->out_fifos[i](cores[i]->packet_in_fifo);

        // ── Global-sync path ──────────────────────────────────────────────
        // MainCtrl (inside Core) writes arrive message → gsync_arrive_fifo
        // GlobalSyncManager reads it, then writes release_id → gsync_release_fifo
        // MainCtrl reads release and resumes dispatch.
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

    // ── start/seq converter thread ────────────────────────────────────────
    SC_THREAD(start_seq_conv);
}

template <typename T>
Acc<T>::~Acc() {
    delete noc_sim;
    delete gsync_mgr;
    delete dram_router;
    for (int i = 0; i < NUM_CORES; ++i) {
        delete cores[i];
        delete owner_req_bufs[i];
        delete owner_resp_fifos[i];
    }
    // Delete last: Cores hold a non-owning pointer to data_manager but never
    // dereference it after their own destruction, so this order is safe.
    delete data_manager;
}

// ── Control converter: Acc sc_in → sc_buffer → all Core sc_in ────────────────
template <typename T>
void Acc<T>::start_seq_conv() {
    while (true) {
        wait(start.default_event());
        start_buf.write(start.read());
    }
}

// ── Explicit template instantiation ──────────────────────────────────────────
template class Acc<float>;
