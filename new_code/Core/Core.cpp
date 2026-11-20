// Core.cpp ── Implementation of the Core hierarchical module.
//
// Wires together:
//   MainCtrl → ArrayUnit / VectorUnit / DMA / TMU
//   Cache    ↔ ArrayUnit / VectorUnit / DMA / TMU
//   DMA      ↔ DRAMSim / Router
//   ArrayUnit↔ DRAMSim
//
// See Core.h for full channel layout and connection policy.

#include "Core/Core.h"

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
Core<T>::Core(sc_module_name name, const std::string& inst_file, int router_id, bool compute,
               DataManager<T>* data_manager_in)
    : sc_module(name),
      compute_(compute),
      // ── Control channels ───────────────────────────────────────────────
      start_buf("start_buf"),
      mc_done_buf("mc_done_buf"),
      // ── Instruction channels (sc_fifo, direct) ─────────────────────────
      array_inst_fifo("array_inst_fifo", 8),
      vec_inst_fifo  ("vec_inst_fifo",   8),
      dma_inst_fifo  ("dma_inst_fifo",   8),
      tmu_inst_fifo  ("tmu_inst_fifo",   8),
      // ── Busy channels (typed) ──────────────────────────────────────────
      array_busy_ch("array_busy_ch"),
      vec_busy_ch  ("vec_busy_ch"),
      dma_busy_ch  ("dma_busy_ch"),
      tmu_busy_ch  ("tmu_busy_ch"),
      // ── Busy channels (base) ───────────────────────────────────────────
      array_busy_base_ch("array_busy_base_ch"),
      vec_busy_base_ch  ("vec_busy_base_ch"),
      dma_busy_base_ch  ("dma_busy_base_ch"),
      tmu_busy_base_ch  ("tmu_busy_base_ch"),
      // ── Cache channels: Port 0 (ArrayUnit) ────────────────────────────
      cache_array_req_fifo ("cache_array_req_fifo",  8),
      cache_array_out_fifo ("cache_array_out_fifo",  8),
      cache_array_in_fifo  ("cache_array_in_fifo",   8),
      cache_array_wreq_fifo("cache_array_wreq_fifo", 8),
      // ── Cache channels: Port 1 (VectorUnit input) ─────────────────────
      cache_vec_in_req_fifo("cache_vec_in_req_fifo", 8),
      cache_vec_in_out_fifo("cache_vec_in_out_fifo", 8),
      cache_stub_in1_fifo  ("cache_stub_in1_fifo",   2),
      cache_stub_wreq1_fifo("cache_stub_wreq1_fifo", 2),
      // ── Cache channels: Port 2 (VectorUnit weight) ────────────────────
      cache_vec_w_req_fifo ("cache_vec_w_req_fifo",  8),
      cache_vec_w_out_fifo ("cache_vec_w_out_fifo",  8),
      cache_stub_in2_fifo  ("cache_stub_in2_fifo",   2),
      cache_stub_wreq2_fifo("cache_stub_wreq2_fifo", 2),
      // ── Cache channels: Port 3 (VectorUnit output) ────────────────────
      cache_stub_req3_fifo ("cache_stub_req3_fifo",  2),
      cache_stub_out3_fifo ("cache_stub_out3_fifo",  2),
      cache_vec_out_in_fifo("cache_vec_out_in_fifo", 8),
      cache_vec_wreq_fifo  ("cache_vec_wreq_fifo",   8),
      // ── Cache channels: Port 4 (DMA) ──────────────────────────────────
      // 2026-08-14 (Bug H
      // continuation): tried widening 8->32 to relieve Phase 4's
      // full-attention chain-forward burst (K+V recv writes + K+V forward
      // reads, ~12+ ops back to back once MAX_XFER_ROWS splits seq_len=512
      // into 2 chunks) saturating cache_dma_in_fifo and permanently
      // stalling DRAM_READ_PUBLISH's nb_write() retry. Widening only
      // delayed the same deadlock to a later sim time (and a later core in
      // the chain) rather than resolving it -- confirmed NOT a sufficient
      // fix on its own. Reverted to 16 (a still-reasonable bump from the
      // original 8, kept as headroom) while the real fix -- restructuring
      // Phase 4's K/V exchange to genuinely stagger the two MAX_XFER_ROWS
      // chunks in time instead of back-to-back -- is designed separately.
      cache_dma_req_fifo   ("cache_dma_req_fifo",    16),
      cache_dma_out_fifo   ("cache_dma_out_fifo",    16),
      cache_dma_in_fifo    ("cache_dma_in_fifo",     16),
      cache_dma_wreq_fifo  ("cache_dma_wreq_fifo",   16),
      // ── Cache channels: Port 5 (TMU) ──────────────────────────────────
      cache_tmu_req_fifo   ("cache_tmu_req_fifo",    8),
      cache_tmu_out_fifo   ("cache_tmu_out_fifo",    8),
      cache_tmu_in_fifo    ("cache_tmu_in_fifo",     8),
      cache_tmu_wreq_fifo  ("cache_tmu_wreq_fifo",   8),
      // ── Cache channels: Port 6 (MainCtrl cache read, stub write side) ──
      cache_stub_wreq6_fifo ("cache_stub_wreq6_fifo",  2),
      // ── Cache channels: Port 7 (ArrayUnit weight-from-cache) ──────────
      cache_array_w_req_fifo("cache_array_w_req_fifo", 8),
      cache_array_w_out_fifo("cache_array_w_out_fifo", 8),
      cache_stub_in7_fifo   ("cache_stub_in7_fifo",    2),
      cache_stub_wreq7_fifo ("cache_stub_wreq7_fifo",  2),
      // ── DRAM request bridges ───────────────────────────────────────────
      dma_dram_req_fifo  ("dma_dram_req_fifo",   16),
      dma_dram_req_buf   ("dma_dram_req_buf"),
      array_dram_req_fifo("array_dram_req_fifo", 16),
      array_dram_req_buf ("array_dram_req_buf"),
      // ── DRAM data FIFOs ────────────────────────────────────────────────
      dram_to_dma_fifo  ("dram_to_dma_fifo",   16),
      dma_to_dram_fifo  ("dma_to_dram_fifo",   16),
      dram_to_array_fifo("dram_to_array_fifo", 16),
      array_to_dram_fifo("array_to_dram_fifo", 16),
      // ── Phase F: fill request/response ────────────────────────────────
      dram_fill_req_fifo ("dram_fill_req_fifo",  16),
      dram_fill_resp_fifo("dram_fill_resp_fifo", 16),
      // ── (Part D): MoE weight miss ────
      // fallback notify
      dma_miss_notify_fifo("dma_miss_notify_fifo", 16),
      // ── DRAM busy ──────────────────────────────────────────────────────
      dram_busy_ch("dram_busy_ch"),
      // ── NoC inst channels ──────────────────────────────────────────────
      dma_noc_inst_fifo    ("dma_noc_inst_fifo",    16),
      router_send_inst_fifo("router_send_inst_fifo"),
      router_recv_inst_fifo("router_recv_inst_fifo"),
      // ── NoC data FIFOs ─────────────────────────────────────────────────
      dma_to_router_fifo("dma_to_router_fifo", 16),
      router_to_dma_fifo("router_to_dma_fifo", 16),
      packet_out_fifo   ("packet_out_fifo",    64),
      packet_in_fifo    ("packet_in_fifo"),
      dynamic_req_fifo  ("dynamic_req_fifo",   1024),
      // ── NoC busy ───────────────────────────────────────────────────────
      noc_busy_ch("noc_busy_ch"),
      // ── Global-sync channels ───────────────────────────────────────────
      gsync_arrive_fifo ("gsync_arrive_fifo",  4),
      gsync_release_fifo("gsync_release_fifo", 4),
      // ── DataManager (not owned; injected by Acc/Acc8x8) ────────────────
      data_manager(data_manager_in)
{
    // ── Instantiate sub-modules ────────────────────────────────────────────
    main_ctrl   = new MainCtrl     ("MainCtrl",  inst_file, router_id);
    array_unit  = new ArrayUnit<T> ("ArrayUnit", 0, compute_);
    vector_unit = new VectorUnit<T>("VectorUnit",0, compute_);
    dma_unit    = new ::DMA<T>     ("DMA");
    tmu_unit    = new ::TMU<T>     ("TMU",        compute_);
    cache       = new Cache<T>     ("Cache",     8);   // 8 ports (port 6: MainCtrl cache read, port 7: ArrayUnit weight-from-cache)
    dram_sim    = new DRAMSim<T>   ("DRAMSim",   data_manager, LLM_SIM_DRAM::NUM_SRC, router_id);
    router      = new Router<T>    ("Router",    router_id);

    // Wire DMA to MainCtrl.
    main_ctrl->setDMA(dma_unit);

    // Phase 3b: wire DMA to the injected DataManager/
    // AddressMapper and this core's own id, so try_dispatch() can resolve
    // DRAM read ownership and redirect cross-core reads to DRAMRouter.
    dma_unit->setDataManager(data_manager);
    dma_unit->setCoreId(router_id);

    // ── Clock ──────────────────────────────────────────────────────────────
    main_ctrl->clk  (clk);
    array_unit->clk (clk);
    vector_unit->clk(clk);
    dma_unit->clk   (clk);
    tmu_unit->clk   (clk);
    cache->clk      (clk);
    dram_sim->clk   (clk);
    router->clk     (clk);

    // ── Control: Core sc_in → sc_buffer → MainCtrl sc_in ──────────────────
    // A converter thread watches Core's sc_in ports and propagates any change
    // into the internal sc_buffers.  MainCtrl reads from the sc_buffers.
    main_ctrl->start(start_buf);
    main_ctrl->done (mc_done_buf);
    SC_THREAD(start_seq_conv);

    // ══════════════════════════════════════════════════════════════════════
    // Instruction path: MainCtrl sc_fifo_out → sc_fifo → unit sc_fifo_in
    // ══════════════════════════════════════════════════════════════════════
    main_ctrl->array_inst_out[0]  (array_inst_fifo);
    array_unit->inst_in           (array_inst_fifo);

    main_ctrl->vector_inst_out[0] (vec_inst_fifo);
    vector_unit->vector_inst_in   (vec_inst_fifo);

    main_ctrl->dma_inst_out[0]    (dma_inst_fifo);
    dma_unit->inst_in             (dma_inst_fifo);

    main_ctrl->tmu_inst_out[0]    (tmu_inst_fifo);
    tmu_unit->inst_buffer_in      (tmu_inst_fifo);

    // ── Global-sync path: MainCtrl ↔ gsync_arrive/release FIFOs ──────────
    // Acc wires these FIFOs on to GlobalSyncManager at the top level.
    main_ctrl->gsync_arrive_out  (gsync_arrive_fifo);
    main_ctrl->gsync_release_in  (gsync_release_fifo);

    // ══════════════════════════════════════════════════════════════════════
    // Busy state: units → typed sc_buffer → converter → base sc_buffer → MC
    // ══════════════════════════════════════════════════════════════════════
    array_unit->array_busy       (array_busy_ch);
    vector_unit->vector_busy     (vec_busy_ch);
    dma_unit->dma_busy           (dma_busy_ch);
    tmu_unit->tmu_busy           (tmu_busy_ch);

    main_ctrl->array_busy_in[0]  (array_busy_base_ch);
    main_ctrl->vector_busy_in[0] (vec_busy_base_ch);
    main_ctrl->dma_busy_in[0]    (dma_busy_base_ch);
    main_ctrl->tmu_busy_in[0]    (tmu_busy_base_ch);

    // ══════════════════════════════════════════════════════════════════════
    // Cache connections (6 ports)
    // ══════════════════════════════════════════════════════════════════════
    // Port 0: ArrayUnit
    array_unit->in_buf_req        (cache_array_req_fifo);
    cache->data_req[0]            (cache_array_req_fifo);
    cache->data_out[0]            (cache_array_out_fifo);
    array_unit->in_buf_in         (cache_array_out_fifo);
    array_unit->out_buf_out       (cache_array_in_fifo);
    cache->data_in[0]             (cache_array_in_fifo);
    array_unit->cache_wreq_out    (cache_array_wreq_fifo);
    cache->wreq_in[0]             (cache_array_wreq_fifo);

    // Port 1: VectorUnit input read
    vector_unit->in_buf_req       (cache_vec_in_req_fifo);
    cache->data_req[1]            (cache_vec_in_req_fifo);
    cache->data_out[1]            (cache_vec_in_out_fifo);
    vector_unit->input_buffer_in  (cache_vec_in_out_fifo);
    cache->data_in[1]             (cache_stub_in1_fifo);   // stub (no writes from vec to port 1)
    cache->wreq_in[1]             (cache_stub_wreq1_fifo); // stub

    // Port 2: VectorUnit weight read
    vector_unit->w_buf_req        (cache_vec_w_req_fifo);
    cache->data_req[2]            (cache_vec_w_req_fifo);
    cache->data_out[2]            (cache_vec_w_out_fifo);
    vector_unit->weight_buffer_in (cache_vec_w_out_fifo);
    cache->data_in[2]             (cache_stub_in2_fifo);   // stub (weight read-only from cache)
    cache->wreq_in[2]             (cache_stub_wreq2_fifo); // stub

    // Port 3: VectorUnit output write
    cache->data_req[3]            (cache_stub_req3_fifo);  // stub
    cache->data_out[3]            (cache_stub_out3_fifo);  // stub drain
    vector_unit->output_buffer_out(cache_vec_out_in_fifo);
    cache->data_in[3]             (cache_vec_out_in_fifo);
    vector_unit->cache_wreq_out   (cache_vec_wreq_fifo);
    cache->wreq_in[3]             (cache_vec_wreq_fifo);

    // Port 4: DMA
    dma_unit->cache_inst_out      (cache_dma_req_fifo);
    cache->data_req[4]            (cache_dma_req_fifo);
    cache->data_out[4]            (cache_dma_out_fifo);
    dma_unit->cache_data_in       (cache_dma_out_fifo);
    dma_unit->cache_data_out      (cache_dma_in_fifo);
    cache->data_in[4]             (cache_dma_in_fifo);
    dma_unit->cache_wreq_out      (cache_dma_wreq_fifo);
    cache->wreq_in[4]             (cache_dma_wreq_fifo);

    // Port 5: TMU
    tmu_unit->cache_req_fifo      (cache_tmu_req_fifo);
    cache->data_req[5]            (cache_tmu_req_fifo);
    cache->data_out[5]            (cache_tmu_out_fifo);
    tmu_unit->input_fifo          (cache_tmu_out_fifo);
    tmu_unit->output_fifo         (cache_tmu_in_fifo);
    cache->data_in[5]             (cache_tmu_in_fifo);
    tmu_unit->cache_wreq_out      (cache_tmu_wreq_fifo);
    cache->wreq_in[5]             (cache_tmu_wreq_fifo);

    // Port 6: MainCtrl cache read（DMA 發 req，Cache 直送 MainCtrl）
    dma_unit->cache_inst_out_mc   (cache_mc_req_fifo);
    cache->data_req[6]            (cache_mc_req_fifo);
    cache->data_out[6]            (cache_mc_out_fifo);
    main_ctrl->cache_data_in      (cache_mc_out_fifo);
    cache->data_in[6]             (cache_stub_in6_fifo);  // stub，MainCtrl 不寫 Cache
    cache->wreq_in[6]             (cache_stub_wreq6_fifo); // stub

    // Port 7: ArrayUnit weight-from-cache read (used when weight_src==CACHE)
    array_unit->w_buf_req         (cache_array_w_req_fifo);
    cache->data_req[7]            (cache_array_w_req_fifo);
    cache->data_out[7]            (cache_array_w_out_fifo);
    array_unit->weight_buf_cache_in(cache_array_w_out_fifo);
    cache->data_in[7]             (cache_stub_in7_fifo);   // stub (weight read-only from cache)
    cache->wreq_in[7]             (cache_stub_wreq7_fifo); // stub

    // ══════════════════════════════════════════════════════════════════════
    // DRAM request bridges: sc_fifo → buffer → DRAMSim.req_port (buffer_in)
    // ══════════════════════════════════════════════════════════════════════
    dma_unit->dram_req_out        (dma_dram_req_fifo);
    dram_sim->req_port[0]         (dma_dram_req_buf);

    array_unit->dram_req_out      (array_dram_req_fifo);
    dram_sim->req_port[1]         (array_dram_req_buf);

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3b: cross-core remote-read bridge.
    // DMA (this core, as requester) → Core's exposed remote_dram_req_fifo/
    // remote_dram_resp_fifo → wired by Acc/Acc8x8 to a single DRAMRouter
    // module shared by all cores (analogous to the gsync_arrive/release
    // wiring below). dram_sim->remote_req_port/remote_read_port (this core,
    // as owner) are also wired by Acc/Acc8x8 directly to the router, since
    // dram_sim is a public Core member -- no extra Core-level fifo needed
    // for that side.
    // ══════════════════════════════════════════════════════════════════════
    dma_unit->remote_dram_req_out (remote_dram_req_fifo);
    dma_unit->remote_dram_resp_in (remote_dram_resp_fifo);

    // ══════════════════════════════════════════════════════════════════════
    // DRAM data FIFOs (sc_fifo_out/in ↔ sc_fifo)
    // ══════════════════════════════════════════════════════════════════════
    dram_sim->read_port[0]        (dram_to_dma_fifo);
    dma_unit->dram_read_in        (dram_to_dma_fifo);

    dma_unit->dram_write_out      (dma_to_dram_fifo);
    dram_sim->write_port[0]       (dma_to_dram_fifo);

    dram_sim->read_port[1]        (dram_to_array_fifo);
    array_unit->weight_buf_in     (dram_to_array_fifo);

    array_unit->dram_write_out    (array_to_dram_fifo);
    dram_sim->write_port[1]       (array_to_dram_fifo);

    // ══════════════════════════════════════════════════════════════════════
    // Phase F: fill request/response, purely
    // local to this core -- see dram_fill_req_fifo/dram_fill_resp_fifo's
    // header comments in Core.h.
    // ══════════════════════════════════════════════════════════════════════
    dram_sim->fill_req_out        (dram_fill_req_fifo);
    dma_unit->dram_fill_req_in    (dram_fill_req_fifo);

    dma_unit->dram_fill_resp_out  (dram_fill_resp_fifo);
    dram_sim->fill_resp_in        (dram_fill_resp_fifo);

    // ══════════════════════════════════════════════════════════════════════
    // (Part D): MoE weight miss fallback
    // notify, purely local to this core -- see dma_miss_notify_fifo's
    // header comment in Core.h.
    // ══════════════════════════════════════════════════════════════════════
    dma_unit->dma_miss_notify_out      (dma_miss_notify_fifo);
    main_ctrl->main_ctrl_miss_notify_in(dma_miss_notify_fifo);

    // DRAM busy (sc_buffer) → DMA.dram_busy_in (sc_in)
    dram_sim->dram_busy_out       (dram_busy_ch);
    dma_unit->dram_busy_in        (dram_busy_ch);

    // ══════════════════════════════════════════════════════════════════════
    // NoC connections
    // ══════════════════════════════════════════════════════════════════════
    // DMA.noc_inst_out → sc_fifo → dispatch thread → fifo → Router
    dma_unit->noc_inst_out        (dma_noc_inst_fifo);
    router->send_inst_in          (router_send_inst_fifo);
    router->recv_inst_in          (router_recv_inst_fifo);

    // NoC data
    dma_unit->noc_send_out        (dma_to_router_fifo);
    router->send_in_ports         (dma_to_router_fifo);
    router->recv_out_ports        (router_to_dma_fifo);
    dma_unit->noc_recv_in         (router_to_dma_fifo);

    // Packet output (exposed for test), stub recv input
    router->send_out_packet       (packet_out_fifo);
    router->recv_in_packet        (packet_in_fifo);

    // Dynamic-request path: Router → MainCtrl
    router->dynamic_notify_out    (dynamic_req_fifo);
    main_ctrl->dynamic_req_in     (dynamic_req_fifo);

    // NoC busy → DMA.noc_busy_in (sc_in)
    router->router_busy           (noc_busy_ch);
    dma_unit->noc_busy_in         (noc_busy_ch);

    // ══════════════════════════════════════════════════════════════════════
    // Register SC_THREADs
    // ══════════════════════════════════════════════════════════════════════
    SC_THREAD(array_busy_conv);
    SC_THREAD(vec_busy_conv);
    SC_THREAD(dma_busy_conv);
    SC_THREAD(tmu_busy_conv);
    SC_THREAD(dma_dram_req_conv);
    SC_THREAD(array_dram_req_conv);
    SC_THREAD(noc_inst_dispatch);
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
Core<T>::~Core() {
    delete main_ctrl;
    delete array_unit;
    delete vector_unit;
    delete dma_unit;
    delete tmu_unit;
    delete cache;
    delete dram_sim;
    delete router;
    // data_manager is injected (owned by Acc/Acc8x8) -- do not delete here.
}

// ══════════════════════════════════════════════════════════════════════════════
// Control signal converter: Core sc_in → internal sc_buffer → MainCtrl sc_in
// ══════════════════════════════════════════════════════════════════════════════

template <typename T>
void Core<T>::start_seq_conv() {
    while (true) {
        wait(start.default_event());
        start_buf.write(start.read());
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Busy state converters: typed sc_buffer → BaseBusyPtr_t sc_buffer
// ══════════════════════════════════════════════════════════════════════════════

template <typename T>
void Core<T>::array_busy_conv() {
    while (true) {
        wait(array_busy_ch.value_changed_event());
        auto val = array_busy_ch.read();
        if (val)
            array_busy_base_ch.write(std::static_pointer_cast<LLM_SIM_BUSY::BaseBusy_t>(val));
    }
}

template <typename T>
void Core<T>::vec_busy_conv() {
    while (true) {
        wait(vec_busy_ch.value_changed_event());
        auto val = vec_busy_ch.read();
        if (val)
            vec_busy_base_ch.write(std::static_pointer_cast<LLM_SIM_BUSY::BaseBusy_t>(val));
    }
}

template <typename T>
void Core<T>::dma_busy_conv() {
    while (true) {
        wait(dma_busy_ch.value_changed_event());
        auto val = dma_busy_ch.read();
        if (val)
            dma_busy_base_ch.write(std::static_pointer_cast<LLM_SIM_BUSY::BaseBusy_t>(val));
    }
}

template <typename T>
void Core<T>::tmu_busy_conv() {
    while (true) {
        wait(tmu_busy_ch.value_changed_event());
        auto val = tmu_busy_ch.read();
        if (val)
            tmu_busy_base_ch.write(std::static_pointer_cast<LLM_SIM_BUSY::BaseBusy_t>(val));
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// DRAM request bridges: sc_fifo → buffer (for DRAMSim.req_port buffer_in)
// ══════════════════════════════════════════════════════════════════════════════

template <typename T>
void Core<T>::dma_dram_req_conv() {
    while (true) {
        wait(dma_dram_req_fifo.data_written_event());
        LLM_SIM_DRAM::ReqPtr_t req;
        if (!dma_dram_req_fifo.nb_read(req) || !req) continue;
        while (!dma_dram_req_buf.nb_write(req))
            wait(dma_dram_req_buf.data_read_event());
    }
}

template <typename T>
void Core<T>::array_dram_req_conv() {
    while (true) {
        wait(array_dram_req_fifo.data_written_event());
        LLM_SIM_DRAM::ReqPtr_t req;
        if (!array_dram_req_fifo.nb_read(req) || !req) continue;
        while (!array_dram_req_buf.nb_write(req))
            wait(array_dram_req_buf.data_read_event());
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// NoC instruction dispatcher
//
// DMA.noc_inst_out carries both send and recv NoCInst_t objects.
// Route based on src_id:
//   src_id >= 0  → Router.send_inst_in (fifo_in)
//   src_id == -1 → Router.recv_inst_in (fifo_in)
// ══════════════════════════════════════════════════════════════════════════════

template <typename T>
void Core<T>::noc_inst_dispatch() {
    // Bug fix (hang investigation, 2026-07-28):
    // dma_noc_inst_fifo is a plain sc_fifo<T> -- its data_written_event()
    // fires ONCE per delta-cycle update, regardless of how many items were
    // written into it during that cycle's evaluate phase (standard sc_fifo
    // semantics: all writes in a cycle are batched into one notify).
    // DMA::start_offchip_fetch() (and other call sites) can write BOTH a
    // send_inst AND a recv_inst to this same fifo within one cycle -- the
    // old code below only read ONE item per wake-up before going back to
    // wait(), permanently stranding the second item (no future write would
    // ever re-trigger data_written_event() for it, since the event doesn't
    // queue -- it's "level-less", one notify per update regardless of
    // count). This silently dropped every off-chip fetch's recv_inst
    // registration whenever its send_inst was written in the same cycle,
    // so the destination core's Router never even knew to watch for that
    // packet_id -- the fetch's response would arrive at the core's Router
    // (packet_pool) and just sit there forever unmatched, hanging the
    // instruction that was waiting on it. Root-caused via temporary
    // instrumentation (since removed): Router::recv_inst_thread never
    // admitted these packet_ids into outstanding_ at all, even though the
    // NoC-level delivery itself (confirmed separately) was succeeding.
    // Fix: drain every item available this cycle, not just one.
    while (true) {
        wait(dma_noc_inst_fifo.data_written_event());
        LLM_SIM_INST::NoCInstPtr_t inst;
        while (dma_noc_inst_fifo.nb_read(inst)) {
            if (!inst) continue;
            if (inst->src_id >= 0)
                router_send_inst_fifo.write(inst);
            else
                router_recv_inst_fifo.write(inst);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiation
// ─────────────────────────────────────────────────────────────────────────────
template class Core<float>;
