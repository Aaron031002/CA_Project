// Core.h ── Top-level hierarchical module that wires together:
//   MainCtrl → ArrayUnit / VectorUnit / DMA / TMU
//   Cache    ↔ ArrayUnit / VectorUnit / DMA / TMU  (shared memory)
//   DMA      ↔ DRAMSim / Router
//   ArrayUnit↔ DRAMSim
//
// Connection policy:
//   sc_fifo_in / sc_fifo_out → sc_fifo<T>   (SystemC built-in)
//   buffer_in  / buffer_out  → buffer<T>    (custom 1-element channel)
//   fifo_in                  → fifo<T>      (custom FIFO channel)
//   sc_in / sc_out           → sc_buffer<T> (sc_signal-compatible)
//
// MainCtrl now emits typed instructions (ArrayInstPtr_t, VectorInstPtr_t,
// DMAInstPtr_t, TMUInstPtr_t) directly via sc_fifo_out ports, so no
// base→typed converter threads are needed on the instruction path.
//
// Busy-state converters (typed → BaseBusyPtr_t) are still required because
// MainCtrl.busy_in accepts BaseBusyPtr_t while units emit typed pointers.
//
// DMA.noc_inst_out (sc_fifo_out<NoCInstPtr_t>) is split by a Core thread:
//   src_id >= 0 → Router.send_inst_in (fifo_in)
//   src_id == -1 → Router.recv_inst_in (fifo_in)
//
// DMA.dram_req_out and ArrayUnit.dram_req_out (sc_fifo_out<ReqPtr_t>) are
// bridged to buffer<ReqPtr_t> for DRAMSim.req_port (buffer_in).
//
// All public channels are exposed so test_core.cpp can observe them.

#ifndef LLM_SIM_CORE_H
#define LLM_SIM_CORE_H

#ifndef SC_INCLUDE_DYNAMIC_PROCESSES   // also set by CMake (acc_lib/core_lib)
#define SC_INCLUDE_DYNAMIC_PROCESSES
#endif
#include <memory>
#include <string>
#include <systemc.h>

#include "Core/MainCtrl/MainCtrl.h"
#include "Core/ArrayUnit/ArrayUnit.h"
#include "Core/VectorUnit/VectorUnit.h"
#include "Core/DMA/DMA.h"
#include "Core/TMU/TMU.h"
#include "Core/Cache/Cache.h"
#include "DRAM_sim/DRAMSim.h"
#include "DRAM_sim/DataManager.h"
#include "NoC_sim/Router/Router.h"
#include "Core/MainCtrl/main_ctrl_params.h"
#include "utils/buffer.h"
#include "utils/fifo.h"
#include "utils/Data.h"
#include "utils/busyState.h"
#include "utils/BaseInst.h"
#include "NoC_sim/noc_utils.h"
#include "Acc/gsync.h"

template <typename T>
class Core : public sc_module {
public:
    SC_HAS_PROCESS(Core);

    // data_manager: non-owning pointer to a DataManager<T> constructed and
    // owned by the top-level Acc/Acc8x8 module (Phase 2).
    // Core does NOT delete it -- see ~Core(). Passing nullptr is only valid
    // if the caller never issues any DRAM read/write instructions.
    Core(sc_module_name name, const std::string& inst_file, int router_id = 0,
         bool compute = true, DataManager<T>* data_manager = nullptr);
    ~Core();

    // ── Clock ──────────────────────────────────────────────────────────────
    sc_in_clk clk;

    // ── Control inputs ─────────────────────────────────────────────────────
    // start: when asserted (true), MainCtrl begins loading the instruction file
    //        and dispatching.  Connected via internal sc_buffer to MainCtrl.
    sc_in<bool> start;

    // ══════════════════════════════════════════════════════════════════════
    // Sub-modules (exposed for test access)
    // ══════════════════════════════════════════════════════════════════════
    MainCtrl*       main_ctrl;
    ArrayUnit<T>*   array_unit;
    VectorUnit<T>*  vector_unit;
    ::DMA<T>*       dma_unit;    // prefix :: to avoid clash with InstType_t::DMA
    ::TMU<T>*       tmu_unit;    // prefix :: to avoid clash with InstType_t::TMU
    Cache<T>*       cache;
    DRAMSim<T>*     dram_sim;
    Router<T>*      router;

    // Must be declared before data_manager so it is initialized first
    // (C++ initializes in declaration order, not initializer-list order).
    // Whether computing units should perform real arithmetic (true) or
    // only simulate Data_t transactions with zeroed values (false).
    const bool compute_;

    // Not owned by Core -- injected by the top-level Acc/Acc8x8, which
    // constructs one DataManager<T> for the whole simulation and is
    // responsible for deleting it. ~Core() must NOT delete this pointer.
    DataManager<T>* data_manager;

    // ══════════════════════════════════════════════════════════════════════
    // Control channels: Core sc_in → sc_buffer → MainCtrl sc_in
    // ══════════════════════════════════════════════════════════════════════
    sc_buffer<bool> start_buf;   // carries start from Core's sc_in to MainCtrl
    sc_buffer<bool> mc_done_buf;  // high when MainCtrl dispatched its last inst

    // ══════════════════════════════════════════════════════════════════════
    // Instruction channels: MainCtrl → execution units (direct sc_fifo)
    // ══════════════════════════════════════════════════════════════════════
    sc_fifo<LLM_SIM_INST::ArrayInstPtr_t>   array_inst_fifo;   // MC→ArrayUnit
    sc_fifo<LLM_SIM_INST::VectorInstPtr_t>  vec_inst_fifo;     // MC→VectorUnit
    sc_fifo<LLM_SIM_INST::DMAInstPtr_t>     dma_inst_fifo;     // MC→DMA
    sc_fifo<LLM_SIM_INST::TMUInstPtr_t>     tmu_inst_fifo;     // MC→TMU

    // ══════════════════════════════════════════════════════════════════════
    // Busy state channels: typed (units) → base (MainCtrl)
    // ══════════════════════════════════════════════════════════════════════
    sc_buffer<LLM_SIM_BUSY::ArrayBusyPtr_t>  array_busy_ch;        // ← ArrayUnit
    sc_buffer<LLM_SIM_BUSY::VectorBusyPtr_t> vec_busy_ch;          // ← VectorUnit
    sc_buffer<LLM_SIM_BUSY::DMABusyPtr_t>    dma_busy_ch;          // ← DMA
    sc_buffer<LLM_SIM_BUSY::TMUBusyPtr_t>    tmu_busy_ch;          // ← TMU
    sc_buffer<LLM_SIM_BUSY::BaseBusyPtr_t>   array_busy_base_ch;   // → MainCtrl
    sc_buffer<LLM_SIM_BUSY::BaseBusyPtr_t>   vec_busy_base_ch;     // → MainCtrl
    sc_buffer<LLM_SIM_BUSY::BaseBusyPtr_t>   dma_busy_base_ch;     // → MainCtrl
    sc_buffer<LLM_SIM_BUSY::BaseBusyPtr_t>   tmu_busy_base_ch;     // → MainCtrl

    // ══════════════════════════════════════════════════════════════════════
    // Cache channels (8 ports)
    // Port 0: ArrayUnit  (req→data_req[0], out→in_buf_in, in←out_buf_out)
    // Port 1: VectorUnit input  (req→data_req[1], out→input_buffer_in)
    // Port 2: VectorUnit weight (req→data_req[2], out→weight_buffer_in)
    // Port 3: VectorUnit output (in←output_buffer_out; req/out are stubs)
    // Port 4: DMA  (req→data_req[4], out→cache_data_in, in←cache_data_out)
    // Port 5: TMU  (req→data_req[5], out→input_fifo, in←output_fifo)
    // Port 6: MainCtrl cache read (req←dma.cache_inst_out_mc, out→cache_data_in)
    // Port 7: ArrayUnit weight (req→w_buf_req, out→weight_buf_cache_in) --
    //         used when an ArrayInst_t's weight_src == CACHE instead of DRAM.
    // ══════════════════════════════════════════════════════════════════════
    // Port 0
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_array_req_fifo;     // array→req[0]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_array_out_fifo;     // cache→array.in_buf_in
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_array_in_fifo;      // array.out_buf_out→cache
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_array_wreq_fifo; // array.cache_wreq_out→wreq_in[0]
    // Port 1
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_vec_in_req_fifo;    // vec.in_buf_req→req[1]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_vec_in_out_fifo;    // cache→vec.input_buffer_in
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_stub_in1_fifo;      // stub data_in[1]
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_stub_wreq1_fifo; // stub wreq_in[1]
    // Port 2
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_vec_w_req_fifo;     // vec.w_buf_req→req[2]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_vec_w_out_fifo;     // cache→vec.weight_buffer_in
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_stub_in2_fifo;      // stub data_in[2]
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_stub_wreq2_fifo; // stub wreq_in[2]
    // Port 3
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_stub_req3_fifo;     // stub data_req[3]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_stub_out3_fifo;     // stub data_out[3]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_vec_out_in_fifo;    // vec.output_buffer_out→cache
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_vec_wreq_fifo;  // vec.cache_wreq_out→wreq_in[3]
    // Port 4
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_dma_req_fifo;       // dma.cache_inst_out→req[4]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_dma_out_fifo;       // cache→dma.cache_data_in
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_dma_in_fifo;        // dma.cache_data_out→cache
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_dma_wreq_fifo;  // dma.cache_wreq_out→wreq_in[4]
    // Port 5
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_tmu_req_fifo;       // tmu.cache_req_fifo→req[5]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_tmu_out_fifo;       // cache→tmu.input_fifo
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_tmu_in_fifo;        // tmu.output_fifo→cache
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_tmu_wreq_fifo;  // tmu.cache_wreq_out→wreq_in[5]
    // Port 6: MainCtrl cache read（DMA 發 req，Cache 直送 MainCtrl）
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_mc_req_fifo;        // dma.cache_inst_out_mc→req[6]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_mc_out_fifo;        // cache→main_ctrl.cache_data_in
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_stub_in6_fifo;      // stub data_in[6]
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_stub_wreq6_fifo; // stub wreq_in[6]
    // Port 7
    sc_fifo<LLM_SIM_INST::CacheReqPtr_t>  cache_array_w_req_fifo;   // array.w_buf_req→req[7]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_array_w_out_fifo;   // cache→array.weight_buf_cache_in
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>   cache_stub_in7_fifo;      // stub data_in[7]
    sc_fifo<LLM_SIM_INST::CacheWriteReqPtr_t> cache_stub_wreq7_fifo; // stub wreq_in[7]

    // ══════════════════════════════════════════════════════════════════════
    // DRAM request bridge: sc_fifo (from unit) → buffer (DRAMSim.req_port)
    // ══════════════════════════════════════════════════════════════════════
    sc_fifo<LLM_SIM_DRAM::ReqPtr_t>        dma_dram_req_fifo;    // DMA.dram_req_out
    buffer<LLM_SIM_DRAM::ReqPtr_t>         dma_dram_req_buf;     // → DRAMSim.req_port[0]
    sc_fifo<LLM_SIM_DRAM::ReqPtr_t>        array_dram_req_fifo;  // ArrayUnit.dram_req_out
    buffer<LLM_SIM_DRAM::ReqPtr_t>         array_dram_req_buf;   // → DRAMSim.req_port[1]

    // ══════════════════════════════════════════════════════════════════════
    // DRAM data FIFOs
    // ══════════════════════════════════════════════════════════════════════
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>    dram_to_dma_fifo;     // DRAMSim.read_port[0]→DMA
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>    dma_to_dram_fifo;     // DMA→DRAMSim.write_port[0]
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>    dram_to_array_fifo;   // DRAMSim.read_port[1]→Array
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>    array_to_dram_fifo;   // Array→DRAMSim.write_port[1]

    // Phase F: fill request/response, purely
    // local to this core (unlike remote_dram_req_fifo/remote_dram_resp_fifo
    // above, which cross to a shared DRAMRouter) -- DRAMSim and DMA are both
    // members of this same Core instance, so wiring them here (rather than
    // delegating to Acc/Acc8x8 like dram_sim->remote_req_port/remote_read_port)
    // matches how dram_to_dma_fifo/dma_to_dram_fifo above are already wired.
    sc_fifo<LLM_SIM_DRAM::DRAMAddrPtr_t>   dram_fill_req_fifo;   // DRAMSim.fill_req_out→DMA
    sc_fifo<std::string>                   dram_fill_resp_fifo;  // DMA→DRAMSim.fill_resp_in

    // (Part D): MoE weight miss fallback
    // notify, purely local to this core (DMA and MainCtrl are both members
    // of this same Core instance -- same rationale as dram_fill_req_fifo
    // above). No response fifo -- see MoeMissNotify_t::data_id's doc
    // comment in DMA.h for why the RPC pull's reply needs no separate
    // MainCtrl -> DMA channel.
    sc_fifo<LLM_SIM_DRAM::MoeMissNotify_t>       dma_miss_notify_fifo;  // DMA.dma_miss_notify_out→MainCtrl

    // DRAM busy state (DRAMSim → DMA)
    sc_buffer<LLM_SIM_BUSY::DRAMBusyPtr_t> dram_busy_ch;

    // ══════════════════════════════════════════════════════════════════════
    // NoC channels
    // ══════════════════════════════════════════════════════════════════════
    // DMA.noc_inst_out is split by noc_inst_dispatch thread:
    //   NoCInst with src_id >= 0 → Router.send_inst_in
    //   NoCInst with src_id == -1 → Router.recv_inst_in
    sc_fifo<LLM_SIM_INST::NoCInstPtr_t>    dma_noc_inst_fifo;     // DMA.noc_inst_out
    fifo<LLM_SIM_INST::NoCInstPtr_t>       router_send_inst_fifo; // → Router.send_inst_in
    fifo<LLM_SIM_INST::NoCInstPtr_t>       router_recv_inst_fifo; // → Router.recv_inst_in

    // NoC data
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>    dma_to_router_fifo;   // DMA.noc_send_out→Router
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>    router_to_dma_fifo;   // Router→DMA.noc_recv_in

    // Packet channels (exposed for test monitoring)
    sc_fifo<LLM_SIM_NOC::PacketPtr_t<T>>   packet_out_fifo;      // Router.send_out_packet
    fifo<LLM_SIM_NOC::PacketPtr_t<T>>      packet_in_fifo;       // stub → Router.recv_in_packet

    // Dynamic-request notification: Router → MainCtrl
    // Router writes DynamicReq_t here when a dynamic packet arrives from NoC.
    sc_fifo<LLM_SIM_NOC::DynamicReq_t>     dynamic_req_fifo;     // Router.dynamic_notify_out → MainCtrl

    // NoC busy state (Router → DMA)
    sc_buffer<LLM_SIM_BUSY::NoCBusyPtr_t>  noc_busy_ch;

    // ══════════════════════════════════════════════════════════════════════
    // Global-sync channels
    // ══════════════════════════════════════════════════════════════════════
    // MainCtrl.gsync_arrive_out → gsync_arrive_fifo → Acc's GlobalSyncManager
    // Acc's GlobalSyncManager   → gsync_release_fifo → MainCtrl.gsync_release_in
    sc_fifo<GSyncArriveMsg_t> gsync_arrive_fifo;   // Core → GlobalSyncManager
    sc_fifo<uint64_t>         gsync_release_fifo;  // GlobalSyncManager → Core

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3b: cross-core DRAM remote-read channels.
    // Same pattern as the global-sync channels above: Core exposes plain
    // fifos, and the top-level Acc/Acc8x8 wires them on to a single
    // DRAMRouter module (one per simulation), plus wires this Core's OWN
    // dram_sim->remote_req_port/remote_read_port so the router can forward
    // requests FROM other cores INTO this core's DRAMSim when this core is
    // the owner.
    // ══════════════════════════════════════════════════════════════════════
    // DMA (this core, as requester) → DRAMRouter
    sc_fifo<LLM_SIM_DRAM::ReqPtr_t>        remote_dram_req_fifo;   // DMA → DRAMRouter.req_in[core_id]
    // DRAMRouter → DMA (this core, as requester)
    sc_fifo<LLM_SIM_DATA::DataPtr_t<T>>    remote_dram_resp_fifo;  // DRAMRouter.resp_out[core_id] → DMA

private:    // ── Control signal converter: sc_in → sc_buffer → MainCtrl ────────────
    void start_seq_conv();

    // ── Busy state converters: typed sc_buffer → BaseBusyPtr_t sc_buffer ──
    void array_busy_conv();
    void vec_busy_conv();
    void dma_busy_conv();
    void tmu_busy_conv();

    // ── DRAM request bridge: sc_fifo → buffer (one per DRAM src) ──────────
    void dma_dram_req_conv();
    void array_dram_req_conv();

    // ── NoC instruction dispatcher: DMA.noc_inst_out → send/recv fifo ─────
    void noc_inst_dispatch();
};

#endif // LLM_SIM_CORE_H
