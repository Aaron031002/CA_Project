#ifndef LLM_SIM_MAINCTRL_H
#define LLM_SIM_MAINCTRL_H

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <systemc.h>

#include "utils/BaseInst.h"
#include "utils/busyState.h"
#include "utils/Data.h"
#include "Core/DMA/DMA.h"
#include "Core/Cache/Cache.h"
#include "Core/MainCtrl/PortBundle.h"
#include "Core/MainCtrl/Decoder/Decoder.h"
#include "Core/MainCtrl/main_ctrl_params.h"
#include "Acc/gsync.h"

using namespace LLM_SIM_INST;
using namespace LLM_SIM_BUSY;

// MainCtrl — instruction dispatcher.
//
// Reads an instruction file at simulation start, then dispatches one
// instruction per clock cycle (strict in-order).  If the target unit's FIFO
// is full the cycle is stalled (the same instruction is retried next cycle).
//
// Units: Array, Vector, DMA, TMU.
// NOC and DRAM are not separate units; they are encoded as sub-types of
// DMA instructions (DMA NoC / DMA DRAM).
//
// TCore instructions are handled inline in run():
//   LOCAL_SYNC  — stalls until all local units report idle (busy==false)
//                 for two consecutive clock cycles.
//   GLOBAL_SYNC — writes a GSyncArriveMsg_t to gsync_arrive_out, then
//                 blocks on gsync_release_in until the matching barrier_id
//                 is returned by GlobalSyncManager.
class MainCtrl : public sc_module {
public:
    SC_HAS_PROCESS(MainCtrl);

    // ── R1 counter breakdown
    // (numerical cross-check extension) ─────────────────────────────────────
    // Process-wide (not per-core) aggregates, summed across every MainCtrl
    // instance in this binary (SystemC is single-threaded/cooperative --
    // no actual concurrent-write hazard from using `static` here, every
    // core's own SC_THREAD runs to a `wait()` before another one resumes).
    // Env-gated by DBG_R1_COUNTERS (see handleGlobalSync's own
    // accumulation site) -- zero cost on a normal run. Readable by the
    // testbench after sc_stop() to print alongside "Total execution
    // cycles" (see test_m1_latency_probe.cpp's own summary block).
    static inline uint64_t r1_barrier_wait_cyc_ = 0;
    static uint64_t r1_barrier_wait_cycles() { return r1_barrier_wait_cyc_; }

    // core_id: identifies this core to the GlobalSyncManager (default 0).
    MainCtrl(sc_module_name name, std::string inst_file_path, int core_id = 0);

    // Clock
    sc_in_clk clk;

    // ── Control inputs ────────────────────────────────────────────────────────
    // start: when asserted (true), MainCtrl begins loading the instruction file
    //        and dispatching.  Tied via sc_buffer inside Core.
    sc_in<bool> start;

    // ── Instruction output FIFOs (one sc_fifo_out per unit) ──────────────────
    sc_vector<sc_fifo_out<ArrayInstPtr_t>>  array_inst_out;
    sc_vector<sc_fifo_out<VectorInstPtr_t>> vector_inst_out;
    sc_vector<sc_fifo_out<DMAInstPtr_t>>    dma_inst_out;
    sc_vector<sc_fifo_out<TMUInstPtr_t>>    tmu_inst_out;

    // ── Busy-state inputs ─────────────────────────────────────────────────────
    // Used by LOCAL_SYNC to detect when all local units are idle.
    sc_vector<sc_in<BaseBusyPtr_t>> array_busy_in;
    sc_vector<sc_in<BaseBusyPtr_t>> vector_busy_in;
    sc_vector<sc_in<BaseBusyPtr_t>> dma_busy_in;
    sc_vector<sc_in<BaseBusyPtr_t>> tmu_busy_in;

    // ── Cache read port (for MainCtrl, via DMA → Cache port 6) ──────────────
    // DMA 發 cache_inst_out_mc req → Cache port 6 → data_out[6] 直連此 port。
    // fetchFromCache() 透過 cache_data_in.read() blocking 等資料。
    sc_fifo_in<LLM_SIM_DATA::DataPtr_t<float>> cache_data_in;

    // ── Dynamic-packet input ──────────────────────────────────────────────────
    // Connected to Router::dynamic_notify_out. CA_2026_FP: the RPC/dynamic-
    // dispatch mechanism that used to preempt local dispatch on arrival has
    // been removed; this port stays wired (Core.cpp) but is never read, and
    // nothing in this worktree ever writes to it (Router no longer detects
    // dynamic packets).
    sc_fifo_in<LLM_SIM_NOC::DynamicReq_t> dynamic_req_in;

    // ── MoE weight miss fallback notification port ───────────────────────────
    // Purely local to this core (DMA and MainCtrl are siblings within the
    // same Core instance -- see Core.cpp). CA_2026_FP: the RPC-pull handler
    // that used to service this has been removed; this port stays wired but
    // is never read, and nothing in this worktree ever writes to it.
    sc_fifo_in<LLM_SIM_DRAM::MoeMissNotify_t>    main_ctrl_miss_notify_in;   // DMA → MainCtrl

    // ── Global-sync channels ──────────────────────────────────────────────────
    // gsync_arrive_out: MainCtrl → GlobalSyncManager (arrive notification).
    // gsync_release_in: GlobalSyncManager → MainCtrl (release notification).
    // Both are wired through Core's gsync_arrive_fifo / gsync_release_fifo.
    sc_fifo_out<GSyncArriveMsg_t> gsync_arrive_out;
    sc_fifo_in<uint64_t>          gsync_release_in;

    // ── Completion output ─────────────────────────────────────────────────────
    // Driven true when the last instruction has been dispatched (run() exits).
    // Wired to Core::mc_done_buf for testbench monitoring.
    sc_out<bool> done;

    // Wire the DMA unit. Must be called before sc_start().
    void setDMA(::DMA<float>* dma) { dma_ = dma; }

private:
    std::string                inst_file_path_;
    int                        core_id_;
    std::vector<BaseInstPtr_t> inst_queue_;
    PortBundle*                port_bundle_;
    Decoder*                   decoder_;

    // SC_THREAD: loads instructions from file, then dispatches one per cycle.
    void run();

    // Stall until all local units are idle for two consecutive clock cycles.
    void handleLocalSync();

    // Send arrive message to GlobalSyncManager, then stall until released.
    void handleGlobalSync(const TCoreInstPtr_t& tc);

    // Returns true only when every local busy-state input reports isBusy()==false.
    bool allUnitsIdle() const;

    // Pointer to the local DMA unit; set via setDMA() before sc_start().
    ::DMA<float>* dma_ = nullptr;
};

#endif