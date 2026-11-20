#ifndef LLM_SIM_DRAM_SIM_H
#define LLM_SIM_DRAM_SIM_H

#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <algorithm>
#include <systemc.h>

#include "DRAM_sim/dram_params.h"
#include "DRAM_sim/DataManager.h"
#include "utils/buffer.h"
#include "utils/fifo.h"
#include "utils/Data.h"
#include "utils/Req.h"
#include "utils/busyState.h"

template <typename T>
class DRAMSim : public sc_module {
public:
    // core_id: Phase 7a -- identifies which core's local DRAM
    // this instance represents, threaded through to DataManager::read_data/
    // write_data so per-core LRU/capacity tracking can be attributed
    // correctly. -1 (default) preserves pre-Phase-7 behavior: no LRU
    // tracking, no eviction, is_resident() always true.
    DRAMSim(sc_module_name name, DataManager<T>* data_manager,
            const int num_src = LLM_SIM_DRAM::NUM_SRC, int core_id = -1);

    // ---- IO ports ----
    sc_in_clk                                              clk;
    sc_vector<buffer_in<LLM_SIM_DRAM::ReqPtr_t>>          req_port;   // one per src
    sc_vector<sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>>    read_port;  // DRAMSim sends read data out
    sc_vector<sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>>     write_port; // DRAMSim receives write data in
    sc_out<LLM_SIM_BUSY::DRAMBusyPtr_t>                    dram_busy_out;

    // ---- Phase 3b: remote-request port ----
    // A single dedicated port pair, separate from req_port/read_port/write_port
    // (which stay purely local: DMA=src0, Array=src1, fixed at elaboration
    // time via NUM_SRC). All cross-core reads forwarded by DRAMRouter arrive
    // here instead -- this keeps local-request semantics (NUM_SRC, arbiter
    // indexing) completely untouched. Read-only: remote access never writes.
    // Requests submitted here flow through the SAME pending_queue_/
    // processor_thread as local requests, so they get the same arrival-order
    // serialization guarantee (no separate race-guard mechanism needed) and
    // the same busy_cycle_/utilization() accounting.
    buffer_in<LLM_SIM_DRAM::ReqPtr_t>                      remote_req_port;
    sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>                remote_read_port;

    // ---- Phase F: fill request/response ----
    // A LOCAL read that misses (this core's own capacity budget evicted the
    // path it nominally owns -- see do_read()) sends the ORIGINAL DRAMAddr_t
    // here so DMA's tick_fill() can size the off-chip fetch to the actual
    // requested slice (not the whole file -- see fill_thread's own comment).
    // The response only needs to carry the path back: DRAMSim already knows
    // core_id_, and recomputes the WHOLE FILE's logical size itself via
    // std::filesystem::file_size (same convention as warm_start()).
    sc_fifo_out<LLM_SIM_DRAM::DRAMAddrPtr_t>               fill_req_out;
    sc_fifo_in<std::string>                                fill_resp_in;

    // ── Utilization tracking (readable after sc_stop) ─────────────────────────
    uint64_t busy_cycle_ = 0;
    double utilization() const {
        const uint64_t total = static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        return total > 0 ? static_cast<double>(busy_cycle_) / static_cast<double>(total) : 0.0;
    }
    uint64_t busy_cycles() const { return busy_cycle_; }

private:
    const int        NUM_SRC_;
    DataManager<T>*  data_manager_;
    const int        core_id_;
    uint64_t         cycle_count_;

    // Sentinel src_id for requests arriving via remote_req_port (Phase 3b),
    // distinguishing them from local sources 0..NUM_SRC_-1 in PendingReq /
    // insert_sorted's tie-break, and in do_read()'s reply routing.
    static constexpr int REMOTE_SRC_ID = -1;

    // req_port[0] is always DMA's local request channel (Core.cpp wires
    // req_port[0]<-DMA, req_port[1]<-ArrayUnit, fixed at elaboration time via
    // NUM_SRC -- see this class's own header comment). Phase F
    // do_read()'s local-miss check always
    // applies to this source unconditionally -- try_dispatch already only
    // routes a read here (local, not remote) once it has confirmed this core
    // is a recorded owner (or the path is unmapped), so a subsequent miss is
    // always genuine.
    //
    // ArrayUnit's own reads (src_id==1) go straight to whichever core's
    // DRAMSim it happens to be wired to, completely bypassing DMA::
    // try_dispatch's AddressMapper-based ownership routing (see
    // ArrayUnit.cpp's own header comment: "reads DRAM directly, never
    // consults AddressMapper") -- so an ArrayUnit read is "local" in the
    // sense of "whichever core's local DRAMSim this instance is", not
    // necessarily in the sense of "this core is dram_map.yaml's owner of
    // this file". do_read() additionally checks owners_of(path) for any
    // non-DMA source before applying the miss check, so ONLY the genuine
    // case -- this core really does own the file (e.g. a TP4 shard core
    // reading its own slice of a multi-owner expert weight file) -- can
    // still miss/fill; a file some OTHER core owns that this core happens to
    // read anyway (e.g. a different core's K-cache during attention) is left
    // exactly as before Phase F: always succeeds, residency is a meaningless
    // question for a file this core was never assigned (confirmed by a real
    // 64-core regression: core 63's ArrayUnit attention computation reads
    // core 56's K-cache file "locally" and must NOT be treated as a miss).
    static constexpr int LOCAL_DMA_SRC_ID = 0;

    // ---- priority-queue entry ----
    struct PendingReq {
        uint64_t                   arrival_cycle;
        int                        src_id;
        LLM_SIM_DRAM::ReqPtr_t     req;
    };

    // Single FIFO ordered by arrival cycle; earlier arrival first, lower src_id breaks ties.
    // No read/write preference — requests are processed in arrival order.
    std::deque<PendingReq> pending_queue_;

    // Phase A: requests parked here while
    // blocked on an in-flight eviction write-back for their path (see
    // do_read's is_evicting() check), instead of being reinserted at the
    // FRONT of pending_queue_ with their original arrival_cycle. The old
    // "reinsert immediately" idiom made a blocked request strict
    // head-of-line: insert_sorted places it back at position 0 (same
    // arrival_cycle as before), so processor_thread picks it again on the
    // very next iteration and spins on it while every other pending
    // request -- local DMA reads (req_port[0]), ArrayUnit weight reads
    // (req_port[1]), and any group-remote reads (remote_req_port) -- waits
    // behind it. Parking here instead lets processor_thread move on to
    // service everything else; arbiter_thread scans this list every cycle
    // and moves entries back into pending_queue_ (with a freshly bumped
    // arrival_cycle, so they don't jump ahead of requests that arrived
    // later) once is_evicting() clears for their (core_id, path).
    std::deque<PendingReq> deferred_;

    sc_event new_req_event_;      // notified when any new request is enqueued
    sc_event busy_update_event_;  // notified by add/remove_pending_req; consumed by busy_writer_thread

    std::vector<LLM_SIM_DRAM::ReqPtr_t> all_pending_reqs_;

    // ---- helpers ----
    static void     insert_sorted(std::deque<PendingReq>& q, const PendingReq& pr);
    uint64_t        get_dram_latency() const;
    void            do_read (const PendingReq& pr);
    void            do_write(const PendingReq& pr);
    void            add_pending_req(LLM_SIM_DRAM::ReqPtr_t req);
    void            remove_pending_req(LLM_SIM_DRAM::ReqPtr_t req);

    // ---- SC threads ----
    void arbiter_thread();    // clocked; polls req_port every cycle and enqueues requests
    void processor_thread();  // event-driven; processes ONE request at a time; arrival order
    void busy_writer_thread();// sole writer to dram_busy_out
    // Phase F: clocked (not folded into processor_thread, which blocks on
    // wait(new_req_event_) whenever pending_queue_ is empty -- a fill-blocked
    // request has already been moved OUT of pending_queue_ into deferred_ by
    // then, so nothing would ever wake processor_thread back up once the
    // fill lands if this lived there instead). Polls fill_resp_in; on arrival:
    // install_resident() -> clear_fill_inflight() -> move every deferred_
    // entry for that path back into pending_queue_ (fresh arrival_cycle,
    // same rationale as arbiter_thread's is_evicting retry) -> notify
    // new_req_event_ so processor_thread picks them up.
    void fill_thread();
};

#endif // LLM_SIM_DRAM_SIM_H
