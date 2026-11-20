// DRAMRouter.h ── Cross-core DRAM read forwarding (Phase 3b/3c).
//
// One instance per simulation, at the Acc/Acc8x8 level -- analogous to
// GlobalSyncManager (Acc/gsync.h): per-core fan-in/fan-out port arrays,
// coordinated by a single clocked poll thread (no blocking wait() inside the
// thread body; state is advanced incrementally every cycle, mirroring
// GlobalSyncManager::process() and DRAMSim::arbiter_thread()).
//
// Wiring (done in Acc.cpp / acc_8x8.cpp):
//   cores[i]->dma_unit  (via Core's remote_dram_req_fifo)  → req_in[i]
//   resp_out[i]  → cores[i]->remote_dram_resp_fifo (→ DMA)
//   owner_req_out[i]   → cores[i]->dram_sim->remote_req_port
//   cores[i]->dram_sim->remote_read_port → owner_resp_in[i]
//
// Responsibilities:
//   1. Resolve owner_of(file_path) and validate is_group(requester, owner)
//      (defense in depth -- DMA already checks this before forwarding here;
//      a violation here means a logic error upstream, hence SC_REPORT_FATAL).
//   2. Forward the request into the owner's DRAMSim via owner_req_out[owner]
//      (one core can only have ONE outstanding remote_req_port write at a
//      time, so requests targeting the same owner are queued and issued one
//      at a time -- but since DRAMSim's own arbiter drains its port every
//      cycle, this never serializes beyond what DRAMSim would do anyway for
//      requests from the same source).
//   3. Once the owner's DRAMSim responds (already including the owner's own
//      real DRAM latency -- a remote read occupies the owner's DRAM
//      bandwidth exactly like a local read), hold the response for the
//      per-edge "rail" latency (Phase 3c) before delivering it to the
//      original requester via resp_out[requester]. The rail resource is
//      keyed by the unordered mesh edge (min(a,b), max(a,b)) so unrelated
//      edges never contend with each other.
//
// No additional race-guard mechanism: forwarded requests join the owner's
// DRAMSim pending_queue_ exactly like local requests, so they inherit the
// same arrival-order serialization DRAMSim already guarantees (confirmed by
// inspection of DRAMSim::processor_thread -- see Phase 3
// discussion). This router adds no locking of its own.

#ifndef LLM_SIM_DRAM_ROUTER_H
#define LLM_SIM_DRAM_ROUTER_H

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <systemc.h>

#include "DRAM_sim/AddressMapper.h"
#include "DRAM_sim/DataManager.h"
#include "utils/Req.h"
#include "utils/Data.h"
#include "utils/buffer.h"

template <typename T>
class DRAMRouter : public sc_module {
public:
    SC_HAS_PROCESS(DRAMRouter);

    // num_cores: total number of cores (== all port-array sizes below)
    DRAMRouter(sc_module_name name, DataManager<T>* data_manager, int num_cores);

    sc_in_clk clk;

    // ── Requester-facing channels (core i, as requester) ──────────────────
    sc_vector<sc_fifo_in<LLM_SIM_DRAM::ReqPtr_t>>       req_in;    // core i → router
    sc_vector<sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>>  resp_out;  // router → core i

    // ── Owner-facing channels (core i, as owner of the requested file) ────
    // owner_req_out is buffer_out (not sc_fifo_out) because it binds
    // directly to DRAMSim::remote_req_port, which is a buffer_in<ReqPtr_t>
    // (single-slot channel, matching DRAMSim's existing local req_port
    // type). owner_resp_in binds to DRAMSim::remote_read_port, an ordinary
    // sc_fifo_out<DataPtr_t<T>>, same as the local read_port.
    sc_vector<buffer_out<LLM_SIM_DRAM::ReqPtr_t>>        owner_req_out;  // router → core i's DRAMSim.remote_req_port
    sc_vector<sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>>    owner_resp_in;  // core i's DRAMSim.remote_read_port → router

    // ── Diagnostics ─────────────────────────────────────────────────────────
    uint64_t forwarded_count  = 0;  // requests forwarded to an owner
    uint64_t delivered_count  = 0;  // responses delivered back to a requester

private:
    DataManager<T>* data_manager_;
    const int       num_cores_;
    uint64_t        cycle_count_ = 0;

    // Per owner core: requests accepted from req_in[] but not yet forwarded
    // to owner_req_out[owner] (owner_req_out is a single-slot buffer, so
    // only one forward can be in flight to a given owner at a time).
    std::vector<std::deque<std::pair<int, LLM_SIM_DRAM::ReqPtr_t>>> to_forward_;

    // Per owner core: (requester, tid) pairs whose request has been
    // forwarded and is awaiting that owner's DRAMSim response.
    //
    // Phase A: matched by tid on arrival, NOT
    // by blind front-of-queue pop. Before Phase A, response order from a
    // single owner DRAMSim always matched forward order (its
    // processor_thread served pending_queue_ in strict arrival order with
    // no way to skip ahead), so popping the front was safe. Phase A gave
    // DRAMSim a deferred_ side-queue for requests blocked on an in-flight
    // eviction write-back (see DRAMSim.h) -- a request parked there can now
    // be overtaken by a later, faster request to the SAME owner, so the
    // owner's responses no longer necessarily arrive in forward order.
    // tid (Req_t::tid, stamped onto the response's Data_t::id by
    // DRAMSim::do_read) is exactly the "transaction id for tracking
    // request-response pairs" its own doc comment says it's for.
    std::vector<std::deque<std::pair<int, uint64_t>>> inflight_requesters_;

    // A response already received from the owner's DRAMSim, waiting out the
    // per-edge rail latency before delivery to resp_out[requester].
    struct PendingDelivery {
        int requester;
        LLM_SIM_DATA::DataPtr_t<T> data;
        uint64_t ready_cycle;
    };
    std::vector<PendingDelivery> pending_delivery_;

    // Rail reservation: edge (min(a,b), max(a,b)) → cycle until which that
    // edge's rail is occupied by an in-flight transfer (Phase 3c).
    std::map<std::pair<int, int>, uint64_t> rail_busy_until_;

    uint64_t rail_latency_for(uint64_t len, int elem_bits) const;
    static std::pair<int, int> edge_key(int a, int b);

    void process();  // single clocked SC_THREAD; see .cpp for the 3 phases
};

#endif // LLM_SIM_DRAM_ROUTER_H
