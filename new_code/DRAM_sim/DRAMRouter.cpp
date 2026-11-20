#include "DRAM_sim/DRAMRouter.h"
#include "DRAM_sim/dram_params.h"
#include "utils/precision_utils.h"

#include <algorithm>
#include <sstream>

template <typename T>
DRAMRouter<T>::DRAMRouter(sc_module_name name, DataManager<T>* data_manager, int num_cores)
    : sc_module(name)
    , req_in("req_in", num_cores)
    , resp_out("resp_out", num_cores)
    , owner_req_out("owner_req_out", num_cores)
    , owner_resp_in("owner_resp_in", num_cores)
    , data_manager_(data_manager)
    , num_cores_(num_cores)
    , to_forward_(num_cores)
    , inflight_requesters_(num_cores)
{
    SC_THREAD(process);
    sensitive << clk.pos();
}

template <typename T>
std::pair<int, int> DRAMRouter<T>::edge_key(int a, int b) {
    return { std::min(a, b), std::max(a, b) };
}

template <typename T>
uint64_t DRAMRouter<T>::rail_latency_for(uint64_t len, int elem_bits) const {
    return static_cast<uint64_t>(LLM_SIM_DRAM::DRAM_RAIL_OVERHEAD)
         + LLM_SIM_PRECISION::transfer_cycles(
               len, elem_bits, static_cast<uint64_t>(LLM_SIM_DRAM::DRAM_RAIL_BANDWIDTH));
}

// ---------------------------------------------------------------------------
// process -- single clocked poll thread (no blocking wait() in the body),
// mirroring GlobalSyncManager::process() / DRAMSim::arbiter_thread(). Four
// phases per cycle: accept new requests, forward queued requests to owners,
// collect owner responses (reserving rail bandwidth), deliver responses
// whose rail latency has elapsed.
// ---------------------------------------------------------------------------
template <typename T>
void DRAMRouter<T>::process() {
    while (true) {
        wait(); // advance one clock cycle
        ++cycle_count_;

        // ---- Phase 1: accept new requests from every requester core ----
        for (int i = 0; i < num_cores_; ++i) {
            LLM_SIM_DRAM::ReqPtr_t req;
            if (!req_in[i].nb_read(req)) continue;

            const std::string& path = req->addr_ptr->file_path;
            // Phase C: prefer the requester's
            // own routing decision (Req_t::target_core, set by
            // DMA::try_dispatch after it already resolved WHICH group
            // member actually has path resident via
            // DataManager::find_resident_group_member()) over re-deriving a
            // single static owner via owner_of() here. owner_of() only
            // returns the first entry of a possibly-multi-owner list and
            // knows nothing about runtime residency -- re-deriving it here
            // could target a core that doesn't actually have the data, or
            // one that isn't even group-adjacent to the requester (this was
            // the B1 hazard: dram_map.yaml's 169 multi-owner entries mean
            // owner_of() frequently returns a core the requester isn't
            // adjacent to). target_core == -1 (default) means the caller
            // didn't set it -- fall back to owner_of() for callers that
            // bypass DMA (direct DRAMRouter unit tests).
            const int owner = (req->target_core >= 0)
                ? req->target_core
                : data_manager_->address_mapper().owner_of(path);

            if (owner == LLM_SIM_DRAM::AddressMapper::OWNER_LOCAL || owner == i) {
                std::ostringstream msg;
                msg << "core " << i << " sent DRAMRouter a request that resolves to "
                    << "itself or an unmapped path (file=" << path << ") -- DMA should "
                    << "never forward this";
                SC_REPORT_FATAL("DRAMRouter", msg.str().c_str());
                continue;  // unreachable in production (FATAL aborts); for test overrides
            }
            if (!data_manager_->address_mapper().is_group(i, owner)) {
                std::ostringstream msg;
                msg << "core " << i << " sent DRAMRouter a request for a file owned by "
                    << "non-group core " << owner << " (file=" << path << ")";
                SC_REPORT_FATAL("DRAMRouter", msg.str().c_str());
                continue;  // unreachable in production (FATAL aborts); for test overrides
            }

            to_forward_[owner].emplace_back(i, req);
        }

        // ---- Phase 2: forward one queued request per owner this cycle ----
        // owner_req_out is a single-slot buffer_out (matches DRAMSim's
        // remote_req_port), so at most one nb_write per owner per cycle can
        // succeed; DRAMSim's own arbiter_thread drains it every cycle, so
        // back-to-back forwards to the same owner (across cycles) are fine.
        for (int owner = 0; owner < num_cores_; ++owner) {
            if (to_forward_[owner].empty()) continue;
            auto& front = to_forward_[owner].front();
            if (owner_req_out[owner].nb_write(front.second)) {
                inflight_requesters_[owner].push_back({front.first, front.second->tid});
                to_forward_[owner].pop_front();
                ++forwarded_count;
            }
        }

        // ---- Phase 3: collect owner responses, reserve rail bandwidth ----
        // Phase A: matched by tid, not queue position -- see
        // inflight_requesters_'s header comment. Falls back to a linear
        // scan (this deque is at most a handful of entries -- bounded by
        // how many remote reads one owner core can plausibly have in
        // flight at once) rather than a map, since the common case (no
        // deferral in play) still finds the match at the front.
        for (int owner = 0; owner < num_cores_; ++owner) {
            LLM_SIM_DATA::DataPtr_t<T> resp;
            if (!owner_resp_in[owner].nb_read(resp)) continue;

            auto& infl = inflight_requesters_[owner];
            typename std::deque<std::pair<int, uint64_t>>::iterator it;

            if (resp == nullptr) {
                // Phase 7 "miss" reply (Phase 3b/7b, DRAMSim::do_read):
                // carries no tid of its own -- this is a pre-existing wire
                // value (DMA::tick_remote_dram and test_dma_offchip_fetch
                // both depend on a bare nullptr meaning "miss", a protocol
                // this router must not change -- see
                // Phase A decision 6).
                // Unambiguous only when exactly one request is in flight to
                // this owner. A genuine capacity-driven miss arriving
                // concurrently with an eviction-deferred request (Phase A's
                // deferred_) to the SAME owner from a DIFFERENT requester
                // cannot be attributed correctly without a protocol change
                // -- both require capacity limiting enabled on the owner
                // core (not reachable before Phase D), and resolving the
                // full miss/defer interplay is Phase F's job (fill-and-
                // serve already has to design this). FATAL rather than
                // silently misroute to the wrong requester.
                if (infl.size() != 1) {
                    std::ostringstream msg;
                    msg << "owner core " << owner << " produced an ambiguous miss "
                        << "response (nullptr) while " << infl.size()
                        << " requests are in flight to it -- cannot attribute without "
                        << "a protocol change (see Phase A)";
                    SC_REPORT_FATAL("DRAMRouter", msg.str().c_str());
                    continue;  // unreachable in production (FATAL aborts); for test overrides
                }
                it = infl.begin();
            } else {
                it = std::find_if(infl.begin(), infl.end(),
                    [&](const std::pair<int, uint64_t>& p) { return p.second == resp->id; });
                if (it == infl.end()) {
                    std::ostringstream msg;
                    msg << "owner core " << owner << " produced a remote-read response "
                        << "(tid=" << resp->id << ") with no matching in-flight requester";
                    SC_REPORT_FATAL("DRAMRouter", msg.str().c_str());
                    continue;  // unreachable in production (FATAL aborts); for test overrides
                }
            }
            const int requester = it->first;
            infl.erase(it);

            // Pre-existing latent gap (not introduced by Phase A, just
            // never exercised before): a miss reply is a bare nullptr, so
            // resp->size_ cannot be dereferenced for it. 0 elements is the
            // correct rail-latency input for a response that carries no
            // payload.
            const uint64_t len = (resp == nullptr || resp->size_.empty()) ? 0
                : static_cast<uint64_t>(resp->size_[0]);
            // Phase E: resp already carries the owner DRAMSim's elem_bits
            // tag (do_read stamps it before handing the Data_t off) -- reuse
            // it directly rather than re-deriving from the path here.
            const int elem_bits = (resp == nullptr) ? 32 : resp->elem_bits;
            const auto edge = edge_key(requester, owner);
            const uint64_t start = std::max(cycle_count_, rail_busy_until_[edge]);
            const uint64_t lat   = rail_latency_for(len, elem_bits);
            const uint64_t ready = start + lat;
            rail_busy_until_[edge] = ready;

            pending_delivery_.push_back(PendingDelivery{requester, resp, ready});
        }

        // ---- Phase 4: deliver responses whose rail latency has elapsed ----
        for (auto it = pending_delivery_.begin(); it != pending_delivery_.end(); ) {
            if (cycle_count_ >= it->ready_cycle) {
                if (resp_out[it->requester].nb_write(it->data)) {
                    ++delivered_count;
                    it = pending_delivery_.erase(it);
                    continue;
                }
                // resp_out full: retry next cycle (should be rare -- DMA
                // only has one remote read in flight at a time per core).
            }
            ++it;
        }
    }
}

template class DRAMRouter<float>;
template class DRAMRouter<int>;
