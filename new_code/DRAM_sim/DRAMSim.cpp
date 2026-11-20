#include "DRAM_sim/DRAMSim.h"
#include "utils/precision_utils.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
template <typename T>
DRAMSim<T>::DRAMSim(sc_module_name name, DataManager<T>* data_manager, const int num_src,
                     int core_id)
    : sc_module(name)
    , NUM_SRC_(num_src)
    , data_manager_(data_manager)
    , core_id_(core_id)
    , cycle_count_(0)
{
    req_port  .init(NUM_SRC_);
    read_port .init(NUM_SRC_);
    write_port.init(NUM_SRC_);

    SC_THREAD(arbiter_thread);
    sensitive << clk.pos();

    SC_THREAD(processor_thread);

    SC_THREAD(busy_writer_thread);

    SC_THREAD(fill_thread);
    sensitive << clk.pos();
}

// ---------------------------------------------------------------------------
// add_pending_req / remove_pending_req
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::add_pending_req(LLM_SIM_DRAM::ReqPtr_t req) {
    all_pending_reqs_.push_back(req);
    busy_update_event_.notify(SC_ZERO_TIME);
}

template <typename T>
void DRAMSim<T>::remove_pending_req(LLM_SIM_DRAM::ReqPtr_t req) {
    auto it = std::find(all_pending_reqs_.begin(), all_pending_reqs_.end(), req);
    if (it != all_pending_reqs_.end())
        all_pending_reqs_.erase(it);
    busy_update_event_.notify(SC_ZERO_TIME);
}

// ---------------------------------------------------------------------------
// busy_writer_thread
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::busy_writer_thread() {
    dram_busy_out.write(std::make_shared<LLM_SIM_BUSY::DRAMBusy_t>());
    while (true) {
        wait(busy_update_event_);
        auto b = std::make_shared<LLM_SIM_BUSY::DRAMBusy_t>();
        b->pending_reqs = all_pending_reqs_;
        b->busy = !all_pending_reqs_.empty();
        dram_busy_out.write(b);
    }
}

// ---------------------------------------------------------------------------
// insert_sorted
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::insert_sorted(std::deque<PendingReq>& q, const PendingReq& pr) {
    auto it = std::find_if(q.begin(), q.end(), [&](const PendingReq& e) {
        if (pr.arrival_cycle != e.arrival_cycle)
            return pr.arrival_cycle < e.arrival_cycle;
        return pr.src_id < e.src_id;
    });
    q.insert(it, pr);
}

// ---------------------------------------------------------------------------
// get_dram_latency
// ---------------------------------------------------------------------------
template <typename T>
uint64_t DRAMSim<T>::get_dram_latency() const {
    return 5;
}

// ---------------------------------------------------------------------------
// arbiter_thread
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::arbiter_thread() {
    while (true) {
        wait(); // wait for rising clock edge
        ++cycle_count_;
        if (!all_pending_reqs_.empty()) ++busy_cycle_;

        if (std::getenv("DEBUG_ARBITER_HEARTBEAT") && cycle_count_ % 200000 == 0) {
            std::cerr << "[arbiter_heartbeat] core=" << core_id_
                      << " cycle=" << cycle_count_
                      << " pending_queue_=" << pending_queue_.size()
                      << " deferred_=" << deferred_.size()
                      << " all_pending_reqs_=" << all_pending_reqs_.size()
                      << "\n";
        }

        for (int i = 0; i < NUM_SRC_; ++i) {
            LLM_SIM_DRAM::ReqPtr_t req;
            if (req_port[i].nb_read(req)) {
                PendingReq pr{cycle_count_, i, req};
                insert_sorted(pending_queue_, pr);
                add_pending_req(req);
                new_req_event_.notify(SC_ZERO_TIME);
            }
        }

        // Phase 3b: remote-originated requests (forwarded by DRAMRouter) join
        // the SAME pending_queue_ as local requests, tagged with REMOTE_SRC_ID
        // so do_read() knows to reply via remote_read_port instead of
        // read_port[src_id]. This gives remote reads the exact same
        // arrival-order serialization and busy_cycle_/utilization accounting
        // as local reads -- no separate mechanism needed.
        {
            LLM_SIM_DRAM::ReqPtr_t req;
            if (remote_req_port.nb_read(req)) {
                PendingReq pr{cycle_count_, REMOTE_SRC_ID, req};
                insert_sorted(pending_queue_, pr);
                add_pending_req(req);
                new_req_event_.notify(SC_ZERO_TIME);
            }
        }

        // Phase A: retry anything parked in deferred_ whose (core_id, path)
        // eviction write-back has since completed (is_evicting() cleared).
        // Bump arrival_cycle to now rather than reusing the original
        // request's arrival_cycle -- it was already blocked once and must
        // not jump ahead of requests that arrived (and are still waiting)
        // in the interim.
        //
        // Phase F: ALSO must not retry an entry still blocked on
        // is_fill_inflight() -- fill_thread is the one that retries those
        // (explicitly, only once its specific path's fill actually lands),
        // not this generic per-cycle scan. Without this second guard, a
        // fill-blocked entry (which was never evicting in the first place,
        // so is_evicting() is trivially false for it) would be retried on
        // the very next cycle regardless of whether the fill has landed,
        // immediately re-hitting the same local-miss check in do_read() and
        // spinning.
        if (!deferred_.empty()) {
            std::deque<PendingReq> still_deferred;
            for (auto& pr : deferred_) {
                const std::string& p = pr.req->addr_ptr->file_path;
                if (!data_manager_->is_evicting(core_id_, p) &&
                    !data_manager_->is_fill_inflight(core_id_, p)) {
                    PendingReq retry_pr{cycle_count_, pr.src_id, pr.req};
                    insert_sorted(pending_queue_, retry_pr);
                    new_req_event_.notify(SC_ZERO_TIME);
                } else {
                    still_deferred.push_back(pr);
                }
            }
            deferred_ = std::move(still_deferred);
        }
    }
}

// ---------------------------------------------------------------------------
// fill_thread -- Phase F: consumes fill_resp_in
// (DMA reporting an off-chip fill has landed for `path`). Cannot live inside
// processor_thread: that thread blocks on wait(new_req_event_) whenever
// pending_queue_ is empty, but a fill-blocked request has already been moved
// OUT of pending_queue_ into deferred_ by the time the fill lands -- nothing
// would ever wake processor_thread back up if this logic lived there.
//
// install_resident() uses the WHOLE FILE's element count (file size on disk
// / sizeof(T)), matching warm_start()'s own convention -- the actual off-chip
// wire transfer DMA performed only moved the originally-requested slice (see
// DMA::tick_fill), but LRU/capacity accounting always tracks whole-file
// residency. This is a deliberate approximation (a partial read installs
// full-file residency credit), documented here.
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::fill_thread() {
    while (true) {
        wait(); // wait for rising clock edge

        std::string path;
        if (!fill_resp_in.nb_read(path)) continue;

        std::error_code ec;
        const auto file_bytes = std::filesystem::file_size(path, ec);
        if (!ec && file_bytes > 0) {
            const uint64_t n_elems = static_cast<uint64_t>(file_bytes) / sizeof(T);
            data_manager_->install_resident(core_id_, path, n_elems);
        } else {
            std::cerr << "[DRAMSim] fill_thread: cannot stat file for install_resident: "
                      << path << std::endl;
        }
        data_manager_->clear_fill_inflight(core_id_, path);

        // Move every deferred_ entry waiting on THIS path back into
        // pending_queue_ (fresh arrival_cycle -- same rationale as
        // arbiter_thread's is_evicting retry above: already blocked once,
        // must not jump ahead of requests that arrived later and are still
        // waiting). Entries deferred for a DIFFERENT reason (is_evicting, or
        // fill-inflight on a DIFFERENT path) are left untouched.
        std::deque<PendingReq> still_deferred;
        bool woke_any = false;
        for (auto& pr : deferred_) {
            if (pr.req->addr_ptr->file_path == path) {
                PendingReq retry_pr{cycle_count_, pr.src_id, pr.req};
                insert_sorted(pending_queue_, retry_pr);
                woke_any = true;
            } else {
                still_deferred.push_back(pr);
            }
        }
        deferred_ = std::move(still_deferred);
        if (woke_any) new_req_event_.notify(SC_ZERO_TIME);
    }
}

// ---------------------------------------------------------------------------
// processor_thread
//   Processes ONE request at a time.  Reads are always preferred over writes.
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::processor_thread() {
    while (true) {
        while (pending_queue_.empty())
            wait(new_req_event_);

        PendingReq pr = pending_queue_.front();
        pending_queue_.pop_front();
        if (pr.req->is_read)
            do_read(pr);
        else
            do_write(pr);
    }
}

// ---------------------------------------------------------------------------
// do_read
//   1. Fetch data from DataManager (row/col -> file positions).
//   2. Wait get_dram_latency() + len/DRAM_BUS_BANDWIDTH clock cycles.
//   3. Send as a single DataPtr_t (size_ = {len}) via non-blocking write.
//
// Phase 7b: a REMOTE read (this core acting as the group
// owner for another core's miss) produces a Phase 7 "miss" response if
// capacity tracking is enabled for this core (core_id_) and the requested
// path has been evicted: respond with nullptr instead of data so the
// requester's DMA can fall back to an off-chip fetch (see
// DMA::tick_remote_dram). Ordinary core-to-core group reads (no capacity
// limit configured, i.e. pre-Phase-7 behavior) are completely unaffected:
// is_resident() always returns true when no limit is set.
//
// Phase F: a LOCAL read (pr.src_id !=
// REMOTE_SRC_ID) can ALSO now genuinely miss -- superseding Phase 7b's
// original "local read always services from disk regardless of LRU/capacity
// state" behavior. If this core's own capacity budget evicted the path it
// nominally owns, the read is parked and an off-chip fill is triggered (see
// the local-miss branch below and fill_thread) rather than silently reading
// straight from disk as if nothing had been evicted.
//
// Phase 7c: if the requested path is currently mid-eviction (write-back in
// flight to its off-chip owner), park this request in deferred_ rather than
// servicing a read against data that's about to move -- not a fatal
// condition. (Phase A: parked in deferred_ instead of reinserted at the
// front of pending_queue_ -- see deferred_'s header comment for why the old
// "reinsert immediately" idiom head-of-line-blocked this DRAMSim instance.)
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::do_read(const PendingReq& pr) {
    const bool is_remote = (pr.src_id == REMOTE_SRC_ID);
    const std::string& path = pr.req->addr_ptr->file_path;

    // Phase A/F: (core_id_, path) is-evicting guard applies to ANY reader of
    // this core's local storage, not just remote ones -- a request that
    // arrives while this path's write-back is in flight must wait regardless
    // of who's asking (broadened from Phase A's original is_remote-only gate
    // since Phase F's local-miss check below would otherwise see a
    // just-evicted path as simply "not resident" and redundantly kick off a
    // fill for data that's still on its way off-chip).
    if (data_manager_->is_evicting(core_id_, path)) {
        deferred_.push_back(pr);
        return;
    }

    if (is_remote && !data_manager_->is_resident(core_id_, path)) {
        // Phase 7 miss: this owner's local DRAM no longer has the file
        // (evicted). Respond with nullptr -- DRAMRouter forwards it
        // unchanged (same path as real data), and the requester's DMA
        // falls back to an off-chip fetch.
        wait(get_dram_latency() * sc_time(1, SC_NS));
        while (!remote_read_port.nb_write(nullptr))
            wait(clk.posedge_event());
        remove_pending_req(pr.req);
        return;
    }

    // Phase F: a LOCAL read can now ALSO
    // genuinely miss -- previously a local read always serviced from disk
    // regardless of LRU/capacity state (see this function's original header
    // comment, now superseded for this case). If capacity tracking is
    // enabled for this core and the path isn't currently resident, this
    // core's own HBM budget evicted it: park the request in deferred_ and
    // trigger an off-chip fill (piggybacking on an already-in-flight one for
    // the same path instead of duplicating it -- see DataManager::
    // is_fill_inflight()'s header comment). fill_thread moves this request
    // back into pending_queue_ once the fill lands.
    //
    // DMA's own local reads (src_id==LOCAL_DMA_SRC_ID) always qualify --
    // try_dispatch already only routes a read here (local, not remote) when
    // this core is a recorded owner (or the path is unmapped), so a
    // subsequent miss here is always a genuine "evicted since routing"
    // situation.
    //
    // Any OTHER local source (today: ArrayUnit, src_id==1) bypasses
    // AddressMapper-based ownership routing entirely by design -- it always
    // reads straight from "whichever core's local DRAMSim it's wired to",
    // regardless of true dram_map.yaml ownership (see LOCAL_DMA_SRC_ID's
    // header comment). That means it can legitimately be reading EITHER (a)
    // a file THIS core genuinely owns (e.g. a TP4 shard core's own slice of
    // a multi-owner expert weight file -- if that got evicted from this
    // core's own capacity, it SHOULD miss and re-fetch, exactly like a DMA
    // read would), or (b) a file some OTHER core owns that this core is
    // reading anyway purely because of ArrayUnit's read-my-own-disk
    // architecture (e.g. a different core's K-cache during attention -- this
    // is NOT a real miss, "resident" is a meaningless question for a file
    // this core was never assigned). owners_of() distinguishes the two: only
    // case (a) -- core_id_ actually appears in the owner list -- is treated
    // as a genuine miss.
    const bool src_can_miss = (pr.src_id == LOCAL_DMA_SRC_ID) || [&] {
        const auto owners = data_manager_->address_mapper().owners_of(path);
        return std::find(owners.begin(), owners.end(), core_id_) != owners.end();
    }();

    if (src_can_miss && data_manager_->has_capacity_limit(core_id_) &&
        !data_manager_->is_resident(core_id_, path)) {
        deferred_.push_back(pr);
        if (!data_manager_->is_fill_inflight(core_id_, path)) {
            data_manager_->mark_fill_inflight(core_id_, path);
            while (!fill_req_out.nb_write(pr.req->addr_ptr))
                wait(clk.posedge_event());
        }
        return;
    }

    std::vector<T> data = data_manager_->read_data(pr.req->addr_ptr, core_id_);

    // (2026-08-12): read_data()
    // returns {} (empty) on any I/O failure (file too short for the
    // requested n_cols, seekg/read past EOF, etc.) -- see its own header
    // comment. Building a Data_t below with size_={pr.req->len} regardless
    // of the ACTUAL data.size() (previously done unconditionally) produces
    // an object whose declared size_ and real value.size() disagree, the
    // same size_/value.size() mismatch class TMUOp::cancat()/split() were
    // hardened against after a real UB incident --
    // except here the mismatched object gets silently handed to every
    // downstream consumer instead of hitting an explicit shape check. Fail
    // loud instead: this only fires on a genuine read failure (a correct
    // read always returns exactly pr.req->len elements, including
    // read_data()'s own out-of-range-row zero-padding), so there is no
    // legitimate short-read case being over-tightened here.
    if (data.size() != pr.req->len) {
        SC_REPORT_ERROR("DRAMSim/do_read",
            ("read_data returned " + std::to_string(data.size()) +
             " element(s), expected " + std::to_string(pr.req->len) +
             " -- file too short or otherwise unreadable: " +
             pr.req->addr_ptr->file_path).c_str());
        remove_pending_req(pr.req);
        return;
    }

    // Phase E: elem_bits_for() is a coarse
    // per-element width (32 fp32, 4 mxfp4) -- appropriate here since this
    // feeds transfer_cycles()'s existing bandwidth-scaling formula (same
    // approximation Q8_0's own elem_bits=8 already accepts), NOT the exact
    // logical_bytes() byte count (that's for LRU/capacity accounting only).
    const int elem_bits = data_manager_->elem_bits_for(pr.req->addr_ptr->file_path);
    const uint64_t latency = get_dram_latency()
                           + LLM_SIM_PRECISION::transfer_cycles(
                                 pr.req->len, elem_bits,
                                 static_cast<uint64_t>(LLM_SIM_DRAM::DRAM_BUS_BANDWIDTH));
    wait(latency * sc_time(1, SC_NS));

    auto data_ptr    = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    data_ptr->id     = pr.req->tid;
    data_ptr->value  = std::move(data);
    data_ptr->size_  = {static_cast<int>(pr.req->len)};
    data_ptr->elem_bits = elem_bits;

    if (is_remote) {
        while (!remote_read_port.nb_write(data_ptr))
            wait(clk.posedge_event());
    } else {
        while (!read_port[pr.src_id].nb_write(data_ptr))
            wait(clk.posedge_event());
    }

    remove_pending_req(pr.req);
}

// ---------------------------------------------------------------------------
// do_write
//   1. Receive data as a single DataPtr_t via non-blocking read.
//   2. Write row-major 1D vector to backing store via DataManager immediately
//      so the file is always committed even if sc_stop() fires during the
//      latency wait (which would otherwise prevent the write from reaching disk).
//   3. Wait get_dram_latency() + len/DRAM_BUS_BANDWIDTH clock cycles (timing).
// ---------------------------------------------------------------------------
template <typename T>
void DRAMSim<T>::do_write(const PendingReq& pr) {
    LLM_SIM_DATA::DataPtr_t<T> data_ptr;
    while (!write_port[pr.src_id].nb_read(data_ptr))
        wait(clk.posedge_event());

    // (Round 18, S512 out_tensor
    // row-truncation investigation): pin diagnostic for the exact
    // (path, data_ptr) pairing do_write() commits -- DBG_PIN_OUT_WRITE gated,
    // only fires for paths containing "out_tensor" to keep overhead near zero.
    if (std::getenv("DBG_PIN_OUT_WRITE") &&
        pr.req->addr_ptr->file_path.find("out_tensor") != std::string::npos) {
        std::cerr << "[PIN-OUT-WRITE] @" << sc_time_stamp()
                  << " core_id=" << core_id_ << " src_id=" << pr.src_id
                  << " path=" << pr.req->addr_ptr->file_path
                  << " row_start=" << pr.req->addr_ptr->row_start
                  << " n_rows=" << pr.req->len
                  << " data_id=" << data_ptr->id
                  << " data_size=" << data_ptr->value.size()
                  << "\n" << std::flush;
    }

    // step 6: destination-file format tag
    // (dram_map.yaml's format=) remains the FIRST authority -- unchanged
    // from Phase E for every already-tagged path (weight/KV cache), so
    // this is a pure additive fallback, not a priority inversion. Only
    // when the path carries NO format tag (elem_bits_for() falls through
    // to the untagged-default 32) do we consult the producer's own
    // Data_t::elem_bits (e.g. a future producer that tags quantized
    // activation output but was never registered in dram_map.yaml) --
    // and only if THAT is itself non-default (!= 32), else stay at 32.
    // This ordering was deliberately chosen (not the reverse) to avoid a
    // real regression risk: if Data_t::elem_bits took priority, any path
    // whose producer forgot to tag it (defaulting to 32) would silently
    // override an already-correct format="f16"/"mxfp4" tag with the wrong
    // value the moment that data next got written.
    const int format_elem_bits = data_manager_->elem_bits_for(pr.req->addr_ptr->file_path);
    const int elem_bits = (format_elem_bits != 32) ? format_elem_bits
                         : (data_ptr->elem_bits != 32) ? data_ptr->elem_bits
                         : 32;

    // (write-time registration): a path
    // dram_map.yaml never mentioned (format_elem_bits==32 AND no owner
    // recorded) whose producer DID tag a non-default Data_t::elem_bits
    // gets that fact persisted into AddressMapper now, not just used for
    // THIS write's own latency/capacity accounting above. Without this,
    // every OTHER access to the same path this run (this core's later
    // reads, or -- DataManager/AddressMapper are the single whole-
    // simulation instance, acc_8x8.h -- any other core's read) would
    // independently re-derive 32 from the still-unregistered path and
    // silently disagree with what THIS write actually used. Gated on
    // "genuinely unregistered", not "elem_bits != 32", so an
    // already-tagged path (weight/KV cache, or a path we ourselves
    // registered on an earlier write) is never touched here -- dram_map.
    // yaml, once it has an opinion, remains authoritative.
    //
    // INVARIANT THIS DOES NOT ENFORCE: the same path must never be
    // write-registered at two DIFFERENT elem_bits across the run's
    // lifetime -- set_owner_and_format() always overwrites unconditionally
    // (last writer wins), so a path legitimately written at varying
    // precisions on different occasions would have its format record
    // silently corrupted for readers after the second write. Every path
    // this mechanism currently applies to (a same-core scratch tensor
    // written once, read back later, e.g. qcache_ring_b_core before its
    // own dram_map_writer.add() tag was added) satisfies "one path -> one
    // elem_bits for the whole run" by construction.
    if (data_ptr->elem_bits != 32 && format_elem_bits == 32) {
        const auto& owners = data_manager_->address_mapper().owners_of(pr.req->addr_ptr->file_path);
        if (owners.empty()) {
            data_manager_->address_mapper().set_owner_and_format(
                pr.req->addr_ptr->file_path, core_id_, data_ptr->elem_bits);
        }
    }

    // Commit to disk before the latency wait so sc_stop() cannot race the write.
    data_manager_->write_data(pr.req->addr_ptr, data_ptr->value, core_id_, elem_bits);

    const uint64_t latency = get_dram_latency()
                           + LLM_SIM_PRECISION::transfer_cycles(
                                 pr.req->len, elem_bits,
                                 static_cast<uint64_t>(LLM_SIM_DRAM::DRAM_BUS_BANDWIDTH));
    wait(latency * sc_time(1, SC_NS));

    remove_pending_req(pr.req);
}

template class DRAMSim<int>;
template class DRAMSim<float>;
