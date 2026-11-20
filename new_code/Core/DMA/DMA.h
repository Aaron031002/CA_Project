#ifndef LLM_SIM_DMA_H
#define LLM_SIM_DMA_H

#include <cstdlib>
#include <deque>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <systemc.h>

#include "Core/DMA/dma_params.h"
#include "utils/BaseInst.h"
#include "utils/buffer.h"
#include "utils/Data.h"
#include "utils/busyState.h"
#include "utils/TransChannel.h"

// Forward declarations only -- DMA.h does not need the full definitions,
// keeping this header lightweight. DMA.cpp includes the real headers.
template <typename T> class DataManager;
namespace LLM_SIM_DRAM {
    class AddressMapper;
    // Mirrors BaseInst.h's existing Req_t/ReqPtr_t forward declaration for
    // the same reason (break the circular include with utils/Req.h) --
    // start_offchip_fetch()'s signature needs to name DRAMAddrPtr_t
    // (Phase B, B-5).
    class DRAMAddr_t;
    using DRAMAddrPtr_t = std::shared_ptr<DRAMAddr_t>;

    // (Part D): sent DMA -> MainCtrl
    // (dma_miss_notify_out/main_ctrl_miss_notify_in) when tick_fill()
    // consults FallbackMapper and finds a CORE destination for a local-HBM
    // miss, instead of the plain off-chip fetch tick_fill() otherwise
    // issues unconditionally. Carries everything MainCtrl::
    // handleMoeMissNotify() needs to construct the RPC pull's target
    // sub-region without querying FallbackMapper a second time -- DMA has
    // already resolved it once, no reason to duplicate that lookup on the
    // MainCtrl side.
    struct MoeMissNotify_t {
        DRAMAddrPtr_t addr;       // path + the ORIGINAL requested sub-region
        int dest_core_id = -1;    // which core's local HBM to pull from
        // Fallback destination's own sub-region within dest_core_id's copy
        // of the file (FallbackMapper::Entry::has_region and friends) --
        // NOT necessarily the same region as addr's row/col (addr describes
        // what the REQUESTING core asked to read; this describes where
        // inside the DESTINATION core's copy that data actually lives).
        // has_region == false means "read dest_core_id's whole file".
        bool has_region = false;
        int64_t row_start = 0, row_end = 0;
        int64_t col_start = 0, col_end = 0;
        // The data_id tick_fill() is using for this fill (same FILL_DATA_ID_
        // BASE-derived allocation scheme as the off-chip path). MainCtrl
        // reuses this SAME data_id as the ret_data_id it registers in
        // rpc_ret_pkt_hash_ for the RPC pull's return leg -- once the
        // reply lands, processDynamicReq's existing (unmodified) ret-pkt-
        // hash branch dispatches a NocRecv that writes the result straight
        // into THIS DMA instance's own data_pool_[data_id], exactly where
        // tick_fill()'s tail loop is already polling. No separate MainCtrl
        // -> DMA response port is needed: the existing NoC recv path IS the
        // hand-back mechanism (mirrors how a normal RPC dispatch's result
        // already lands in the leader's own data_pool_ via the same
        // mechanism, see handleGrant's TP-1 remote branch).
        uint64_t data_id = 0;
    };

    // operator<< required by sc_fifo<MoeMissNotify_t> for internal tracing
    // (mirrors NoC_sim/noc_utils.h's DynamicReq_t). Deliberately does NOT
    // dereference addr (DRAMAddr_t is only forward-declared at this point
    // in the header, same reason start_offchip_fetch()'s own signature
    // above can only NAME DRAMAddrPtr_t, not use it) -- dest_core_id/
    // data_id are enough for a trace line's purposes.
    inline std::ostream& operator<<(std::ostream& os, const MoeMissNotify_t& n) {
        os << "{dest_core_id=" << n.dest_core_id << " data_id=" << n.data_id << "}";
        return os;
    }
    inline void sc_trace(sc_core::sc_trace_file*, const MoeMissNotify_t&,
                          const std::string&) {}
}

template <typename T>
class DMA: public sc_module {
    public:
        SC_HAS_PROCESS(DMA);
        DMA(sc_module_name name);
        sc_in_clk clk;

        // ── DRAM ports ──────────────────────────────────────────────────────
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>    dram_read_in;
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>   dram_write_out;
        sc_fifo_out<LLM_SIM_DRAM::ReqPtr_t>       dram_req_out;
        sc_in<LLM_SIM_BUSY::DRAMBusyPtr_t>        dram_busy_in;

        // ── Phase 3b: cross-core remote-read ports ───────────
        // Used only for DRAM reads whose AddressMapper::owner_of() resolves
        // to a different (but group-adjacent) core. Local reads/writes are
        // completely unaffected and still use dram_req_out/dram_read_in
        // above. See try_dispatch()'s owner-lookup branch.
        sc_fifo_out<LLM_SIM_DRAM::ReqPtr_t>       remote_dram_req_out;  // DMA → Core → DRAMRouter
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>    remote_dram_resp_in;  // DRAMRouter → Core → DMA

        // ── Phase F: fill request/response ──
        // Purely local to this core (DRAMSim and DMA are siblings within the
        // same Core instance -- see Core.cpp). A LOCAL read that misses
        // (DRAMSim::do_read finds its own capacity budget evicted a path it
        // nominally owns) sends the request's DRAMAddr_t here; tick_fill()
        // fetches it from off-chip (reusing start_offchip_fetch()'s existing
        // machinery) and reports the path back once it lands.
        sc_fifo_in<LLM_SIM_DRAM::DRAMAddrPtr_t>   dram_fill_req_in;     // DRAMSim → DMA
        sc_fifo_out<std::string>                  dram_fill_resp_out;   // DMA → DRAMSim

        // ── (Part D/E): MoE weight miss ───
        // fallback via RPC pull. Purely local to this core (MainCtrl and DMA
        // are siblings within the same Core instance, exactly like the fill
        // ports above). When tick_fill() consults FallbackMapper and finds a
        // CORE destination (rather than the plain off-chip default), it
        // sends a MoeMissNotify_t here INSTEAD of calling
        // start_offchip_fetch() -- MainCtrl::handleMoeMissNotify() issues
        // the RPC pull over the NoC and registers the SAME data_id
        // MoeMissNotify_t carries in rpc_ret_pkt_hash_; the pull's reply
        // then lands directly in THIS DMA instance's own data_pool_[data_id]
        // via the ordinary NoC-recv path processDynamicReq's (unmodified)
        // ret-pkt-hash branch already dispatches -- no separate MainCtrl ->
        // DMA response port is needed (see MoeMissNotify_t::data_id's doc
        // comment in the namespace block above for the full mechanism).
        sc_fifo_out<LLM_SIM_DRAM::MoeMissNotify_t>   dma_miss_notify_out;  // DMA → MainCtrl

        // Wire the global DataManager (for AddressMapper::owner_of/is_group)
        // and this core's own id, so try_dispatch() can decide local vs.
        // remote. Mirrors MainCtrl::setDMA/setCache's raw-pointer escape
        // hatch pattern (MainCtrl.h) -- must be called before sc_start().
        void setDataManager(DataManager<T>* dm) { data_manager_ = dm; }
        void setCoreId(int core_id) { core_id_ = core_id; }

        // ── NoC ports ────────────────────────────────────────────────────────
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>    noc_recv_in;
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>   noc_send_out;
        sc_fifo_out<LLM_SIM_INST::NoCInstPtr_t>   noc_inst_out;
        sc_in<LLM_SIM_BUSY::NoCBusyPtr_t>         noc_busy_in;

        // ── Cache ports ──────────────────────────────────────────────────────
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>   cache_data_out;
        sc_fifo_out<LLM_SIM_INST::CacheReqPtr_t>  cache_inst_out;
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>    cache_data_in;
        sc_fifo_out<LLM_SIM_INST::CacheReqPtr_t>  cache_inst_out_mc; // MainCtrl cache read → Cache port 6
        // Write-request metadata for cache_data_out's target Cache port
        // (port 4). try_publish_to_cache() is the ONLY sanctioned writer --
        // see its doc comment for why both FIFOs must be written together.
        sc_fifo_out<LLM_SIM_INST::CacheWriteReqPtr_t> cache_wreq_out;

        // ── Instruction input & busy output ──────────────────────────────────
        sc_fifo_in<LLM_SIM_INST::DMAInstPtr_t>    inst_in;
        sc_out<LLM_SIM_BUSY::DMABusyPtr_t>        dma_busy;

    // ── Data pool synchronous read (used by MainCtrl::handleRPCDispatch) ────────
    // Returns nullptr if data_id is not yet in the pool.
    LLM_SIM_DATA::DataPtr_t<T> lookupData(uint64_t data_id) const {
        auto it = data_pool_.find(data_id);
        return (it != data_pool_.end()) ? it->second : nullptr;
    }

    // Remove a data_id from the pool so the next NocSend re-fetches from cache.
    void eraseData(uint64_t data_id) {
        data_pool_.erase(data_id);
    }

    // Debug: return the packet_id of the recv currently in transfer, or (if
    // none is actively transferring) the oldest one still waiting to arrive.
    // 0 if nothing pending at all.
    uint64_t noc_recv_waiting_pkt() const {
        if (recv_xfer_active_) return recv_xfer_entry_.packet_id;
        if (!pending_recvs_.empty()) return pending_recvs_.front().packet_id;
        return 0;
    }

    // Debug: return a string describing all busy slots.
    std::string debug_slots() const {
        // (2026-08-13, seq_len=1024
        // full-attention chain-forward deadlock investigation): sub-phase
        // now shown explicitly (was collapsed to one label per slot, e.g.
        // DRAM_READ_WAIT/TIMER/PUBLISH all printed "DRAM_R"), so a stuck
        // slot's exact wait condition (WAIT = nothing submitted yet /
        // dram_read_in empty; TIMER = submitted, waiting on start_timer's
        // fin_sig_; PUBLISH = timer done, waiting on cache_data_out being
        // writable) can be told apart without guessing from the surrounding
        // arbiter/queue state alone.
        auto phase_str = [](SubPhase p) -> const char* {
            switch(p) {
                case SubPhase::IDLE: return "IDLE";
                case SubPhase::DRAM_WRITE: return "DRAM_W_WAIT";
                case SubPhase::DRAM_WRITE_PUSH: return "DRAM_W_PUSH";
                case SubPhase::DRAM_READ_WAIT: return "DRAM_R_WAIT";
                case SubPhase::DRAM_READ_TIMER: return "DRAM_R_TIMER";
                case SubPhase::DRAM_READ_PUBLISH: return "DRAM_R_PUBLISH";
                case SubPhase::NOC_SEND: return "NOC_SEND_WAIT";
                case SubPhase::NOC_SEND_PUSH: return "NOC_SEND_PUSH";
                case SubPhase::CACHE_WRITE: return "CACHE_W_WAIT";
                case SubPhase::CACHE_WRITE_PUSH: return "CACHE_W_PUSH";
                case SubPhase::CACHE_READ_WAIT: return "CACHE_R_WAIT";
                case SubPhase::CACHE_READ_TIMER: return "CACHE_R_TIMER";
                case SubPhase::CACHE_COPY_READ_WAIT: return "CACHE_COPY_R_WAIT";
                case SubPhase::CACHE_COPY_READ_TIMER: return "CACHE_COPY_R_TIMER";
                case SubPhase::CACHE_COPY_WRITE_PUSH: return "CACHE_COPY_W_PUSH";
                case SubPhase::REMOTE_DRAM_WAIT: return "REMOTE_DRAM_WAIT";
                case SubPhase::REMOTE_DRAM_TIMER: return "REMOTE_DRAM_TIMER";
                case SubPhase::REMOTE_DRAM_PUBLISH: return "REMOTE_DRAM_PUBLISH";
                default: return "?";
            }
        };
        auto pkt_id = [](const SlotState& s) -> uint64_t {
            if (s.cur_inst && s.cur_inst->noc_inst_ptr)
                return s.cur_inst->noc_inst_ptr->packet_id;
            return 0;
        };
        std::string r;
        if (dram_slot_.phase    != SubPhase::IDLE) r += std::string("DRAM=")     + phase_str(dram_slot_.phase)     + " ";
        // Phase B: noc_recv_slot_/offchip_fetch_ merged into pending_recvs_ +
        // a single shared transfer state (see PendingRecv's header comment).
        if (recv_xfer_active_ || !pending_recvs_.empty()) {
            r += "NOC_RECV=" + std::string(recv_xfer_active_ ? "XFER" : "WAIT") +
                 "(pkt=" + std::to_string(noc_recv_waiting_pkt()) +
                 ", pending=" + std::to_string(pending_recvs_.size()) + ") ";
        }
        if (noc_send_slot_.phase!= SubPhase::IDLE) r += std::string("NOC_SEND=") + phase_str(noc_send_slot_.phase) + "(pkt=" + std::to_string(pkt_id(noc_send_slot_)) + ") ";
        if (cache_slot_.phase   != SubPhase::IDLE) {
            uint64_t cwid = cache_slot_.cur_inst ? cache_slot_.cur_inst->data_id : 0;
            r += std::string("CACHE=") + phase_str(cache_slot_.phase) + "(wid=" + std::to_string(cwid) + ") ";
        }
        if (remote_dram_slot_.phase != SubPhase::IDLE) r += std::string("REMOTE_DRAM=") + phase_str(remote_dram_slot_.phase) + " ";
        if (r.empty()) r = "all_idle";
        return r;
    }

    // ── Utilization tracking (readable after sc_stop) ─────────────────────────
    uint64_t busy_cycle_ = 0;
    double utilization() const {
        const uint64_t total = static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        return total > 0 ? static_cast<double>(busy_cycle_) / static_cast<double>(total) : 0.0;
    }
    uint64_t busy_cycles() const { return busy_cycle_; }

    private:
        // ── Per-slot sub-phase ────────────────────────────────────────────────
        //   IDLE              – slot is free
        //   DRAM_WRITE        – req sent; timer running → push pool[data_id] to dram_write_out
        //   DRAM_READ_WAIT    – req sent; polling dram_read_in
        //   DRAM_READ_TIMER   – data staged; timer running → store in pool
        //   NOC_SEND          – inst sent; timer running → push pool[data_id] to noc_send_out
        //   CACHE_WRITE       – req sent; timer running → push pool[data_id] to cache_data_out
        //   CACHE_READ_WAIT   – req sent; polling cache_data_in
        //   CACHE_READ_TIMER  – data staged; timer running → store in pool
        //
        // NoC recv (formerly its own NOC_RECV_WAIT/TIMER/PUBLISH SubPhase
        // sequence) is handled separately below by pending_recvs_/
        // RecvXferPhase -- see that section's comment for why (Phase B).
        enum class SubPhase {
            IDLE,
            DRAM_WRITE, DRAM_WRITE_PUSH, DRAM_READ_WAIT, DRAM_READ_TIMER, DRAM_READ_PUBLISH,
            NOC_SEND, NOC_SEND_PUSH,
            CACHE_WRITE, CACHE_WRITE_PUSH, CACHE_READ_WAIT, CACHE_READ_TIMER,
            // cache_copy: read src from cache into pool, then write pool to cache as dst_id
            CACHE_COPY_READ_WAIT, CACHE_COPY_READ_TIMER,
            CACHE_COPY_WRITE_PUSH,
            // Phase 3b: remote-read (cross-core, owner != this core). Shape
            // mirrors DRAM_READ_WAIT/DRAM_READ_TIMER/DRAM_READ_PUBLISH exactly,
            // just sourced from remote_dram_resp_in instead of dram_read_in.
            REMOTE_DRAM_WAIT, REMOTE_DRAM_TIMER, REMOTE_DRAM_PUBLISH
        };

        struct SlotState {
            SubPhase                         phase    = SubPhase::IDLE;
            LLM_SIM_INST::DMAInstPtr_t       cur_inst = nullptr;
            LLM_SIM_DATA::DataPtr_t<T>       staged   = nullptr;
        };

        // Who cache_slot_'s CACHE_READ_WAIT/CACHE_READ_TIMER prefetch is
        // currently running on behalf of -- CACHE_READ_TIMER's completion
        // uses this to hand the fetched Data_t directly back to that single
        // requester (staging it into dram_slot_.staged / noc_send_slot_.
        // staged, whichever asked) instead of a shared data_pool_[id] entry.
        // see Round 11: two
        // independent, concurrently-timing slots (dram_slot_ and
        // noc_send_slot_) each issuing their own prefetch for the same
        // data_id and racing to consume a shared data_pool_[id] was the
        // actual root cause of the SWA-layer deadlock (id=1007 in that
        // investigation) -- whichever slot's prefetch/timer finished first
        // would erase the id out from under the other, which would then wait
        // on a data_pool_ entry that no longer existed, forever. A prefetch
        // result must only ever be visible to (and consumable by) the single
        // slot that requested it.
        //
        // DATA_POOL covers the genuine standalone "Cache READ" instruction
        // (is_cache branch of try_dispatch()) and CACHE_COPY_READ_WAIT/TIMER
        // -- both are already a single, self-contained cache_slot_ flow with
        // no second concurrently-timing slot able to race on the same
        // data_id, so publishing to the shared data_pool_ (their existing,
        // correct behavior -- e.g. a later Cache WRITE instruction for the
        // same data_id genuinely expects to find it there) is unaffected.
        enum class PrefetchFor { NONE, DRAM_WRITE, NOC_SEND, DATA_POOL };
        PrefetchFor cache_prefetch_for_ = PrefetchFor::NONE;

        // Resource slots (one per exclusive resource type):
        //   DRAM_SLOT=0        – local DRAM read/write (exclusive: only one at a time)
        //   NOC_RECV_SLOT=1    – NoC receive TRANSFER (see pending_recvs_ below --
        //                        this index now only names the TransChannel
        //                        timer channel for whichever recv is actively
        //                        transferring; admission/arbitration across
        //                        multiple outstanding recvs happens in
        //                        pending_recvs_, not here)
        //   NOC_SEND_SLOT=2    – NoC send  (recv and send may run simultaneously)
        //   CACHE_SLOT=3       – Cache read/write
        //   REMOTE_DRAM_SLOT=4 – cross-core DRAM read (Phase 3b); independent
        //                        of DRAM_SLOT so a local read/write and an
        //                        in-flight remote read can proceed in parallel
        //   OFFCHIP_FETCH_SLOT=5 – reserved, unused (off-chip fetch never
        //                        drove a TransChannel timer of its own even
        //                        before Phase B; kept to avoid renumbering)
        static constexpr int DRAM_SLOT          = 0;
        static constexpr int NOC_RECV_SLOT      = 1;
        static constexpr int NOC_SEND_SLOT      = 2;
        static constexpr int CACHE_SLOT         = 3;
        static constexpr int REMOTE_DRAM_SLOT    = 4;
        static constexpr int OFFCHIP_FETCH_SLOT  = 5;
        static constexpr int NUM_SLOTS           = 6;

        SlotState dram_slot_;
        SlotState noc_send_slot_;
        SlotState cache_slot_;
        SlotState remote_dram_slot_;

        // ── Phase B: NoC recv, unified ──────
        // Merges the old noc_recv_slot_ (instruction-driven NOC_RECV) and
        // offchip_fetch_ (Phase 7b's DMA-synthesized off-chip fetch, Flow A)
        // into one admission table + one shared transfer state, so both can
        // have MULTIPLE requests simultaneously outstanding (waiting for
        // their packet to arrive) instead of forcing strict one-at-a-time
        // serialization through a single slot. Physically: a core still has
        // only one NoC ingress channel, so only one payload actually
        // transfers at a time (recv_xfer_active_ below) -- pending_recvs_ is
        // the separate "what am I expecting" resource, analogous to a NIC's
        // receive-descriptor ring or an NoC router's outstanding-transaction
        // table (see dma_params.h's MAX_PEND_PKT comment).
        struct PendingRecv {
            enum class Kind {
                INSTRUCTION,  // ordinary "DMA NoC recv" from the instruction stream
                SYNTHESIZED   // DMA-synthesized (off-chip fetch today; Phase F fill later)
            };
            Kind        kind = Kind::INSTRUCTION;
            uint64_t    packet_id = 0;
            uint64_t    data_id   = 0;   // landing slot in data_pool_
            LLM_SIM_INST::DMAInstPtr_t inst = nullptr;  // INSTRUCTION only, for cal_lat(); nullptr for SYNTHESIZED
            bool        publish_to_cache = true;        // INSTRUCTION=true; SYNTHESIZED=false (Flow A never publishes -- test_dma_offchip_fetch.cpp depends on this)
            std::string path;            // SYNTHESIZED only, diagnostic (debug_slots()/future Phase F use)
        };
        std::vector<PendingRecv> pending_recvs_;

        // The single active transfer, if any (payload actually moving through
        // the one physical ingress channel this cycle). While active, no new
        // packet is admitted from noc_recv_in -- exactly one at a time,
        // matching the old noc_recv_slot_'s NOC_RECV_WAIT->TIMER->PUBLISH
        // sequence. SYNTHESIZED entries never occupy this (they land the
        // instant their packet arrives, with no added transfer-timer cost --
        // preserves Flow A's pre-Phase-B behavior exactly, see tick_noc_recv()).
        enum class RecvXferPhase { IDLE, TIMER, PUBLISH };
        RecvXferPhase              recv_xfer_phase_ = RecvXferPhase::IDLE;
        bool                       recv_xfer_active_ = false;
        PendingRecv                recv_xfer_entry_;   // valid only while recv_xfer_active_
        LLM_SIM_DATA::DataPtr_t<T> recv_xfer_staged_ = nullptr;

        bool can_admit_recv(typename PendingRecv::Kind k) const;
        void remove_pending_recv(uint64_t packet_id);

        uint64_t next_offchip_fetch_pkt_id_ = 0;  // initialised in constructor from core_id_

        // Phase 7b: true when remote_dram_slot_ observed a group-owner miss
        // (nullptr) but the SYNTHESIZED pending-recv quota was full, so
        // starting the actual fetch had to be deferred to a later cycle.
        // remote_dram_slot_ stays in REMOTE_DRAM_WAIT (not consuming
        // remote_dram_resp_in again) until start_offchip_fetch() succeeds.
        bool remote_dram_miss_pending_ = false;

        // Phase F (extension): turns Flow A's
        // group-wide-miss off-chip fetch into a proper read-through cache --
        // once the fetch this map is tracking actually lands, install
        // residency for THIS core (the requester), so a LATER read of the
        // same path by this core hits locally instead of repeating the whole
        // group-scan/remote-read/off-chip-fetch sequence. Flow A's own
        // mechanics (landing in data_pool_[data_id], not publishing to
        // cache_data_out, etc.) are otherwise completely untouched -- this is
        // purely an ADDITIONAL side effect once the SAME landing that Flow A
        // already relies on occurs, tracked here because tick_remote_dram()
        // itself returns its slot to IDLE immediately after starting the
        // fetch and does not wait for it to land.
        std::unordered_map<uint64_t, std::string> group_fetch_pending_;
        void tick_group_cache_install();

        // ── Phase A: KV eviction write-back
        // dispatch -- a THIRD, independent off-chip flow (call it Flow C),
        // distinct from Flow A (start_offchip_fetch/pending_recvs_ above,
        // existing, untouched) and the Phase F fill-and-serve flow (not yet
        // implemented). Polls
        // this core's own DataManager::pop_pending_writeback(core_id_, ...)
        // every cycle; when non-empty, sends the popped KV bytes off-chip as
        // a real DRAM WRITE (Req_t::is_read=false) whose payload rides in
        // the SAME packet as its NoCInst_t -- OffChipPacketAdapter's
        // depacketize_thread reads a write's data straight out of
        // pkt->payload.data, unlike a read (which gets its response via a
        // separate recv leg). Fire-and-forget: no response is awaited
        // (matches the write-back design decision recorded in
        // -- the far side committing to disk is
        // not this core's concern once the packet is on the wire).
        //
        // Deliberately reuses noc_inst_out/noc_send_out (the same pair
        // try_dispatch()'s ordinary NOC_SEND path and start_offchip_fetch()
        // both write to) rather than adding new ports -- see tick_writeback()
        // for why that is safe: it only ever starts a new (inst, data) pair
        // while noc_send_slot_ is IDLE, and never starts a second pair while
        // one is still in flight, which is sufficient to keep this
        // mechanism's inst/data ever staying in relative lock-step with
        // whatever else is using those two fifos.
        struct WritebackState {
            bool        active    = false;  // an EvictedKv has been popped and is being sent
            bool        inst_sent = false;  // inst half went out; only the payload remains
            int         core_id   = -1;
            std::string path;
            std::vector<T> data;
        };
        WritebackState writeback_;
        uint64_t next_writeback_pkt_id_ = 0;  // lazily bumped to this core's OFFCHIP_WRITEBACK_PKT_ID_BASE range on first use (mirrors next_offchip_fetch_pkt_id_ above)

        void tick_writeback();

        // ── Phase F: fill-and-serve, a
        // FOURTH independent off-chip flow (Flow D): DRAMSim's own local-miss
        // requests, distinct from Flow A (off-chip fetch, serves ONE waiting
        // instruction), Flow B (n/a -- Flow A already covers "fetch for a
        // read"), and Flow C (KV write-back, tick_writeback() above). Reuses
        // start_offchip_fetch()'s existing off-chip round-trip machinery
        // (same pending_recvs_/Kind::SYNTHESIZED landing DRAMSim's do_read/
        // fill_thread never needs to know about) -- the only new thing here
        // is what happens once the data lands: report the path back via
        // dram_fill_resp_out instead of leaving it for an instruction to
        // consume.
        //
        // FillSlot.data_id is a synthetic landing slot in data_pool_ (no real
        // instruction data_id to reuse, unlike Flow A) -- allocated from
        // FILL_DATA_ID_BASE's dedicated per-core range so it can never
        // collide with a real WID. Multiple fills may be in flight
        // concurrently (bounded by fill_slots_'s size, itself bounded by how
        // many SYNTHESIZED entries can fit under pending_recvs_'s
        // MAX_PEND_PKT quota -- start_offchip_fetch() naturally enforces
        // this, so no separate cap is needed here).
        struct FillSlot {
            bool        active = false;  // fetch admitted; in flight or landed-but-not-yet-replied
            bool        landed = false;  // data has arrived in data_pool_[data_id]
            std::string path;
            uint64_t    data_id = 0;
        };
        std::vector<FillSlot> fill_slots_;
        // Requests read from dram_fill_req_in but not yet admitted (waiting
        // for a free pending_recvs_ SYNTHESIZED slot) -- mirrors
        // remote_dram_miss_pending_'s retry-next-cycle idiom, generalized to
        // a queue since multiple local misses can arrive before any one of
        // them is admitted.
        std::deque<LLM_SIM_DRAM::DRAMAddrPtr_t> fill_req_queue_;
        uint64_t next_fill_data_id_ = 0;  // lazily bumped to this core's FILL_DATA_ID_BASE range on first use (mirrors next_writeback_pkt_id_)

        void tick_fill();

        // Injected (not owned) -- see setDataManager()/setCoreId() above.
        DataManager<T>* data_manager_ = nullptr;
        int             core_id_      = -1;

        // (TOKENS_PER_CORE>1 hang investigation,
        // 2026-08-12): DBG_NOC_TRACE=1 (same env var as NoC_sim.cpp's trace,
        // deliberately shared so one env var lights up both layers) dumps
        // pending_recvs_ admission attempts/stalls and tick_noc_recv()
        // attribution, so the Core/DMA side of a hang can be correlated
        // against NoC_sim.cpp's send/recv completion trace by packet_id.
        const bool dbg_trace_ = (std::getenv("DBG_NOC_TRACE") != nullptr);

        // Instruction read from inst_in but not yet dispatched (held when
        // bandwidth is full or the required resource slot is busy)
        LLM_SIM_INST::DMAInstPtr_t pending_inst_ = nullptr;

        // (2026-08-12): tracks how many
        // consecutive try_dispatch() calls have failed to clear the SAME
        // pending_inst_ (identity via raw pointer), so DBG_NOC_TRACE can
        // report "stuck on this instruction for N cycles, type=..." for
        // whichever resource class (DRAM/cache/NoC send) is the blocker --
        // a single choke point covering every try_dispatch() branch instead
        // of instrumenting each one individually.
        LLM_SIM_INST::DMAInst_t*   pending_inst_stuck_on_ = nullptr;
        uint64_t                   pending_inst_stuck_count_ = 0;

        // data pool: maps data_id → DataPtr_t
        std::unordered_map<uint64_t, LLM_SIM_DATA::DataPtr_t<T>> data_pool_;

        // ── TransChannel (NUM_SLOTS independent timer channels) ───────────────
        TransChannel*                  ch_;
        sc_vector<sc_signal<bool>>     SC_NAMED(start_sig_);
        sc_vector<sc_signal<uint64_t>> SC_NAMED(lat_sig_);
        sc_vector<sc_signal<bool>>     SC_NAMED(fin_sig_);
        sc_vector<sc_signal<bool>>     SC_NAMED(busy_sig_);

        void run();
        void start_timer(int slot, uint64_t lat);
        uint64_t cal_lat(const LLM_SIM_INST::DMAInstPtr_t& inst) const;
        int  active_channels() const;
        void try_dispatch();
        void tick_dram();
        void tick_remote_dram();
        void tick_noc_recv();
        void tick_noc_send();
        void tick_cache();
        void sync_busy();

        // Phase 7b: begin an off-chip fetch for addr's file, landing the
        // result in data_pool_[data_id] once it arrives. Called from
        // tick_remote_dram() when a group-owner miss (nullptr response) is
        // observed. No-op (returns false) if the SYNTHESIZED pending-recv
        // quota is full -- caller must hold the miss and retry later
        // (mirrors every other slot's busy-check-and-retry convention).
        //
        // Phase B (B-5): takes the ORIGINAL DRAMAddrPtr_t rather than a
        // separately-passed (path, len) pair -- the old signature discarded
        // row/col offsets and reconstructed a synthetic "row 0, cols
        // [0,len)" access, which read the wrong region for any request that
        // wasn't already row-0 (the common case for K/V cache tensor-mode
        // accesses).
        bool start_offchip_fetch(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr, uint64_t data_id);

        // Phase F (extension): thin wrapper
        // around start_offchip_fetch() that ALSO registers the fetch in
        // group_fetch_pending_ (gated on has_capacity_limit(), same as
        // tick_remote_dram()'s two call sites used to do inline) -- factored
        // out so try_dispatch()'s known-group-wide-miss shortcut (below) can
        // share the exact same "start fetch + track for read-through cache"
        // logic instead of duplicating it.
        bool start_group_miss_fetch(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr, uint64_t data_id);

        // The only sanctioned way to publish to cache_data_out: writes
        // cache_wreq_out THEN cache_data_out, both via nb_write, and only
        // commits if BOTH have free space this cycle (checked up front via
        // num_free()) -- otherwise writes neither and returns false, so the
        // two FIFOs can never go out of sync (a request with no matching
        // data, or vice versa). Every one of DMA.cpp's five existing
        // cache_data_out.nb_write(...) call sites must route through this
        // instead of calling nb_write directly.
        bool try_publish_to_cache(const LLM_SIM_DATA::DataPtr_t<T>& data);
};


#endif
