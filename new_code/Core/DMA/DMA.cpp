#include "Core/DMA/DMA.h"
#include "utils/Req.h"  // full Req_t definition needed for is_read access
#include "utils/precision_utils.h"
#include "DRAM_sim/DataManager.h"    // DataManager<T>::address_mapper(), full def needed here
#include "DRAM_sim/AddressMapper.h"  // AddressMapper::OWNER_LOCAL / owner_of / is_group
#include "DRAM_sim/OffchipMapper.h"  // OffchipMapper::OWNER_NONE / owner_of (Phase 7b)
#include "DRAM_sim/FallbackMapper.h" // FallbackMapper::Entry/DestKind (Part D)
#include "NoC_sim/noc_param.h"       // OFFCHIP_FETCH_PKT_ID_BASE/_POOL_SIZE (Phase 7b)
#include <algorithm>                  // std::find_if (Phase B: pending_recvs_ lookup)
#include <filesystem>                 // std::filesystem::file_size (Flow A group-cache install)
#include <iostream>
#include <cstdlib>

template <typename T>
DMA<T>::DMA(sc_module_name name)
    : sc_module(name)
{
    start_sig_.init(NUM_SLOTS);
    lat_sig_.init(NUM_SLOTS);
    fin_sig_.init(NUM_SLOTS);
    busy_sig_.init(NUM_SLOTS);

    ch_ = new TransChannel("ch", NUM_SLOTS);
    ch_->clk(clk);
    for (int i = 0; i < NUM_SLOTS; ++i) {
        ch_->start[i](start_sig_[i]);
        ch_->lat_in[i](lat_sig_[i]);
        ch_->fin[i](fin_sig_[i]);
        ch_->busy[i](busy_sig_[i]);
    }

    SC_THREAD(run);
    sensitive << clk.pos();
}

// ─────────────────────────────────────────────────────────────────────────────
// start_timer  –  pulse start_sig_[slot] high; TransChannel picks it up on
//                 the following posedge and runs the countdown.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::start_timer(int slot, uint64_t lat) {
    start_sig_[slot].write(true);
    lat_sig_[slot].write(lat);
}

// ─────────────────────────────────────────────────────────────────────────────
// cal_lat  –  overhead + max(ceil(len/RBW), ceil(len/WBW)), scaled by
//             inst->elem_bits (§2). elem_bits==32
//             (default) reduces exactly to the original formula.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
uint64_t DMA<T>::cal_lat(const LLM_SIM_INST::DMAInstPtr_t& inst) const {
    uint64_t len = static_cast<uint64_t>(inst->length);
    // Phase E: inst->elem_bits is an ISA-
    // encoded, decode-time-hardcoded field (DMADec.cpp always emits 32) --
    // per Decision 3, don't touch the ISA;
    // instead, for a DRAM instruction, look up the destination file's real
    // format at dispatch time via DataManager (which try_dispatch already
    // holds a pointer to). Non-DRAM instructions (NoC/Cache) are unaffected,
    // keeping inst->elem_bits unchanged for them.
    int elem_bits = inst->elem_bits;
    if (inst->is_dram && data_manager_ != nullptr) {
        elem_bits = data_manager_->elem_bits_for(inst->dram_req_ptr->addr_ptr->file_path);
    }
    uint64_t rl  = LLM_SIM_PRECISION::transfer_cycles(len, elem_bits, LLM_SIM_DMA::RBW);
    uint64_t wl  = LLM_SIM_PRECISION::transfer_cycles(len, elem_bits, LLM_SIM_DMA::WBW);
    return LLM_SIM_DMA::overhead + std::max(rl, wl);
}

// ─────────────────────────────────────────────────────────────────────────────
// active_channels  –  count how many resource slots are currently busy.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
int DMA<T>::active_channels() const {
    int cnt = 0;
    if (dram_slot_.phase        != SubPhase::IDLE) ++cnt;
    // Phase B: only an ACTUAL payload transfer counts as busy, not a recv
    // merely waiting in pending_recvs_ for its packet to arrive -- waiting
    // costs a descriptor slot, not ingress bandwidth (see dma_params.h's
    // MAX_PEND_PKT comment). Counting pending-but-not-yet-transferring recvs
    // here would let a handful of pending recvs alone saturate
    // NUM_MAX_CHANNEL and starve DRAM/other dispatch for no physical reason.
    if (recv_xfer_active_) ++cnt;
    if (noc_send_slot_.phase    != SubPhase::IDLE) ++cnt;
    if (cache_slot_.phase       != SubPhase::IDLE) ++cnt;
    if (remote_dram_slot_.phase != SubPhase::IDLE) ++cnt;
    return cnt;
}

// ─────────────────────────────────────────────────────────────────────────────
// sync_busy  –  publish the current busy state to dma_busy port.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::sync_busy() {
    auto b = std::make_shared<LLM_SIM_BUSY::DMABusy_t>();
    // Phase B: a recv admitted into pending_recvs_ but not yet transferring
    // is still unfinished work from a kernel's point of view -- MainCtrl::
    // handleLocalSync()/allUnitsIdle() reads b->busy (via isBusy()) to decide
    // when a LOCAL_SYNC barrier may release, and must NOT see "idle" while a
    // dispatched "DMA NoC recv" instruction's packet hasn't arrived yet, or
    // the barrier would release before that recv's data actually lands.
    // This is DELIBERATELY broader than active_channels()'s bandwidth-gate
    // definition just below (which only counts an ACTIVE transfer, on
    // purpose -- see that function's comment) -- the two serve different
    // purposes and must not be conflated.
    const bool recv_outstanding = recv_xfer_active_ || !pending_recvs_.empty();
    // (Round 17/18, S512 seq_len=512
    // GLOBAL_SYNC-too-early investigation): active_channels()/recv_outstanding
    // reflect this DMA's OWN slot-phase bookkeeping, but the signal carrying
    // this busy flag out to MainCtrl (Core::dma_busy_conv()'s event-driven
    // relay) can lag by a delta cycle or more. If that lag straddles the
    // brief per-write IDLE gap between two back-to-back DMA<->Cache
    // transfers, handleLocalSync()'s "2 consecutive idle cycles" check
    // could pass while a Cache-side FIFO still holds an entry this DMA
    // hasn't drained yet -- letting GLOBAL_SYNC release before that
    // transfer is truly done. Folding the three DMA<->Cache FIFOs'
    // occupancy into busy directly (via each port's own num_available/
    // num_free -- no FIFO-depth constant needed) keeps busy=true until
    // they're actually drained, at the cost of at most one extra wait
    // cycle if the signal itself is stale -- never an incorrect early
    // release. cache_data_in (this DMA's read side) is the primary risk:
    // Cache has already produced a reply but this DMA's own tick hasn't
    // consumed it yet. cache_inst_out/cache_data_out (this DMA's write
    // side) use num_free()==0 (fully saturated) as a conservative
    // backpressure signal -- the Cache side is falling behind.
    const bool cache_io_pending =
        (cache_data_in.num_available() > 0) ||
        (cache_inst_out.num_free() == 0) ||
        (cache_data_out.num_free() == 0);
    b->busy        = (active_channels() > 0) || recv_outstanding || cache_io_pending;
    b->dram_read   = (dram_slot_.phase == SubPhase::DRAM_READ_WAIT  ||
                      dram_slot_.phase == SubPhase::DRAM_READ_TIMER ||
                      dram_slot_.phase == SubPhase::DRAM_READ_PUBLISH ||
                      remote_dram_slot_.phase == SubPhase::REMOTE_DRAM_WAIT  ||
                      remote_dram_slot_.phase == SubPhase::REMOTE_DRAM_TIMER ||
                      remote_dram_slot_.phase == SubPhase::REMOTE_DRAM_PUBLISH);
    b->dram_write  = (dram_slot_.phase == SubPhase::DRAM_WRITE ||
                      dram_slot_.phase == SubPhase::DRAM_WRITE_PUSH);
    b->noc_recv    = recv_outstanding;
    b->noc_send    = (noc_send_slot_.phase == SubPhase::NOC_SEND ||
                      noc_send_slot_.phase == SubPhase::NOC_SEND_PUSH);
    b->cache_read  = (cache_slot_.phase == SubPhase::CACHE_READ_WAIT ||
                      cache_slot_.phase == SubPhase::CACHE_READ_TIMER ||
                      cache_slot_.phase == SubPhase::CACHE_COPY_READ_WAIT ||
                      cache_slot_.phase == SubPhase::CACHE_COPY_READ_TIMER);
    b->cache_write = (cache_slot_.phase == SubPhase::CACHE_WRITE ||
                      cache_slot_.phase == SubPhase::CACHE_WRITE_PUSH ||
                      cache_slot_.phase == SubPhase::CACHE_COPY_WRITE_PUSH);
    dma_busy.write(b);
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_dram  –  advance the DRAM resource slot state machine.
//
//   DRAM_WRITE:        wait for timer fin → push pool[data_id] to dram_write_out
//   DRAM_READ_WAIT:    poll dram_read_in → stage data → DRAM_READ_TIMER
//   DRAM_READ_TIMER:   wait for timer fin → store staged in pool[data_id]
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_dram() {
    switch (dram_slot_.phase) {

        case SubPhase::DRAM_WRITE:
            if (fin_sig_[DRAM_SLOT].read()) {
                dram_slot_.phase = SubPhase::DRAM_WRITE_PUSH;
            }
            break;

        case SubPhase::DRAM_WRITE_PUSH: {
            // dram_slot_.staged was populated before admission (try_dispatch()
            // only lets a write past DRAM exclusive once its own prefetch
            // handed the data back here -- see cache_prefetch_for_'s doc
            // comment in DMA.h), so it is always non-null by this point.
            if (dram_write_out.nb_write(dram_slot_.staged)) {
                dram_slot_.staged = nullptr;
                dram_slot_.phase  = SubPhase::IDLE;
            }
            break;
        }

        case SubPhase::DRAM_READ_WAIT: {
            LLM_SIM_DATA::DataPtr_t<T> chunk;
            if (dram_read_in.nb_read(chunk)) {
                dram_slot_.staged = chunk;
                start_timer(DRAM_SLOT, cal_lat(dram_slot_.cur_inst));
                dram_slot_.phase  = SubPhase::DRAM_READ_TIMER;
            }
            break;
        }

        case SubPhase::DRAM_READ_TIMER:
            if (fin_sig_[DRAM_SLOT].read()) {
                dram_slot_.staged->id = dram_slot_.cur_inst->data_id;
                data_pool_[dram_slot_.cur_inst->data_id] = dram_slot_.staged;
                dram_slot_.phase = SubPhase::DRAM_READ_PUBLISH;
            }
            break;

        case SubPhase::DRAM_READ_PUBLISH:
            if (try_publish_to_cache(dram_slot_.staged)) {
                // (Round 18, S512/S256/
                // S128/S64 out_tensor row-corruption investigation): a DRAM
                // READ leaves dram_slot_.phase back at IDLE but previously
                // left dram_slot_.staged pointing at the just-published read
                // result -- try_dispatch()'s DRAM WRITE admission (the
                // `!dram_slot_.staged` check) uses that same field as its
                // sole "do I need to prefetch?" signal, so the NEXT write
                // this core issues (regardless of data_id) would see a
                // stale non-null staged, skip its own prefetch entirely, and
                // silently write out this READ's leftover payload instead of
                // its own -- confirmed via a precise (core, cache_slot_)
                // pin trace: a Phase 15 output write's prefetch never fired
                // at all, and the previously-staged data belonged to an
                // unrelated dynamic MoE-kernel activation read earlier in
                // the same core's instruction stream (root cause of the
                // 736/704-sized garbage rows observed in real_out_*.bin at
                // every seq_len, not just seq_len=512 -- S64 masked the
                // damage because that data_id happened to get correctly
                // rewritten on some later retry; S128/S256/S512 did not).
                dram_slot_.staged = nullptr;
                dram_slot_.phase = SubPhase::IDLE;
            }
            break;

        default: break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_remote_dram  –  advance the remote-DRAM (cross-core read) slot state
//                       machine (Phase 3b). Mirrors tick_dram()'s
//                       DRAM_READ_WAIT/DRAM_READ_TIMER/DRAM_READ_PUBLISH
//                       shape exactly, just sourced from remote_dram_resp_in
//                       (fed by DRAMRouter) instead of dram_read_in.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_remote_dram() {
    switch (remote_dram_slot_.phase) {

        case SubPhase::REMOTE_DRAM_WAIT: {
            LLM_SIM_DATA::DataPtr_t<T> chunk;
            if (remote_dram_resp_in.nb_read(chunk)) {
                if (chunk == nullptr) {
                    // Phase 7b: group owner miss (its local DRAM evicted this
                    // path -- see DRAMSim::do_read's is_resident() check).
                    // Fall back to an off-chip fetch instead of publishing a
                    // null result. start_offchip_fetch() registers a new
                    // pending_recvs_ entry (Phase B) for the fetch's own
                    // recv leg; this slot goes back to IDLE immediately so
                    // it can service the next remote read while the fetch
                    // is in flight.
                    const auto& addr = remote_dram_slot_.cur_inst->dram_req_ptr->addr_ptr;
                    const uint64_t data_id = remote_dram_slot_.cur_inst->data_id;
                    // Phase F (extension): start_group_miss_fetch() also
                    // tracks this fetch (if capacity is configured) so
                    // tick_group_cache_install() can install residency for
                    // this core once it lands -- turns this group-wide-miss
                    // fetch into a read-through cache instead of a one-off
                    // serve (Flow A's own landing mechanics are otherwise
                    // untouched).
                    if (start_group_miss_fetch(addr, data_id)) {
                        remote_dram_slot_.phase = SubPhase::IDLE;
                    } else {
                        // SYNTHESIZED pending-recv quota is full -- stay in
                        // REMOTE_DRAM_WAIT and retry start_offchip_fetch()
                        // next cycle without re-reading remote_dram_resp_in
                        // (the miss response is already consumed; cur_inst/
                        // miss state is remembered via
                        // remote_dram_miss_pending_ below). Must NOT be set
                        // on the success branch above: cur_inst is reused
                        // (this same SlotState object) for the next dispatch
                        // once phase goes back to IDLE, and a stale `true`
                        // here would spuriously reinterpret that unrelated
                        // future dispatch's first missed nb_read as "the
                        // deferred fetch retry", firing an incorrect
                        // off-chip fetch for it.
                        remote_dram_miss_pending_ = true;
                    }
                    break;
                }
                remote_dram_slot_.staged = chunk;
                start_timer(REMOTE_DRAM_SLOT, cal_lat(remote_dram_slot_.cur_inst));
                remote_dram_slot_.phase  = SubPhase::REMOTE_DRAM_TIMER;
            } else if (remote_dram_miss_pending_) {
                // Retry starting the deferred off-chip fetch (see above).
                const auto& addr = remote_dram_slot_.cur_inst->dram_req_ptr->addr_ptr;
                const uint64_t data_id = remote_dram_slot_.cur_inst->data_id;
                if (start_group_miss_fetch(addr, data_id)) {
                    remote_dram_miss_pending_ = false;
                    remote_dram_slot_.phase = SubPhase::IDLE;
                }
            }
            break;
        }

        case SubPhase::REMOTE_DRAM_TIMER:
            if (fin_sig_[REMOTE_DRAM_SLOT].read()) {
                remote_dram_slot_.staged->id = remote_dram_slot_.cur_inst->data_id;
                data_pool_[remote_dram_slot_.cur_inst->data_id] = remote_dram_slot_.staged;
                remote_dram_slot_.phase = SubPhase::REMOTE_DRAM_PUBLISH;
            }
            break;

        case SubPhase::REMOTE_DRAM_PUBLISH:
            if (try_publish_to_cache(remote_dram_slot_.staged)) {
                remote_dram_slot_.phase = SubPhase::IDLE;
            }
            break;

        default: break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_group_cache_install  –  Phase F (extension): once a
// Flow A group-wide-miss fetch tracked in
// group_fetch_pending_ actually lands in data_pool_, install residency for
// THIS core (the requester) -- so a later read of the same path by this core
// hits locally (try_dispatch's capacity_aware_resident check) instead of
// repeating the whole group-scan/remote-read/off-chip-fetch sequence every
// time. Deliberately does NOT erase the data_pool_ entry -- Flow A's own
// existing consumer (e.g. MainCtrl's lookupData()) still needs it exactly as
// before; this is purely an additional side effect on the same landing.
//
// Gated on has_capacity_limit(core_id_), matching Flow D's own gating in
// DRAMSim::do_read -- if this core has no capacity budget configured at all,
// installing "residency" would just self-evict immediately (capacity_bytes
// defaults to 0), which is a harmless no-op but would spuriously bump
// eviction_count() for cores/tests that never opted into capacity tracking.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_group_cache_install() {
    // Nothing is ever inserted into group_fetch_pending_ unless
    // has_capacity_limit(core_id_) already held at that time (see both
    // tick_remote_dram() call sites) -- capacity configuration is static for
    // the whole run, so it is safe to skip re-checking it here.
    if (group_fetch_pending_.empty()) return;
    for (auto it = group_fetch_pending_.begin(); it != group_fetch_pending_.end(); ) {
        if (data_pool_.find(it->first) == data_pool_.end()) { ++it; continue; }  // still in flight
        std::error_code ec;
        const auto file_bytes = std::filesystem::file_size(it->second, ec);
        if (!ec && file_bytes > 0) {
            const uint64_t n_elems = static_cast<uint64_t>(file_bytes) / sizeof(T);
            data_manager_->install_resident(core_id_, it->second, n_elems);
        }
        it = group_fetch_pending_.erase(it);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// can_admit_recv / remove_pending_recv  –  Phase B admission helpers for
// pending_recvs_. INSTRUCTION recvs are capped
// at MAX_PEND_PKT - PEND_PKT_RESERVED so ordinary traffic can never starve a
// DMA-synthesized flow (off-chip fetch / Phase F fill) of the slot it needs
// to retrieve the very data those instruction recvs might be waiting on.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
bool DMA<T>::can_admit_recv(typename PendingRecv::Kind k) const {
    const int cap = (k == PendingRecv::Kind::SYNTHESIZED)
                    ? LLM_SIM_DMA::MAX_PEND_PKT
                    : LLM_SIM_DMA::MAX_PEND_PKT - LLM_SIM_DMA::PEND_PKT_RESERVED;
    return static_cast<int>(pending_recvs_.size()) < cap;
}

template <typename T>
void DMA<T>::remove_pending_recv(uint64_t packet_id) {
    for (auto it = pending_recvs_.begin(); it != pending_recvs_.end(); ++it) {
        if (it->packet_id == packet_id) {
            pending_recvs_.erase(it);
            return;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_noc_recv  –  Phase B: the single,
// unified advance function for all NoC recv traffic -- both instruction-
// driven NOC_RECV and DMA-synthesized flows (off-chip fetch today), which
// used to be two independent functions (tick_noc_recv/tick_offchip_fetch)
// each blindly nb_read-ing the SAME noc_recv_in fifo. Whichever ran first in
// a given cycle could steal the other's payload if both had a recv in
// flight simultaneously (D1 in the design discussion) -- reading exactly
// once per cycle here, then attributing the arrival via packet_id against
// pending_recvs_, removes that race entirely.
//
// Two independent pieces of state:
//   1. recv_xfer_active_/recv_xfer_phase_ -- the single active TRANSFER
//      (only one payload physically moves through the one NoC ingress
//      channel at a time). Only ever set for Kind::INSTRUCTION entries --
//      see below for why SYNTHESIZED never uses it.
//   2. pending_recvs_ -- everything currently registered as "expected but
//      not yet arrived", independent of whether a transfer is in progress
//      for a DIFFERENT entry.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_noc_recv() {
    // An INSTRUCTION transfer's timer is running -- advance it exactly like
    // the old NOC_RECV_TIMER/NOC_RECV_PUBLISH phases. Occupies this cycle:
    // no new packet is admitted from noc_recv_in while this is in progress
    // (matches the single-physical-channel constraint).
    if (recv_xfer_active_) {
        switch (recv_xfer_phase_) {
            case RecvXferPhase::TIMER:
                if (fin_sig_[NOC_RECV_SLOT].read()) {
                    recv_xfer_staged_->id = recv_xfer_entry_.data_id;
                    data_pool_[recv_xfer_entry_.data_id] = recv_xfer_staged_;
                    recv_xfer_phase_ = RecvXferPhase::PUBLISH;
                }
                break;
            case RecvXferPhase::PUBLISH:
                if (try_publish_to_cache(recv_xfer_staged_)) {
                    remove_pending_recv(recv_xfer_entry_.packet_id);
                    recv_xfer_active_ = false;
                    recv_xfer_phase_  = RecvXferPhase::IDLE;
                } else if (dbg_trace_) {
                    static thread_local uint64_t last_logged_cyc = ~0ull;
                    const uint64_t now = static_cast<uint64_t>(sc_core::sc_time_stamp().to_double());
                    if (now != last_logged_cyc) {
                        last_logged_cyc = now;
                        std::cerr << "[DMA-TRACE] t=" << now << " c=" << core_id_
                                  << " RECV_PUBLISH_BLOCKED data_id=" << recv_xfer_entry_.data_id
                                  << " pkt=" << recv_xfer_entry_.packet_id
                                  << "\n" << std::flush;
                    }
                }
                break;
            default: break;
        }
        return;
    }

    if (pending_recvs_.empty()) {
        // (2026-08-12): if this fires
        // repeatedly while the NoC side (NoC_sim.cpp's DBG_NOC_TRACE) shows
        // a packet already delivered to this core's out_fifo, the packet is
        // stuck sitting in the NoC->DMA sc_fifo because nothing has been
        // admitted into pending_recvs_ to read it out -- i.e. the hang is
        // upstream of here (try_dispatch never got to issue this core's next
        // DMA NoC recv instruction), not in this function.
        if (dbg_trace_) {
            static thread_local uint64_t last_logged_cyc = ~0ull;
            const uint64_t now = static_cast<uint64_t>(sc_core::sc_time_stamp().to_double());
            if (now != last_logged_cyc) {
                last_logged_cyc = now;
                std::cerr << "[DMA-TRACE] t=" << now << " c=" << core_id_
                          << " TICK_NOC_RECV_IDLE pending_recvs_ empty\n" << std::flush;
            }
        }
        return;
    }

    LLM_SIM_DATA::DataPtr_t<T> chunk;
    if (!noc_recv_in.nb_read(chunk)) return;

    // TEMP DEBUG (non-determinism investigation,
    // 2026-08-10): dump exact sim-time + packet_id + core_id for every
    // packet admitted here, to compare two independent runs of the SAME
    // fixture and see whether packet arrival ORDER/TIMING genuinely differs
    // between runs (the user's hypothesis) or stays identical (pointing
    // elsewhere for the observed cycle-count variance in odd/full-mode
    // layers).
    if (std::getenv("DEBUG_DUMP_NOC_RECV_ORDER")) {
        std::cerr << "[NOC-RECV-ORDER] t=" << sc_core::sc_time_stamp().to_double()
                  << " core=" << core_id_ << " packet_id=" << chunk->packet_id
                  << " pending_count=" << pending_recvs_.size() << "\n" << std::flush;
    }

    // Attribution: with only one entry outstanding there is no ambiguity, so
    // don't require packet_id to match -- this is what keeps
    // test_dma_offchip_fetch.cpp passing unmodified (its synthetic responses
    // never set packet_id, since it stands in for Router/NoCSim by hand).
    // With >=2 outstanding, packet_id is the only reliable attribution (see
    // Data_t::packet_id's header comment) -- an unmatched packet_id here
    // means the arbitration logic itself is wrong, not a data problem, hence
    // FATAL rather than a silent drop.
    typename std::vector<PendingRecv>::iterator match;
    if (pending_recvs_.size() == 1) {
        match = pending_recvs_.begin();
    } else {
        match = std::find_if(pending_recvs_.begin(), pending_recvs_.end(),
            [&](const PendingRecv& pr) { return pr.packet_id == chunk->packet_id; });
        if (match == pending_recvs_.end()) {
            SC_REPORT_FATAL("DMA/NOC_RECV",
                ("core " + std::to_string(core_id_) +
                 " received a packet (packet_id=" + std::to_string(chunk->packet_id) +
                 ") matching no pending recv").c_str());
            return;  // unreachable in production (FATAL aborts); for test overrides
        }
    }

    if (match->kind == PendingRecv::Kind::SYNTHESIZED) {
        // Preserves the original tick_offchip_fetch() behavior exactly:
        // land immediately, no transfer timer, never published to
        // cache_data_out (test_dma_offchip_fetch.cpp depends on both).
        chunk->id = match->data_id;
        data_pool_[match->data_id] = chunk;
        remove_pending_recv(match->packet_id);
        return;
    }

    // INSTRUCTION: stage + start the timer, exactly like the old
    // NOC_RECV_WAIT->NOC_RECV_TIMER transition.
    if (dbg_trace_) {
        std::cerr << "[DMA-TRACE] t=" << sc_core::sc_time_stamp().to_double()
                  << " c=" << core_id_ << " RECV_LANDED pkt=" << chunk->packet_id
                  << " pending_recvs_.size()=" << pending_recvs_.size()
                  << "\n" << std::flush;
    }
    recv_xfer_entry_  = *match;
    recv_xfer_staged_ = chunk;
    start_timer(NOC_RECV_SLOT, cal_lat(match->inst));
    recv_xfer_phase_  = RecvXferPhase::TIMER;
    recv_xfer_active_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_noc_send  –  advance the NoC-send slot state machine.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_noc_send() {
    switch (noc_send_slot_.phase) {

        case SubPhase::NOC_SEND:
            if (fin_sig_[NOC_SEND_SLOT].read()) {
                noc_send_slot_.phase = SubPhase::NOC_SEND_PUSH;
            }
            break;

        case SubPhase::NOC_SEND_PUSH: {
            const auto& ni = noc_send_slot_.cur_inst->noc_inst_ptr;
            // noc_send_slot_.staged was populated before admission
            // (try_dispatch() only lets a real-payload send past NOC_SEND
            // exclusive once its own prefetch handed the data back here --
            // see cache_prefetch_for_'s doc comment in DMA.h), so it is
            // always non-null here for a non-control-packet send.
            // (2026-08-12, seq_len>=256
            // TMUOp shape-mismatch investigation): dump every real (non-control)
            // NoC send's actual payload size at the exact moment it's pushed
            // into the NoC, so a corrupted (size_/value.size() mismatched)
            // Data_t can be attributed to "already broken when picked up here"
            // vs. "broken somewhere between here and the receiver".
            if (std::getenv("DBG_RACE_TRACE")) {
                int size_prod = 1;
                for (int s : noc_send_slot_.staged->size_) size_prod *= s;
                std::cerr << "[NOC-SEND-PUSH] @" << sc_time_stamp()
                          << " c=" << core_id_
                          << " data_id=" << noc_send_slot_.cur_inst->data_id
                          << " packet_id=" << (ni ? ni->packet_id : 0)
                          << " dst_ids=[";
                if (ni) for (int d : ni->dst_ids) std::cerr << d << " ";
                std::cerr << "] value.size()=" << noc_send_slot_.staged->value.size()
                          << " size_=[";
                for (int s : noc_send_slot_.staged->size_) std::cerr << s << ",";
                std::cerr << "] size_prod=" << size_prod
                          << (size_prod != static_cast<int>(noc_send_slot_.staged->value.size()) ? " MISMATCH" : "")
                          << "\n" << std::flush;
            }
            if (noc_send_out.nb_write(noc_send_slot_.staged)) {
                noc_send_slot_.staged = nullptr;
                noc_send_slot_.phase  = SubPhase::IDLE;
            }
            break;
        }

        default: break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// start_offchip_fetch  –  Phase 7b: begin fetching addr's file from its
// off-chip owner node (resolved via DataManager::offchip_mapper()).
// Constructs a send-side NoCInst_t (dst_ids={offchip_node_id},
// offchip_dram_req = a read Req_t) and pushes it through the ordinary
// noc_inst_out/noc_send_out pipe (Router::data2packet copies offchip_dram_req
// straight into the outgoing Packet_t -- see Router.cpp). The matching
// response is retrieved by tick_noc_recv() (Phase B: merged with the
// instruction-driven recv path) via a second, recv-side NoCInst_t using the
// same packet_id, mirroring how RPC dispatch's return leg works.
//
// Returns false (no-op) if the SYNTHESIZED pending-recv quota is already
// full -- caller must retry on a later cycle.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
bool DMA<T>::start_offchip_fetch(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr, uint64_t data_id) {
    if (!can_admit_recv(PendingRecv::Kind::SYNTHESIZED)) return false;

    const std::string& path = addr->file_path;
    const int offchip_node_id = data_manager_->offchip_mapper().owner_of(path);
    if (offchip_node_id == LLM_SIM_DRAM::OffchipMapper::OWNER_NONE) {
        SC_REPORT_FATAL("DMA/OFFCHIP_FETCH",
            ("core " + std::to_string(core_id_) +
             " off-chip fetch miss with no offchip_map.yaml owner for file=" +
             path + " (landing data_id=" + std::to_string(data_id) + ")").c_str());
        return false;  // unreachable in production (FATAL aborts); for test overrides
    }

    // packet_id: circular per-core pool, entirely DMA-internal (never seen
    // by MainCtrl/RPCDispatch's own allocator -- see noc_param.h's
    // OFFCHIP_FETCH_PKT_ID_BASE comment for why this needs its own range).
    const uint64_t base = LLM_SIM_NOC::OFFCHIP_FETCH_PKT_ID_BASE +
        static_cast<uint64_t>(core_id_) * LLM_SIM_NOC::OFFCHIP_FETCH_PKT_ID_POOL_SIZE;
    if (next_offchip_fetch_pkt_id_ < base) next_offchip_fetch_pkt_id_ = base;
    const uint64_t packet_id = next_offchip_fetch_pkt_id_;
    next_offchip_fetch_pkt_id_ = base +
        (next_offchip_fetch_pkt_id_ - base + 1) % LLM_SIM_NOC::OFFCHIP_FETCH_PKT_ID_POOL_SIZE;

    // Phase B (B-5): forward the ORIGINAL addr unchanged instead of
    // reconstructing a synthetic "row 0, cols [0,len)" access -- the old
    // code read the wrong region for any request that wasn't already row-0
    // (the common case for K/V cache tensor-mode accesses, and for any
    // multi-row request). OffChipDRAM::read_data already handles multi-row
    // addresses correctly (same byte-offset convention as DataManager), so
    // there is no need to special-case anything here.
    auto req = std::make_shared<LLM_SIM_DRAM::Req_t>(addr, /*is_read=*/true);
    req->tid = packet_id;

    auto send_inst = std::make_shared<LLM_SIM_INST::NoCInst_t>(
        core_id_, std::vector<int>{offchip_node_id}, packet_id);
    send_inst->offchip_dram_req = req;

    // Recv-side NoCInst_t: submitted ONCE here (not resubmitted every cycle
    // -- mirrors try_dispatch()'s NOC_RECV branch, which also writes its
    // recv inst exactly once). Router's recv_inst_thread delivers via
    // recv_out_ports -> this core's noc_recv_in whenever the packet arrives
    // (Phase B: non-blocking, see Router.cpp) -- no further action needed
    // from DMA until then.
    auto recv_inst = std::make_shared<LLM_SIM_INST::NoCInst_t>(
        packet_id, data_id, std::vector<int>{static_cast<int>(req->len)});

    // Phase B (B-5): check space for BOTH noc_inst_out writes (send half +
    // recv half) and the noc_send_out payload BEFORE writing anything, so a
    // mid-sequence failure can never happen. The old code wrote send_inst,
    // then the payload with its return value discarded (a full noc_send_out
    // silently dropped the payload while the already-sent send_inst
    // permanently desynced Router::send_thread's positional inst/payload
    // pairing -- every subsequent send on this core would then pair with
    // the wrong payload), then finally checked the recv_inst write -- too
    // late to recover from the first two even when it caught a failure.
    if (noc_inst_out.num_free() < 2 || noc_send_out.num_free() < 1) return false;

    noc_inst_out.nb_write(send_inst);
    // Fire-and-forget payload: the actual request content rides in
    // offchip_dram_req (copied by Router::data2packet), not in this
    // Data_t's value -- an empty vector is sufficient and matches how
    // do_write's DRAM request path also splits request-metadata (Req_t)
    // from payload (Data_t) into two separate sends.
    auto empty_payload = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    noc_send_out.nb_write(empty_payload);
    noc_inst_out.nb_write(recv_inst);

    pending_recvs_.push_back(PendingRecv{
        PendingRecv::Kind::SYNTHESIZED, packet_id, data_id,
        /*inst=*/nullptr, /*publish_to_cache=*/false, path});
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// start_group_miss_fetch  –  Phase F (extension):
// thin wrapper factored out of tick_remote_dram()'s two former inline call
// sites so try_dispatch()'s known-group-wide-miss shortcut (see try_dispatch,
// below) can share the exact same "start the fetch, then track it for the
// read-through-cache install" logic.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
bool DMA<T>::start_group_miss_fetch(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr, uint64_t data_id) {
    if (!start_offchip_fetch(addr, data_id)) return false;
    if (data_manager_->has_capacity_limit(core_id_))
        group_fetch_pending_[data_id] = addr->file_path;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_writeback  –  Phase A: Flow C, KV
// eviction write-back dispatch. See WritebackState's declaration in DMA.h
// for the full design rationale (why it's safe to share noc_inst_out/
// noc_send_out with the ordinary NOC_SEND path and start_offchip_fetch).
//
// Two steps, gated on writeback_.active/.inst_sent so a fifo-full retry
// never re-sends the inst half:
//   1. Not active: try to pop a pending write-back for this core. Only
//      START one while noc_send_slot_ is IDLE -- if an ordinary NOC_SEND's
//      inst has already gone out but its data is still pending (timer-
//      gated in tick_noc_send/NOC_SEND_PUSH), pushing our own inst+data
//      into the same two fifos right now could let our data arrive first
//      and get mis-paired with that unrelated, still-unmatched inst. Once
//      IDLE, nothing is left in that half-pushed state, so it's safe to
//      begin a new (inst, data) pair -- and since we never start a SECOND
//      pair while ours is still in flight (single WritebackState, not a
//      queue of them), our own inst and data always stay adjacent in
//      push-order relative to anything dispatched after us.
//   2. Active: send the inst half once (if not already sent), then keep
//      retrying the data half until it lands; clear_evicting() once both
//      have gone out (fire-and-forget -- see class comment).
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_writeback() {
    if (data_manager_ == nullptr) return;  // not every DMA testbench wires a DataManager (see try_dispatch's own nullptr check)

    if (!writeback_.active) {
        if (noc_send_slot_.phase != SubPhase::IDLE) return;  // see rationale above

        typename DataManager<T>::EvictedKv ev;
        if (!data_manager_->pop_pending_writeback(core_id_, ev)) return;

        writeback_.active    = true;
        writeback_.inst_sent = false;
        writeback_.core_id   = ev.core_id;
        writeback_.path      = ev.path;
        writeback_.data      = std::move(ev.data);
    }

    if (!writeback_.inst_sent) {
        const int offchip_node_id = data_manager_->offchip_mapper().owner_of(writeback_.path);
        if (offchip_node_id == LLM_SIM_DRAM::OffchipMapper::OWNER_NONE) {
            SC_REPORT_FATAL("DMA/WRITEBACK",
                ("core " + std::to_string(core_id_) +
                 " KV eviction write-back with no offchip_map.yaml owner for file=" +
                 writeback_.path).c_str());
            return;  // unreachable in production (FATAL aborts); for test overrides
        }

        const uint64_t base = LLM_SIM_NOC::OFFCHIP_WRITEBACK_PKT_ID_BASE +
            static_cast<uint64_t>(core_id_) * LLM_SIM_NOC::OFFCHIP_WRITEBACK_PKT_ID_POOL_SIZE;
        if (next_writeback_pkt_id_ < base) next_writeback_pkt_id_ = base;
        const uint64_t packet_id = next_writeback_pkt_id_;
        next_writeback_pkt_id_ = base +
            (next_writeback_pkt_id_ - base + 1) % LLM_SIM_NOC::OFFCHIP_WRITEBACK_PKT_ID_POOL_SIZE;

        auto addr = std::make_shared<LLM_SIM_DRAM::DRAMAddr_t>(
            writeback_.path, /*row_start=*/0, /*row_end=*/1, /*col_start=*/0,
            /*col_end=*/static_cast<int64_t>(writeback_.data.size()));
        auto req = std::make_shared<LLM_SIM_DRAM::Req_t>(addr, /*is_read=*/false);
        req->tid = packet_id;

        auto send_inst = std::make_shared<LLM_SIM_INST::NoCInst_t>(
            core_id_, std::vector<int>{offchip_node_id}, packet_id);
        send_inst->offchip_dram_req = req;

        if (!noc_inst_out.nb_write(send_inst)) return;  // retry next cycle, nothing committed yet
        writeback_.inst_sent = true;
    }

    // Payload rides in the SAME packet as the inst above (Router::data2packet
    // copies whatever Data_t is read from send_in_ports into payload.data
    // unconditionally -- OffChipPacketAdapter::depacketize_thread relies on
    // exactly this for a write's data, unlike a read's fire-and-forget empty
    // payload in start_offchip_fetch).
    auto payload = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    payload->value = writeback_.data;
    payload->size_ = {static_cast<int>(writeback_.data.size())};
    if (!noc_send_out.nb_write(payload)) return;  // retry next cycle; inst already sent, do not resend it

    data_manager_->clear_evicting(writeback_.core_id, writeback_.path);
    writeback_ = WritebackState{};
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_fill  –  Phase F: advance Flow D
// (DRAMSim local-miss fills). Reuses start_offchip_fetch()'s existing
// off-chip round-trip (same pending_recvs_/Kind::SYNTHESIZED admission and
// landing DRAMSim never needs to know about) with a synthetic data_id from
// FILL_DATA_ID_BASE's dedicated range -- once that lands in data_pool_,
// report the path back via dram_fill_resp_out (DRAMSim's fill_thread does
// the actual install_resident() + deferred_ retry; DMA's job here is only
// the off-chip fetch itself).
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_fill() {
    // Drain at most one new fill request per cycle from DRAMSim.
    LLM_SIM_DRAM::DRAMAddrPtr_t new_addr;
    if (dram_fill_req_in.nb_read(new_addr)) {
        fill_req_queue_.push_back(new_addr);
    }

    // Try to admit the oldest still-queued request (retry next cycle if the
    // SYNTHESIZED pending-recv quota is full -- mirrors
    // remote_dram_miss_pending_'s retry idiom).
    if (!fill_req_queue_.empty()) {
        const auto& addr = fill_req_queue_.front();

        const uint64_t base = LLM_SIM_NOC::FILL_DATA_ID_BASE +
            static_cast<uint64_t>(core_id_) * LLM_SIM_NOC::FILL_DATA_ID_POOL_SIZE;
        if (next_fill_data_id_ < base) next_fill_data_id_ = base;
        const uint64_t data_id = next_fill_data_id_;

        // (Part D): consult FallbackMapper
        // BEFORE defaulting to the plain off-chip fetch below -- a CORE
        // destination is sent to MainCtrl as an RPC-pull notification
        // instead of start_offchip_fetch(); an OFFCHIP destination (or no
        // entry at all, i.e. FallbackMapper has "no opinion") falls through
        // to the existing, unmodified off-chip path.
        LLM_SIM_DRAM::FallbackMapper::Entry fb_entry;
        const bool use_core_fallback = data_manager_ != nullptr &&
            data_manager_->fallback_mapper().lookup(addr->file_path, core_id_, &fb_entry) &&
            fb_entry.dest_kind == LLM_SIM_DRAM::FallbackMapper::DestKind::CORE;

        bool admitted = false;
        if (use_core_fallback) {
            LLM_SIM_DRAM::MoeMissNotify_t notify;
            notify.addr = addr;
            notify.dest_core_id = fb_entry.dest_core_id;
            notify.has_region = fb_entry.has_region;
            notify.row_start = fb_entry.row_start;
            notify.row_end   = fb_entry.row_end;
            notify.col_start = fb_entry.col_start;
            notify.col_end   = fb_entry.col_end;
            notify.data_id = data_id;
            admitted = dma_miss_notify_out.nb_write(notify);
            // No pending_recvs_/SYNTHESIZED entry to register here (unlike
            // start_offchip_fetch()) -- MainCtrl::handleMoeMissNotify()
            // registers THIS SAME data_id in rpc_ret_pkt_hash_, so the RPC
            // pull's reply lands in data_pool_[data_id] via the ordinary
            // NoC-recv path processDynamicReq's ret-pkt-hash branch already
            // dispatches, exactly like any other RPC result landing in a
            // leader's own data_pool_ -- no separate response port needed.
        } else {
            admitted = start_offchip_fetch(addr, data_id);
        }

        if (admitted) {
            next_fill_data_id_ = base +
                (next_fill_data_id_ - base + 1) % LLM_SIM_NOC::FILL_DATA_ID_POOL_SIZE;

            FillSlot slot;
            slot.active  = true;
            slot.path    = addr->file_path;
            slot.data_id = data_id;
            bool placed = false;
            for (auto& s : fill_slots_) {
                if (!s.active) { s = slot; placed = true; break; }
            }
            if (!placed) fill_slots_.push_back(slot);
            fill_req_queue_.pop_front();
        }
    }

    // Check every active slot for landed data, then try to report it back.
    for (auto& slot : fill_slots_) {
        if (!slot.active) continue;
        if (!slot.landed) {
            auto it = data_pool_.find(slot.data_id);
            if (it == data_pool_.end()) continue;  // still in flight
            data_pool_.erase(it);  // no instruction will ever consume this synthetic id
            slot.landed = true;
        }
        if (dram_fill_resp_out.nb_write(slot.path)) {
            slot.active = false;
            slot.landed = false;
        }
        // else: retry the reply next cycle (data already erased from
        // data_pool_ -- safe, slot.path/landed already record everything
        // needed to retry without re-touching data_pool_).
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// try_publish_to_cache  –  the only sanctioned way to publish to
// cache_data_out (Cache port 4). Writes cache_wreq_out (the metadata Cache
// needs to compute latency/reserve capacity BEFORE reading the payload)
// strictly before cache_data_out, same cycle, and only if both FIFOs
// currently have free space -- checked up front via num_free() so this
// never writes one and not the other, which would desync the two FIFOs'
// one-request-per-datum correspondence. All five of this file's cache
// publish sites (DRAM_READ_PUBLISH, REMOTE_DRAM_PUBLISH, NoC recv PUBLISH,
// CACHE_WRITE_PUSH, CACHE_COPY_WRITE_PUSH) route through this.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
bool DMA<T>::try_publish_to_cache(const LLM_SIM_DATA::DataPtr_t<T>& data) {
    if (cache_wreq_out.num_free() == 0 || cache_data_out.num_free() == 0) return false;
    bool req_ok = cache_wreq_out.nb_write(LLM_SIM_DATA::make_cache_write_req(data));
    bool data_ok = cache_data_out.nb_write(data);
    // Both num_free() checks passed above, so both nb_write calls below are
    // guaranteed to succeed -- no partial-write state is reachable here.
    return req_ok && data_ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// tick_cache  –  advance the Cache resource slot state machine.
//
//   CACHE_WRITE:       wait for timer fin → push pool[data_id] to cache_data_out
//   CACHE_READ_WAIT:   poll cache_data_in → stage data → CACHE_READ_TIMER
//   CACHE_READ_TIMER:  wait for timer fin → store staged in pool[data_id]
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::tick_cache() {
    switch (cache_slot_.phase) {

        case SubPhase::CACHE_WRITE:
            if (fin_sig_[CACHE_SLOT].read()) {
                cache_slot_.phase = SubPhase::CACHE_WRITE_PUSH;
            }
            break;

        case SubPhase::CACHE_WRITE_PUSH: {
            auto it = data_pool_.find(cache_slot_.cur_inst->data_id);
            if (it == data_pool_.end()) {
                SC_REPORT_WARNING("DMA/CACHE_WRITE", "data_id not in pool");
                cache_slot_.phase = SubPhase::IDLE;
                break;
            }
            if (try_publish_to_cache(it->second)) {
                cache_slot_.phase = SubPhase::IDLE;
            }
            break;
        }

        case SubPhase::CACHE_READ_WAIT: {
            LLM_SIM_DATA::DataPtr_t<T> chunk;
            if (cache_data_in.nb_read(chunk)) {
                cache_slot_.staged = chunk;
                start_timer(CACHE_SLOT, cal_lat(cache_slot_.cur_inst));
                cache_slot_.phase  = SubPhase::CACHE_READ_TIMER;
            }
            break;
        }

        case SubPhase::CACHE_READ_TIMER:
            if (fin_sig_[CACHE_SLOT].read()) {
                if (std::getenv("DBG_PIN_PREFETCH") && core_id_ ==
                    std::atoi(std::getenv("DBG_PIN_PREFETCH"))) {
                    std::cerr << "[PIN-PREFETCH-DONE] @" << sc_time_stamp()
                              << " c=" << core_id_
                              << " requested_data_id=" << cache_slot_.cur_inst->data_id
                              << " received_chunk_id=" << cache_slot_.staged->id
                              << " received_chunk_size=" << cache_slot_.staged->value.size()
                              << " prefetch_for=" << static_cast<int>(cache_prefetch_for_)
                              << "\n" << std::flush;
                }
                // (data-operation-inst-array-vector-tmu plan) Deliberately NOT
                // re-stamping cache_slot_.staged->id = cache_slot_.cur_inst->
                // data_id here anymore: under rename-on-terminal-read, if THIS
                // very prefetch read is the one that triggers a rename, Cache
                // has already set staged->id to the new_data_id before this
                // code runs (SystemC runs Cache's thread to completion before
                // yielding -- guaranteed visible, not a race). Stamping the
                // old requested id back on would corrupt that field for any
                // future reader of cache_data[new_data_id] that trusts .id
                // (mirrors CACHE_COPY_READ_TIMER's existing clone-before-
                // mutate comment below, same aliasing hazard). Nothing
                // downstream in this file re-reads staged->id -- data_pool_/
                // noc_send_slot_/dram_slot_ consumers all key by the
                // externally-tracked requested id, not by staged->id.
                // Hand the fetched Data_t directly to whichever single slot
                // requested this prefetch -- never through a shared
                // data_pool_[id] entry another concurrently-timing slot
                // could also be racing to fetch/consume/erase (see
                // cache_prefetch_for_'s doc comment in DMA.h).
                switch (cache_prefetch_for_) {
                    case PrefetchFor::DRAM_WRITE:
                        dram_slot_.staged = cache_slot_.staged;
                        break;
                    case PrefetchFor::NOC_SEND:
                        noc_send_slot_.staged = cache_slot_.staged;
                        break;
                    case PrefetchFor::DATA_POOL:
                        // Genuine standalone "Cache READ" instruction -- see
                        // PrefetchFor's doc comment in DMA.h for why this one
                        // case still legitimately publishes to data_pool_.
                        data_pool_[cache_slot_.cur_inst->data_id] = cache_slot_.staged;
                        break;
                    case PrefetchFor::NONE:
                        break;  // unreachable: only set alongside CACHE_READ_WAIT admission
                }
                cache_prefetch_for_ = PrefetchFor::NONE;
                cache_slot_.phase = SubPhase::IDLE;
            }
            break;

        // ── cache_copy phases ────────────────────────────────────────────────
        // Phase 1: read src (data_id) from cache into pool
        case SubPhase::CACHE_COPY_READ_WAIT: {
            LLM_SIM_DATA::DataPtr_t<T> chunk;
            if (cache_data_in.nb_read(chunk)) {
                cache_slot_.staged = chunk;
                start_timer(CACHE_SLOT, cal_lat(cache_slot_.cur_inst));
                cache_slot_.phase  = SubPhase::CACHE_COPY_READ_TIMER;
            }
            break;
        }

        case SubPhase::CACHE_COPY_READ_TIMER:
            if (fin_sig_[CACHE_SLOT].read()) {
                const uint64_t dst = cache_slot_.cur_inst->dst_data_id;
                // Copy-on-write: never mutate the source's own shared Data_t in
                // place (cache_slot_.staged may still be aliased by the source
                // cache_data_ entry / other holders) -- clone before rewriting id/
                // size_ so the source's own metadata is untouched.
                auto copied = std::make_shared<LLM_SIM_DATA::Data_t<T>>(*cache_slot_.staged);
                copied->id = dst;
                // dst_size (activation-shape normalization bug):
                // mirrors Router::packet2data's "receiver's declared shape wins"
                // convention -- without this, a purely local cache-copy silently
                // kept whatever size_ the source happened to carry (e.g. DRAMSim's
                // 1-D {len} stamp on every DRAM read), while the NoC path for the
                // SAME logical tensor always normalized to the declared shape,
                // producing two different ndims for what should be one tensor.
                if (!cache_slot_.cur_inst->dst_size.empty()) {
                    if (std::getenv("DBG_RACE_TRACE")) {
                        int size_prod = 1;
                        for (int s : cache_slot_.cur_inst->dst_size) size_prod *= s;
                        std::cerr << "[CACHE-COPY] @" << sc_time_stamp()
                                  << " c=" << core_id_
                                  << " src=" << cache_slot_.cur_inst->data_id
                                  << " dst=" << dst
                                  << " src_value.size()=" << copied->value.size()
                                  << " dst_size=[";
                        for (int s : cache_slot_.cur_inst->dst_size) std::cerr << s << ",";
                        std::cerr << "] dst_size_prod=" << size_prod
                                  << (size_prod != static_cast<int>(copied->value.size()) ? " MISMATCH" : "")
                                  << "\n" << std::flush;
                    }
                    copied->size_ = cache_slot_.cur_inst->dst_size;
                }
                cache_slot_.staged = copied;
                data_pool_[dst] = copied;
                cache_slot_.phase = SubPhase::CACHE_COPY_WRITE_PUSH;
            }
            break;

        case SubPhase::CACHE_COPY_WRITE_PUSH: {
            const uint64_t dst = cache_slot_.cur_inst->dst_data_id;
            auto it = data_pool_.find(dst);
            if (it == data_pool_.end()) {
                SC_REPORT_WARNING("DMA/CACHE_COPY", "dst not in pool");
                cache_slot_.phase = SubPhase::IDLE;
                break;
            }
            if (try_publish_to_cache(it->second)) {
                cache_slot_.phase = SubPhase::IDLE;
            }
            break;
        }


        default: break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// try_dispatch  –  attempt to dispatch one instruction to a free resource slot.
//
// Bandwidth rules enforced here:
//   1. DRAM read/write: exclusive  – dram_slot_ must be IDLE
//   2. NoC recv / send: independent slots (can run simultaneously)
//   3. Cache: exclusive per slot   – cache_slot_ must be IDLE
//   4. Global: total active_channels() < NUM_MAX_CHANNEL
//
// Cache direction is inferred from the data pool:
//   data_id present → write to cache;  absent → read from cache.
//
// Instructions not yet dispatchable are held in pending_inst_ across cycles.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::try_dispatch() {
    // Fetch a new instruction only when the pending slot is empty
    if (!pending_inst_) {
        LLM_SIM_INST::DMAInstPtr_t inst;
        if (!inst_in.nb_read(inst)) return;
        pending_inst_ = inst;
    }

    // (2026-08-12): identity-track pending_inst_
    // across calls (raw pointer compare) so a single choke point can report
    // "stuck on this instruction for N cycles" regardless of which branch
    // below (DRAM/cache/NoC send/recv) is the actual blocker. Logs at 101,
    // then every 1000 (matches handleLocalSync()'s DBG_NOC_TRACE-lowered
    // period so the two traces are comparable at the same timescale).
    if (dbg_trace_) {
        if (pending_inst_.get() == pending_inst_stuck_on_) {
            ++pending_inst_stuck_count_;
        } else {
            pending_inst_stuck_on_ = pending_inst_.get();
            pending_inst_stuck_count_ = 1;
        }
        if (pending_inst_stuck_count_ == 101 || pending_inst_stuck_count_ % 1000 == 0) {
            const auto& pi = pending_inst_;
            std::cerr << "[DMA-TRACE] t=" << sc_core::sc_time_stamp().to_double()
                      << " c=" << core_id_ << " TRY_DISPATCH_STUCK"
                      << " stuck_count=" << pending_inst_stuck_count_
                      << " is_noc=" << pi->is_noc << " is_dram=" << pi->is_dram
                      << " is_cache=" << pi->is_cache << " is_cache_copy=" << pi->is_cache_copy
                      << " is_cache_mc=" << pi->is_cache_mc
                      << " data_id=" << pi->data_id
                      << " active_channels=" << active_channels()
                      << " slots=[" << debug_slots() << "]"
                      << "\n" << std::flush;
        }
    }

    // Global bandwidth gate
    auto inst = pending_inst_;

    if (active_channels() >= LLM_SIM_DMA::NUM_MAX_CHANNEL) {
        return;
    }

    if (inst->is_dram) {
        // ── Phase C owner lookup, rewritten
        // from the static owner_of()-based rule -- this simulator's
        // convention is that a core only ever WRITES its own local DRAM
        // (see gen_core.py's docstring), so writes never need redirecting
        // and always take the local path below.
        //
        // OLD rule: "owner_of() != core_id_ -> remote". owner_of() is
        // static (just the first entry of a possibly-multi-owner list) and
        // ignores every OTHER owner in that list -- a broadcast file like
        // attn_norm.bin (owners: [0..63]) resolves to owner_of()==0 for
        // EVERY reader, so core 2 reading its OWN legitimately-owned copy
        // would have been redirected to core 0 regardless, and core 0 is
        // not 4-neighbor-adjacent to core 2 -> SC_REPORT_FATAL. This was
        // unreachable in production only because dram_map_path was always
        // empty (owner_of() always OWNER_LOCAL); wiring the map with the
        // old rule would abort within the first few instructions of any
        // core reading a broadcast weight.
        //
        // NEW rule -- checks static ownership (owners_of()) first, NOT
        // is_resident() alone:
        //   !is_mapped(path) || core_id_ ∈ owners_of(path)
        //     || (capacity tracking enabled for core_id_ && is_resident(core_id_, path))
        //   -> local
        //   else -> remote, preferring a group member that Phase D/F's
        //     capacity-aware residency says currently has it, falling back
        //     to the map's owner_of() when capacity tracking isn't enabled
        //   (still true for every core through Phase C -- Phase D is what
        //   turns it on)
        //
        // is_resident() ALONE cannot be the "am I local" test here: with
        // capacity tracking disabled (has_capacity_limit()==false, true for
        // every core before Phase D wires it into Acc8x8), is_resident()
        // trivially returns true for EVERY core on EVERY path regardless of
        // actual ownership (see its own doc comment: "tracking not enabled
        // -> always resident") -- using it as the sole local-vs-remote test
        // would collapse EVERY read to "local" unconditionally, silently
        // breaking the very redirect mechanism test_dma_offchip_fetch.cpp
        // and test_dma_concurrent_recv.cpp depend on (both run with capacity
        // deliberately left at 0). Checking owners_of() first keeps the
        // static-ownership answer meaningful with or without capacity
        // tracking, while the is_resident()-based group scan only ever
        // engages once Phase D actually enables it.
        bool is_remote_read = false;
        // Phase F (extension): true when the
        // group scan below already covered owner_core itself (because
        // owner_core happens to be one of core_id_'s OWN group members) and
        // found it non-resident -- we already KNOW, for free, that an actual
        // remote read to owner_core would come back nullptr. This does NOT
        // reach beyond core_id_'s own group in any new way: it only fires
        // when owner_core was ALREADY a member of the group just scanned
        // (is_group(core_id_, owner_core) below is the SAME adjacency check
        // the FATAL guard already enforces for any remote read) -- it never
        // causes a read of, or even a status check on, a core outside
        // core_id_'s group that wasn't already going to be touched anyway.
        bool remote_is_known_miss = false;
        int  owner_core     = LLM_SIM_DRAM::AddressMapper::OWNER_LOCAL;
        if (inst->dram_req_ptr->is_read && data_manager_ != nullptr) {
            const std::string& path = inst->dram_req_ptr->addr_ptr->file_path;
            auto& mapper = data_manager_->address_mapper();
            if (mapper.is_mapped(path)) {
                const auto owners = mapper.owners_of(path);
                const bool i_am_owner =
                    std::find(owners.begin(), owners.end(), core_id_) != owners.end();
                const bool capacity_on = data_manager_->has_capacity_limit(core_id_);
                const bool capacity_aware_resident =
                    capacity_on && data_manager_->is_resident(core_id_, path);
                if (!i_am_owner && !capacity_aware_resident) {
                    // compile-time switch
                    // to skip the group-remote scan entirely (fall straight
                    // through to off-chip/lowest-level-memory fallback as if
                    // every group member were non-resident), gated here at
                    // the sole production call site rather than inside
                    // find_resident_group_member() itself -- keeps that
                    // function's own semantics (a plain group-scan query)
                    // unpolluted by an unrelated bypass flag. The mechanism
                    // itself (find_resident_group_member(), the group
                    // bitmap, DRAMRouter's group-gated path) is left fully
                    // intact.
                    const bool try_group_remote =
                        capacity_on && !data_manager_->skip_group_remote(core_id_);
                    const int j = try_group_remote
                        ? data_manager_->find_resident_group_member(core_id_, path)
                        : LLM_SIM_DRAM::AddressMapper::OWNER_LOCAL;
                    if (j != LLM_SIM_DRAM::AddressMapper::OWNER_LOCAL) {
                        owner_core = j;
                    } else {
                        owner_core = mapper.owner_of(path);
                        // The group scan above already checked EVERY member
                        // of core_id_'s own group (including owner_core, if
                        // it happens to be one) and found none resident. If
                        // owner_core IS one of those group members, we
                        // already know for free it would miss -- no need to
                        // pay a real remote-read round trip just to learn
                        // that again. Gated by try_group_remote (not just
                        // capacity_on) so skip_group_remote also skips this
                        // derived shortcut -- it exists only as an
                        // optimization ON TOP OF the group scan this switch
                        // disables, so it must be skipped along with it.
                        if (try_group_remote && mapper.is_group(core_id_, owner_core)) {
                            remote_is_known_miss = true;
                        }
                    }
                    is_remote_read = true;
                }
            }
        }

        if (is_remote_read) {
            if (remote_dram_slot_.phase != SubPhase::IDLE) return;  // remote DRAM exclusive
            if (!data_manager_->address_mapper().is_group(core_id_, owner_core)) {
                SC_REPORT_FATAL("DMA/REMOTE_DRAM",
                    ("core " + std::to_string(core_id_) +
                     " attempted a remote read of a file owned by non-group core " +
                     std::to_string(owner_core) +
                     " (file=" + inst->dram_req_ptr->addr_ptr->file_path + ")").c_str());
                return;  // unreachable in production: owner_core came directly from
                         // find_resident_group_member()'s own group scan, so it is
                         // group-adjacent by construction; kept as defense-in-depth.
            }

            if (remote_is_known_miss) {
                // Phase F (extension): skip the remote read we already know
                // would come back nullptr -- go straight to the off-chip
                // fetch, exactly as if tick_remote_dram() had already
                // received that nullptr (see its own nullptr branch), just
                // without paying for the round trip.
                if (!start_group_miss_fetch(inst->dram_req_ptr->addr_ptr, inst->data_id))
                    return;  // SYNTHESIZED quota full -- retry this whole dispatch next cycle
                pending_inst_ = nullptr;  // successfully dispatched
                return;
            }

            // Phase C: tell DRAMRouter directly which core to forward to,
            // instead of letting it re-derive a (possibly different, or
            // non-group-adjacent) owner via its own owner_of() call.
            inst->dram_req_ptr->target_core = owner_core;
            if (!remote_dram_req_out.nb_write(inst->dram_req_ptr)) return;
            data_manager_->note_remote_read();  // Phase D observability counter
            remote_dram_slot_.cur_inst = inst;
            remote_dram_slot_.staged   = nullptr;
            remote_dram_slot_.phase    = SubPhase::REMOTE_DRAM_WAIT;
            pending_inst_ = nullptr;  // successfully dispatched
            return;
        }

        if (dram_slot_.phase != SubPhase::IDLE) return;  // DRAM exclusive
        // For DRAM write: this exact instruction must own a private prefetch
        // of its own -- never read a shared data_pool_[id] entry, which a
        // concurrently-timing NoC send prefetching the SAME id could equally
        // be racing to fetch/consume/erase (see cache_prefetch_for_'s doc
        // comment in DMA.h; this replaced the earlier data_pool_-based
        // staging, which caused a genuine SWA-layer deadlock this way).
        // dram_pending_staged_ survives across repeated try_dispatch() calls
        // for this same pending_inst_ (not reset elsewhere), so once the
        // prefetch below completes and CACHE_READ_TIMER hands it back via
        // dram_slot_.staged, this check sees it non-null and proceeds.
        if (!inst->dram_req_ptr->is_read && inst->data_id > 0 && !dram_slot_.staged) {
            if (cache_slot_.phase != SubPhase::IDLE) return;  // wait for cache slot
            auto cache_req = std::make_shared<LLM_SIM_INST::CacheReq_t>(inst->data_id, inst->truncate, inst->new_data_id);
            auto cache_read_inst = std::make_shared<LLM_SIM_INST::DMAInst_t>(cache_req, inst->data_id, 0);
            if (!cache_inst_out.nb_write(cache_req)) return;
            if (std::getenv("DBG_PIN_PREFETCH") && core_id_ ==
                std::atoi(std::getenv("DBG_PIN_PREFETCH"))) {
                std::cerr << "[PIN-PREFETCH-REQ] @" << sc_time_stamp()
                          << " c=" << core_id_ << " for=DRAM_WRITE"
                          << " inst_data_id=" << inst->data_id
                          << " pending_inst_.get()=" << pending_inst_.get()
                          << "\n" << std::flush;
            }
            cache_slot_.cur_inst = cache_read_inst;
            cache_slot_.staged   = nullptr;
            cache_slot_.phase    = SubPhase::CACHE_READ_WAIT;
            cache_prefetch_for_  = PrefetchFor::DRAM_WRITE;
            return;  // keep pending_inst_ as DRAM write; retry after prefetch
        }
        if (!dram_req_out.nb_write(inst->dram_req_ptr)) return;
        dram_slot_.cur_inst = inst;
        // dram_slot_.staged already holds the freshly-prefetched Data_t for
        // a write (set just above via cache_prefetch_for_'s handoff) --
        // must NOT be cleared here. A read has none yet; DRAM_READ_TIMER
        // populates it.
        if (inst->dram_req_ptr->is_read) {
            dram_slot_.staged = nullptr;
            dram_slot_.phase  = SubPhase::DRAM_READ_WAIT;
        } else {
            start_timer(DRAM_SLOT, cal_lat(inst));
            dram_slot_.phase = SubPhase::DRAM_WRITE;
        }

    } else if (inst->is_noc) {
        if (inst->noc_inst_ptr->src_id < 0) {
            // Recv -- Phase B: admit into pending_recvs_ instead of a
            // single-occupancy slot check (see can_admit_recv()'s header
            // comment for the MAX_PEND_PKT/PEND_PKT_RESERVED reasoning).
            if (!can_admit_recv(PendingRecv::Kind::INSTRUCTION)) {
                if (dbg_trace_) {
                    std::cerr << "[DMA-TRACE] t=" << sc_core::sc_time_stamp().to_double()
                              << " c=" << core_id_
                              << " RECV_ADMIT_BLOCKED pkt=" << inst->noc_inst_ptr->packet_id
                              << " pending_recvs_.size()=" << pending_recvs_.size()
                              << " cap=" << (LLM_SIM_DMA::MAX_PEND_PKT - LLM_SIM_DMA::PEND_PKT_RESERVED)
                              << "\n" << std::flush;
                }
                return;
            }
            if (!noc_inst_out.nb_write(inst->noc_inst_ptr)) {
                if (dbg_trace_) {
                    std::cerr << "[DMA-TRACE] t=" << sc_core::sc_time_stamp().to_double()
                              << " c=" << core_id_
                              << " RECV_NOC_INST_OUT_FULL pkt=" << inst->noc_inst_ptr->packet_id
                              << "\n" << std::flush;
                }
                return;
            }
            if (dbg_trace_) {
                std::cerr << "[DMA-TRACE] t=" << sc_core::sc_time_stamp().to_double()
                          << " c=" << core_id_
                          << " RECV_ADMITTED pkt=" << inst->noc_inst_ptr->packet_id
                          << " data_id=" << inst->data_id
                          << " pending_recvs_.size()=" << (pending_recvs_.size() + 1)
                          << "\n" << std::flush;
            }
            pending_recvs_.push_back(PendingRecv{
                PendingRecv::Kind::INSTRUCTION, inst->noc_inst_ptr->packet_id,
                inst->data_id, inst, /*publish_to_cache=*/true, ""});
        } else {
            // Send - this exact instruction must own a private prefetch of
            // its own -- never read a shared data_pool_[id] entry a
            // concurrently-timing DRAM write prefetching the SAME id could
            // equally be racing to fetch/consume/erase (see
            // cache_prefetch_for_'s doc comment in DMA.h).
            if (noc_send_slot_.phase != SubPhase::IDLE) return;
            if (inst->data_id > 0 && !noc_send_slot_.staged) {
                if (cache_slot_.phase != SubPhase::IDLE) return;  // wait for cache slot
                auto cache_req = std::make_shared<LLM_SIM_INST::CacheReq_t>(inst->data_id, inst->truncate, inst->new_data_id);
                auto cache_read_inst = std::make_shared<LLM_SIM_INST::DMAInst_t>(cache_req, inst->data_id, 0);
                if (!cache_inst_out.nb_write(cache_req)) return;
                cache_slot_.cur_inst = cache_read_inst;
                cache_slot_.staged   = nullptr;
                cache_slot_.phase    = SubPhase::CACHE_READ_WAIT;
                cache_prefetch_for_  = PrefetchFor::NOC_SEND;
                return;  // keep pending_inst_ as NoC send; retry after prefetch
            }
            if (!noc_inst_out.nb_write(inst->noc_inst_ptr)) return;
            noc_send_slot_.cur_inst = inst;
            // noc_send_slot_.staged already holds the freshly-prefetched
            // Data_t for a real payload send (set just above); a control
            // packet never had one to begin with (is_ctrl_pkt short-
            // circuited the branch above) -- leave it as whatever it was
            // (nullptr in that case, never read by NOC_SEND_PUSH's control-
            // packet branch).
            start_timer(NOC_SEND_SLOT, cal_lat(inst));
            noc_send_slot_.phase    = SubPhase::NOC_SEND;
        }

    } else if (inst->is_cache_mc) {
        // MainCtrl cache read: 發 req 給 Cache port 6，結果直送 MainCtrl.cache_data_in。
        // 不占用 cache_slot_，發完立刻完成。
        auto cache_req = std::make_shared<LLM_SIM_INST::CacheReq_t>(inst->data_id, false);
        if (!cache_inst_out_mc.nb_write(cache_req)) return;
        // 立刻完成，不設 slot state
    } else if (inst->is_cache) {
        if (cache_slot_.phase != SubPhase::IDLE) return;
        if (!cache_inst_out.nb_write(inst->cache_req_ptr)) return;
        cache_slot_.cur_inst = inst;
        cache_slot_.staged   = nullptr;
        // Direction: if data_id already in pool → write; else → read
        if (data_pool_.count(inst->data_id)) {
            start_timer(CACHE_SLOT, cal_lat(inst));
            cache_slot_.phase = SubPhase::CACHE_WRITE;
        } else {
            cache_slot_.phase   = SubPhase::CACHE_READ_WAIT;
            cache_prefetch_for_ = PrefetchFor::DATA_POOL;
        }
    } else if (inst->is_cache_copy) {
        // cache copy: read cache[src=data_id] → pool → write pool to cache[dst=dst_data_id]
        if (cache_slot_.phase != SubPhase::IDLE) return;
        // Issue a CacheReq for the src to start the read
        auto rreq = std::make_shared<LLM_SIM_INST::CacheReq_t>(inst->data_id, false);
        if (!cache_inst_out.nb_write(rreq)) return;
        cache_slot_.cur_inst = inst;
        cache_slot_.staged   = nullptr;
        cache_slot_.phase    = SubPhase::CACHE_COPY_READ_WAIT;
    }

    pending_inst_ = nullptr;  // successfully dispatched
}

// ─────────────────────────────────────────────────────────────────────────────
// run  –  main SC_THREAD; advances all slot state machines every posedge.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void DMA<T>::run() {
    sync_busy();  // publish initial idle state

    while (true) {
        wait(clk.posedge_event());

        // Clear all timer start pulses that were set in the previous cycle
        for (int i = 0; i < NUM_SLOTS; ++i)
            start_sig_[i].write(false);

        // Advance each resource slot
        tick_dram();
        tick_remote_dram();
        tick_group_cache_install();
        tick_noc_recv();
        tick_noc_send();
        tick_cache();
        tick_writeback();
        tick_fill();

        // Attempt to dispatch the next pending instruction
        try_dispatch();

        // Refresh busy output once per cycle
        sync_busy();

        // Phase B: utilization accounting uses the same broadened "any
        // outstanding work" definition as sync_busy()'s b->busy, not the
        // narrower bandwidth-gate active_channels() -- a recv merely
        // pending in pending_recvs_ is genuinely unfinished work and should
        // count toward busy_cycle_/utilization() consistently with what
        // MainCtrl observes via isBusy().
        if (active_channels() > 0 || recv_xfer_active_ || !pending_recvs_.empty())
            ++busy_cycle_;
    }
}

template class DMA<float>;
template class DMA<int>;

