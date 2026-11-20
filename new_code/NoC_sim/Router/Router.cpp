#include "Router.h"
#include <algorithm>
#include <cstdio>


template <typename T>
Router<T>::Router(sc_module_name name, int id) : sc_module(name), router_id(id) {
    current_busy_ = std::make_shared<LLM_SIM_BUSY::NoCBusy_t>();
    SC_THREAD(send_thread);
    sensitive << clk.pos();
    SC_THREAD(recv_packet_thread);
    sensitive << clk.pos();
    SC_THREAD(recv_inst_thread);
    sensitive << clk.pos();
}

// ── Send path ─────────────────────────────────────────────────────────────────
// Waits until both a send NoCInst and a Data payload are available.
// Builds a Packet, records the inst in pending_noc_send_insts (in-place mutation
// so readers of router_busy always see the updated list via the stored shared_ptr),
// then forwards the Packet to the NoC.  Only removes the inst from pending after
// send_out_packet.nb_write() succeeds, preserving FIFO order.
template <typename T>
void Router<T>::send_thread() {
    while (true) {
        wait(); // posedge clk
        if (send_inst_in.empty() || send_in_ports.num_available() == 0)
            continue;

        LLM_SIM_INST::NoCInstPtr_t inst_ptr;
        send_inst_in.nb_read(inst_ptr);

        LLM_SIM_DATA::DataPtr_t<T> data_ptr;
        send_in_ports.nb_read(data_ptr);

        // Record as in-flight.  recv_inst_thread is the sole driver of router_busy;
        // mutating current_busy_ in place is visible to all readers without an
        // extra port write.
        current_busy_->pending_noc_send_insts.push_back(inst_ptr);

        auto packet_ptr = data2packet(data_ptr, inst_ptr);

        // Transmit.  Retry every cycle if send_out_packet FIFO is full.
        // Processing one inst at a time naturally preserves send order.
        int retry_cnt = 0;
        while (!send_out_packet.nb_write(packet_ptr)) {
            ++retry_cnt;
            wait();
        }

        // Successfully sent → clear from pending.
        auto& pending = current_busy_->pending_noc_send_insts;
        auto it = std::find(pending.begin(), pending.end(), inst_ptr);
        if (it != pending.end()) pending.erase(it);
    }
}

// ── Packet buffering ──────────────────────────────────────────────────────────
// Continuously drains recv_in_packet into packet_pool (keyed by packet_id).
// Runs independently from recv_inst_thread so that packets accumulate in the pool
// even when recv_inst_thread is blocked waiting for a different packet.
template <typename T>
void Router<T>::recv_packet_thread() {
    while (true) {
        wait(); // posedge clk
        if (recv_in_packet.empty())
            continue;

        LLM_SIM_NOC::PacketPtr_t<T> packet_ptr;
        recv_in_packet.nb_read(packet_ptr);
        packet_pool[packet_ptr->packet_id] = packet_ptr;
    }
}

// ── Recv instruction processing ───────────────────────────────────────────────
// Phase B (B-4): non-blocking, multi-outstanding.
// Was: block on ONE specific packet_id (while (packet_pool.find(...) ==
// end()) wait();) before even looking at the next queued recv inst -- a slow
// arrival (e.g. an off-chip fetch response, tens of thousands of cycles)
// would starve every other recv this core is waiting on. Now: admit every
// newly-arrived recv inst into outstanding_ every cycle, then scan
// outstanding_ (in submission order) for the first one whose packet has
// actually arrived and deliver just that one -- an inst submitted earlier
// but not yet ready never blocks a later one that IS ready.
template <typename T>
void Router<T>::recv_inst_thread() {
    router_busy.write(current_busy_);

    while (true) {
        wait(); // posedge clk

        // Admit every recv inst that arrived this cycle (or accumulated
        // while we were busy) -- recv_inst_in only ever carries this core's
        // own DMA's recv requests, so outstanding_ is naturally bounded by
        // DMA's own MAX_PEND_PKT (dma_params.h).
        bool admitted_any = false;
        LLM_SIM_INST::NoCInstPtr_t new_inst;
        while (recv_inst_in.nb_read(new_inst)) {
            outstanding_.push_back(new_inst);
            current_busy_->pending_noc_recv_insts.push_back(new_inst);
            admitted_any = true;
        }
        if (admitted_any) router_busy.write(current_busy_);

        // Deliver at most one per cycle -- recv_out_ports is a single fifo
        // (one physical NoC ingress channel worth of bandwidth).
        for (auto it = outstanding_.begin(); it != outstanding_.end(); ++it) {
            auto pool_it = packet_pool.find((*it)->packet_id);
            if (pool_it == packet_pool.end()) continue;  // not arrived yet -- check the next outstanding entry

            auto data_ptr = packet2data(pool_it->second, *it);
            data_ptr->id  = (*it)->id;

            if (!recv_out_ports.nb_write(data_ptr)) break;  // recv_out_ports full -- retry next cycle, nothing consumed

            packet_pool.erase(pool_it);
            LLM_SIM_INST::NoCInstPtr_t delivered = *it;
            outstanding_.erase(it);

            auto& pending = current_busy_->pending_noc_recv_insts;
            auto pit = std::find(pending.begin(), pending.end(), delivered);
            if (pit != pending.end()) pending.erase(pit);
            router_busy.write(current_busy_);
            break;
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
template <typename T>
LLM_SIM_NOC::PacketPtr_t<T> Router<T>::data2packet(
    const LLM_SIM_DATA::DataPtr_t<T>& data_ptr,
    const LLM_SIM_INST::NoCInstPtr_t& inst_ptr)
{
    auto packet_ptr = std::make_shared<LLM_SIM_NOC::Packet_t<T>>();
    packet_ptr->payload.data         = data_ptr->value;
    // packet_ptr->payload.dst_addr_ptr = inst_ptr->dst_buf;
    packet_ptr->src_id               = router_id;
    packet_ptr->packet_id            = inst_ptr->packet_id;
    packet_ptr->dst_ids              = inst_ptr->dst_ids;
    // flit_size is in BYTES. Use data_ptr->elem_bits
    // instead of the hardcoded sizeof(T)*8 so quantized payloads (e.g.
    // Q8_0, elem_bits=8) report their true wire size. elem_bits defaults to
    // sizeof(T)*8, so this is unchanged for all existing (non-quantized) data.
    packet_ptr->flit_size            = static_cast<int>(
        (static_cast<uint64_t>(data_ptr->elem_bits) * data_ptr->value.size() + 7) / 8);
    // Phase E: carry elem_bits itself across
    // the hop, not just its already-applied effect on flit_size -- see
    // Packet_t::elem_bits's doc comment.
    packet_ptr->elem_bits             = data_ptr->elem_bits;
    packet_ptr->cast_type            = inst_ptr->cast_type;
    packet_ptr->op_id                = inst_ptr->op_id;
    packet_ptr->offchip_dram_req        = inst_ptr->offchip_dram_req;
    return packet_ptr;
}

template <typename T>
LLM_SIM_DATA::DataPtr_t<T> Router<T>::packet2data(
    const LLM_SIM_NOC::PacketPtr_t<T>& packet_ptr,
    const LLM_SIM_INST::NoCInstPtr_t& inst_ptr)
{
    auto data_ptr      = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
    data_ptr->value    = packet_ptr->payload.data;
    // inst_ptr->data_size is the receiver's declared shape; fall back to a
    // flat {payload.size()} when it's empty (not provided by the sender's
    // instruction) rather than let an empty size_ reach Cache's write-
    // request mechanism, which aborts on exactly that. A declared shape
    // that disagrees with the real payload size is NOT papered over here --
    // Cache::validate_write() catches that mismatch at commit time, which is
    // what should surface a genuine inst-gen shape bug instead of silently
    // trusting whichever side happens to be wrong.
    data_ptr->size_    = !inst_ptr->data_size.empty()
        ? inst_ptr->data_size
        : std::vector<int>{static_cast<int>(packet_ptr->payload.data.size())};
    // Phase B (B-1): carry packet_id one hop
    // further to the consumer instead of dropping it here -- a receiver with
    // multiple simultaneously-outstanding recv requests needs this to
    // attribute an arrival to the correct one.
    data_ptr->packet_id = packet_ptr->packet_id;
    // Phase E: restore elem_bits from the
    // packet -- without this, every packet reconstructs a Data_t at the
    // default 32 bits regardless of what the sender actually tagged (the
    // biggest gap in the elem_bits plumbing prior to this fix).
    data_ptr->elem_bits = packet_ptr->elem_bits;
    return data_ptr;
}


// Template instantiation
template class Router<int>;
template class Router<float>;

