#include "DRAM_sim/OffChipPacketAdapter.h"

template <typename T>
OffChipPacketAdapter<T>::OffChipPacketAdapter(sc_module_name name, int node_id)
    : sc_module(name), node_id_(node_id)
{
    SC_THREAD(depacketize_thread);
    sensitive << clk.pos();

    SC_THREAD(packetize_thread);
    sensitive << clk.pos();
}

// ---------------------------------------------------------------------------
// depacketize_thread -- NoC -> OffChipDRAM
// ---------------------------------------------------------------------------
template <typename T>
void OffChipPacketAdapter<T>::depacketize_thread() {
    while (true) {
        LLM_SIM_NOC::PacketPtr_t<T> pkt;
        while (!noc_in.nb_read(pkt))
            wait(clk.posedge_event());

        if (!pkt->offchip_dram_req) {
            SC_REPORT_WARNING("OffChipPacketAdapter",
                ("node " + std::to_string(node_id_) +
                 " received a packet with no offchip_dram_req payload -- dropped").c_str());
            continue;
        }

        const auto& req = pkt->offchip_dram_req;

        if (req->is_read) {
            // Remember who to reply to; packetize_thread pops this in FIFO
            // order when OffChipDRAM's response comes back (correct because
            // OffChipDRAM processes exactly one request at a time -- see
            // its class comment).
            pending_reads_.emplace_back(pkt->src_id, pkt->packet_id);
        }

        while (!dram_req_out.nb_write(req))
            wait(clk.posedge_event());

        if (!req->is_read) {
            // Write request: the payload rides in the SAME packet
            // (Payload_t<T>::data), so forward it right after the request --
            // OffChipDRAM's write_port expects exactly this pairing (mirrors
            // DRAMSim::do_write's req_port + write_port pattern).
            auto data_ptr = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
            data_ptr->id    = req->tid;
            data_ptr->value = pkt->payload.data;
            data_ptr->size_ = {static_cast<int>(req->len)};
            while (!dram_write_out.nb_write(data_ptr))
                wait(clk.posedge_event());
        }
    }
}

// ---------------------------------------------------------------------------
// packetize_thread -- OffChipDRAM -> NoC
// ---------------------------------------------------------------------------
template <typename T>
void OffChipPacketAdapter<T>::packetize_thread() {
    while (true) {
        LLM_SIM_DATA::DataPtr_t<T> data;
        while (!dram_read_in.nb_read(data))
            wait(clk.posedge_event());

        if (pending_reads_.empty()) {
            SC_REPORT_FATAL("OffChipPacketAdapter",
                ("node " + std::to_string(node_id_) +
                 " produced a read response with no matching pending requester").c_str());
            continue; // unreachable in production (FATAL aborts); for test overrides
        }
        const auto [requester_core, packet_id] = pending_reads_.front();
        pending_reads_.pop_front();

        auto pkt = std::make_shared<LLM_SIM_NOC::Packet_t<T>>();
        pkt->packet_id   = packet_id; // see header: reply reuses the request's packet_id (Phase 7 may revisit)
        pkt->src_id      = node_id_;
        pkt->dst_ids     = {requester_core};
        pkt->payload.data = data->value;
        // Phase E: use data->elem_bits (now
        // correctly tagged by OffChipDRAM::do_read) instead of the fixed
        // sizeof(T) -- mirrors Router::data2packet's own flit_size formula.
        pkt->flit_size   = static_cast<int>(
            (static_cast<uint64_t>(data->elem_bits) * data->value.size() + 7) / 8);
        pkt->elem_bits   = data->elem_bits;
        pkt->cast_type   = LLM_SIM_NOC::CastType::UNICAST;

        while (!noc_out.nb_write(pkt))
            wait(clk.posedge_event());
    }
}

template class OffChipPacketAdapter<int>;
template class OffChipPacketAdapter<float>;
