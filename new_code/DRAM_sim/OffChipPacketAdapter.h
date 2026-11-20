#ifndef LLM_SIM_OFFCHIP_PACKET_ADAPTER_H
#define LLM_SIM_OFFCHIP_PACKET_ADAPTER_H

#include <cstdint>
#include <deque>
#include <utility>
#include <systemc.h>

#include "NoC_sim/noc_utils.h"
#include "utils/buffer.h"
#include "utils/fifo.h"
#include "utils/Data.h"
#include "utils/Req.h"

// ---------------------------------------------------------------------------
// OffChipPacketAdapter<T> (Phase 6b)
//
// The single packetizer/depacketizer unit sitting between one off-chip DRAM
// node and the NoC, per the user's request ("一個 packetizer/depacketizer
// 的 unit"). Implemented as ONE module (not two separate classes) because
// its two directions share a small FIFO correlator (see pending_reads_
// below) -- splitting them into independent sc_modules would mean sharing
// that state across a module boundary, which the project's port-only
// communication convention disallows.
//
// Depacketize direction (NoC -> OffChipDRAM):
//   Reads a Packet_t<T> addressed to this off-chip node from NoCSim's
//   out_fifos[node_id]. The packet's offchip_dram_req field (see
//   noc_utils.h) already IS a fully-formed Req_t (DRAMAddr_t + is_read +
//   tid + len) -- reconstructed by whatever sent the packet (Phase 7's
//   DMA-side off-chip fetch path), not by this adapter. Forwards it
//   directly to OffChipDRAM.req_port. For writes, also forwards the
//   packet's payload.data as a Data_t<T> to OffChipDRAM.write_port.
//
// Packetize direction (OffChipDRAM -> NoC):
//   Reads OffChipDRAM's read_port response (only reads produce a response;
//   writes are fire-and-forget, matching how DRAMSim::do_write itself never
//   sends anything back over a channel either), wraps it into a new
//   Packet_t<T> addressed back to the original requester core
//   (dst_ids={requester}), and pushes it to NoCSim.in_ports[node_id].
//
// Correlation: OffChipDRAM (Phase 6b) has exactly one upstream source and
// processes strictly one request at a time end-to-end (see its class
// comment), so a plain FIFO of (requester core_id, packet_id) -- pushed
// when a read request is forwarded, popped when its response comes back --
// is sufficient to know which core/packet_id to address each response to.
// Same idiom as DRAMRouter's inflight_requesters_ (Phase 3b).
//
// Needs-decision (flagged for Phase 7, which defines how the requesting
// core's DMA actually issues the off-chip fetch and consumes the
// response): the response packet currently reuses the SAME packet_id as
// the request. Phase 7 may want a distinct convention (e.g. request/
// response packet_id pairs, mirroring the RPC dispatch/return pattern)
// once it wires this into DMA/MainCtrl's real instruction-driven path.
// ---------------------------------------------------------------------------
template <typename T>
class OffChipPacketAdapter : public sc_module {
public:
    SC_HAS_PROCESS(OffChipPacketAdapter);

    OffChipPacketAdapter(sc_module_name name, int node_id);

    sc_in_clk clk;

    // ── NoC side (this off-chip node's ports on NoCSim) ──────────────────
    // Port TYPES here deliberately mirror Router<T>'s NoC-facing ports
    // exactly (recv_in_packet is fifo_in<T>, send_out_packet is
    // sc_fifo_out<T>) -- NoCSim::out_fifos is the custom fifo_out<T>/
    // fifo_in<T> channel pair (utils/fifo.h), while NoCSim::in_ports is the
    // built-in sc_fifo_in<T>/sc_fifo_out<T> pair. Getting this backwards
    // would fail to compile at the binding call site in Acc8x8/Acc.
    fifo_in<LLM_SIM_NOC::PacketPtr_t<T>>     noc_in;  // <- NoCSim.out_fifos[node_id]
    sc_fifo_out<LLM_SIM_NOC::PacketPtr_t<T>> noc_out; // -> NoCSim.in_ports[node_id]

    // ── OffChipDRAM side ──────────────────────────────────────────────────
    buffer_out<LLM_SIM_DRAM::ReqPtr_t>       dram_req_out;   // -> OffChipDRAM.req_port
    sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>  dram_write_out; // -> OffChipDRAM.write_port
    sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>   dram_read_in;   // <- OffChipDRAM.read_port

    int node_id() const { return node_id_; }

private:
    const int node_id_;

    // FIFO of (requester core_id, packet_id) for outstanding READ requests
    // only (writes are fire-and-forget, never pushed here).
    std::deque<std::pair<int, uint64_t>> pending_reads_;

    void depacketize_thread();
    void packetize_thread();
};

#endif // LLM_SIM_OFFCHIP_PACKET_ADAPTER_H
