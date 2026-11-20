#ifndef LLM_SIM_NOC_UTILS_H
#define LLM_SIM_NOC_UTILS_H

#include <iostream>
#include <systemc.h>
#include <vector>
#include <queue>
#include <array>
#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "utils/bufferAddr.h"

// Forward declaration only (not #include "utils/Req.h") to avoid a circular
// include: utils/Req.h -> utils/BaseInst.h -> NoC_sim/noc_utils.h. Packet_t
// below only holds a shared_ptr<Req_t> (a pointer member), which a forward
// declaration is sufficient for -- every file that actually constructs/
// destroys a Packet_t<T> (Router.cpp, DMA.cpp, NoC_sim.cpp, ...) already
// transitively includes the full utils/Req.h for other reasons (they all
// handle DRAM requests directly).
namespace LLM_SIM_DRAM {
    class Req_t;
    using ReqPtr_t = std::shared_ptr<Req_t>;
}

namespace LLM_SIM_NOC {

    const int MESH_WIDTH = 8;
    const int FLIT_CYCLE = 3;
    const int LANES = 32;
    
    enum class DIR { // more direction may be added in the future
        E = 0, S = 1, W = 2, N = 3, NUM_DIR = 4, LOCAL = 5
    };
    

    struct Hop_t {
        int curr_id; // passing router_id
        LLM_SIM_NOC::DIR dir;
    };
    struct UnionHop_t {
        int curr_id; // passing router_id
        std::vector<std::pair<LLM_SIM_NOC::DIR, uint64_t>> dir_dist;
        // std::unordered_map<DIR, uint64_t> dir2dist;
    };
    
    // global states DS declaration
    // Backpressure model: each output-port queue entry represents one packet's
    // in-flight service at that port. `start_at` is the ABSOLUTE cycle the
    // packet begins service on the port; `dur` is the service duration (flit
    // cycles). The port is occupied for [start_at, start_at + dur) and frees
    // at start_at + dur. Both times are absolute so insert/check/port_finish
    // all agree (previously ready_at was stored relative-to-now but compared as
    // absolute -- see backpressure plan).
    struct OutEntry_t {
        uint64_t start_at;   // absolute cycle service begins on this port
        uint64_t dur;        // service duration in cycles (frees at start_at+dur)
    };

    // queue related types
    using OutQ_t = std::queue<OutEntry_t>;
    using AllOutQ_t = std::vector<std::vector<std::queue<OutEntry_t>>>; // [router_id][dir]
    // path related types
    using PathUnion_t = std::unordered_map<int, LLM_SIM_NOC::UnionHop_t>; // passed router_id to hops
    using Path_t = std::vector<LLM_SIM_NOC::Hop_t>; // unicast path
    using AllPath_t = std::unordered_map<int, LLM_SIM_NOC::Path_t>; // dst_id to path_t
    
    // Packet structures
    enum class CastType {
        UNICAST,
        MULTICAST
    };

    template<typename T>
    struct Payload_t {
        Payload_t() {};
        std::vector<T> data;
        LLM_SIM_BUF::BaseAddrPtr_t dst_addr_ptr;
    };

    template <typename T>
    struct Packet_t {
        Packet_t() {};
        uint64_t packet_id  = 0;
        bool     is_dynamic = false;  // true → Router notifies MainCtrl to run op_id
        uint64_t op_id      = 0;      // predefined operation to execute (valid when is_dynamic)
        Payload_t<T> payload;
        int      src_id     = -1;
        std::vector<int> dst_ids; // dst core ids
        int      flit_size  = 0;
        // Phase E: carries Data_t::elem_bits
        // across the NoC hop -- without this, packet2data() has no way to
        // recover the sender's elem_bits and every packet silently reverts to
        // 32 bits (sizeof(T)*8 default) on the receiving end, even though
        // data2packet() already reads the true elem_bits to size flit_size.
        int      elem_bits  = sizeof(T) * 8;
        CastType cast_type  = CastType::UNICAST;

        // Phase 6b: DRAM request/response riding inside a NoC
        // packet addressed to/from an off-chip DRAM node (dst_ids/src_id in
        // [OFFCHIP_ID_BASE, OFFCHIP_ID_BASE+NUM_OFFCHIP_NODES)). Null for
        // every ordinary core-to-core packet -- zero behavior change for
        // existing traffic. Depacketizer reads this to reconstruct the
        // DRAMAddr_t/Req_t it forwards to OffChipDRAM; Packetizer sets it
        // (is_read=true request) when wrapping OffChipDRAM's response for
        // the return trip so the requester's Depacketizer-side logic (DMA's
        // remote-dram path) can recover addr/len bookkeeping if needed.
        LLM_SIM_DRAM::ReqPtr_t offchip_dram_req = nullptr;
    };

    template <typename T>
    using PacketPtr_t = std::shared_ptr<Packet_t<T>>;

    struct TransState_t {
        int src_id;
        std::unordered_map<int, uint64_t> dst2time; // dst_id to hops left
    };

    // ── Dynamic-packet notification ──────────────────────────────────────────
    // Sent by Router to MainCtrl via dynamic_notify_out when a packet with
    // is_dynamic == true is received from the NoC.  MainCtrl uses this to
    // dispatch the corresponding predefined operation with priority over its
    // local instruction stream. CA_2026_FP: nothing in this worktree ever
    // sets is_dynamic true or writes to dynamic_notify_out, so this type
    // exists only to keep the dynamic_req_in/dynamic_notify_out port wiring
    // (Core.cpp) well-typed.
    struct DynamicReq_t {
        uint64_t packet_id = 0;  // identifies the triggering packet
        uint64_t op_id     = 0;  // selects the predefined instruction sequence
    };
    // operator<< required by sc_fifo<DynamicReq_t> for internal tracing
    inline std::ostream& operator<<(std::ostream& os, const DynamicReq_t& r) {
        os << "{packet_id=" << r.packet_id << " op_id=" << r.op_id << "}";
        return os;
    }
    // sc_trace no-op so sc_fifo<DynamicReq_t> compiles cleanly
    inline void sc_trace(sc_core::sc_trace_file*, const DynamicReq_t&,
                         const std::string&) {}
    
}

#endif