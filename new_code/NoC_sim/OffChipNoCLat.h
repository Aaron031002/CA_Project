#ifndef LLM_SIM_NOC_OFFCHIP_NOCLAT_H
#define LLM_SIM_NOC_OFFCHIP_NOCLAT_H

#include <unordered_map>
#include <vector>
#include <utility>

#include "NoC_sim/NoCLat.h"
#include "NoC_sim/OffChipRouteAlgo.h"
#include "NoC_sim/noc_utils.h"
#include "NoC_sim/noc_param.h"

// ---------------------------------------------------------------------------
// OffChipNoCLat (Phase 6a)
//
// Deliberately separate from NoCLat -- see OffChipRouteAlgo's header comment
// for why off-chip ids never flow through the ordinary mesh routing/latency
// classes. Composes NoCLat to reuse its congestion-aware per-hop formula for
// the mesh-side portion of the path (src core -> edge core of the
// destination's row), then adds a FIXED, congestion-independent
// OFFCHIP_LINK_LATENCY for the final off-chip leg -- that leg models a
// dedicated physical channel, not a shared mesh link, so it must never be
// looked up in / inserted into alloutq.
// ---------------------------------------------------------------------------
class OffChipNoCLat {
public:
    // pf_fix: research switch S3, passed to the composed NoCLat (off-chip
    // packets share alloutq with core packets, so they use the same formula).
    explicit OffChipNoCLat(int mesh_width = LLM_SIM_NOC::MESH_WIDTH, bool pf_fix = false)
        : noc_lat_(mesh_width, pf_fix) {}

    // allpath_to_edge: from OffChipRouteAlgo::get_path_to_offchip(), keyed by
    // EDGE CORE id (not the off-chip id itself). This function remaps the
    // key to offchip_id internally before delegating to NoCLat::get_latency
    // (which looks entries up by packet.dst_ids, and packet.dst_ids carries
    // the off-chip id the caller actually addressed).
    //
    // Returns: {dst_id(=offchip_id) -> total latency including the fixed
    // off-chip leg, hop_vec} where hop_vec contains ONLY the mesh-side hops
    // (no synthetic off-chip hop) -- callers insert hop_vec into alloutq
    // exactly as they would for a normal core-to-core packet; the off-chip
    // leg's fixed cost is never congestion-tracked.
    // `now`: see NoCLat::get_latency's doc comment -- the caller's cycle
    // counter, not sc_time_stamp(); forwarded as-is to the composed NoCLat.
    template <typename T>
    std::pair<std::unordered_map<int, uint64_t>, std::vector<LLM_SIM_NOC::UnionHop_t>>
    get_latency(const LLM_SIM_NOC::AllOutQ_t& alloutq,
                const LLM_SIM_NOC::Packet_t<T>& packet,
                const LLM_SIM_NOC::AllPath_t& allpath_to_edge,
                int offchip_id, uint64_t now) const {
        // Remap the single edge-core-keyed entry to offchip_id so
        // NoCLat::get_latency's `allpath.find(dst_id)` (dst_id taken from
        // packet.dst_ids, which carries the off-chip id) succeeds.
        LLM_SIM_NOC::AllPath_t remapped;
        for (const auto& [edge_id, path] : allpath_to_edge) {
            (void)edge_id;
            remapped[offchip_id] = path;
        }

        auto [lat, hop_vec] = noc_lat_.get_latency(alloutq, packet, remapped, now);

        auto it = lat.find(offchip_id);
        if (it != lat.end()) {
            it->second += static_cast<uint64_t>(LLM_SIM_NOC::OFFCHIP_LINK_LATENCY);
        }
        return {lat, hop_vec};
    }

    // Reverse of get_latency() above, for an off-chip fetch RESPONSE (src_id
    // == the off-chip node's own id, dst_id == the real requesting core).
    // Bug fix (hang investigation, 2026-07-28):
    // see OffChipRouteAlgo::get_path_from_offchip()'s header comment for the
    // full root-cause story. No remapping needed here (unlike get_latency()
    // above) -- allpath_from_edge is already keyed by dst_id (the real core
    // the packet is addressed to), which is exactly what NoCLat::get_latency
    // looks up via packet.dst_ids.
    template <typename T>
    std::pair<std::unordered_map<int, uint64_t>, std::vector<LLM_SIM_NOC::UnionHop_t>>
    get_latency_from_offchip(const LLM_SIM_NOC::AllOutQ_t& alloutq,
                              const LLM_SIM_NOC::Packet_t<T>& packet,
                              const LLM_SIM_NOC::AllPath_t& allpath_from_edge,
                              int offchip_id, uint64_t now) const {
        (void)offchip_id; // kept for signature symmetry with get_latency() above; unused here
        auto [lat, hop_vec] = noc_lat_.get_latency(alloutq, packet, allpath_from_edge, now);
        for (auto& [dst, l] : lat) {
            l += static_cast<uint64_t>(LLM_SIM_NOC::OFFCHIP_LINK_LATENCY);
        }
        return {lat, hop_vec};
    }

private:
    NoCLat noc_lat_; // composed; reuses the existing congestion-aware formula
};

#endif // LLM_SIM_NOC_OFFCHIP_NOCLAT_H
