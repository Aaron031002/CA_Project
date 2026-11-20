#include "NoC_sim/OffChipRouteAlgo.h"

#include <stdexcept>
#include <string>

OffChipRouteAlgo::OffChipRouteAlgo(int mesh_width)
    : route_algo_("XY", mesh_width), mesh_width_(mesh_width) {}

int OffChipRouteAlgo::edge_core_id(int offchip_id) const {
    if (!is_offchip_id(offchip_id) ||
        offchip_id >= LLM_SIM_NOC::OFFCHIP_ID_BASE + LLM_SIM_NOC::NUM_OFFCHIP_NODES) {
        throw std::invalid_argument(
            "OffChipRouteAlgo: offchip_id " + std::to_string(offchip_id) +
            " out of range [" + std::to_string(LLM_SIM_NOC::OFFCHIP_ID_BASE) + ", " +
            std::to_string(LLM_SIM_NOC::OFFCHIP_ID_BASE + LLM_SIM_NOC::NUM_OFFCHIP_NODES) + ")");
    }
    const int row = offchip_id - LLM_SIM_NOC::OFFCHIP_ID_BASE;
    return row * mesh_width_ + (mesh_width_ - 1); // rightmost core in that row
}

std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t>
OffChipRouteAlgo::get_path_to_offchip(const int& src_id, const int& offchip_id) {
    const int edge_id = edge_core_id(offchip_id);

    if (src_id == edge_id) {
        // Already at the edge core: no mesh-side hops needed at all: the
        // caller (OffChipNoCLat) will still add the fixed off-chip link
        // latency on top of this empty path.
        LLM_SIM_NOC::PathUnion_t empty_union;
        LLM_SIM_NOC::AllPath_t empty_path;
        empty_path[edge_id] = {}; // present but empty path for edge_id
        return {empty_union, empty_path};
    }

    return route_algo_.get_path(src_id, edge_id);
}

std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t>
OffChipRouteAlgo::get_path_from_offchip(const int& offchip_id, const int& dst_id) {
    const int edge_id = edge_core_id(offchip_id);

    if (edge_id == dst_id) {
        // Response is addressed to the edge core itself: no mesh-side hops
        // needed, symmetric to get_path_to_offchip()'s own edge_id==src_id case.
        LLM_SIM_NOC::PathUnion_t empty_union;
        LLM_SIM_NOC::AllPath_t empty_path;
        empty_path[dst_id] = {};
        return {empty_union, empty_path};
    }

    // edge_id and dst_id are both real core ids -- never passes an
    // off-chip id into route_algo_/id2coord.
    return route_algo_.get_path(edge_id, dst_id);
}
