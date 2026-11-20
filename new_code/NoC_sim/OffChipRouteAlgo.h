#ifndef LLM_SIM_ROUTER_OFFCHIP_ROUTEALGO_H
#define LLM_SIM_ROUTER_OFFCHIP_ROUTEALGO_H

#include "NoC_sim/RouteAlgo.h"
#include "NoC_sim/noc_utils.h"
#include "NoC_sim/noc_param.h"

// ---------------------------------------------------------------------------
// OffChipRouteAlgo (Phase 6a)
//
// Deliberately SEPARATE from RouteAlgo -- an off-chip DRAM node's id
// (LLM_SIM_NOC::OFFCHIP_ID_BASE + row, i.e. 64..71) cannot be expressed as a
// mesh coordinate the way a core id can (RouteAlgo::id2coord's id%8/id/8
// formula is for the 8x8 core mesh only and must never receive an off-chip
// id). RouteAlgo/NoCLat are intentionally left untouched.
//
// Model: an off-chip DRAM node is NOT a point inside the mesh -- it's a
// dedicated I/O channel attached to the rightmost core of its row. The path
// from any core (x,y) to its row's off-chip node is:
//   1. the ordinary congestion-tracked XY mesh path from (x,y) to the edge
//      core (mesh_width-1, y) -- reuses RouteAlgo's existing hop-walk/
//      congestion-path-union logic unmodified (composition, not inheritance).
//   2. a final, fixed-cost "off-chip link" leg from the edge core to the
//      off-chip node itself. This leg is NOT part of the mesh congestion
//      model (see OffChipNoCLat) and is therefore not represented as a
//      Hop_t in the returned path -- callers add the fixed latency
//      separately.
// ---------------------------------------------------------------------------
class OffChipRouteAlgo {
public:
    explicit OffChipRouteAlgo(int mesh_width = LLM_SIM_NOC::MESH_WIDTH);

    // Returns the mesh-side (congestion-tracked) path from src_id (a core id,
    // 0..63) to the EDGE CORE of offchip_id's row -- i.e. core
    // (mesh_width-1, row), NOT the off-chip node itself. The path/union map
    // keys are still core ids; callers that need the result keyed by
    // offchip_id (e.g. for NoCLat::get_latency's dst_id lookup) must remap --
    // see OffChipNoCLat::get_latency, which does this automatically.
    //
    // src_id must be a core id (0..63). offchip_id must be in
    // [OFFCHIP_ID_BASE, OFFCHIP_ID_BASE + NUM_OFFCHIP_NODES). Throws
    // std::invalid_argument on an out-of-range offchip_id (a programming
    // error upstream, not a runtime condition to route around).
    std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t>
    get_path_to_offchip(const int& src_id, const int& offchip_id);

    // Reverse of get_path_to_offchip() above: the mesh-side (congestion-
    // tracked) path from the EDGE CORE of offchip_id's row to dst_id (a real
    // core id, 0..63) -- i.e. the leg an off-chip fetch RESPONSE travels
    // after its fixed off-chip link leg lands it at the edge core. Bug fix
    // (hang investigation, 2026-07-28): before
    // this existed, off-chip response packets (src_id == an off-chip node's
    // own id, 64..71) fell through to NoCSim's ordinary route_algo/noc_lat,
    // which computed src_id's mesh coordinate via id%mesh_width/id/
    // mesh_width -- for an off-chip id this produces an out-of-range row
    // (e.g. id2coord(66) with mesh_width=8 gives y=8, outside the valid 0..7
    // mesh rows), aliasing the computed path through OTHER off-chip nodes'
    // ids as if they were ordinary mesh routers and corrupting their shared
    // alloutq congestion-tracking state. This function (and its caller in
    // NoC_sim.cpp) keeps the entire computation in valid core-id space
    // (edge_id, dst_id) exactly like get_path_to_offchip() does for the
    // forward direction, so this can never happen.
    //
    // dst_id must be a core id (0..63). offchip_id must be in
    // [OFFCHIP_ID_BASE, OFFCHIP_ID_BASE + NUM_OFFCHIP_NODES) -- same
    // validation as get_path_to_offchip() (edge_core_id() throws).
    std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t>
    get_path_from_offchip(const int& offchip_id, const int& dst_id);

    // The core id of the rightmost core in offchip_id's row -- the fixed
    // "edge" every path funnels through before the final off-chip leg.
    int edge_core_id(int offchip_id) const;

    static bool is_offchip_id(int id) { return id >= LLM_SIM_NOC::OFFCHIP_ID_BASE; }

private:
    RouteAlgo route_algo_; // composed; reuses the existing XY hop-walk as-is
    const int mesh_width_;
};

#endif // LLM_SIM_ROUTER_OFFCHIP_ROUTEALGO_H
