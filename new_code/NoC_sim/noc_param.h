#ifndef LLM_SIM_NOC_PARAM_H
#define LLM_SIM_NOC_PARAM_H

// AUTO-GENERATED from config/hardware.yaml by tools/gen_config_headers.py.
// Do not edit by hand -- edit the YAML and re-run the generator instead.

#include <systemc.h>

namespace LLM_SIM_NOC {
    constexpr int NOC_WIDTH = 8;      // Network width (X dimension), core mesh only
    constexpr int NOC_HEIGHT = 8;     // Network height (Y dimension), core mesh only
    constexpr int NUM_OFFCHIP_NODES = 8; // Phase 6: one off-chip DRAM node per mesh row
    // Total NoCSim node count = 64 cores + 8 off-chip DRAM nodes.
    // Off-chip nodes use a SEPARATE id range (NOC_WIDTH*NOC_HEIGHT .. +NUM_OFFCHIP_NODES-1,
    // i.e. 64..71) -- they are deliberately NOT part of the 8x8 core mesh's
    // coordinate system (RouteAlgo::id2coord's id%8/id/8 formula is untouched
    // and never receives an off-chip id). See OffChipRouteAlgo/OffChipNoCLat.
    // NODE_NUM is only this DEFAULT mesh's node count, not a limit: NoCSim
    // sizes all of its per-node state from the mesh it is constructed with,
    // so an Acc8x8 mesh (no off-chip nodes) may have any width x height.
    constexpr int NODE_NUM = NOC_WIDTH * NOC_HEIGHT + NUM_OFFCHIP_NODES;
    constexpr int NOC_LANES = 8;      // Data lanes per link
    constexpr int BUFFER_DEPTH = 4;   // Router buffer depth
    constexpr int PACKET_SIZE = 32;   // Default packet size in flits
    constexpr int MAX_HOPS = 9;       // Max mesh-side hops (N-1 per dim, XY, no torus wrap):
                                       // worst case is corner-to-corner = (NOC_WIDTH-1)+(NOC_HEIGHT-1) = 14;
                                       // kept as a soft/documentation constant, not enforced anywhere today.

    // Phase 6: off-chip DRAM node id range. A packet destination id >= this
    // base is an off-chip node, never a core -- OffChipRouteAlgo/OffChipNoCLat
    // handle these; RouteAlgo/NoCLat only ever see ids in [0, OFFCHIP_ID_BASE).
    // Off-chip nodes therefore need a mesh of exactly OFFCHIP_ID_BASE cores
    // (NoCSim checks this); meshes without off-chip nodes are unaffected.
    constexpr int OFFCHIP_ID_BASE = NOC_WIDTH * NOC_HEIGHT; // 64

    // Fixed, congestion-independent latency (cycles) for the final "off-chip
    // link" leg (edge-of-mesh core -> its row's off-chip DRAM node). This
    // leg is NOT tracked in alloutq -- it models a dedicated physical
    // channel, not a shared mesh link, per the Phase 6 design decision.
    constexpr int OFFCHIP_LINK_LATENCY = 20;

    // Research switch S1 (noc.endpoint_rule): NoCSim's default endpoint_port.
    // false = `transit` (today: a source and every destination are held for a
    // packet's whole transit); true = `port` (per-node inject / eject ports).
    constexpr bool ENDPOINT_PORT_RULE = false;

    // Research switch S2 (noc.multicast_tree): NoCSim's default mc_tree.
    // false = today (a link shared by k destinations of one multicast is
    // reserved k times); true = each tree link once.
    constexpr bool MULTICAST_TREE = false;

    // S2's fork mode (noc.multicast_fork): NoCSim's default mc_fork_async.
    // false = `sync` (forwarding through an intermediate destination waits for
    // its eject); true = `async` (the local copy is ejected apart).
    constexpr bool MULTICAST_FORK_ASYNC = false;

    // Research switch S3 (noc.port_finish_fix): NoCSim's default pf_fix.
    // false = today (an entry already in service is restarted at `now` and
    // charged its full duration again); true = a port is free when its last
    // entry ends.
    constexpr bool PORT_FINISH_FIX = false;

    // Research switch S4 (noc.backfill): NoCSim's default backfill (needs
    // ENDPOINT_PORT_RULE). false = today (a hop starts after the end of its
    // port's queue); true = the earliest gap that fits its whole service.
    constexpr bool BACKFILL = false;

    // Phase 7b: packet_id namespace reserved for DMA-internal
    // off-chip fetch/write-back requests (constructed inside DMA itself,
    // not MainCtrl/RPCDispatch's runtime pool -- see main_ctrl_params.h's
    // DYNAMIC_PKT_BASE=50000/DYNAMIC_POOL_SIZE=2048, capping RPC's range at
    // 50000+64*2048=181072;
    // DYNAMIC_PKT_BASE raised from 10000 to 50000 on 2026-08-12 after a real
    // packet_id collision bug (AGG_GRANT vs the dynamic pool), then
    // DYNAMIC_POOL_SIZE raised from 256 to 2048 on 2026-08-13 (Bug F) after a
    // SECOND, independent collision -- see main_ctrl_params.h's own
    // DYNAMIC_PKT_BASE/DYNAMIC_POOL_SIZE comments for both incidents.
    // Deliberately far above that ceiling (and far
    // below LLM_SIM_MAINCTRL::DYNAMIC_DATA_ID_BASE=1000000, a different
    // namespace entirely -- data_id, not packet_id -- so there is no
    // numeric overlap risk even if the two were ever compared by mistake).
    // Each core gets a fixed-size circular block:
    //   [OFFCHIP_FETCH_PKT_ID_BASE + core_id*OFFCHIP_FETCH_PKT_ID_POOL_SIZE,
    //    OFFCHIP_FETCH_PKT_ID_BASE + (core_id+1)*OFFCHIP_FETCH_PKT_ID_POOL_SIZE)
    // matching the per-core-partitioned allocator shape MainCtrl already
    // uses for its own RPC pool.
    constexpr uint64_t OFFCHIP_FETCH_PKT_ID_BASE = 20000000;
    constexpr uint64_t OFFCHIP_FETCH_PKT_ID_POOL_SIZE = 64;

    // Phase A: separate packet_id namespace
    // for DMA-internal KV eviction write-backs, so a core's in-flight
    // off-chip fetch and write-back never collide on the same packet_id.
    // Same per-core circular-block shape as OFFCHIP_FETCH_PKT_ID_BASE above.
    constexpr uint64_t OFFCHIP_WRITEBACK_PKT_ID_BASE = 21000000;
    constexpr uint64_t OFFCHIP_WRITEBACK_PKT_ID_POOL_SIZE = 64;

    // Phase F: synthetic data_id namespace for
    // fill requests (DRAMSim -> DMA -> off-chip -> DRAMSim, no real
    // instruction data_id to land into, unlike Flow A's off-chip fetch which
    // reuses the requesting instruction's own data_id). Same per-core
    // circular-block shape as the two pools above; deliberately far above
    // LLM_SIM_MAINCTRL::DYNAMIC_DATA_ID_BASE=1000000 (max range
    // 1000000+64*256=1016384) so there is no numeric overlap risk.
    constexpr uint64_t FILL_DATA_ID_BASE = 2000000;
    constexpr uint64_t FILL_DATA_ID_POOL_SIZE = 64;

    // Congestion thresholds
    constexpr double CONGESTION_THRESHOLD_LOW = 0.3;
    constexpr double CONGESTION_THRESHOLD_HIGH = 0.7;

    // Timing parameters
    constexpr int ROUTER_DELAY = 1;   // Cycles per router hop
    constexpr int LINK_DELAY = 1;     // Cycles per link traversal
}

#endif
