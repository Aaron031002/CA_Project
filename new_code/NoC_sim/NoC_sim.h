#ifndef LLM_SIM_NOC_SIM_H
#define LLM_SIM_NOC_SIM_H

#include <cstdlib>
#include <iostream>
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <systemc.h>

// #include "Router.h"
#include "NoC_sim/noc_param.h"
#include "NoC_sim/noc_utils.h"
#include "NoC_sim/NoCLat.h"
#include "NoC_sim/RouteAlgo.h"
#include "NoC_sim/OffChipRouteAlgo.h"
#include "NoC_sim/OffChipNoCLat.h"
#include "utils/fifo.h"

/*
 * NoC Simulator for LLM workloads
 * Features:
 * - Adaptive routing with congestion awareness
 * - SystemC-based cycle-accurate simulation
 * - Configurable mesh topology
 * - Performance statistics collection
 * - Multiple routing algorithms (XY, Adaptive, Congestion-aware)
 */


template <typename T>
class NoCSim : public sc_module {
    public:
        // mesh_width/mesh_height: core mesh dimensions (X/Y). Both default to
        // the hardware.yaml-derived LLM_SIM_NOC::NOC_WIDTH/NOC_HEIGHT, so any
        // caller that doesn't pass them keeps today's configured size.
        // Rectangular (mesh_width != mesh_height) meshes are supported for
        // both "XY" and "TORUS_XY" topology -- RouteAlgo::xy_route's hop-walk
        // only ever needs the width for id<->coord conversion, and
        // torus_xy_route wraps the X axis on mesh_width and the Y axis on
        // mesh_height independently (see RouteAlgo's own constructor/
        // torus_xy_route comments), so a rectangular torus wraps correctly
        // on both axes, not just a square one.
        // topology: "XY" (default plain mesh) or "TORUS_XY" (2D mesh torus with wrap-around routing)
        //   -- applies to the core mesh only; off-chip DRAM nodes (Phase 6)
        //      never use this routing algorithm, see OffChipRouteAlgo.
        // num_offchip_nodes: number of off-chip DRAM nodes appended after the
        //   mesh_width*mesh_height core ids (ids mesh_width*mesh_height ..
        //   +num_offchip_nodes-1). Default LLM_SIM_NOC::NUM_OFFCHIP_NODES
        //   (8, one per mesh row) -- independent of mesh_height; callers
        //   changing mesh_height should reconsider this default themselves.
        // endpoint_port: research switch S1, the endpoint rule (default from
        //   hardware.yaml noc.endpoint_rule; `transit` = false = today's rule).
        //   false: send_busy / recv_busy hold a source and every destination
        //     for a packet's WHOLE transit (one packet per endpoint in flight).
        //   true:  each node has an INJECT and an EJECT port, reserved like a
        //     link port (same window as the first / last hop's link service);
        //     a source admits its next packet once this one has left its inject
        //     port, a destination takes packets back to back. Core-to-core only:
        //     an off-chip packet is FATAL under it.
        // mc_tree: research switch S2, multicast tree reservation (default from
        //   hardware.yaml noc.multicast_tree; false = today). Today a link shared
        //   by k destinations of one multicast packet is reserved k times; with
        //   it, once (see NoCLat::get_latency_port). Applies under both rules.
        // mc_fork_async: S2's fork mode (default from hardware.yaml
        //   noc.multicast_fork; false = `sync`; needs mc_tree and endpoint_port).
        //   sync: forwarding through an intermediate destination waits for its
        //   eject; async: it does not, the local copy is ejected apart.
        // pf_fix: research switch S3, the port_finish_time fix (default from
        //   hardware.yaml noc.port_finish_fix; false = today). Today an entry
        //   already in service is restarted at `now` and charged its full
        //   duration again; with it a port is free when its last entry ends.
        //   Applies under both rules and to off-chip packets (they share alloutq).
        // backfill: research switch S4 (default from hardware.yaml noc.backfill;
        //   false = today; needs endpoint_port). Each hop takes the earliest gap
        //   on its port(s) that fits its whole service, instead of the end of the
        //   port's queue (see NoCLat::get_latency_port); every finished entry
        //   leaves its queue each cycle, not only those at the front.
    explicit NoCSim(sc_module_name name, int mesh_width = LLM_SIM_NOC::NOC_WIDTH,
                   int mesh_height = LLM_SIM_NOC::NOC_HEIGHT,
                   const std::string& topology = "XY",
                   int num_offchip_nodes = LLM_SIM_NOC::NUM_OFFCHIP_NODES,
                   bool backpressure_enabled = true,
                   bool endpoint_port = LLM_SIM_NOC::ENDPOINT_PORT_RULE,
                   bool mc_tree = LLM_SIM_NOC::MULTICAST_TREE,
                   bool mc_fork_async = LLM_SIM_NOC::MULTICAST_FORK_ASYNC,
                   bool pf_fix = LLM_SIM_NOC::PORT_FINISH_FIX,
                   bool backfill = LLM_SIM_NOC::BACKFILL);
        // IO ports
        sc_in_clk SC_NAMED(clk);
        sc_vector<sc_fifo_in<LLM_SIM_NOC::PacketPtr_t<T>>> in_ports;
        sc_vector<fifo_out<LLM_SIM_NOC::PacketPtr_t<T>>> out_fifos;

    // ── Utilization tracking (readable after sc_stop) ─────────────────────────
    uint64_t busy_cycle_ = 0;
    double utilization() const {
        const uint64_t total = static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        return total > 0 ? static_cast<double>(busy_cycle_) / static_cast<double>(total) : 0.0;
    }
    uint64_t busy_cycles() const { return busy_cycle_; }

    // ── Backpressure tracking (readable after sc_stop) ─────────────────────────
    // stall_cycles_: cycles where a pending packet was held back at admission
    // because some port on its path was at BUFFER_DEPTH occupancy.
    // occupancy_sum_/occupancy_samples_: running average of per-port queue
    // depth across all (router,dir) pairs, sampled once per cycle.
    // peak_inject_depth_: high-water mark of any single source's input_buffer.
    uint64_t stall_cycles_ = 0;
    uint64_t max_port_occupancy_ = 0;
    uint64_t occupancy_sum_ = 0;
    uint64_t occupancy_samples_ = 0;
    uint64_t peak_inject_depth_ = 0;
    uint64_t stall_cycles() const { return stall_cycles_; }
    uint64_t max_port_occupancy() const { return max_port_occupancy_; }
    double avg_port_occupancy() const {
        return occupancy_samples_ > 0
            ? static_cast<double>(occupancy_sum_) / static_cast<double>(occupancy_samples_)
            : 0.0;
    }
    uint64_t peak_inject_depth() const { return peak_inject_depth_; }
    bool backpressure_enabled() const { return bp_enabled_; }
    bool endpoint_port() const { return endpoint_port_; }
    bool mc_tree() const { return mc_tree_; }
    bool mc_fork_async() const { return mc_fork_async_; }
    bool pf_fix() const { return pf_fix_; }
    bool backfill() const { return backfill_; }

    // ── R1 counter breakdown
    // (numerical cross-check extension): NoC hop-flight time, aggregated
    // across every packet this NoCSim instance ever admitted, readable
    // after sc_stop() alongside busy_cycles()/stall_cycles() above.
    // total_hop_flight_cyc_ sums each admitted packet's own hop_svc_cycle()
    // (the exact cycle count that packet occupies the NoC in flight, same
    // value insert_alloutq() uses for congestion tracking -- see run()'s
    // own admission block) -- NOT wall-clock send-to-receive latency (which
    // would also include admission-queue stall time, already separately
    // counted by stall_cycles_ above). Deliberately env-gated off by
    // default (DBG_R1_COUNTERS) -- see run()'s own accumulation site for
    // why this is a single aggregate, not per-core.
    uint64_t total_hop_flight_cyc_ = 0;
    uint64_t total_packets_admitted_ = 0;
    uint64_t total_hop_flight_cycles() const { return total_hop_flight_cyc_; }
    uint64_t total_packets_admitted() const { return total_packets_admitted_; }

    private:
        const bool bp_enabled_;
        const bool endpoint_port_;
        const bool mc_tree_;
        const bool mc_fork_async_;
        const bool pf_fix_;
        const bool backfill_;
        // (TOKENS_PER_CORE>1 hang investigation,
        // 2026-08-12): DBG_NOC_TRACE=1 dumps every admission attempt/stall/
        // completion this run() loop makes, so a hang can be reconstructed as
        // an explicit wait-for graph (src waiting on busy dst(s)) instead of
        // just observing "core X stuck" from the Core-side SYNC-STUCK dump.
        // getenv() is cached once here rather than called every cycle inside
        // run() (which is a SC_THREAD hot loop).
        const bool dbg_trace_ = (std::getenv("DBG_NOC_TRACE") != nullptr);
        // R1 counter breakdown (see total_hop_flight_cyc_'s own doc
        // comment above) -- separate flag from dbg_trace_ since this is a
        // cheap unconditional counter increment (no string formatting/I-O),
        // not a print, but still gated so a normal run pays zero cost.
        const bool dbg_r1_counters_ = (std::getenv("DBG_R1_COUNTERS") != nullptr);
        std::shared_ptr<NoCLat> noc_lat;
        std::shared_ptr<RouteAlgo> route_algo;
        // Phase 6: off-chip DRAM nodes never use route_algo/noc_lat above --
        // see OffChipRouteAlgo/OffChipNoCLat header comments for why.
        std::shared_ptr<OffChipRouteAlgo> offchip_route_algo;
        std::shared_ptr<OffChipNoCLat>    offchip_noc_lat;
        LLM_SIM_NOC::AllOutQ_t alloutq;
        const int mesh_width_;
        const int mesh_height_;
        const int num_offchip_nodes_;
        const int node_num_;
        // Cycle counter driving all alloutq/get_latency "now" comparisons.
        // Deliberately decoupled from sc_time_stamp() -- the latter advances by
        // the raw simulation-time-resolution tick count per posedge (e.g. 1000
        // for a 1ns clock at the SystemC-default 1ps resolution), which dwarfs
        // the small "cycle count" scale of hop_svc_cycle()/FLIT_CYCLE. Using it
        // directly as "now" made every alloutq port-occupancy entry expire
        // before the next posedge's check_alloutq ever saw it, so congestion
        // could only ever be observed within a single admission cycle. This
        // counter increments by exactly 1 per posedge, matching the cycle-scale
        // units used everywhere else in the latency model, so a port's queued
        // occupancy now correctly persists across cycles until it's served.
        uint64_t cycle_count_ = 0;
        // busy state, one entry per node (sized to node_num_ in the
        // constructor, so the mesh size has no compile-time limit)
        std::vector<bool> send_busy;
        std::vector<bool> recv_busy;
        // Off-chip DRAM nodes, if this instance has any, take the ids right
        // after its cores: [mesh_width_*mesh_height_, node_num_). Decided per
        // instance rather than with the compile-time OFFCHIP_ID_BASE, which
        // is only the default (hardware.yaml) mesh's core count.
        bool is_offchip(int id) const {
            return num_offchip_nodes_ > 0 && id >= mesh_width_ * mesh_height_;
        }
        std::unordered_map<int, LLM_SIM_NOC::PacketPtr_t<T>> pending_packets;
        std::unordered_map<int, LLM_SIM_NOC::TransState_t> trans_states;
        // Input buffer: drain in_ports eagerly each cycle to avoid losing packets
        std::unordered_map<int, std::queue<LLM_SIM_NOC::PacketPtr_t<T>>> input_buffer;
        // Endpoint rule `port` (endpoint_port_) only: per-node inject / eject
        // port queues, kept apart from alloutq so its per-port occupancy
        // statistics are unchanged; the cycle each source may admit again;
        // the packets in flight, keyed by admission order (several per source).
        struct InFlight_t {
            LLM_SIM_NOC::PacketPtr_t<T> packet;
            std::unordered_map<int, uint64_t> dst2time; // dst_id to cycles left
        };
        std::vector<LLM_SIM_NOC::OutQ_t> inject_q_;
        std::vector<LLM_SIM_NOC::OutQ_t> eject_q_;
        std::vector<uint64_t> release_at_;
        std::map<uint64_t, InFlight_t> in_flight_;
        uint64_t admit_seq_ = 0;
        // Buffers for each node
        void run();
        // one source's admission attempt under the `port` rule (run()'s src loop)
        void admit_port(int src_num);
        void complete_port();
        void check_alloutq(LLM_SIM_NOC::AllOutQ_t& alloutq, uint64_t now);
        void insert_alloutq(LLM_SIM_NOC::AllOutQ_t& alloutq, const std::vector<LLM_SIM_NOC::UnionHop_t>& hop_vec, const uint64_t packet_remaining_time);
        // helper function
        uint64_t hop_svc_cycle(const int& flit_size) const;
};

#endif