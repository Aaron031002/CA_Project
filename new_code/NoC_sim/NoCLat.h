#ifndef LLM_SIM_NOCLAT_H
#define LLM_SIM_NOCLAT_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <systemc.h>

#include "NoC_sim/noc_utils.h"

class NoCLat {
    public:
        // pf_fix: research switch S3 (NoCSim's pf_fix), see port_finish_time().
        explicit NoCLat(int mesh_width = LLM_SIM_NOC::MESH_WIDTH, bool pf_fix = false);
        // user APIs
        // `now` is the caller's cycle counter (NOT sc_time_stamp() -- see
        // NoCSim::cycle_count_ for why: hop_svc_cycle()/FLIT_CYCLE are small
        // cycle-scale integers, and comparing them against raw sc_time_stamp()
        // ticks made every alloutq entry look expired by the very next posedge,
        // silently disabling cross-cycle congestion tracking). Must match the
        // units used to populate `alloutq` (via NoCSim::insert_alloutq) and to
        // expire it (via NoCSim::check_alloutq).
        // mc_tree: research switch S2 (NoCSim's mc_tree), see get_latency_port.
        template <typename T>
        std::pair<std::unordered_map<int, uint64_t>, std::vector<LLM_SIM_NOC::UnionHop_t>>
        get_latency(const LLM_SIM_NOC::AllOutQ_t& alloutq, const LLM_SIM_NOC::Packet_t<T>& packet, const LLM_SIM_NOC::AllPath_t& allpath, uint64_t now,
                    bool mc_tree = false) const {
            std::unordered_map<int, uint64_t> latencies;
            std::unordered_map<int, LLM_SIM_NOC::UnionHop_t> hop_map;
            std::map<std::pair<int, int>, uint64_t> tree; // S2: (router, dir) -> arrival after it
            // check all unicast paths from the same source
            for(size_t dst_idx = 0; dst_idx < packet.dst_ids.size(); dst_idx++) {
                const int dst_id = packet.dst_ids[dst_idx];
                if(allpath.find(dst_id) == allpath.end()) continue;
                const auto& path = allpath.at(dst_id);
                uint64_t current_arrival = now; // actual arrival time at current router node
                for(const auto& hop: path) { // congestion estimation with accumulated arrival
                    if(mc_tree) {
                        const auto m = tree.find({hop.curr_id, static_cast<int>(hop.dir)});
                        if(m != tree.end()) { current_arrival = m->second; continue; }
                    }
                    // Add boundary check for alloutq access
                    if(hop.curr_id >= 0 && static_cast<size_t>(hop.curr_id) < alloutq.size() && 
                       static_cast<int>(hop.dir) >= 0 && static_cast<size_t>(hop.dir) < alloutq[hop.curr_id].size()) {
                        auto port_finish = port_finish_time(alloutq[hop.curr_id][static_cast<size_t>(hop.dir)], now);
                        uint64_t service_start = std::max(current_arrival, port_finish);
                        uint64_t departure = service_start + hop_svc_cycle(packet.flit_size);
                        // Merge into hop_map: one UnionHop_t per router
                        if(hop_map.find(hop.curr_id) == hop_map.end()) {
                            hop_map[hop.curr_id].curr_id = hop.curr_id;
                        }
                        // Store the ABSOLUTE cycle service begins on this port
                        // (matches OutEntry_t::start_at) so insert_alloutq/
                        // check_alloutq/port_finish_time all agree on units.
                        hop_map[hop.curr_id].dir_dist.emplace_back(hop.dir, service_start);
                        current_arrival = departure + LLM_SIM_NOC::FLIT_CYCLE; // link traversal to next router
                        if(mc_tree) tree[{hop.curr_id, static_cast<int>(hop.dir)}] = current_arrival;
                    } else {
                        std::cerr << "[ERROR] Invalid alloutq access: hop.curr_id=" << hop.curr_id 
                                  << ", hop.dir=" << static_cast<int>(hop.dir) 
                                  << ", alloutq.size()=" << alloutq.size() << std::endl;
                    }
                }
                latencies[dst_id] = current_arrival - now;
            }
            // Convert hop_map to vector
            std::vector<LLM_SIM_NOC::UnionHop_t> hop_vec;
            hop_vec.reserve(hop_map.size());
            for(auto& [id, uhop] : hop_map) {
                hop_vec.push_back(std::move(uhop));
            }
            return {latencies, hop_vec};
        }

        // Research switch S1 (endpoint rule `port`; NoCSim's endpoint_port): the
        // same walk as get_latency(), plus each node's INJECT and EJECT port.
        // The first hop of every destination's path also waits for the source's
        // inject port, the last hop for that destination's eject port, in the
        // SAME window as the hop's link service (an uncontended packet's latency
        // is unchanged). get_latency() above is left untouched for the default
        // rule. Returns, besides the latencies and the link hops, the inject
        // window's start (the earliest first-hop start; `has_inject` is false
        // when every destination is the source itself) and each destination's
        // eject window start; the caller reserves both for hop_svc_cycle().
        //
        // mc_tree: research switch S2. Today a link shared by k destinations of
        // one multicast is walked and reserved k times (each destination's path
        // alone, duplicates kept in hop_map). With it each tree link is computed
        // and reserved once: a destination whose walk reaches a link already in
        // the tree continues from that link's arrival. XY from one source is a
        // tree (each router entered by one link), so under the port rule a tree
        // link that enters one of the packet's destinations also holds that
        // destination's eject port, whichever destination's walk reaches it
        // first; each destination's eject is reserved at most once.
        //
        // backfill: research switch S4 (needs the port rule). Today a hop starts
        // after the END of every reservation already on its port, so a later
        // packet never uses an idle gap before an earlier packet's (later)
        // window and links idle while packets wait. With it each hop takes the
        // EARLIEST GAP, at or after its arrival, in which its whole service fits
        // on all its resources (link, plus inject on the first hop, plus eject
        // when it holds one). Still reserved at admission; an earlier
        // reservation is never moved (conservative backfilling).
        //
        // async_fork: S2's fork mode (NoCSim's mc_fork_async; needs mc_tree). With
        // mc_tree alone a tree link entering an INTERMEDIATE destination (one the
        // tree passes through to reach others) holds that destination's eject in
        // the same window: forwarding waits on ejection (a synchronous fork). With
        // async_fork the link does not wait; the local copy takes the eject's
        // earliest window at or after the link's (the same window when the eject
        // is free) and that destination's arrival is the eject window's end + one
        // link traversal. Leaf destinations keep the coupling.
        struct PortLatency_t {
            std::unordered_map<int, uint64_t> latencies;
            std::vector<LLM_SIM_NOC::UnionHop_t> hop_vec;
            bool has_inject = false;
            uint64_t inject_start = 0;
            std::vector<std::pair<int, uint64_t>> eject_start; // (dst, start), dst order
        };
        template <typename T>
        PortLatency_t
        get_latency_port(const LLM_SIM_NOC::AllOutQ_t& alloutq, const LLM_SIM_NOC::OutQ_t& inject_q,
                         const std::vector<LLM_SIM_NOC::OutQ_t>& eject_q,
                         const LLM_SIM_NOC::Packet_t<T>& packet, const LLM_SIM_NOC::AllPath_t& allpath,
                         uint64_t now, bool mc_tree = false, bool backfill = false,
                         bool async_fork = false) const {
            PortLatency_t out;
            std::unordered_map<int, LLM_SIM_NOC::UnionHop_t> hop_map;
            std::map<std::pair<int, int>, uint64_t> tree; // S2: (router, dir) -> arrival after it
            std::set<int> ejected;                        // S2: destinations whose eject is reserved
            std::set<int> inter;                          // S2 async: destinations passed through
            std::map<int, uint64_t> deliver;              // S2 async: a forked copy's arrival
            if(mc_tree && async_fork) {
                for(int d0 : packet.dst_ids) {
                    const auto p0 = allpath.find(d0);
                    if(p0 == allpath.end()) continue;
                    for(size_t h0 = 0; h0 + 1 < p0->second.size(); ++h0) {
                        const int r0 = p0->second[h0].curr_id;
                        if(std::find(packet.dst_ids.begin(), packet.dst_ids.end(), r0) != packet.dst_ids.end())
                            inter.insert(r0);
                    }
                }
            }
            const uint64_t inject_finish = port_finish_time(inject_q, now);
            for(size_t dst_idx = 0; dst_idx < packet.dst_ids.size(); dst_idx++) {
                const int dst_id = packet.dst_ids[dst_idx];
                if(allpath.find(dst_id) == allpath.end()) continue;
                const auto& path = allpath.at(dst_id);
                uint64_t current_arrival = now;
                for(size_t h = 0; h < path.size(); ++h) {
                    const auto& hop = path[h];
                    if(mc_tree) {
                        const auto m = tree.find({hop.curr_id, static_cast<int>(hop.dir)});
                        if(m != tree.end()) { current_arrival = m->second; continue; }
                    }
                    // the destination whose eject port this hop holds, or -1
                    int ej = -1;
                    if(mc_tree) {
                        if(!ejected.count(hop.curr_id) &&
                           std::find(packet.dst_ids.begin(), packet.dst_ids.end(), hop.curr_id) != packet.dst_ids.end())
                            ej = hop.curr_id;
                    } else if(h + 1 == path.size()) {
                        ej = dst_id;
                    }
                    int fork = -1;                        // S2 async: the copy ejected apart
                    if(ej >= 0 && inter.count(ej)) { fork = ej; ej = -1; }
                    if(hop.curr_id >= 0 && static_cast<size_t>(hop.curr_id) < alloutq.size() &&
                       static_cast<int>(hop.dir) >= 0 && static_cast<size_t>(hop.dir) < alloutq[hop.curr_id].size()) {
                        const auto& link_q = alloutq[hop.curr_id][static_cast<size_t>(hop.dir)];
                        uint64_t service_start;
                        if(backfill) {
                            std::vector<const LLM_SIM_NOC::OutQ_t*> res{&link_q};
                            if(h == 0) res.push_back(&inject_q);
                            if(ej >= 0) res.push_back(&eject_q[static_cast<size_t>(ej)]);
                            service_start = earliest_gap(res, current_arrival, hop_svc_cycle(packet.flit_size));
                        } else {
                            service_start = std::max(current_arrival, port_finish_time(link_q, now));
                            if(h == 0) service_start = std::max(service_start, inject_finish);
                            if(ej >= 0)
                                service_start = std::max(service_start,
                                                         port_finish_time(eject_q[static_cast<size_t>(ej)], now));
                        }
                        if(ej >= 0) {
                            out.eject_start.emplace_back(ej, service_start);
                            ejected.insert(ej);
                        }
                        if(fork >= 0) {
                            const auto& fq = eject_q[static_cast<size_t>(fork)];
                            const uint64_t es = backfill
                                ? earliest_gap({&fq}, service_start, hop_svc_cycle(packet.flit_size))
                                : std::max(service_start, port_finish_time(fq, now));
                            out.eject_start.emplace_back(fork, es);
                            ejected.insert(fork);
                            deliver[fork] = es + hop_svc_cycle(packet.flit_size) + LLM_SIM_NOC::FLIT_CYCLE;
                        }
                        if(h == 0) {
                            out.inject_start = out.has_inject ? std::min(out.inject_start, service_start)
                                                              : service_start;
                            out.has_inject = true;
                        }
                        uint64_t departure = service_start + hop_svc_cycle(packet.flit_size);
                        if(hop_map.find(hop.curr_id) == hop_map.end()) {
                            hop_map[hop.curr_id].curr_id = hop.curr_id;
                        }
                        hop_map[hop.curr_id].dir_dist.emplace_back(hop.dir, service_start);
                        current_arrival = departure + LLM_SIM_NOC::FLIT_CYCLE;
                        if(mc_tree) tree[{hop.curr_id, static_cast<int>(hop.dir)}] = current_arrival;
                    } else {
                        std::cerr << "[ERROR] Invalid alloutq access: hop.curr_id=" << hop.curr_id
                                  << ", hop.dir=" << static_cast<int>(hop.dir)
                                  << ", alloutq.size()=" << alloutq.size() << std::endl;
                    }
                }
                out.latencies[dst_id] = current_arrival - now;
            }
            for(const auto& [x, tx] : deliver) out.latencies[x] = tx - now;
            out.hop_vec.reserve(hop_map.size());
            for(auto& [id, uhop] : hop_map) {
                out.hop_vec.push_back(std::move(uhop));
            }
            return out;
        }
    private:
        // latency data structure
        // assitant methods
        uint64_t physical_latency(const int& src, const int& dst) const;
        uint64_t port_finish_time(const LLM_SIM_NOC::OutQ_t& queue, uint64_t now) const;
        // S4: the earliest start >= t at which [start, start + s) overlaps no
        // entry of any of the queues (each entry occupies [start_at, start_at + dur)).
        uint64_t earliest_gap(const std::vector<const LLM_SIM_NOC::OutQ_t*>& queues, uint64_t t, uint64_t s) const;
        uint64_t hop_svc_cycle(const int& flit_size) const;
        // distance calculations
        inline uint64_t mandist(const int & src_id, const int & dst_id) const;
        // id to coord mapping functions
        inline std::pair<int, int> id2coor(const int& id) const;

        const int mesh_width_;
        const bool pf_fix_;
};

#endif