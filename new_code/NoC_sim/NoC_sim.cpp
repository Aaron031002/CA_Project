#include "NoC_sim/NoC_sim.h"


template <typename T>
NoCSim<T>::NoCSim(sc_module_name name, int mesh_width, int mesh_height,
                   const std::string& topology,
                   int num_offchip_nodes, bool backpressure_enabled, bool endpoint_port, bool mc_tree,
                   bool mc_fork_async, bool pf_fix, bool backfill)
    : sc_module(name), bp_enabled_(backpressure_enabled), endpoint_port_(endpoint_port), mc_tree_(mc_tree),
      mc_fork_async_(mc_fork_async),
      pf_fix_(pf_fix), backfill_(backfill),
      mesh_width_(mesh_width), mesh_height_(mesh_height),
      num_offchip_nodes_(num_offchip_nodes),
      node_num_(mesh_width * mesh_height + num_offchip_nodes) {
    send_busy.assign(node_num_, false);
    recv_busy.assign(node_num_, false);
    // OffChipRouteAlgo/OffChipNoCLat/OffchipMapper number the off-chip nodes
    // from the compile-time LLM_SIM_NOC::OFFCHIP_ID_BASE (the hardware.yaml
    // mesh's core count), so off-chip nodes are only consistent on a mesh of
    // exactly that many cores. Meshes without off-chip nodes (num_offchip_
    // nodes=0, e.g. Acc8x8) may have any size.
    if (num_offchip_nodes_ > 0 && mesh_width_ * mesh_height_ != LLM_SIM_NOC::OFFCHIP_ID_BASE) {
        SC_REPORT_FATAL("NoCSim",
            ("off-chip DRAM nodes need a mesh of exactly " +
             std::to_string(LLM_SIM_NOC::OFFCHIP_ID_BASE) + " cores (hardware.yaml noc.width x "
             "noc.height), got " + std::to_string(mesh_width_) + "x" +
             std::to_string(mesh_height_)).c_str());
    }
    // Initialize NoCLat and RouteAlgo with shared_ptr (core mesh only).
    // NoCLat/RouteAlgo::id2coord only ever need the mesh WIDTH for
    // id<->coord conversion -- correct for any rectangular mesh under
    // plain XY routing. RouteAlgo also takes mesh_height_ for
    // TORUS_XY's Y-axis wrap-around distance, so a rectangular torus
    // wraps correctly on both axes independently (see RouteAlgo::
    // torus_xy_route).
    noc_lat = std::make_shared<NoCLat>(mesh_width_, pf_fix_);
    route_algo = std::make_shared<RouteAlgo>(topology, mesh_width_, mesh_height_);
    // Phase 6: off-chip DRAM nodes get their own, separate routing/latency
    // classes -- see OffChipRouteAlgo/OffChipNoCLat.
    offchip_route_algo = std::make_shared<OffChipRouteAlgo>(mesh_width_);
    offchip_noc_lat    = std::make_shared<OffChipNoCLat>(mesh_width_, pf_fix_);

    // Initialize alloutq with proper dimensions [node_num_][NUM_DIR]
    alloutq.clear();
    alloutq.resize(node_num_);
    for(size_t i = 0; i < alloutq.size(); i++) {
        alloutq[i].resize(static_cast<size_t>(LLM_SIM_NOC::DIR::NUM_DIR));
    }

    in_ports.init(node_num_);
    out_fifos.init(node_num_);

    if (mc_fork_async_ && !(mc_tree_ && endpoint_port_)) {
        SC_REPORT_FATAL("NoCSim", "multicast_fork async (S2) needs multicast_tree and the endpoint rule `port`");
    }
    if (backfill_ && !endpoint_port_) {
        SC_REPORT_FATAL("NoCSim", "backfill (research switch S4) needs the endpoint rule `port` (S1)");
    }

    // Research switch S1 (endpoint rule `port`): see the constructor's doc comment.
    inject_q_.resize(node_num_);
    eject_q_.resize(node_num_);
    release_at_.assign(node_num_, 0);

    // Register the main process
    SC_THREAD(run);
    sensitive << clk.pos();
}

// main function
template <typename T>
void NoCSim<T>::run() {
    while(true) {
        wait();
        ++cycle_count_;
        check_alloutq(alloutq, cycle_count_); // check if any packet finishes at this cycle and update states
        if (endpoint_port_) { // the inject / eject ports expire like alloutq's ports
            for (auto* qs : {&inject_q_, &eject_q_})
                for (auto& q : *qs)
                    while (!q.empty() && q.front().start_at + q.front().dur <= cycle_count_) q.pop();
        }
        if (backfill_) { // S4: entries are not time-ordered, so every finished one leaves
            auto drop_finished = [this](LLM_SIM_NOC::OutQ_t& q) {
                LLM_SIM_NOC::OutQ_t keep;
                for (; !q.empty(); q.pop())
                    if (q.front().start_at + q.front().dur > cycle_count_) keep.push(q.front());
                q.swap(keep);
            };
            for (auto& router : alloutq) for (auto& q : router) drop_finished(q);
            for (auto* qs : {&inject_q_, &eject_q_}) for (auto& q : *qs) drop_finished(q);
        }

        // ── Sample per-port occupancy for backpressure metrics ────────────────
        for(size_t router_id = 0; router_id < alloutq.size(); ++router_id) {
            for(size_t dir = 0; dir < alloutq[router_id].size(); ++dir) {
                const uint64_t occ = alloutq[router_id][dir].size();
                occupancy_sum_ += occ;
                ++occupancy_samples_;
                if(occ > max_port_occupancy_) max_port_occupancy_ = occ;
            }
        }

        // ── Eagerly drain all in_ports into input_buffer ──────────────────────
        // This ensures packets are not lost if in_ports is consumed between cycles.
        for(size_t src_num = 0; src_num < static_cast<size_t>(node_num_); ++src_num) {
            LLM_SIM_NOC::PacketPtr_t<T> pkt;
            while(in_ports[src_num].nb_read(pkt)) {
                // Every destination must be a node of this mesh: an id from
                // an instruction (e.g. `DMA NoC send <pkt> <data> 99` on a
                // 4x2 mesh) would otherwise index past the per-node state.
                for (int d : pkt->dst_ids) {
                    if (d < 0 || d >= node_num_) {
                        SC_REPORT_FATAL("NoCSim",
                            ("node " + std::to_string(src_num) + " sent packet " +
                             std::to_string(pkt->packet_id) + " to node " + std::to_string(d) +
                             ", but this " + std::to_string(mesh_width_) + "x" +
                             std::to_string(mesh_height_) + " mesh only has nodes 0.." +
                             std::to_string(node_num_ - 1)).c_str());
                        return;
                    }
                }
                input_buffer[src_num].push(pkt);
            }
            const uint64_t depth = input_buffer[src_num].size();
            if(depth > peak_inject_depth_) peak_inject_depth_ = depth;
        }

        for(size_t src_num = 0; src_num < static_cast<size_t>(node_num_); ++src_num) {
            if (endpoint_port_) { admit_port(static_cast<int>(src_num)); continue; }
            if(!this->send_busy[src_num] && pending_packets.find(src_num) == pending_packets.end()) { // if not busy and has packet
                if(input_buffer.find(src_num) == input_buffer.end() || input_buffer[src_num].empty()) {
                    continue; // no packet available
                }
                pending_packets[src_num] = input_buffer[src_num].front();
                input_buffer[src_num].pop();
            }

            else if(!this->send_busy[src_num]) { // found but still not send yet
                bool any_dst_busy = false;
                for(size_t dst_idx = 0; dst_idx < pending_packets[src_num]->dst_ids.size(); ++dst_idx) {
                    int dst_id = pending_packets[src_num]->dst_ids[dst_idx];
                    if(this->recv_busy[dst_id]) {
                        any_dst_busy = true;
                        break;
                    }
                }
                if(any_dst_busy) {
                    // (2026-08-12): report the
                    // FIRST busy dst found above (loop broke there) -- enough
                    // to reconstruct "src is waiting on dst" edges across many
                    // cycles into a wait-for graph, without paying the cost of
                    // rescanning all dst_ids again just for the trace. Gated
                    // to every 1000 cycles per src (not every cycle) -- this
                    // signal already did its job once (proved the NoC layer
                    // completes every multicast) and
                    // was the dominant source of the
                    // earlier trace's multi-million-line volume, drowning out
                    // newer DMA/local_sync signals. STALL_PORT_FULL/ADMIT_OK/
                    // DST_DONE/SRC_DONE stay ungated (already infrequent).
                    if (dbg_trace_ && cycle_count_ % 1000 == 0) {
                        int first_busy_dst = -1;
                        for (int d : pending_packets[src_num]->dst_ids) {
                            if (this->recv_busy[d]) { first_busy_dst = d; break; }
                        }
                        std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                                  << " STALL_DST_BUSY src=" << src_num
                                  << " pkt=" << pending_packets[src_num]->packet_id
                                  << " blocked_by_dst=" << first_busy_dst
                                  << " ndst=" << pending_packets[src_num]->dst_ids.size()
                                  << "\n" << std::flush;
                    }
                    continue; // wait until all destinations are free
                }
                this->send_busy[src_num] = true; // mark src as busy
                for(size_t dst_idx = 0; dst_idx < pending_packets[src_num]->dst_ids.size(); ++dst_idx) {
                    this->recv_busy[pending_packets[src_num]->dst_ids[dst_idx]] = true; // mark all dst as busy
                }
                if (dbg_trace_) {
                    // BUSY_MARKED, not ADMIT: send_busy/recv_busy are set here,
                    // but the has_offchip/src_is_offchip guards and the
                    // any_port_full backpressure check below can still undo
                    // these marks and retry -- see ADMIT_OK/STALL_PORT_FULL
                    // for the point where a packet actually starts moving.
                    std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                              << " BUSY_MARKED src=" << src_num
                              << " pkt=" << pending_packets[src_num]->packet_id
                              << " dsts=[";
                    for (int d : pending_packets[src_num]->dst_ids) std::cerr << d << " ";
                    std::cerr << "]\n" << std::flush;
                }

                // Phase 6: a packet either targets ordinary cores (ids
                // < OFFCHIP_ID_BASE) via the normal route_algo/noc_lat, or
                // exactly one off-chip DRAM node (id >= OFFCHIP_ID_BASE) via
                // OffChipRouteAlgo/OffChipNoCLat. Mixing the two in one
                // packet is not a supported/expected use case (off-chip
                // fetches are point-to-point requests) -- fail loudly rather
                // than silently mis-route if it ever happens.
                bool has_offchip = false, has_core = false;
                for (int d : pending_packets[src_num]->dst_ids) {
                    if (is_offchip(d)) has_offchip = true;
                    else has_core = true;
                }
                if (has_offchip && has_core) {
                    SC_REPORT_FATAL("NoCSim",
                        "packet mixes core and off-chip DRAM destinations in dst_ids -- not supported");
                    // unreachable in production (FATAL aborts); for test overrides that
                    // install a non-aborting handler, bail out of this src_num's
                    // processing rather than falling through with invalid state
                    // (also undo the recv_busy marks above so those dst nodes
                    // don't get stuck busy forever).
                    for (int d : pending_packets[src_num]->dst_ids) this->recv_busy[d] = false;
                    this->send_busy[src_num] = false;
                    pending_packets.erase(src_num);
                    continue;
                }

                // Bug fix (hang investigation,
                // 2026-07-28): a packet whose SOURCE is an off-chip node id
                // (64..71) is an off-chip fetch RESPONSE traveling back to
                // the original requesting core. This is a third, distinct
                // case from the has_offchip (core -> off-chip) and plain
                // core -> core cases above/below -- it must be checked
                // FIRST and routed via OffChipRouteAlgo/OffChipNoCLat's
                // reverse-direction methods, never the plain route_algo/
                // noc_lat (which computes src_num's mesh coordinate via
                // id%mesh_width/id/mesh_width -- for an off-chip id this
                // produces an out-of-range row, e.g. id2coord(66) with
                // mesh_width=8 gives y=8, aliasing the computed path through
                // OTHER off-chip nodes' ids as if they were ordinary mesh
                // routers and corrupting their shared alloutq congestion
                // state). Confirmed via direct instrumentation: a single
                // in-flight off-chip response never surfaced this (no
                // congestion to collide with), but 2+ simultaneous ones
                // corrupted each other's alloutq entries badly enough to
                // permanently deadlock the entire off-chip subsystem.
                const bool src_is_offchip = is_offchip(static_cast<int>(src_num));
                if (src_is_offchip && (has_offchip || pending_packets[src_num]->dst_ids.size() != 1)) {
                    SC_REPORT_FATAL("NoCSim",
                        "off-chip-source packet must target exactly one core (never another off-chip node)");
                    // unreachable in production (FATAL aborts); for test overrides, see above.
                    for (int d : pending_packets[src_num]->dst_ids) this->recv_busy[d] = false;
                    this->send_busy[src_num] = false;
                    pending_packets.erase(src_num);
                    continue;
                }

                std::unordered_map<int, uint64_t> lat;
                std::vector<LLM_SIM_NOC::UnionHop_t> hop_vec;
                if (src_is_offchip) {
                    const int dst_core = pending_packets[src_num]->dst_ids[0];
                    auto [pu, ap] = offchip_route_algo->get_path_from_offchip(static_cast<int>(src_num), dst_core);
                    (void)pu; // unused, mirrors the discarded PathUnion_t elsewhere in this function
                    std::tie(lat, hop_vec) =
                        offchip_noc_lat->get_latency_from_offchip(alloutq, *(pending_packets[src_num]), ap,
                                                                    static_cast<int>(src_num), cycle_count_);
                } else if (has_offchip) {
                    if (pending_packets[src_num]->dst_ids.size() != 1) {
                        SC_REPORT_FATAL("NoCSim",
                            "off-chip DRAM packets must target exactly one destination");
                        // unreachable in production (FATAL aborts); for test overrides, see above.
                        for (int d : pending_packets[src_num]->dst_ids) this->recv_busy[d] = false;
                        this->send_busy[src_num] = false;
                        pending_packets.erase(src_num);
                        continue;
                    }
                    const int offchip_id = pending_packets[src_num]->dst_ids[0];
                    auto [pu, ap] = offchip_route_algo->get_path_to_offchip(src_num, offchip_id);
                    (void)pu; // unused, mirrors route_algo->get_path()'s discarded PathUnion_t below
                    std::tie(lat, hop_vec) =
                        offchip_noc_lat->get_latency(alloutq, *(pending_packets[src_num]), ap, offchip_id, cycle_count_);
                } else {
                    // get latency and path for all receiver
                    auto [pathunion, allpath] = route_algo->get_path(src_num, pending_packets[src_num]->dst_ids);
                    (void)pathunion; // unused (was already discarded before Phase 6)
                    std::tie(lat, hop_vec) = noc_lat->get_latency(alloutq, *(pending_packets[src_num]), allpath, cycle_count_, mc_tree_);
                }

                // ── Backpressure: reject admission if any port along the path
                // is already at BUFFER_DEPTH occupancy. Undo the busy marks set
                // above and leave the packet in pending_packets so it retries
                // next cycle once the downstream port drains (finite buffer,
                // no route/latency state was mutated -- alloutq/hop_svc_time are
                // computed fresh on the retry).
                if (bp_enabled_) {
                    bool any_port_full = false;
                    for (const auto& hop : hop_vec) {
                        for (const auto& [dir, start_at] : hop.dir_dist) {
                            (void)start_at;
                            if (alloutq[hop.curr_id][static_cast<size_t>(dir)].size() >=
                                static_cast<size_t>(LLM_SIM_NOC::BUFFER_DEPTH)) {
                                any_port_full = true;
                                break;
                            }
                        }
                        if (any_port_full) break;
                    }
                    if (any_port_full) {
                        if (dbg_trace_) {
                            std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                                      << " STALL_PORT_FULL src=" << src_num
                                      << " pkt=" << pending_packets[src_num]->packet_id
                                      << " (busy marks undone, retry next cycle)\n" << std::flush;
                        }
                        for (int d : pending_packets[src_num]->dst_ids) this->recv_busy[d] = false;
                        this->send_busy[src_num] = false;
                        ++stall_cycles_;
                        continue;
                    }
                }

                if (dbg_trace_) {
                    std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                              << " ADMIT_OK src=" << src_num
                              << " pkt=" << pending_packets[src_num]->packet_id
                              << " flit_size=" << pending_packets[src_num]->flit_size
                              << "\n" << std::flush;
                }

                std::cout << "[NOC-PKT] @" << sc_time_stamp() << " cyc=" << cycle_count_
                          << " pkt " << pending_packets[src_num]->packet_id
                          << " is sent by core " << src_num
                          << "\n" << std::flush;

                const uint64_t hop_svc_time = hop_svc_cycle(pending_packets[src_num]->flit_size);
                insert_alloutq(alloutq, hop_vec, hop_svc_time); // insert the new hops into alloutq for future congestion tracking

                if (dbg_r1_counters_) {
                    total_hop_flight_cyc_ += hop_svc_time;
                    ++total_packets_admitted_;
                }


                LLM_SIM_NOC::TransState_t trans_state;
                trans_state.src_id = src_num;
                trans_state.dst2time = lat;
                trans_states[src_num] = trans_state; // store the transaction state
            }
        }
        
        if (endpoint_port_) complete_port(); // trans_states stays empty under it

        // Check all trans_states for completed transactions (outside the src loop)
        for(auto& [src_id, trans_state] : trans_states) {
            for(auto it = trans_state.dst2time.begin(); it != trans_state.dst2time.end();) {
                if(it->second <= 0) {
                    this->recv_busy[it->first] = false; // mark dst as not busy
                    bool ok = out_fifos[it->first].nb_write(pending_packets[src_id]); // write the packet to output fifo
                    (void)ok;
                    std::cout << "[NOC-PKT] @" << sc_time_stamp() << " cyc=" << cycle_count_
                              << " pkt " << pending_packets[src_id]->packet_id
                              << " is received by core " << it->first
                              << "\n" << std::flush;
                    if (dbg_trace_) {
                        std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                                  << " DST_DONE src=" << src_id
                                  << " dst=" << it->first
                                  << " pkt=" << pending_packets[src_id]->packet_id
                                  << " write_ok=" << ok << "\n" << std::flush;
                    }
                    it = trans_state.dst2time.erase(it); // erase returns next iterator
                } else {
                    it->second -= 1; // decrease the remaining time for each dst
                    ++it; // manually increment iterator
                }
            }

            // If all destinations for this source are completed, clean up the source state
            if(trans_state.dst2time.empty()) {
                if (dbg_trace_) {
                    std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                              << " SRC_DONE src=" << src_id << "\n" << std::flush;
                }
                this->send_busy[src_id] = false; // mark src as not busy
                pending_packets.erase(src_id); // clear the pending packet
            }
        }
        
        // Clean up empty trans_states entries
        for(auto it = trans_states.begin(); it != trans_states.end();) {
            if(it->second.dst2time.empty()) {
                it = trans_states.erase(it);
            } else {
                ++it;
            }
        }

        // Count busy cycle when any node is actively sending
        for(int i = 0; i < node_num_; ++i) {
            if(send_busy[i]) { ++busy_cycle_; break; }
        }
    }
}


// Research switch S1, endpoint rule `port`: one source's turn in run()'s
// admission loop. Today's pull-then-admit-next-cycle step is kept (a source
// pulls its next packet while the previous one is still in its inject port, so
// there is no bubble and an uncontended packet is unchanged); there is no
// send_busy / recv_busy. Mirrors the `transit` branch's admission otherwise.
template <typename T>
void NoCSim<T>::admit_port(int src_num) {
    if (pending_packets.find(src_num) == pending_packets.end()) {
        if (input_buffer.find(src_num) == input_buffer.end() || input_buffer[src_num].empty()) return;
        pending_packets[src_num] = input_buffer[src_num].front();
        input_buffer[src_num].pop();
        return;
    }
    if (cycle_count_ < release_at_[src_num]) return; // the previous packet still holds the inject port
    const auto pkt = pending_packets[src_num];

    bool offchip = is_offchip(src_num);
    for (int d : pkt->dst_ids) offchip = offchip || is_offchip(d);
    if (offchip) {
        SC_REPORT_FATAL("NoCSim",
            "endpoint rule `port` (research switch S1) covers core-to-core packets only; "
            "an off-chip packet reached it -- run with the `transit` rule");
        return;
    }

    auto [pathunion, allpath] = route_algo->get_path(src_num, pkt->dst_ids);
    (void)pathunion;
    const auto pl = noc_lat->get_latency_port(alloutq, inject_q_[src_num], eject_q_, *pkt, allpath, cycle_count_,
                                              mc_tree_, backfill_, mc_fork_async_);

    // Backpressure: as the `transit` branch, plus the inject and eject queues.
    if (bp_enabled_) {
        const size_t depth = static_cast<size_t>(LLM_SIM_NOC::BUFFER_DEPTH);
        bool any_port_full = pl.has_inject && inject_q_[src_num].size() >= depth;
        for (const auto& [dst, start_at] : pl.eject_start) {
            (void)start_at;
            any_port_full = any_port_full || eject_q_[dst].size() >= depth;
        }
        for (const auto& hop : pl.hop_vec)
            for (const auto& [dir, start_at] : hop.dir_dist) {
                (void)start_at;
                any_port_full = any_port_full || alloutq[hop.curr_id][static_cast<size_t>(dir)].size() >= depth;
            }
        if (any_port_full) {
            if (dbg_trace_) {
                std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                          << " STALL_PORT_FULL src=" << src_num
                          << " pkt=" << pkt->packet_id << " (retry next cycle)\n" << std::flush;
            }
            ++stall_cycles_;
            return;
        }
    }

    if (dbg_trace_) {
        std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                  << " ADMIT_OK src=" << src_num
                  << " pkt=" << pkt->packet_id
                  << " flit_size=" << pkt->flit_size
                  << "\n" << std::flush;
    }
    std::cout << "[NOC-PKT] @" << sc_time_stamp() << " cyc=" << cycle_count_
              << " pkt " << pkt->packet_id
              << " is sent by core " << src_num
              << "\n" << std::flush;

    const uint64_t hop_svc_time = hop_svc_cycle(pkt->flit_size);
    insert_alloutq(alloutq, pl.hop_vec, hop_svc_time);
    if (pl.has_inject) inject_q_[src_num].push({pl.inject_start, hop_svc_time});
    for (const auto& [dst, start_at] : pl.eject_start) eject_q_[dst].push({start_at, hop_svc_time});
    if (dbg_r1_counters_) {
        total_hop_flight_cyc_ += hop_svc_time;
        ++total_packets_admitted_;
    }

    release_at_[src_num] = pl.has_inject ? pl.inject_start + hop_svc_time : cycle_count_ + 1;
    InFlight_t f;
    f.packet = pkt;
    f.dst2time = pl.latencies;
    in_flight_.emplace(admit_seq_++, std::move(f));
    pending_packets.erase(src_num);
}

// Research switch S1: delivery, as the `transit` branch's completion loop but
// per packet in flight (admission order) instead of per source.
template <typename T>
void NoCSim<T>::complete_port() {
    for (auto pit = in_flight_.begin(); pit != in_flight_.end();) {
        auto& f = pit->second;
        for (auto it = f.dst2time.begin(); it != f.dst2time.end();) {
            if (it->second <= 0) {
                bool ok = out_fifos[it->first].nb_write(f.packet);
                (void)ok;
                std::cout << "[NOC-PKT] @" << sc_time_stamp() << " cyc=" << cycle_count_
                          << " pkt " << f.packet->packet_id
                          << " is received by core " << it->first
                          << "\n" << std::flush;
                if (dbg_trace_) {
                    std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                              << " DST_DONE src=" << f.packet->src_id
                              << " dst=" << it->first
                              << " pkt=" << f.packet->packet_id
                              << " write_ok=" << ok << "\n" << std::flush;
                }
                it = f.dst2time.erase(it);
            } else {
                it->second -= 1;
                ++it;
            }
        }
        if (f.dst2time.empty()) {
            if (dbg_trace_) {
                std::cerr << "[NOC-TRACE] cyc=" << cycle_count_
                          << " PKT_DONE pkt=" << f.packet->packet_id << "\n" << std::flush;
            }
            pit = in_flight_.erase(pit);
        } else {
            ++pit;
        }
    }
    if (!in_flight_.empty()) ++busy_cycle_; // `transit` counts a busy source instead
}


// helper functions
template <typename T>
void NoCSim<T>::check_alloutq(LLM_SIM_NOC::AllOutQ_t& alloutq, uint64_t now) {
    // Entries use absolute {start_at, dur}, both in cycle_count_ units; a port
    // frees at start_at + dur.
    for(size_t router_id = 0; router_id < alloutq.size(); ++router_id) {
        for(size_t dir = 0; dir < alloutq[router_id].size(); ++dir) {
            while(!alloutq[router_id][dir].empty() &&
                  alloutq[router_id][dir].front().start_at + alloutq[router_id][dir].front().dur <= now) {
                alloutq[router_id][dir].pop();
            }
        }
    }
}

template <typename T>
void NoCSim<T>::insert_alloutq(LLM_SIM_NOC::AllOutQ_t& alloutq, const std::vector<LLM_SIM_NOC::UnionHop_t>& hop_vec, const uint64_t packet_remaining_time) {
    for(const auto& hop: hop_vec) {
        for(const auto& [dir, start_at]: hop.dir_dist) {
            alloutq[hop.curr_id][static_cast<size_t>(dir)].push({start_at, packet_remaining_time});
        }
    }
}

template <typename T>
uint64_t NoCSim<T>::hop_svc_cycle(const int& flit_size) const {
    const uint64_t L = std::max(1, flit_size);
    const uint64_t svc = LLM_SIM_NOC::FLIT_CYCLE + (L - 1) / LLM_SIM_NOC::LANES;
    return svc;
}


template class NoCSim<int>;
template class NoCSim<float>;