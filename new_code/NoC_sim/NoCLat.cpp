#include "NoCLat.h"

NoCLat::NoCLat(int mesh_width, bool pf_fix) : mesh_width_(mesh_width), pf_fix_(pf_fix) {}

// assistant methods
uint64_t NoCLat::physical_latency(const int& src, const int& dst) const {
    // Assume each hop takes 2 cycles and each flit takes 1 cycle to process
    return LLM_SIM_NOC::FLIT_CYCLE * mandist(src, dst);

}

uint64_t NoCLat::hop_svc_cycle(const int& flit_size) const {
    const uint64_t L = std::max(1, flit_size);
    const uint64_t svc = LLM_SIM_NOC::FLIT_CYCLE + (L - 1) / LLM_SIM_NOC::LANES;
    return svc;
}


uint64_t NoCLat::port_finish_time(const LLM_SIM_NOC::OutQ_t& queue, uint64_t now) const {
    // Serialize the already-queued services on this output port and return the
    // absolute cycle the port becomes free after all of them complete. Each
    // entry occupies [start_at, start_at + dur); a later entry cannot begin
    // service before both its own start_at and the previous entry's completion.
    // `now` is the caller's cycle counter (see NoCLat::get_latency's doc comment
    // for why this must not be sc_time_stamp()).
    //
    // Research switch S3 (pf_fix_): the loop below restarts an entry already in
    // service at `now` and charges its full duration again (an entry [71, 137)
    // seen at now = 134 frees the port at 200, not 137), contradicting the
    // comment above. With pf_fix_ the port is free when its last entry ends.
    auto ready_time = now;
    auto queue_copy = queue;  // Create a copy to iterate (std::queue has no const iteration)
    while(!queue_copy.empty()) {
        const auto entry = queue_copy.front();
        queue_copy.pop();
        if(pf_fix_) {
            ready_time = std::max(ready_time, entry.start_at + entry.dur);
            continue;
        }
        const uint64_t service_start = std::max(ready_time, entry.start_at);
        ready_time = service_start + entry.dur;
    }
    return ready_time;
}

uint64_t NoCLat::earliest_gap(const std::vector<const LLM_SIM_NOC::OutQ_t*>& queues, uint64_t t, uint64_t s) const {
    std::vector<LLM_SIM_NOC::OutEntry_t> entries;
    for(const auto* q : queues) {
        auto copy = *q; // std::queue has no const iteration
        while(!copy.empty()) { entries.push_back(copy.front()); copy.pop(); }
    }
    // Moving t to the end of an overlapping entry never skips a feasible start
    // (every start before that end still overlaps it), so this is the least one.
    bool moved = true;
    while(moved) {
        moved = false;
        for(const auto& e : entries) {
            if(t < e.start_at + e.dur && e.start_at < t + s) {
                t = e.start_at + e.dur;
                moved = true;
            }
        }
    }
    return t;
}

// dist calculations
inline uint64_t NoCLat::mandist(const int& src_id, const int& dst_id) const {
    std::pair<int, int> src_coor = id2coor(src_id);
    std::pair<int, int> dst_coor = id2coor(dst_id);
    return abs(src_coor.first - dst_coor.first) + abs(src_coor.second - dst_coor.second);
}

inline std::pair<int, int> NoCLat::id2coor(const int& id) const {
    int x = id % mesh_width_;
    int y = id / mesh_width_;
    return {x, y};
}