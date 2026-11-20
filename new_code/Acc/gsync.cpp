// gsync.cpp ── GlobalSyncManager implementation.

#include "Acc/gsync.h"
#include <iostream>

GlobalSyncManager::GlobalSyncManager(sc_module_name name, int num_cores)
    : sc_module(name),
      arrive_in("arrive_in", num_cores),
      release_out("release_out", num_cores),
      num_cores_(num_cores)
{
    SC_THREAD(process);
    sensitive << clk.pos();
}

void GlobalSyncManager::process() {
    while (true) {
        wait();  // advance one clock cycle
        for (int i = 0; i < num_cores_; ++i) {
            GSyncArriveMsg_t msg;
            while (arrive_in[i].nb_read(msg)) {
                processArrive(i, msg);
            }
        }
    }
}

void GlobalSyncManager::processArrive(int core_id, const GSyncArriveMsg_t& msg) {
    const uint64_t bid = msg.barrier_id;
    auto& state = barriers_[bid];

    // First arrival: register the expected participant set.
    if (state.expected.empty()) {
        for (int p : msg.participants) {
            state.expected.insert(p);
        }
        if (state.expected.empty()) {
            SC_REPORT_WARNING("GlobalSyncManager",
                ("barrier_id=" + std::to_string(bid) +
                 " arrived from core " + std::to_string(core_id) +
                 " with empty participant list — releasing immediately").c_str());
            barriers_.erase(bid);
            if (!release_out[core_id].nb_write(bid)) {
                SC_REPORT_ERROR("GlobalSyncManager",
                    "release_out FIFO full (empty-participant early release)");
            }
            return;
        }
    }

    // Guard: core must be in the expected set.
    if (state.expected.find(core_id) == state.expected.end()) {
        SC_REPORT_WARNING("GlobalSyncManager",
            ("core " + std::to_string(core_id) +
             " arrived at barrier " + std::to_string(bid) +
             " but is not in the expected participant set — ignored").c_str());
        return;
    }

    // Duplicate arrive guard.
    if (state.arrived.find(core_id) != state.arrived.end()) {
        SC_REPORT_WARNING("GlobalSyncManager",
            ("core " + std::to_string(core_id) +
             " arrived at barrier " + std::to_string(bid) + " twice — ignored").c_str());
        return;
    }

    // Participant consistency check: every arriving core must declare the same
    // participant set as the first arrival (which established state.expected).
    {
        std::unordered_set<int> declared(msg.participants.begin(), msg.participants.end());
        if (declared != state.expected) {
            std::string err = "participant mismatch at barrier_id=" +
                              std::to_string(bid) + " from core " +
                              std::to_string(core_id) +
                              ": ground-truth expected {";
            for (int p : state.expected) err += std::to_string(p) + " ";
            err += "} but this core declared {";
            for (int p : declared) err += std::to_string(p) + " ";
            err += "}";
            SC_REPORT_FATAL("GlobalSyncManager", err.c_str());
            return;  // unreachable in production (FATAL aborts); for test overrides
        }
    }

    state.arrived.insert(core_id);
    ++arrived_count;

    // Release all participants once the quorum is met.
    if (state.arrived == state.expected) {
        for (int p : state.expected) {
            if (!release_out[p].nb_write(bid)) {
                SC_REPORT_ERROR("GlobalSyncManager",
                    ("release_out FIFO full for core " + std::to_string(p) +
                     " at barrier " + std::to_string(bid) +
                     " — each core should have at most one in-flight global sync").c_str());
            }
        }
        ++released_count;
        barriers_.erase(bid);
    }
}
