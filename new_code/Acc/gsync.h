// gsync.h ── GlobalSyncManager: distributed barrier manager for multi-core sync.
//
// Each core sends a GSyncArriveMsg_t on arrive_in[core_id] when it executes a
// GLOBAL_SYNC instruction.  The manager waits until every core listed in
// participants has arrived, then writes the barrier_id back on each
// participant's release_out[core_id] FIFO.
//
// Constraints:
//   • Each core may have at most ONE in-flight GLOBAL_SYNC at any moment.
//     This is enforced naturally because MainCtrl stalls (blocks dispatch)
//     after writing the arrive message and before reading the release.
//   • Participant sets for the same barrier_id must be identical across all
//     arriving cores.  The first arriving core's participant list is used as
//     ground truth; subsequent arrivals are silently accepted.
//
// Wiring (done in Acc.cpp):
//   main_ctrl->gsync_arrive_out  →  core->gsync_arrive_fifo  →  arrive_in[i]
//   release_out[i]               →  core->gsync_release_fifo →  main_ctrl->gsync_release_in

#ifndef LLM_SIM_GSYNC_H
#define LLM_SIM_GSYNC_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <systemc.h>

// ── Arrive message sent by each core ─────────────────────────────────────────
struct GSyncArriveMsg_t {
    uint64_t         barrier_id   = 0;
    int              core_id      = -1;
    std::vector<int> participants;          // all core IDs expected at this barrier

    bool operator==(const GSyncArriveMsg_t& o) const {
        return barrier_id == o.barrier_id && core_id == o.core_id;
    }
};
// sc_trace no-op (GSyncArriveMsg_t is not trivially traceable)
inline void sc_trace(sc_core::sc_trace_file*, const GSyncArriveMsg_t&, const std::string&) {}

inline std::ostream& operator<<(std::ostream& os, const GSyncArriveMsg_t& m) {
    os << "GSyncArriveMsg_t{barrier_id=" << m.barrier_id
       << ", core_id=" << m.core_id << "}";
    return os;
}

// ── GlobalSyncManager ─────────────────────────────────────────────────────────
class GlobalSyncManager : public sc_module {
public:
    SC_HAS_PROCESS(GlobalSyncManager);

    // num_cores: total number of cores (== arrive_in.size() == release_out.size())
    GlobalSyncManager(sc_module_name name, int num_cores);

    // ── Clock ──────────────────────────────────────────────────────────────
    sc_in_clk clk;

    // ── Arrive channels (core → manager) ──────────────────────────────────
    // Core i writes a GSyncArriveMsg_t here when it hits a GLOBAL_SYNC.
    sc_vector<sc_fifo_in<GSyncArriveMsg_t>>  arrive_in;

    // ── Release channels (manager → core) ─────────────────────────────────
    // Manager writes the barrier_id here once all participants have arrived.
    sc_vector<sc_fifo_out<uint64_t>>         release_out;

    // ── Diagnostics ───────────────────────────────────────────────────────
    // Incremented each time a barrier is fully released.
    int released_count = 0;
    // Total number of arrive messages processed.
    int arrived_count = 0;

private:
    struct BarrierState_t {
        std::unordered_set<int> expected;  // participant set (from first arrival)
        std::unordered_set<int> arrived;   // cores that have checked in so far
    };

    std::unordered_map<uint64_t, BarrierState_t> barriers_;
    const int num_cores_;

    // SC_THREAD: polls all arrive_in FIFOs every clock cycle.
    void process();

    // Process one arrive message; releases all participants when quorum is met.
    void processArrive(int core_id, const GSyncArriveMsg_t& msg);
};

#endif // LLM_SIM_GSYNC_H
