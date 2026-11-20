#ifndef LLM_SIM_OFFCHIP_DRAM_H
#define LLM_SIM_OFFCHIP_DRAM_H

#include <cstdint>
#include <string>
#include <systemc.h>

#include "DRAM_sim/offchip_dram_params.h"
#include "utils/buffer.h"
#include "utils/Data.h"
#include "utils/Req.h"
#include "utils/busyState.h"

template <typename T> class DataManager;

// ---------------------------------------------------------------------------
// OffChipDRAM<T> (Phase 6b)
//
// One instance per off-chip DRAM node (8 total, ids 64..71, one per mesh
// row -- see LLM_SIM_NOC::OFFCHIP_ID_BASE). Deliberately modeled as a
// simplified DRAMSim<T>: own file-backed storage (byte-offset read/write,
// same technique as DataManager<T>::read_data/write_data) and own latency
// formula (offchip_dram_params.h's OFFCHIP_DRAM_OVERHEAD/BANDWIDTH), but NOT
// literally reusing DataManager<T>/DRAMSim<T> -- those classes' KV-cache
// init (init_kv_files) and multi-source arbitration are specific to the
// per-core local DRAM scenario and don't apply here.
//
// Unlike DRAMSim (which serves NUM_SRC=2 local sources -- DMA and Array --
// via an arrival-order arbiter+queue), an off-chip node has exactly ONE
// upstream source: its own Depacketizer, which is itself the single funnel
// for every core's off-chip requests to this row's node. A single
// poll-process-respond thread is therefore sufficient -- no multi-source
// arbitration is needed.
//
// Each of the 8 nodes owns an entirely independent instance (not a
// singleton like local DRAM's DataManager) -- confirmed design decision,
// since each node is physically distinct backing storage.
// ---------------------------------------------------------------------------
template <typename T>
class OffChipDRAM : public sc_module {
public:
    // node_id: this node's NoC id (LLM_SIM_NOC::OFFCHIP_ID_BASE + row, 64..71).
    // compute: if false, all file I/O is skipped (read returns zero-filled
    //          data, write is a no-op) -- mirrors DataManager<T>'s compute_ flag.
    OffChipDRAM(sc_module_name name, int node_id, bool compute = true);

    sc_in_clk clk;

    // ---- IO ports (single source: this node's Depacketizer) ----
    buffer_in<LLM_SIM_DRAM::ReqPtr_t>        req_port;   // Depacketizer -> here
    sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>  read_port;  // here -> Packetizer (read responses)
    sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>   write_port; // Depacketizer -> here (write payloads)

    int node_id() const { return node_id_; }

    // Phase E: non-owning, optional -- used
    // ONLY to look up a path's elem_bits_for() format tag for latency
    // accounting. OffChipDRAM's own file I/O (read_data/write_data above)
    // stays completely independent of DataManager, per this class's own
    // design-decision comment. Unset (nullptr, default) means every path
    // resolves to the fp32 default (32 bits), i.e. no behavior change for
    // any caller that doesn't wire this up.
    void setDataManager(DataManager<T>* dm) { data_manager_ = dm; }

    // ── Utilization tracking (readable after sc_stop) ─────────────────────
    uint64_t busy_cycle_ = 0;
    double utilization() const {
        const uint64_t total = static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        return total > 0 ? static_cast<double>(busy_cycle_) / static_cast<double>(total) : 0.0;
    }
    uint64_t busy_cycles() const { return busy_cycle_; }

private:
    const int  node_id_;
    const bool compute_;
    DataManager<T>* data_manager_ = nullptr;

    std::vector<T> read_data(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr);
    void           write_data(const LLM_SIM_DRAM::DRAMAddrPtr_t& addr_ptr,
                              const std::vector<T>& data);
    uint64_t       get_offchip_latency() const;

    void do_read(const LLM_SIM_DRAM::ReqPtr_t& req);
    void do_write(const LLM_SIM_DRAM::ReqPtr_t& req);

    void run(); // single SC_THREAD: poll req_port, do_read/do_write, wait latency, respond
};

#endif // LLM_SIM_OFFCHIP_DRAM_H
