#ifndef LLM_SIM_TMU_H
#define LLM_SIM_TMU_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <systemc.h>

#include "utils/BaseInst.h"
#include "utils/busyState.h"
#include "utils/Data.h"
#include "Core/TMU/TMUOp.h"
#include "Core/TMU/tmu_params.h"
#include "utils/buffer.h"

template<typename T>
class TMU: public sc_module {
    public:
        TMU(sc_module_name name, bool compute = true);
        // ports declare
        sc_in_clk clk;
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>> input_fifo;
        sc_fifo_out<LLM_SIM_INST::CacheReqPtr_t> cache_req_fifo;
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>> output_fifo;
        // Write-request metadata for output_fifo's target Cache port, sent
        // the same cycle and strictly before output_fifo itself.
        sc_fifo_out<LLM_SIM_INST::CacheWriteReqPtr_t> cache_wreq_out;
        sc_fifo_in<LLM_SIM_INST::TMUInstPtr_t> inst_buffer_in; // input instruction buffer
        sc_out<LLM_SIM_BUSY::TMUBusyPtr_t> tmu_busy; // busy signal output
    // ── Utilization tracking (readable after sc_stop) ─────────────────────────
    uint64_t busy_cycle_ = 0;
    double utilization() const {
        const uint64_t total = static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        return total > 0 ? static_cast<double>(busy_cycle_) / static_cast<double>(total) : 0.0;
    }
    uint64_t busy_cycles() const { return busy_cycle_; }

    private:
        TMUOp<T>* tmu_op; // the TMU operation class that implements the real TMU operations (e.g., scatter, split, concat)
        const bool compute_;
        // main thread function
        void run();
};




#endif