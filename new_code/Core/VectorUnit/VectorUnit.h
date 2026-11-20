#ifndef LLM_SIM_VECTORUNIT_H
#define LLM_SIM_VECTORUNIT_H

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <utility>
#include <systemc.h>

#include "Core/VectorUnit/vector_params.h"
#include "utils/BaseInst.h"
#include "utils/fifo.h"
#include "Core/VectorUnit/VectorOp.h"
#include "utils/Data.h"
#include "utils/busyState.h"

template <typename T>
class VectorUnit: public sc_module {

    public:
        VectorUnit(sc_module_name name, const int id, bool compute = true);
        // IO ports
        sc_in_clk clk;
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>> input_buffer_in;
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>> weight_buffer_in;
        sc_fifo_out<LLM_SIM_INST::CacheReqPtr_t> in_buf_req; // send request for input buffer
        sc_fifo_out<LLM_SIM_INST::CacheReqPtr_t> w_buf_req; // send request for weight buffer
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>> output_buffer_out;
        // Write-request metadata for output_buffer_out's target Cache port,
        // sent the same cycle and strictly before output_buffer_out itself.
        sc_fifo_out<LLM_SIM_INST::CacheWriteReqPtr_t> cache_wreq_out;
        // control signals
        sc_out<LLM_SIM_BUSY::VectorBusyPtr_t> vector_busy;
        sc_fifo_in<LLM_SIM_INST::VectorInstPtr_t> vector_inst_in;
        
    // ── Utilization tracking (readable after sc_stop) ─────────────────────────
    uint64_t busy_cycle_ = 0;
    double utilization() const {
        const uint64_t total = static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        return total > 0 ? static_cast<double>(busy_cycle_) / static_cast<double>(total) : 0.0;
    }
    uint64_t busy_cycles() const { return busy_cycle_; }

    private:
        void run();
        const int id_;
        const bool compute_;
        VectorOp<T>* vector_op_;
        // LLM_SIM_INST::VectorInst_t current_inst_;
};



#endif