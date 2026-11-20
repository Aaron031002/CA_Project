#ifndef LLM_SIM_ARRAYUNIT_H
#define LLM_SIM_ARRAYUNIT_H

#include <iostream>
#include <string>
#include <vector>
#include <systemc.h>

#include "utils/BaseInst.h"
#include "utils/Data.h"
#include "utils/Req.h"
#include "Core/ArrayUnit/ArrayOp.h"
#include "Core/ArrayUnit/array_params.h"
#include "utils/bufferAddr.h"
#include "utils/busyState.h"

template <typename T>
class ArrayUnit : public sc_core::sc_module {
    public:
        // constructor
        ArrayUnit(sc_module_name name, const int id, bool compute = true);
        // ports
        sc_in_clk clk;
        // sc_in<LLM_SIM_DATA::DataPtr_t<T>>       in_buf_in;
        // sc_out<LLM_SIM_DATA::DataPtr_t<T>>      out_buf_out;
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>  in_buf_in;
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>> out_buf_out;
        // Write-request metadata for out_buf_out's target Cache port, sent
        // the same cycle and strictly before out_buf_out itself so Cache can
        // compute latency/capacity ahead of reading the payload.
        sc_fifo_out<LLM_SIM_INST::CacheWriteReqPtr_t> cache_wreq_out;
        sc_fifo_out<LLM_SIM_INST::CacheReqPtr_t> in_buf_req;
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>  weight_buf_in;
        sc_fifo_out<LLM_SIM_INST::CacheReqPtr_t> w_buf_req;        // request weight from Cache (weight_src==CACHE)
        sc_fifo_in<LLM_SIM_DATA::DataPtr_t<T>>  weight_buf_cache_in; // weight data from Cache
        sc_fifo_in<LLM_SIM_INST::ArrayInstPtr_t> inst_in;
        sc_fifo_out<LLM_SIM_DRAM::ReqPtr_t>          dram_req_out;   // send request to DRAM
        sc_fifo_out<LLM_SIM_DATA::DataPtr_t<T>>        dram_write_out; // send write data to DRAM
        sc_out<LLM_SIM_BUSY::ArrayBusyPtr_t> array_busy;
    // ── Utilization tracking (readable after sc_stop) ─────────────────────────
    uint64_t busy_cycle_ = 0;
    double utilization() const {
        const uint64_t total = static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        return total > 0 ? static_cast<double>(busy_cycle_) / static_cast<double>(total) : 0.0;
    }
    uint64_t busy_cycles() const { return busy_cycle_; }

    // ── R1 counter breakdown
    // (numerical cross-check extension) ─────────────────────────────────────
    // Process-wide aggregates (every ArrayUnit<float> instance in this
    // binary shares one T-instantiation's static, safe under SystemC's
    // single-threaded cooperative scheduling -- no core's ArrayUnit::run()
    // resumes concurrently with another's). compute_lat/fetch_lat are the
    // exact same two terms ArrayUnit::run() itself already computes (see
    // this file's own run() body just above weight_buffer's declaration)
    // -- compute_lat is pure GEMM compute work (ceil_div(M,ROWS)*
    // ceil_div(N,COLS)*K), fetch_lat is weight-fetch wait (DRAM or Cache).
    // Env-gated by DBG_R1_COUNTERS, zero cost on a normal run.
    static inline uint64_t r1_compute_lat_cyc_ = 0;
    static inline uint64_t r1_fetch_lat_cyc_ = 0;
    static uint64_t r1_compute_lat_cycles() { return r1_compute_lat_cyc_; }
    static uint64_t r1_fetch_lat_cycles() { return r1_fetch_lat_cyc_; }

    private:
        void run();
        int get_overhead();
        const int id_;
        const bool compute_;
        LLM_SIM_DRAM::ReqPtr_t packet_req(const LLM_SIM_INST::ArrayInst_t& curr_inst);
        ArrayOp<T>* array_op;
        LLM_SIM_DATA::Data_t<T> weight_buffer;
};


#endif