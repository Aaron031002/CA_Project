#include "Core/ArrayUnit/ArrayUnit.h"
#include "DRAM_sim/dram_params.h"
#include "Core/Cache/cache_params.h"
#include "utils/precision_utils.h"
#include <algorithm>
#include <cmath>

template <typename T>
ArrayUnit<T>::ArrayUnit(sc_module_name name, const int id, bool compute)
    : sc_module(name), id_(id), compute_(compute) {
    array_op = new ArrayOp<T>();
    SC_THREAD(run);
    sensitive << clk.pos();
}

template <typename T>
void ArrayUnit<T>::run() {
    // Initialize busy signals
    array_busy.write(std::make_shared<LLM_SIM_BUSY::ArrayBusy_t>(id_));

    while (true) {
        wait(); // posedge clk

        if (inst_in.num_available() == 0) continue;

        // Read instruction
        LLM_SIM_INST::ArrayInstPtr_t inst_ptr;
        inst_ptr = inst_in.read();
        const uint64_t cycle_start =
            static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        // Pull up busy signals on the same cycle
        // auto busy_on = std::make_shared<LLM_SIM_BUSY::ArrayBusy_t>(id_);
        auto busy_on = *array_busy.read();
        busy_on.busy = true;
        array_busy.write(std::make_shared<LLM_SIM_BUSY::ArrayBusy_t>(busy_on));
        // Weight source for this instruction: DRAM (default, unchanged path)
        // or Cache (mirrors VectorUnit's cache weight-read pattern).
        const bool weight_from_cache =
            (inst_ptr->weight_src == LLM_SIM_INST::ArrayInst_t::WeightSrc_t::CACHE);

        // Input is fetched first; the weight request is only issued once the
        // input has arrived, so weight read + compute latency start counting
        // from the input's arrival (no input/weight fetch overlap).
        in_buf_req.write(std::make_shared<LLM_SIM_INST::CacheReq_t>(inst_ptr->in_id, inst_ptr->in_reserved, inst_ptr->new_in_id));

        // (2026-08-12): mirrors VectorUnit::run()'s
        // pre-existing [VEC-STUCK]/[VEC-STUCK-W] diagnostic, which ArrayUnit never
        // had -- busy is pulled up (line 34, above) before these two waits, so if
        // either one blocks for a long time the busy signal stays asserted with no
        // visibility into why (this is the mechanism behind handleLocalSync()'s
        // "arr=1 ... slots=[all_idle]" observation: DMA reports idle, but the array
        // unit's busy signal alone is enough to keep local_sync() from completing).
        const bool dbg_wait_trace = std::getenv("DBG_NOC_TRACE") != nullptr;
        const uint64_t wait_first_log = dbg_wait_trace ? 50000 : 1000000;
        const uint64_t wait_period    = dbg_wait_trace ? 500000 : 5000000;

        {
            uint64_t in_wait = 0;
            while (in_buf_in.num_available() == 0) {
                wait();
                if (++in_wait == wait_first_log || in_wait % wait_period == 0) {
                    std::cerr << "[ARR-STUCK] @" << sc_time_stamp()
                              << " mod=" << name()
                              << " wait_in_id=" << inst_ptr->in_id
                              << " w_id=" << inst_ptr->w_id
                              << " wait=" << in_wait << "\n" << std::flush;
                }
            }
        }
        LLM_SIM_DATA::DataPtr_t<T> in_ptr = in_buf_in.read();

        if (weight_from_cache) {
            w_buf_req.write(std::make_shared<LLM_SIM_INST::CacheReq_t>(inst_ptr->w_id, inst_ptr->w_reserved, inst_ptr->new_w_id));
        } else {
            dram_req_out.write(packet_req(*inst_ptr));
        }

        LLM_SIM_DATA::DataPtr_t<T> chunk;
        if (weight_from_cache) {
            uint64_t w_wait = 0;
            while (weight_buf_cache_in.num_available() == 0) {
                wait();
                if (++w_wait == wait_first_log || w_wait % wait_period == 0) {
                    std::cerr << "[ARR-STUCK-W] @" << sc_time_stamp()
                              << " mod=" << name()
                              << " wait_w_id=" << inst_ptr->w_id
                              << " in_id=" << inst_ptr->in_id
                              << " wait=" << w_wait << "\n" << std::flush;
                }
            }
            chunk = weight_buf_cache_in.read();
        } else {
            uint64_t w_wait = 0;
            while (weight_buf_in.num_available() == 0) {
                wait();
                if (++w_wait == wait_first_log || w_wait % wait_period == 0) {
                    std::cerr << "[ARR-STUCK-DRAM-W] @" << sc_time_stamp()
                              << " mod=" << name()
                              << " wait_w_id=" << inst_ptr->w_id
                              << " in_id=" << inst_ptr->in_id
                              << " wait=" << w_wait << "\n" << std::flush;
                }
            }
            chunk = weight_buf_in.read();
        }
        weight_buffer.value = chunk->value;

        // Execute array operation
        LLM_SIM_DATA::Data_t<T> result;
        if (compute_) {
            result = array_op->execute(inst_ptr, *in_ptr, weight_buffer);
        } else {
            // Transaction-only mode: produce a zero-filled output with the
            // correct shape so downstream units can continue their transactions.
            using AM = LLM_SIM_INST::ArrayInst_t::ArrayMode_t;
            const int row = inst_ptr->row, col = inst_ptr->col;
            int out_size = 0;
            // size_ mirrors ArrayOp::NORMAL/DOT_PRODUCT/QK/SV's own out.size_
            // assignment exactly (ArrayOp.cpp:85,120,189,334) -- the Cache
            // write-request mechanism aborts on an empty/wrong shape, so
            // this transaction-only path must tag the same shape the
            // compute path would.
            std::vector<int> out_shape;
            switch (inst_ptr->mode) {
                case AM::NORMAL:
                case AM::DOT_PRODUCT: {
                    const int total_M = static_cast<int>(in_ptr->value.size()) / row;
                    out_size = total_M * col;
                    out_shape = {total_M, col};
                    break;
                }
                case AM::QK: {
                    const int kv_heads = inst_ptr->kv_head_num > 0 ? inst_ptr->kv_head_num : 1;
                    const int head_dim = col / kv_heads;
                    const int seq_len  = row;
                    int q_tokens = 1;
                    int q_total_dim = static_cast<int>(in_ptr->value.size());
                    if (in_ptr->size_.size() >= 2) {
                        q_tokens    = in_ptr->size_[0];
                        q_total_dim = in_ptr->size_[1];
                    }
                    const int q_heads = q_total_dim / head_dim;
                    out_size = q_tokens * q_heads * seq_len;
                    out_shape = {q_tokens * q_heads, seq_len};
                    break;
                }
                case AM::SV: {
                    const int kv_heads  = inst_ptr->kv_head_num > 0 ? inst_ptr->kv_head_num : 1;
                    const int head_dim  = col / kv_heads;
                    const int score_cols = inst_ptr->has_sink ? row + 1 : row;
                    const int n_groups  = static_cast<int>(in_ptr->value.size()) / score_cols;
                    out_size = n_groups * head_dim;
                    out_shape = {n_groups, head_dim};
                    break;
                }
                default:
                    out_size = static_cast<int>(in_ptr->value.size());
                    out_shape = {out_size};
                    break;
            }
            result.id    = inst_ptr->out_id;
            result.value.assign(out_size, T{});
            result.size_ = out_shape;
        }

        // Compute latency: dataflow-aware systolic-array model
        // (§A).
        //   WEIGHT_STATIONARY (NORMAL/DOT_PRODUCT): K*N weight tile stays
        //     resident in the ARRAY_ROWS x ARRAY_COLS grid, M tokens stream
        //     through -> compute_lat = ceil(K/ROWS) * ceil(N/COLS) * M.
        //   OUTPUT_STATIONARY (QK/SV): K/V differ per token position so
        //     nothing can stay resident; the M*N output tile accumulates
        //     while K streams -> compute_lat = ceil(M/ROWS) * ceil(N/COLS) * K.
        // (M, N, K) per mode mirror the actual output shape and true
        // contraction dimension of each op (see array_params.h comment and
        // ArrayOp::QK / ArrayOp::SV docs above for the K/V layout this reads).
        {
            using AM = LLM_SIM_INST::ArrayInst_t::ArrayMode_t;
            using DF = LLM_SIM_INST::ArrayInst_t::Dataflow_t;
            const int row = inst_ptr->row;
            const int col = inst_ptr->col;
            int M = 1, N = 1, K = 1;

            switch (inst_ptr->mode) {
                case AM::NORMAL:
                case AM::DOT_PRODUCT: {
                    K = row;
                    N = col;
                    M = static_cast<int>(in_ptr->value.size()) / (K > 0 ? K : 1);
                    break;
                }
                case AM::QK: {
                    const int kv_heads = inst_ptr->kv_head_num > 0 ? inst_ptr->kv_head_num : 1;
                    const int head_dim = col / kv_heads;
                    const int seq_len  = row;
                    int q_tokens = 1;
                    int q_total_dim = static_cast<int>(in_ptr->value.size());
                    if (in_ptr->size_.size() >= 2) {
                        q_tokens    = in_ptr->size_[0];
                        q_total_dim = in_ptr->size_[1];
                    }
                    const int q_heads = head_dim > 0 ? q_total_dim / head_dim : 1;
                    M = q_tokens * q_heads;
                    N = seq_len;
                    K = head_dim;
                    break;
                }
                case AM::SV: {
                    const int kv_heads   = inst_ptr->kv_head_num > 0 ? inst_ptr->kv_head_num : 1;
                    const int head_dim   = col / kv_heads;
                    const int seq_len    = row;
                    const int score_cols = inst_ptr->has_sink ? seq_len + 1 : seq_len;
                    const int n_groups   = score_cols > 0
                        ? static_cast<int>(in_ptr->value.size()) / score_cols : 1;
                    M = n_groups;
                    N = head_dim;
                    K = seq_len;
                    break;
                }
                default:
                    K = row;
                    N = col;
                    M = static_cast<int>(in_ptr->value.size()) / (K > 0 ? K : 1);
                    break;
            }

            const auto ceil_div = [](int a, int b) -> uint64_t {
                return b > 0 ? static_cast<uint64_t>((a + b - 1) / b) : 0;
            };

            uint64_t compute_lat = 0;
            if (inst_ptr->resolved_dataflow() == DF::OUTPUT_STATIONARY) {
                compute_lat = ceil_div(M, LLM_SIM_ARRAY::ARRAY_ROWS)
                            * ceil_div(N, LLM_SIM_ARRAY::ARRAY_COLS)
                            * static_cast<uint64_t>(K);
            } else {
                compute_lat = ceil_div(K, LLM_SIM_ARRAY::ARRAY_ROWS)
                            * ceil_div(N, LLM_SIM_ARRAY::ARRAY_COLS)
                            * static_cast<uint64_t>(M);
            }

            // Mixed-precision MAC throughput:
            // scale compute_lat by the square-law speedup for this op's P_mode,
            // derived from the wider of the two operands' nominal elem_bits
            // (weight chunk and activation in_ptr). Narrower-than-16-bit
            // operands (e.g. Q4_0, elem_bits=4) shrink compute_lat; anything
            // still tagged at the default 32 (untagged/full-precision float)
            // grows it, since a 16-bit MAC cannot complete a 32-bit multiply
            // in one native cycle.
            const int p_mode = LLM_SIM_ARRAY::resolve_p_mode(
                std::max(chunk->elem_bits, in_ptr->elem_bits));
            const double speedup = LLM_SIM_ARRAY::mac_speedup(p_mode);
            compute_lat = static_cast<uint64_t>(
                std::ceil(static_cast<double>(compute_lat) / speedup));

            // Weight-fetch latency was already waited (DRAM or Cache, above);
            // compute remaining compute cycles beyond that overlap window.
            uint64_t fetch_lat;
            if (weight_from_cache) {
                // Mirrors Cache::latency_cal exactly (Core/Cache/Cache.cpp) --
                // no DRAM-style fixed row-open overhead since Cache has none.
                int sz = 1;
                for (int d : chunk->size_) sz *= d;
                const uint64_t bits = static_cast<uint64_t>(sz) *
                    static_cast<uint64_t>(chunk->elem_bits);
                const uint64_t cycles = bits / LLM_SIM_CACHE::CACHE_BUS_WIDTH_BITS;
                fetch_lat = cycles > 0 ? cycles : 1;
            } else {
                // Phase E: chunk->elem_bits is
                // already tagged correctly by DRAMSim::do_read (whatever
                // format the weight file was recorded as) -- no separate
                // DataManager lookup needed here, same pattern as the
                // weight_from_cache branch above already used for
                // chunk->elem_bits.
                fetch_lat = static_cast<uint64_t>(5) +
                    LLM_SIM_PRECISION::transfer_cycles(
                        chunk->value.size(), chunk->elem_bits,
                        static_cast<uint64_t>(LLM_SIM_DRAM::DRAM_BUS_BANDWIDTH));
            }
            if (std::getenv("DBG_R1_COUNTERS")) {
                r1_compute_lat_cyc_ += compute_lat;
                r1_fetch_lat_cyc_ += fetch_lat;
            }
            const uint64_t extra = (compute_lat > fetch_lat) ? compute_lat - fetch_lat : 0;
            wait((extra + LLM_SIM_ARRAY::ARRAY_FILL_DRAIN + LLM_SIM_ARRAY::WEIGHT_BUF_OVERHEAD)
                 * sc_time(1, SC_NS));
        }

        busy_cycle_ += static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS)) - cycle_start;
        // Write result to output signal
        auto out_ptr = std::make_shared<LLM_SIM_DATA::Data_t<T>>(std::move(result));
        // DBG_PIN_ID/DBG_PIN_CORE (same convention as Cache.cpp's
        // dbg_pin_matches): S1024
        // QK-output-never-in-Cache investigation -- confirms whether
        // out_ptr->id actually matches the instruction's declared out_id
        // (inst_ptr->out_id) right before the write, and whether this write
        // call is ever reached at all, before out_buf_out.write() below.
        if (std::getenv("DBG_PIN_ID")) {
            static const uint64_t pin_id = static_cast<uint64_t>(std::atoll(std::getenv("DBG_PIN_ID")));
            const char* pin_core_env = std::getenv("DBG_PIN_CORE");
            const bool core_match = !pin_core_env ||
                std::string(name()).find(pin_core_env) != std::string::npos;
            if (core_match && (out_ptr->id == pin_id || inst_ptr->out_id == pin_id)) {
                std::cerr << "[PIN-ARR-OUT] @" << sc_time_stamp()
                          << " mod=" << name()
                          << " inst_out_id=" << inst_ptr->out_id
                          << " out_ptr_id=" << out_ptr->id
                          << " value.size()=" << out_ptr->value.size()
                          << " fifo_free=" << out_buf_out.num_free()
                          << "\n" << std::flush;
            }
        }
        cache_wreq_out.write(LLM_SIM_DATA::make_cache_write_req(out_ptr));
        out_buf_out.write(out_ptr);
        if (std::getenv("DBG_PIN_ID")) {
            static const uint64_t pin_id = static_cast<uint64_t>(std::atoll(std::getenv("DBG_PIN_ID")));
            const char* pin_core_env = std::getenv("DBG_PIN_CORE");
            const bool core_match = !pin_core_env ||
                std::string(name()).find(pin_core_env) != std::string::npos;
            if (core_match && inst_ptr->out_id == pin_id) {
                std::cerr << "[PIN-ARR-OUT-WRITTEN] @" << sc_time_stamp()
                          << " mod=" << name()
                          << " inst_out_id=" << inst_ptr->out_id
                          << "\n" << std::flush;
            }
        }

        // Free weight buffer memory
        weight_buffer.value.clear();
        weight_buffer.value.shrink_to_fit();

        auto busy_off = *array_busy.read();
        busy_off.busy = false;
        array_busy.write(std::make_shared<LLM_SIM_BUSY::ArrayBusy_t>(busy_off));
    }
}

template <typename T>
int ArrayUnit<T>::get_overhead() {
    return 5;
}

template <typename T>
LLM_SIM_DRAM::ReqPtr_t ArrayUnit<T>::packet_req(const LLM_SIM_INST::ArrayInst_t& curr_inst) {
    return curr_inst.req_ptr;
}

template class ArrayUnit<float>;
