#include "Core/VectorUnit/VectorUnit.h"


template <typename T>
VectorUnit<T>::VectorUnit(sc_module_name name, const int id, bool compute)
    : sc_module(name), id_(id), compute_(compute) {
    vector_op_ = new VectorOp<T>();
    SC_THREAD(run);
    sensitive << clk.pos();
}


template <typename T>
void VectorUnit<T>::run() {
    // Initialize busy signal so reads don't dereference nullptr
    vector_busy.write(std::make_shared<LLM_SIM_BUSY::VectorBusy_t>(id_));

    while(true) {
        wait(); // Wait for clock edge
        if (vector_inst_in.num_available() == 0) {
            continue; // No instruction, stay idle
        }
        // main operation
        LLM_SIM_INST::VectorInstPtr_t inst_ptr;
        if (vector_inst_in.nb_read(inst_ptr)) {
            const uint64_t cycle_start =
                static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
            // Mark busy
            auto busy_on = std::make_shared<LLM_SIM_BUSY::VectorBusy_t>(id_);
            busy_on->busy = true;
            vector_busy.write(busy_on);

            // Send requests for input and weight buffers before reading.
            // ROPE computes cos/sin on-the-fly; it does not consume a weight buffer entry.
            in_buf_req.write(std::make_shared<LLM_SIM_INST::CacheReq_t>(inst_ptr->in_id, inst_ptr->in_reserved, inst_ptr->new_in_id));

            {
                // (2026-08-12): DBG_NOC_TRACE=1
                // lowers this pre-existing diagnostic's period from 1e6/5e6 down
                // to 50000/500000 -- the TOKENS_PER_CORE>1 livelock investigation
                // only reaches ~440us (~440000 cycles) of sim time in a wall-clock-
                // affordable run, never enough to trip the original 1e6 threshold
                // even when a unit really is stuck the whole time.
                const bool dbg = std::getenv("DBG_NOC_TRACE") != nullptr;
                const uint64_t first_log = dbg ? 50000 : 1000000;
                const uint64_t period    = dbg ? 500000 : 5000000;
                uint64_t vec_wait = 0;
                while (input_buffer_in.num_available() == 0) {
                    wait();
                    if (++vec_wait == first_log || vec_wait % period == 0) {
                        std::cerr << "[VEC-STUCK] @" << sc_time_stamp()
                                  << " mod=" << name()
                                  << " wait_in_id=" << inst_ptr->in_id
                                  << " w_id=" << inst_ptr->w_id
                                  << " wait=" << vec_wait << "\n" << std::flush;
                    }
                }
            }
            auto input_data = input_buffer_in.read();

            LLM_SIM_DATA::DataPtr_t<T> weight_data;
            {
                using VM = LLM_SIM_INST::VectorInst_t::VectorMode_t;
                bool needs_weight = (inst_ptr->mode == VM::ADD     ||
                                     inst_ptr->mode == VM::SUB     ||
                                     inst_ptr->mode == VM::MUL     ||
                                     inst_ptr->mode == VM::SWIGLU  ||
                                     inst_ptr->mode == VM::RMS     ||
                                     inst_ptr->mode == VM::BIAS_ADD||
                                     inst_ptr->mode == VM::DOT_PRODUCT ||
                                     inst_ptr->mode == VM::SPARSE_GATE ||
                                     inst_ptr->mode == VM::QUANTIZE_DEQUANT_Q8_0_EXT_SCALE ||
                                     inst_ptr->mode == VM::QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX ||
                                     // SOFTMAX: only fetch a weight buffer when an
                                     // attn-sink tensor id was actually supplied
                                     // (w_id>0 or w_reserved); router softmax calls
                                     // (w_id=0, w_reserved=false) skip this entirely.
                                     (inst_ptr->mode == VM::SOFTMAX &&
                                      (inst_ptr->w_id > 0 || inst_ptr->w_reserved)));
                if (needs_weight) {
                    w_buf_req.write(std::make_shared<LLM_SIM_INST::CacheReq_t>(inst_ptr->w_id, inst_ptr->w_reserved, inst_ptr->new_w_id));
                    {
                        const bool dbg = std::getenv("DBG_NOC_TRACE") != nullptr;
                        const uint64_t first_log = dbg ? 50000 : 1000000;
                        const uint64_t period    = dbg ? 500000 : 5000000;
                        uint64_t w_wait = 0;
                        while (weight_buffer_in.num_available() == 0) {
                            wait();
                            if (++w_wait == first_log || w_wait % period == 0) {
                                std::cerr << "[VEC-STUCK-W] @" << sc_time_stamp()
                                          << " mod=" << name()
                                          << " wait_w_id=" << inst_ptr->w_id
                                          << " in_id=" << inst_ptr->in_id
                                          << " wait=" << w_wait << "\n" << std::flush;
                            }
                        }
                        weight_data = weight_buffer_in.read();
                    }
                }
            }

            // Write-request-before-data: send Cache the shape/elem_bits
            // metadata for an output BEFORE the output itself, same cycle,
            // so Cache can compute latency/reserve capacity ahead of ever
            // reading the payload (see utils/Data.h's make_cache_write_req).
            auto emit = [this](const LLM_SIM_DATA::DataPtr_t<T>& out) {
                cache_wreq_out.write(LLM_SIM_DATA::make_cache_write_req(out));
                output_buffer_out.write(out);
            };

            if (compute_) {
                // Execute operation — returns a vector of DataPtr_t
                // (1 item for most ops; 2 for TOP_K: [indices, values], IDs set inside execute())
                auto [output_vec, latency] = vector_op_->execute(*inst_ptr, input_data, weight_data);
                wait(latency * sc_time(1, SC_NS)); // Simulate operation latency

                for (auto& out : output_vec) {
                    emit(out);
                }
            } else {
                // Transaction-only: compute real latency without arithmetic, zero-fill output shapes
                const uint64_t latency = vector_op_->latency_only(*inst_ptr, input_data, weight_data);
                wait(latency * sc_time(1, SC_NS));

                using VM = LLM_SIM_INST::VectorInst_t::VectorMode_t;
                // step 1: mirror VectorOp::
                // execute()'s elem_bits=8 tagging for the Q8_0 modes so the
                // transaction-only (compute_=false, e.g. shard_test timing
                // probes) path reports the same latency-relevant elem_bits
                // as the real-compute path, keeping the two in sync.
                const bool is_q8_mode = (inst_ptr->mode == VM::QUANTIZE_DEQUANT_Q8_0 ||
                                          inst_ptr->mode == VM::QUANTIZE_DEQUANT_Q8_0_EXT_SCALE ||
                                          inst_ptr->mode == VM::QUANTIZE_DEQUANT_Q8_0_BOUNDARY_FIX);
                auto make_zero = [is_q8_mode](uint64_t id, size_t sz) {
                    auto p = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
                    p->id = id;
                    p->value.assign(sz, T{});
                    p->size_ = {static_cast<int>(sz)};
                    if (is_q8_mode) p->elem_bits = 8;
                    return p;
                };
                // ROPE is handled early in execute() with an inst-field-
                // specific shape ({dim}) that output_elem_count() doesn't
                // carry the fields for -- mirrored here directly so
                // transaction-only matches compute_=true exactly.
                if (inst_ptr->mode == VM::TOP_K) {
                    const int n_experts = (inst_ptr->w_id > 0) ? static_cast<int>(inst_ptr->w_id) : 0;
                    const size_t out_sz = (n_experts > 0)
                        ? (input_data->value.size() / static_cast<size_t>(n_experts)) * static_cast<size_t>(inst_ptr->dim)
                        : static_cast<size_t>(inst_ptr->dim);
                    emit(make_zero(inst_ptr->out_id,  out_sz));
                    emit(make_zero(inst_ptr->out_id2, out_sz));
                } else if (inst_ptr->mode == VM::ROPE) {
                    emit(make_zero(inst_ptr->out_id, static_cast<size_t>(inst_ptr->dim)));
                } else {
                    const size_t out_sz = VectorOp<T>::output_elem_count(*inst_ptr, input_data);
                    auto out = make_zero(inst_ptr->out_id, out_sz);
                    // Multi-dimensional shapes (ADD_REDUCE/DOT_PRODUCT axis
                    // reduce, SPARSE_GATE's [chunk,n_experts], etc.) need
                    // assign_output_shape's real switch, not the flat
                    // {out_sz} make_zero defaults to.
                    VectorOp<T>::assign_output_shape(*inst_ptr, input_data, out);
                    emit(out);
                }
            }

            busy_cycle_ += static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS)) - cycle_start;
            // Clear busy
            auto busy_off = std::make_shared<LLM_SIM_BUSY::VectorBusy_t>(id_);
            busy_off->busy = false;
            vector_busy.write(busy_off);
        }
    }
}

template class VectorUnit<float>;
// template class VectorUnit<double>;
