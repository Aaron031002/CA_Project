#include "Core/TMU/TMU.h"

using namespace LLM_SIM_DATA;
using namespace LLM_SIM_BUSY;
// Use explicit using-declarations to avoid ambiguity with LLM_SIM_INST::TMU enum
using LLM_SIM_INST::TMUInstPtr_t;
using LLM_SIM_INST::CacheReqPtr_t;
using LLM_SIM_INST::CacheReq_t;

template <typename T>
TMU<T>::TMU(sc_module_name name, bool compute) : sc_module(name), compute_(compute) {
    tmu_op = new TMUOp<T>();
    SC_THREAD(run);
    sensitive << clk.pos();
    tmu_busy.initialize(std::make_shared<TMUBusy_t>());
}

template <typename T>
void TMU<T>::run() {
    while (true) {
        wait(); // wait for the next clock cycle
        if (inst_buffer_in.num_available() == 0) {
            continue; // if no instruction, skip this cycle
        }
        // read the instruction and input data
        TMUInstPtr_t inst_ptr;
        if (!inst_buffer_in.nb_read(inst_ptr)) {
            continue;
        }
        const uint64_t cycle_start =
            static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS));
        auto in_num = inst_ptr->input_ids.size();
        // Issue cache requests and drain returned data in a pipelined fashion.
        // This avoids deadlock when TMU fan-in is large (e.g. concat with 64 inputs)
        // and cache response FIFO depth is smaller.
        std::vector<DataPtr_t<T>> input_data(in_num, nullptr);
        size_t req_sent = 0;
        size_t resp_recv = 0;
        uint64_t idle_spins = 0;
        while (resp_recv < in_num) {
            bool progressed = false;

            if (req_sent < in_num) {
                const uint64_t new_id = (req_sent < inst_ptr->new_input_ids.size())
                                             ? inst_ptr->new_input_ids[req_sent] : 0;
                CacheReqPtr_t cache_req_ptr = std::make_shared<CacheReq_t>(
                    inst_ptr->input_ids[req_sent], inst_ptr->input_reserved[req_sent], new_id);
                if (cache_req_fifo.nb_write(cache_req_ptr)) {
                    ++req_sent;
                    progressed = true;
                }
            }

            DataPtr_t<T> data_ptr;
            if (input_fifo.nb_read(data_ptr)) {
                // Position-based, not id-content-based: Cache serializes this
                // port to one in-flight request at a time (a miss retries in
                // place, never lets a later request overtake), so responses
                // arrive in exactly the order requests were sent -- the
                // resp_recv-th response is unambiguously for the resp_recv-th
                // request. This also must NOT compare data_ptr->id against
                // input_ids[resp_recv]: a rename-on-terminal-read (new_id
                // above) mutates the delivered object's own id field, which
                // would break an id-content match here even though the
                // response is still correctly the resp_recv-th one. Also
                // fixes the pre-existing "same id repeated N times in one
                // fan-in list" deadlock the old id-content search was
                // vulnerable to.
                input_data[resp_recv] = data_ptr;
                ++resp_recv;
                progressed = true;
            }

            if (!progressed) {
                ++idle_spins;
                wait();
            } else {
                idle_spins = 0;
            }
        }

        // Execute the instruction or produce zero-filled transaction outputs.
        std::vector<DataPtr_t<T>> output_data;
        uint64_t latency = 0;
        if (compute_) {
            std::tie(output_data, latency) = tmu_op->execute(inst_ptr, input_data);
        } else {
            // Transaction-only: compute real latency without data movement,
            // zero-fill outputs using output_meta()'s shape/elem_bits (which
            // mirrors execute()'s own per-mode out->size_/elem_bits exactly)
            // so Cache's write-request mechanism prices this write the same
            // way it would for a compute_=true run.
            latency = tmu_op->latency_only(inst_ptr, input_data);
            auto meta = tmu_op->output_meta(inst_ptr, input_data);
            output_data.resize(meta.size());
            for (size_t i = 0; i < meta.size(); ++i) {
                auto out = std::make_shared<LLM_SIM_DATA::Data_t<T>>();
                int total = 1;
                for (int d : meta[i].shape) total *= d;
                out->value.assign(total, T{});
                out->size_ = meta[i].shape;
                out->elem_bits = meta[i].elem_bits;
                output_data[i] = out;
            }
        }
        // change busy state
        auto busy_on = std::make_shared<TMUBusy_t>();
        busy_on->busy = true;
        busy_on->in_data_ids = inst_ptr->input_ids;
        busy_on->out_data_ids = inst_ptr->output_ids;
        tmu_busy.write(busy_on);
        wait(latency * sc_time(1, SC_NS)); // wait for the latency of the operation
        // write the output data to the output FIFO and send cache reqs if needed
        for (size_t i = 0; i < output_data.size(); i++) {
            if (i < inst_ptr->output_ids.size()) {
                output_data[i]->id = inst_ptr->output_ids[i];
            }
            // Write-request-before-data: Cache must see this write's
            // shape/elem_bits before it ever reads the payload off
            // output_fifo, so the request always goes out first, same cycle.
            cache_wreq_out.write(LLM_SIM_DATA::make_cache_write_req(output_data[i]));
            output_fifo.write(output_data[i]);
        }
        // change busy state back to idle
        busy_cycle_ += static_cast<uint64_t>(sc_time_stamp() / sc_time(1, SC_NS)) - cycle_start;
        auto busy_off = std::make_shared<TMUBusy_t>();
        busy_off->busy = false;
        tmu_busy.write(busy_off);
    }
}   

template class TMU<int>;
template class TMU<float>;